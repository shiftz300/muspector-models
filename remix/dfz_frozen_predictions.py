"""One reusable, signed, full-prefix DFZ prediction-only cache.

Only immutable float32 core predictions are persisted. Dry and Wet audio stay
in the original corpus; source weights, code, all audio bytes and prediction
tensor bytes are checked before any saved cache is reused.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import torch

from .asrnn_effects import effect_controls, effect_files, read_effect_pair
from .fit_asrnn_effect_output import _partition
from .stable_effect import load_stable_effect


SOURCE = Path("remix/runs/dfz-nonlinear-readout-phase9/candidate.pt")
CACHE = Path("remix/runs/dfz-frozen-predictions-phase9.pt")
FRAMES, MAX_CACHE_BYTES = 144_000, 166_000_000
CODE_NAMES = ("dfz_frozen_predictions.py", "stable_effect.py", "stable_nonlinear_readout.py",
              "asrnn_effects.py", "asrnn_data.py", "fit_asrnn_effect_output.py")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(prediction):
    if prediction.device.type != "cpu" or prediction.dtype != torch.float32 or not prediction.is_contiguous():
        raise ValueError("prediction checksum requires contiguous CPU float32")
    # A buffer view avoids allocating another 166 MB copy for hashing.
    return hashlib.sha256(memoryview(prediction.numpy()).cast("B")).hexdigest()


def partitions():
    fit, calibration = _partition(effect_files(Path("data/corpus/asrnn-physical-effects"), "dfz", "train"))
    corners = {(first, second) for first in (0., .5, 1.) for second in (0., .5, 1.)}
    for paths, per_corner in ((fit, 26), (calibration, 6)):
        groups = Counter(tuple(float(value) for value in effect_controls(path, "dfz")) for path in paths)
        if set(groups) != corners or set(groups.values()) != {per_corner}:
            raise ValueError("DFZ requires fixed fit234/cal54 and 9 equal-sized control groups")
    return fit, calibration


def fast_core_block(base, dry, controls, state=None):
    """Exact grid-only schema7 branch, carrying the complete recurrent prefix."""
    hidden, state = base.base.encode(dry, controls, state)
    original = base.base.output_layer(hidden).squeeze(-1) + base.readout.at_training_knots(hidden, controls)
    return original, state


def prediction_signature(fit, calibration):
    return {"schema": 1, "source": str(SOURCE), "source_sha256": file_sha256(SOURCE),
            "extractor_code_sha256": {name: file_sha256(Path(__file__).parent / name) for name in CODE_NAMES},
            "fast_core_block_sha256": hashlib.sha256(inspect.getsource(fast_core_block).encode()).hexdigest(),
            "partitions": {name: [{"path": str(path), "sha256": file_sha256(path)} for path in paths]
                           for name, paths in (("fit", fit), ("calibration", calibration))},
            "frames_per_recording": FRAMES, "prediction_dtype": "float32", "sample_rate": 48_000,
            "full_causal_prefix_from_sample": 0, "recurrent_state_reset": "recording-start-only",
            "controls": "exact-3x3-audited-training-knots", "extractor_batch_size": 4,
            "extractor_block_frames": 8_192, "compute": "mps", "cpu_threads": torch.get_num_threads(),
            "torch": str(torch.__version__), "numpy": str(np.__version__),
            "wet_used_as_model_input": False, "automatic_normalization": False}


def verify_payload(payload, expected):
    if set(payload) != {"schema", "signature", "predictions", "prediction_sha256"} or payload["schema"] != 1:
        raise ValueError("prediction-only cache schema differs or contains extra data")
    if payload["signature"] != expected:
        raise ValueError("frozen prediction cache source/audio/code/numeric signature differs")
    prediction = payload["predictions"]
    count = sum(len(rows) for rows in expected["partitions"].values())
    if (not isinstance(prediction, torch.Tensor) or prediction.dtype != torch.float32
            or prediction.device.type != "cpu" or not prediction.is_contiguous()
            or prediction.shape != (count, expected["frames_per_recording"])):
        raise ValueError("prediction-only cache tensor geometry differs")
    if not torch.isfinite(prediction).all() or tensor_sha256(prediction) != payload["prediction_sha256"]:
        raise ValueError("prediction-only cache tensor is nonfinite or corrupted")
    return prediction


@torch.inference_mode()
def compute_predictions(base, paths, target):
    predictions = torch.empty(len(paths), FRAMES, dtype=torch.float32)
    for offset in range(0, len(paths), 4):
        selected = paths[offset:offset + 4]
        pairs = [read_effect_pair(path, "dfz") for path in selected]
        if any(len(dry) != FRAMES for dry, _, _ in pairs):
            raise ValueError("DFZ prediction cache requires complete 144000-frame recordings")
        dry = torch.from_numpy(np.stack([pair[0] for pair in pairs]))
        controls = torch.from_numpy(np.stack([pair[2] for pair in pairs])).to(target)
        if not torch.equal(controls * 2, (controls * 2).round()):
            raise ValueError("fast core path is only valid at exact training knots")
        state = None
        for start in range(0, FRAMES, 8_192):
            original, state = fast_core_block(base, dry[:, start:start + 8_192].to(target), controls, state)
            predictions[offset:offset + len(selected), start:start + original.shape[1]].copy_(original.cpu())
        if offset % 32 == 0 or offset + len(selected) == len(paths):
            print(json.dumps({"stage": "prediction-only-full-prefix-cache", "completed": offset + len(selected),
                              "total": len(paths)}), flush=True)
    return predictions


def load_or_create_predictions(base, fit, calibration, expected, cache=CACHE):
    cache = Path(cache)
    if prediction_signature(fit, calibration) != expected:
        raise ValueError("source/audio/code changed before cache access")
    created = not cache.exists()
    if created:
        predictions = compute_predictions(base, [*fit, *calibration], torch.device("mps"))
        if prediction_signature(fit, calibration) != expected:
            raise ValueError("source/audio/code changed during prediction extraction")
        payload = {"schema": 1, "signature": expected, "predictions": predictions,
                   "prediction_sha256": tensor_sha256(predictions)}
        verify_payload(payload, expected)
        cache.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation never overwrites an existing or interrupted cache.
        # An interrupted file fails loading rather than silently being reused.
        with cache.open("xb") as handle:
            torch.save(payload, handle)
    if cache.stat().st_size > MAX_CACHE_BYTES:
        raise ValueError("prediction-only cache exceeds the 166 MB bound")
    # Loading maps the tensor, avoiding a second full copy during validation.
    payload = torch.load(cache, map_location="cpu", weights_only=True, mmap=True)
    predictions = verify_payload(payload, expected)
    digest = file_sha256(cache)
    if prediction_signature(fit, calibration) != expected:
        raise ValueError("source/audio/code changed while reading prediction cache")
    return predictions, {"path": str(cache), "sha256": digest, "bytes": cache.stat().st_size,
                         "prediction_sha256": payload["prediction_sha256"], "created": created,
                         "contains_dry_or_wet": False}


def load_training_rows(paths, predictions):
    """Audio is only read from the signed corpus; p views add no second cache."""
    if len(paths) != len(predictions):
        raise ValueError("prediction cache rows do not match the audio manifest")
    rows = []
    for index, path in enumerate(paths):
        dry, wet, controls = read_effect_pair(path, "dfz")
        if len(dry) != FRAMES:
            raise ValueError("DFZ complete recording geometry differs")
        active = np.flatnonzero(np.abs(dry) > max(1e-4, float(np.abs(dry).max()) * .01))
        rows.append({"dry": torch.from_numpy(dry), "wet": torch.from_numpy(wet),
                     "original": predictions[index], "controls": torch.from_numpy(controls),
                     "attack": int(path.stem.split(",")[0]), "path": str(path),
                     "onset": int(active[0]) if len(active) else 1_024,
                     "peak_index": int(np.abs(wet[1_024:]).argmax()) + 1_024})
    return rows
