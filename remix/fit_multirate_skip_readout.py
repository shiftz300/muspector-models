"""Provisional frozen-core all-layer skip readout; no model admission.

Fits zero-anchored additive projections of all ten zero-origin block updates,
not gain on the source output. Uses fit234 only, deterministic stride16 samples
plus each fit Wet peak's +/-128 frames, with true preceding features for pre95.
All54 calibration recordings are scored completely. No feature/audio cache.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .asrnn_effects import read_effect_pair
from .evaluate_multirate_fuzz import load_candidate
from .fit_multirate_readout import anchored_solution, summarize
from .precheck_multirate_fuzz import sha256
from .train_dfz_core_tcn import selection_key


RIDGES = (1e-4, 1e-3, 1e-2, .1)


def fit_positions(target, phase):
    peak = int(np.argmax(np.abs(target)))
    event = np.arange(max(1, peak-128), min(len(target), peak+129))
    selected = np.union1d(np.arange(1+phase, len(target), 16), event)
    positions = np.union1d(selected, selected-1)
    # Deliberately stated fit supervision, not an unbiased full-wave estimate:
    # uniform representatives have16x weight, peak-near frames32x.
    priority = np.where(np.abs(selected-peak) <= 128, 32., 16.)
    return selected, positions, priority


@torch.inference_mode()
def fit(source, output):
    if output.exists():
        raise ValueError("new all-layer readout screen required")
    model, payload, parent = load_candidate(source)
    if model.audio.width != 32 or len(model.audio.projections) != 10:
        raise ValueError("ten32-channel zero-origin updates required")
    source_hash, parent_hash = sha256(source), sha256(source.with_name("metrics.json"))
    names = (Path(__file__).name, "evaluate_multirate_fuzz.py", "fit_multirate_readout.py", "train_dfz_core_tcn.py", "asrnn_effects.py")
    sources = {**parent["source_sha256"], **{name: sha256(Path(__file__).parent / name) for name in names}}
    provenance = parent["audio_provenance"]
    if set(provenance) != {"fit", "calibration"} or len(provenance["fit"]) != 234 or len(provenance["calibration"]) != 54:
        raise ValueError("original234/54 partition required")

    def unchanged():
        if sha256(source) != source_hash or sha256(source.with_name("metrics.json")) != parent_hash:
            raise ValueError("frozen source changed")
        if any(sha256(Path(__file__).parent / name) != digest for name, digest in sources.items()):
            raise ValueError("feature/fit source changed")
        if any(sha256(row["path"]) != row["sha256"] for rows in provenance.values() for row in rows):
            raise ValueError("original audio changed")

    def read(record, split, phase=0):
        path = Path(record["path"])
        if "train" not in path.parts or (int(path.stem.split(",")[-1]) % 5 == 0) != (split == "calibration"):
            raise ValueError("no eval or changed partition allowed")
        dry, wet, knobs = read_effect_pair(path, "dfz")
        target = wet[1024:]
        if split == "fit":
            selected, positions, priority = fit_positions(target, phase)
        else:
            selected = positions = np.arange(len(target))
            priority = None
        captured, handles = [], []
        for projection in model.audio.projections:
            handles.append(projection.register_forward_pre_hook(lambda module, args:
                captured.append(args[0][0, :, positions+1024].T.contiguous().numpy())))
        try:
            original = model(torch.from_numpy(dry)[None], torch.from_numpy(knobs)[None])[0][0, 1024:].numpy()
        finally:
            for handle in handles:
                handle.remove()
        if len(captured) != 10 or any(value.shape != (len(positions), 32) for value in captured):
            raise ValueError("all-layer feature geometry differs")
        features = np.concatenate(captured, axis=1)
        a, b = (knobs*2).astype(int)
        return features, target, original, selected, positions, priority, int(a*3+b), int(a*50)

    unchanged()
    started = time.perf_counter()
    gram, rhs, counts = np.zeros((9, 320, 320)), np.zeros((9, 320)), np.zeros(9, dtype=int)
    for index, record in enumerate(provenance["fit"]):
        features, target, original, selected, positions, priority, corner, _ = read(record, "fit", index%16)
        current, previous = np.searchsorted(positions, selected), np.searchsorted(positions, selected-1)
        x, previous_x = features[current].astype(np.float64), features[previous].astype(np.float64)
        y, p = target.astype(np.float64), original.astype(np.float64)
        weights = priority*(1+4*(np.abs(y[selected])/max(float(np.max(np.abs(y))), 1e-5))**4)
        weights /= len(y)*max(float(np.mean(y*y)), 1e-5)
        gram[corner] += x.T @ (x*weights[:, None])
        rhs[corner] += x.T @ ((y[selected]-p[selected])*weights)
        xp = x-.95*previous_x
        yp, pp = y[selected]-.95*y[selected-1], p[selected]-.95*p[selected-1]
        all_pre = y[1:]-.95*y[:-1]
        weights = .1*priority/((len(y)-1)*max(float(np.mean(all_pre*all_pre)), 1e-5))
        gram[corner] += xp.T @ (xp*weights[:, None])
        rhs[corner] += xp.T @ ((yp-pp)*weights)
        counts[corner] += 1
        if (index+1)%26 == 0:
            print(json.dumps({"fit_files": index+1, "elapsed_seconds": time.perf_counter()-started}), flush=True)
    if counts.tolist() != [26]*9:
        raise ValueError("fit corners incomplete")
    zero = np.zeros(320)
    menus = [("unchanged", None, np.zeros((9, 320), dtype=np.float32))]
    for mode in ("shared", "nine-corner"):
        for ridge in RIDGES:
            weights = (np.tile(anchored_solution(gram.sum(0), rhs.sum(0), zero, ridge), (9, 1)) if mode == "shared" else
                       np.stack([anchored_solution(a, b, zero, ridge) for a, b in zip(gram, rhs)]))
            if not np.isfinite(weights).all():
                raise ValueError("nonfinite skip projection")
            menus.append((mode, ridge, weights.astype(np.float32)))
    targets, controls, predictions = [], [], [[] for _ in menus]
    for index, record in enumerate(provenance["calibration"]):
        features, target, original, _, _, _, corner, first_control = read(record, "calibration")
        values = original[:, None]+features @ np.stack([weights[corner] for _, _, weights in menus], axis=1)
        targets.append(target)
        controls.append(first_control)
        for column in range(len(menus)):
            predictions[column].append(values[:, column].copy())
        if (index+1)%9 == 0:
            print(json.dumps({"calibration_files": index+1}), flush=True)
    history = [{"mode": mode, "ridge": ridge, "provisional_calibration": summarize(values, targets, controls)}
               for (mode, ridge, _), values in zip(menus, predictions)]
    selected = min(range(len(history)), key=lambda index: selection_key(history[index]["provisional_calibration"]))
    unchanged()
    output.mkdir(parents=True)
    report = {"schema": 1, "purpose": "all-layer-skip-readout-screen-not-model-admission", "history": history,
              "selected": selected, "selection": history[selected], "parent_checkpoint_sha256": source_hash,
              "parent_report_sha256": parent_hash, "parent_architecture": payload["architecture"],
              "source_sha256": sources, "audio_provenance": provenance, "source_and_audio_reverified": True,
              "sampling": "fit-only stride16 rotating phase plus Wet-peak+/-128; priority16/32; true preceding feature for pre95",
              "features": "all ten32-channel block updates BEFORE residual projections; no source-output gain feature",
              "fit_files_per_corner": counts.tolist(), "full_model_replay_required": True, "features_cached_to_disk": False,
              "clip_specific_parameters": False, "physical_audio_devices_used": False, "official_eval_opened": False,
              "source_audio_modified": False, "automatic_gain_or_normalization": False, "admitted": False,
              "elapsed_seconds": time.perf_counter()-started}
    if selected:
        path = output / "candidate-skip.pt"
        torch.save({"experimental_schema": 1, "purpose": "unadmitted-all-layer-readout-only", "parent_checkpoint_sha256": source_hash,
                    "weight": torch.from_numpy(menus[selected][2].reshape(9, 10, 32)), "bias": False, "mode": menus[selected][0]}, path)
        report["readout_sha256"] = sha256(path)
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"selected": selected, "selection": report["selection"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    fit(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
