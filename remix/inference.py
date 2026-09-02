#!/usr/bin/env python3
"""Canonical offline inference for the complete Remixer bundle.

Inspector supplies the active effect families. Remixer consumes a time-aligned
Clean/Wet pair and recovers their order plus named physical controls. Source
files are read-only; resampling and downmixing happen only on analysis copies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile
import torch
from scipy.signal import resample_poly

from .delay_model import DelayControlEstimator, apply_delay_residual
from .drive_model import DriveControlEstimator, apply_drive_estimate
from .model import PairedEstimator
from .order_search import decode_controls, rank_topologies, search_margin
from .physics import apply_delay_hints, deconvolution_context
from .quality import analysis_pair, checked_audio, checked_sample_rate
from .reverb_model import ReverbControlEstimator, apply_reverb_estimate, reverb_features
from .spec import KINDS, ChainSpec, ranked_topologies
from .train import RUN, device


INFERENCE_SCHEMA = 1
BUNDLE = RUN / "remixer-bundle.pt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_effects(value: str | Sequence[str]) -> tuple[str, ...]:
    values = tuple(
        item.strip().lower()
        for item in (value.split(",") if isinstance(value, str) else value)
        if item.strip()
    )
    if len(values) != len(set(values)):
        raise ValueError(f"active effect families must be unique, got {values}")
    unknown = [item for item in values if item not in KINDS]
    if unknown:
        raise ValueError(f"unknown effect families: {unknown}; expected {KINDS}")
    return values


def _analysis_copy(
    path: Path,
    target_rate: int,
    frames: int,
    offset_seconds: float,
) -> tuple[np.ndarray, dict]:
    audio, source_rate = soundfile.read(path, dtype="float32", always_2d=True)
    checked_audio(audio, name=str(path))
    source_channels = int(audio.shape[1])
    mono = audio.mean(axis=1, dtype=np.float32)
    if source_rate != target_rate:
        common = math.gcd(source_rate, target_rate)
        mono = resample_poly(
            mono,
            target_rate // common,
            source_rate // common,
        ).astype(np.float32)
    start = round(offset_seconds * target_rate)
    end = start + frames
    segment = mono[start:end]
    available = len(segment)
    if available == 0:
        raise ValueError(f"analysis offset is beyond the end of {path}")
    if available < frames:
        segment = np.pad(segment, (0, frames - available))
    return np.asarray(segment, dtype=np.float32), {
        "path": str(path),
        "sha256": sha256(path),
        "sample_rate": int(source_rate),
        "channels": source_channels,
        "frames": int(len(audio)),
        "analysis_frames_from_source": available,
    }


def load_analysis_pair(
    clean: Path,
    wet: Path,
    sample_rate: int,
    seconds: int,
    offset_seconds: float,
    *,
    preserve_levels: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    checked_sample_rate(sample_rate)
    if seconds <= 0 or offset_seconds < 0.0 or not math.isfinite(offset_seconds):
        raise ValueError("analysis seconds must be positive and offset must be finite/non-negative")
    frames = sample_rate * seconds
    dry, dry_source = _analysis_copy(clean, sample_rate, frames, offset_seconds)
    effected, wet_source = _analysis_copy(wet, sample_rate, frames, offset_seconds)
    if not preserve_levels:
        dry, effected = analysis_pair(dry, effected)
    denominator = float(np.linalg.norm(dry) * np.linalg.norm(effected))
    correlation = float(np.dot(dry, effected) / denominator) if denominator else 0.0
    return dry, effected, {
        "clean": dry_source,
        "wet": wet_source,
        "analysis_sample_rate": sample_rate,
        "analysis_seconds": seconds,
        "offset_seconds": offset_seconds,
        "alignment_policy": "shared time offset; no automatic shift or time stretch",
        "analysis_level_policy": "preserve-input-levels" if preserve_levels else "joint-peak-normalization",
        "zero_lag_correlation": correlation,
    }


class RemixerRuntime:
    def __init__(self, bundle_path: Path = BUNDLE, target: torch.device | None = None) -> None:
        self.bundle_path = bundle_path
        self.bundle_hash = sha256(bundle_path)
        self.target = device() if target is None else target
        payload = torch.load(bundle_path, map_location=self.target, weights_only=True)
        if payload.get("bundle_schema") != 1:
            raise ValueError(f"unsupported Remixer bundle schema: {payload.get('bundle_schema')}")
        quality = payload.get("audio_quality", {})
        required = {
            "source_audio_mutation": "forbidden",
            "automatic_normalization": False,
            "automatic_limiting": False,
            "lossy_reencoding": False,
            "bypass_max_absolute_error": 0.0,
        }
        if any(quality.get(name) != expected for name, expected in required.items()):
            raise ValueError("bundle does not satisfy the loss-preserving audio-quality contract")
        self.sample_rate = int(payload["sample_rate"])
        self.analysis_seconds = int(payload["analysis_seconds"])
        self.quality = quality
        self.order_search = payload["order_search"]
        self.main = PairedEstimator().to(self.target)
        self.drive = DriveControlEstimator().to(self.target)
        self.delay = DelayControlEstimator().to(self.target)
        self.reverb = ReverbControlEstimator().to(self.target)
        for name, model in (
            ("main", self.main),
            ("drive", self.drive),
            ("delay", self.delay),
            ("reverb", self.reverb),
        ):
            model.load_state_dict(payload["states"][name])
            model.eval()

    def _prepare_analysis_pair(self, dry: np.ndarray, wet: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Canonical legacy frontend; specialized audited runtimes may override."""
        return analysis_pair(dry, wet)

    def _model_analysis_batch(self,dry: np.ndarray,wet: np.ndarray) -> tuple[torch.Tensor,torch.Tensor]:
        return (torch.from_numpy(dry).unsqueeze(0).to(self.target),
                torch.from_numpy(wet).unsqueeze(0).to(self.target))

    @torch.no_grad()
    def infer(self, dry: np.ndarray, wet: np.ndarray, active: Sequence[str]) -> dict:
        topology = parse_effects(active)
        dry_copy, wet_copy = self._prepare_analysis_pair(dry, wet)
        expected = self.sample_rate * self.analysis_seconds
        if dry_copy.ndim != 1 or len(dry_copy) != expected:
            raise ValueError(f"expected {expected} mono analysis samples, got {dry_copy.shape}")
        if max(float(np.sqrt(np.mean(dry_copy * dry_copy))), float(np.sqrt(np.mean(wet_copy * wet_copy)))) < 1.0e-6:
            raise ValueError("analysis pair contains no usable signal")
        if not topology:
            return {
                "schema": INFERENCE_SCHEMA,
                "bundle_sha256": self.bundle_hash,
                "audio_quality": self.quality,
                "decision": "bypass",
                "active_effects": [],
                "chain": ChainSpec(()).document(),
            }

        dry_tensor,wet_tensor = self._model_analysis_batch(dry_copy,wet_copy)
        raw = self.main(dry_tensor, wet_tensor)
        context = deconvolution_context(dry_tensor, wet_tensor)
        estimate = apply_delay_hints(raw, dry_tensor, wet_tensor, context)
        impulse = reverb_features(dry_tensor, wet_tensor, context)
        estimate = apply_delay_residual(
            estimate, dry_tensor, wet_tensor, self.delay, impulse, context
        )
        estimate = apply_drive_estimate(estimate, dry_tensor, wet_tensor, self.drive)
        estimate = apply_reverb_estimate(
            estimate, dry_tensor, wet_tensor, self.reverb, impulse
        )
        order_logits = estimate.order_logits[0].cpu().numpy()
        controls = torch.sigmoid(estimate.control_logits[0]).cpu().numpy()
        classifier = ranked_topologies(topology, order_logits)[0][0]
        ranked = rank_topologies(
            dry_copy,
            wet_copy,
            topology,
            controls,
            self.sample_rate,
            tuple(self.order_search["renderers"]),
        )
        search = ranked[0][0]
        margin = search_margin(ranked)
        threshold = float(self.order_search["margin_threshold"])
        use_search = len(topology) > 1 and margin >= threshold
        selected = search if use_search else classifier
        errors = dict(ranked)
        chain = ChainSpec(decode_controls(selected, controls))
        chain.validate()
        return {
            "schema": INFERENCE_SCHEMA,
            "bundle_sha256": self.bundle_hash,
            "audio_quality": self.quality,
            "active_effects": list(topology),
            "decision": (
                "single_effect"
                if len(topology) == 1
                else ("dsp_search" if use_search else "classifier")
            ),
            "chain": chain.document(),
            "normalized_controls": controls.tolist(),
            "order": {
                "classifier": list(classifier),
                "search": list(search),
                "selected": list(selected),
                "search_margin": margin if len(ranked) > 1 else None,
                "margin_threshold": threshold,
                "classifier_reconstruction_error": errors[classifier],
                "selected_reconstruction_error": errors[selected],
                "ranked": [
                    {"topology": list(candidate), "reconstruction_error": error}
                    for candidate, error in ranked
                ],
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--wet", type=Path, required=True)
    parser.add_argument(
        "--effects",
        required=True,
        help="comma-separated active families supplied by Inspector: drive,delay,reverb",
    )
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument("--offset-seconds", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    runtime = RemixerRuntime(args.bundle)
    dry, wet, input_report = load_analysis_pair(
        args.clean,
        args.wet,
        runtime.sample_rate,
        runtime.analysis_seconds,
        args.offset_seconds,
    )
    report = runtime.infer(dry, wet, parse_effects(args.effects))
    report["input"] = input_report
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
