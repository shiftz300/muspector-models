"""Paired Clean/Wet runtime for the packaged routed family detector."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import onnxruntime
import torch

from .quality import checked_audio


RATE = 44_100
FRAMES = 220_500
FFT = 2_048
HOP = 1_024
MELS = 128
MEL_FRAMES = 216
FAMILIES = ("drive", "delay", "reverb")
FLOORS = np.asarray((0.05, 0.55, 0.59), dtype=np.float64)
SCALES = {
    "drive_delay": np.asarray((1.556447149, 0.481221408, 1.188732624)),
    "reverb": np.asarray((2.313131571, 0.506870329, 1.665598392)),
}
BIASES = {
    "drive_delay": np.asarray((-3.630986691, -0.329666764, -4.136132240)),
    "reverb": np.asarray((-0.863560617, -0.273496419, -7.076041698)),
}
VERIFIER_SCALE = 0.3535915296757063
VERIFIER_BIAS = -0.18368981986045507
VERIFIER_THRESHOLD = 0.342
FILES = (
    "drive-delay-encoder.onnx",
    "drive-delay-head.onnx",
    "reverb-encoder.onnx",
    "reverb-head.onnx",
    "reverb-verifier.onnx",
    "routed-device-profile.bin",
    "README.md",
    "LICENSES.md",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sigmoid(value: np.ndarray | float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))


def _top_two(values: list[float]) -> float:
    selected = sorted(values, reverse=True)[:2]
    return float(sum(selected) / len(selected))


def _margin(active: bool, score: float, threshold: float) -> float:
    return score - threshold if active else threshold - score


def _filters(device: torch.device) -> torch.Tensor:
    lower = 2595.0 * math.log10(1.0 + 30.0 / 700.0)
    upper = 2595.0 * math.log10(1.0 + 16_000.0 / 700.0)
    points = torch.linspace(lower, upper, MELS + 2, device=device)
    frequencies = 700.0 * (10.0 ** (points / 2595.0) - 1.0)
    bins = torch.linspace(0.0, RATE / 2.0, FFT // 2 + 1, device=device)
    result = torch.zeros(MELS, FFT // 2 + 1, device=device)
    for band in range(MELS):
        left, center, right = frequencies[band : band + 3]
        rise = (bins - left) / (center - left)
        fall = (right - bins) / (right - center)
        result[band] = torch.minimum(rise, fall).clamp_min(0.0)
        result[band] *= 2.0 / (right - left)
    return result


@torch.inference_mode()
def frontend(audio: np.ndarray) -> np.ndarray:
    value = checked_audio(audio, name="family analysis copy")
    if value.ndim != 1 or len(value) != FRAMES:
        raise ValueError(f"expected {FRAMES} mono samples, got {value.shape}")
    waveform = torch.from_numpy(value.copy()).unsqueeze(0)
    window = torch.hann_window(FFT, periodic=True)
    spectrum = torch.stft(
        waveform,
        n_fft=FFT,
        hop_length=HOP,
        win_length=FFT,
        window=window,
        center=True,
        pad_mode="constant",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    power = spectrum.abs().square()
    mel = torch.matmul(_filters(waveform.device), power).clamp_min(1.0e-10)
    mel = 10.0 * torch.log10(mel)
    peak = mel.amax(dim=(1, 2), keepdim=True)
    mel = torch.maximum(mel, peak - 80.0)
    mean = mel.mean(dim=(1, 2), keepdim=True)
    deviation = mel.std(dim=(1, 2), keepdim=True).clamp_min(1.0e-5)
    normalized = ((mel - mean) / deviation).unsqueeze(1)
    if tuple(normalized.shape) != (1, 1, MELS, MEL_FRAMES):
        raise ValueError(f"family frontend geometry changed: {tuple(normalized.shape)}")
    return normalized.numpy().astype(np.float32, copy=False)


def _session(path: Path) -> onnxruntime.InferenceSession:
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    return onnxruntime.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )


class FamilyRuntime:
    def __init__(self, root: Path, manifest: Path):
        self.root = Path(root).resolve()
        self.manifest_path = Path(manifest).resolve()
        package = json.loads(self.manifest_path.read_text())
        if package.get("id") != "family" or package.get("version") != "1.0.0":
            raise ValueError("unsupported family package")
        artifacts = {row["path"]: row for row in package.get("artifacts", [])}
        if set(artifacts) != set(FILES):
            raise ValueError("family artifact set differs")
        self.hashes = {}
        for name in FILES:
            path = self.root / name
            row = artifacts[name]
            if path.stat().st_size != row["bytes"] or _sha256(path) != row["sha256"]:
                raise ValueError(f"family artifact provenance mismatch: {name}")
            self.hashes[name] = row["sha256"]
        self.manifest_sha256 = _sha256(self.manifest_path)
        self.sessions = {
            "drive_delay_encoder": _session(self.root / "drive-delay-encoder.onnx"),
            "drive_delay_head": _session(self.root / "drive-delay-head.onnx"),
            "reverb_encoder": _session(self.root / "reverb-encoder.onnx"),
            "reverb_head": _session(self.root / "reverb-head.onnx"),
            "verifier": _session(self.root / "reverb-verifier.onnx"),
        }

    def _encode(self, branch: str, mel: np.ndarray) -> np.ndarray:
        embedding, logits = self.sessions[f"{branch}_encoder"].run(None, {"mel": mel})
        query = np.concatenate((embedding[0], logits[0])).astype(np.float32, copy=False)
        if query.shape != (259,):
            raise ValueError("family encoder output geometry changed")
        return query

    def _probabilities(
        self, branch: str, query: np.ndarray, mean: np.ndarray, deviation: np.ndarray
    ) -> np.ndarray:
        features = np.concatenate(
            (query, mean, query - mean, np.abs(query - mean), deviation)
        ).astype(np.float32, copy=False)[None]
        logits = self.sessions[f"{branch}_head"].run(
            None, {"relative_features": features}
        )[0][0]
        return _sigmoid(logits.astype(np.float64) * SCALES[branch] + BIASES[branch])

    def infer(self, dry: np.ndarray, wet: np.ndarray) -> dict:
        dry_copy = checked_audio(dry, name="dry family copy")
        wet_copy = checked_audio(wet, name="wet family copy")
        if dry_copy.shape != wet_copy.shape:
            raise ValueError("family pair geometry differs")
        before = hashlib.sha256(dry_copy.tobytes() + wet_copy.tobytes()).hexdigest()
        clean_mel, wet_mel = frontend(dry_copy), frontend(wet_copy)
        clean = {
            branch: self._encode(branch, clean_mel)
            for branch in ("drive_delay", "reverb")
        }
        profile = {
            branch: (query.copy(), np.full(query.shape, 1.0e-4, dtype=np.float32))
            for branch, query in clean.items()
        }
        self_scores = {
            branch: self._probabilities(branch, clean[branch], *profile[branch])
            for branch in profile
        }
        thresholds = FLOORS.copy()
        thresholds[:2] = np.clip(self_scores["drive_delay"][:2] + 0.02, FLOORS[:2], 0.95)
        thresholds[2] = float(
            np.clip(self_scores["reverb"][2] + 0.02, FLOORS[2], 0.95)
        )
        wet_queries = {
            branch: self._encode(branch, wet_mel)
            for branch in ("drive_delay", "reverb")
        }
        drive_delay = self._probabilities(
            "drive_delay", wet_queries["drive_delay"], *profile["drive_delay"]
        )
        reverb = self._probabilities(
            "reverb", wet_queries["reverb"], *profile["reverb"]
        )
        bands = wet_mel.reshape(MELS, MEL_FRAMES).reshape(8, 16, MEL_FRAMES).mean(1)
        verifier_logit = self.sessions["verifier"].run(
            None,
            {
                "temporal_bands": bands.astype(np.float32, copy=False)[None],
                "pair_probabilities": reverb.astype(np.float32, copy=False)[None],
            },
        )[0][0]
        verifier = float(_sigmoid(float(verifier_logit) * VERIFIER_SCALE + VERIFIER_BIAS))
        raw = np.asarray((drive_delay[0], drive_delay[1], reverb[2]), dtype=np.float64)
        detected = raw >= thresholds
        detected[2] = bool(detected[2] and verifier >= VERIFIER_THRESHOLD)
        scores = raw.copy()
        scores[2] = min(scores[2], verifier)
        margins = np.empty(3, dtype=np.float64)
        margins[:2] = [
            _margin(bool(detected[index]), float(raw[index]), float(thresholds[index]))
            for index in range(2)
        ]
        if detected[2]:
            margins[2] = min(raw[2] - thresholds[2], verifier - VERIFIER_THRESHOLD)
        else:
            margins[2] = max(thresholds[2] - raw[2], VERIFIER_THRESHOLD - verifier, 0.0)
        unchanged = before == hashlib.sha256(dry_copy.tobytes() + wet_copy.tobytes()).hexdigest()
        return {
            "schema": 1,
            "active": [name for name, value in zip(FAMILIES, detected) if value],
            "scores": dict(zip(FAMILIES, map(float, scores))),
            "thresholds": dict(zip(FAMILIES, map(float, thresholds))),
            "margins": dict(zip(FAMILIES, map(float, margins))),
            "minimum_margin": float(margins.min()),
            "reverb_verifier": verifier,
            "manifest_sha256": self.manifest_sha256,
            "source_audio_modified": not unchanged,
            "physical_audio_devices_used": False,
        }

    def assert_artifacts_unchanged(self) -> None:
        if _sha256(self.manifest_path) != self.manifest_sha256:
            raise ValueError("family manifest changed")
        for name, digest in self.hashes.items():
            if _sha256(self.root / name) != digest:
                raise ValueError(f"family artifact changed: {name}")
