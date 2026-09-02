"""Overlap-add arbitrary-length runtime for independent real inverse stages."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .real_chain_nl import load_drive, load_reverb, stage


KINDS = ("drive", "reverb")


class Runtime:
    def __init__(self, drive: Path, reverb: Path, frame: int = 32768, hop: int = 16384) -> None:
        if frame <= 0 or hop <= 0 or frame != 2 * hop:
            raise ValueError("runtime requires a positive 50-percent-overlap frame")
        self.drive = load_drive(drive)
        self.reverb = load_reverb(reverb)
        if int(self.drive["frames"]) != frame or int(self.reverb["frames"]) != frame:
            raise ValueError("runtime model frame geometry differs")
        self.frame, self.hop = frame, hop
        index = np.arange(frame, dtype=np.float64)
        self.window = np.square(np.sin(np.pi * (index + 0.5) / frame)).astype(np.float32)

    def one(self, audio: np.ndarray, kind: str) -> np.ndarray:
        source = np.asarray(audio, dtype=np.float32)
        if source.ndim != 1 or not len(source) or not np.isfinite(source).all():
            raise ValueError("runtime expects finite non-empty mono audio")
        if kind not in KINDS:
            raise ValueError(f"unsupported inverse stage: {kind}")
        left = self.hop
        right = max(self.hop, self.frame - len(source) - left)
        padded_length = len(source) + left + right
        right += (-(padded_length - self.frame)) % self.hop
        mode = "reflect" if len(source) > 1 else "edge"
        padded = np.pad(source, (left, right), mode=mode)
        output = np.zeros(len(padded), dtype=np.float64)
        weight = np.zeros(len(padded), dtype=np.float64)
        for start in range(0, len(padded) - self.frame + 1, self.hop):
            restored = stage(padded[start : start + self.frame], kind, self.drive, self.reverb)
            output[start : start + self.frame] += restored * self.window
            weight[start : start + self.frame] += self.window
        result = np.asarray(output[left : left + len(source)] / np.maximum(weight[left : left + len(source)], 1.0e-8), dtype=np.float32)
        if result.shape != source.shape or not np.isfinite(result).all():
            raise FloatingPointError("runtime output geometry or finiteness failed")
        return result

    def run(self, audio: np.ndarray, forward: tuple[str, ...]) -> np.ndarray:
        if len(forward) != len(set(forward)) or not set(forward) <= set(KINDS):
            raise ValueError(f"unsupported or repeated forward order: {forward}")
        source = np.asarray(audio, dtype=np.float32)
        if not forward:
            return source.copy()
        value = source
        for kind in reversed(forward):
            value = self.one(value, kind)
        return value
