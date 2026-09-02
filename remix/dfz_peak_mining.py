"""Fit-only current-prediction peak mining; not a trainer or runtime schema.

Call only after verifying the immutable p cache with dfz_frozen_predictions.
The caller pauses optimizer updates during each call.  Mining freezes a copy
of the completed readout, never runs the core, and persists no audio/weights.
Wet is read only for diagnostic reductions, never for the current-prediction
event indices or features. Returned detached indices select ordinary
same-index Wet-supervised windows in a separately authorized trainer.
"""
from __future__ import annotations

from collections import Counter
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .dfz_dynamic_readout import CausalTapBank


REFRESH_STEPS = tuple(range(0, 4_000, 250))
AUDIT_STEP = 4_000
MAX_BATCH_SIZE, MAX_BLOCK_FRAMES = 9, 4_096


def _path_fields(path):
    try:
        blend, filter_value, take = (int(value) for value in Path(path).stem.split(","))
    except (TypeError, ValueError) as error:
        raise ValueError("DFZ path must encode blend,filter,take") from error
    if blend not in (0, 50, 100) or filter_value not in (0, 50, 100) or take <= 0:
        raise ValueError("unexpected audited DFZ controls or take")
    return blend, filter_value, take


def module_sha256(module):
    """Named shape/dtype/content SHA, including the fit-only RMS buffer."""
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        if not torch.isfinite(value).all():
            raise ValueError("nonfinite readout snapshot")
        header = json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":"))
        digest.update(header.encode() + b"\0")
        digest.update(memoryview(value.numpy()).cast("B"))
    return digest.hexdigest()


def _check_optimizer_step(readout, optimizer, completed_updates):
    parameters = list(readout.parameters())
    optimized = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    if len({id(p) for p in optimized}) != len(optimized) or {id(p) for p in parameters} != {id(p) for p in optimized}:
        raise ValueError("optimizer must own exactly this readout's parameters")
    for parameter in parameters:
        state = optimizer.state.get(parameter, {})
        if completed_updates == 0:
            if state:
                raise ValueError("step-zero snapshot requires an unused optimizer")
        else:
            counter = state.get("step")
            if counter is None or float(counter) != completed_updates:
                raise ValueError("snapshot must follow the exact completed optimizer update")


@dataclass(frozen=True)
class PredictedExtrema:
    path: str
    positive_index: int
    positive_value: float
    negative_index: int
    negative_value: float
    wet_positive_index: int
    wet_positive_value: float
    wet_negative_index: int
    wet_negative_value: float
    squared_error_sum: float
    wet_energy_sum: float
    scored_frames: int
    absolute_peak_error: float
    file_esr: float


@dataclass(frozen=True)
class PeakMiningResult:
    completed_updates: int
    purpose: str
    readout_sha256: str
    source_sha256: str
    prediction_sha256: str
    frames_per_recording: int
    events: tuple[PredictedExtrema, ...]
    metrics: dict

    def metadata(self):
        """Only indices, scalar fit metrics and provenance; no tensors/audio."""
        return asdict(self)


def _fit_metrics(events):
    peaks = np.asarray([event.absolute_peak_error for event in events], dtype=np.float64)
    esr = np.asarray([event.file_esr for event in events], dtype=np.float64)
    ratios = np.asarray([max(abs(event.positive_value), abs(event.negative_value)) /
                         max(abs(event.wet_positive_value), abs(event.wet_negative_value), 1e-8)
                         for event in events], dtype=np.float64)
    by_blend = {}
    for blend in (0, 50, 100):
        group = [event for event in events if _path_fields(event.path)[0] == blend]
        by_blend[str(blend)] = {"files": len(group), "mean_esr": float(np.mean([e.file_esr for e in group])),
                               "absolute_peak_error_p95": float(np.quantile([e.absolute_peak_error for e in group], .95))}
    squared_error_sum = sum(event.squared_error_sum for event in events)
    wet_energy_sum = sum(event.wet_energy_sum for event in events)
    return {"examples": len(events), "domain": "fit-only-diagnostic-not-admission",
            "reduction": "streamed-float64-CPU-over-float32-current-predictions",
            "squared_error_sum": squared_error_sum, "wet_energy_sum": wet_energy_sum,
            "global_esr": squared_error_sum / max(wet_energy_sum, 1e-12),
            "mean_per_file_esr": float(esr.mean()), "p95_per_file_esr": float(np.quantile(esr, .95)),
            "absolute_peak_error_p95": float(np.quantile(peaks, .95)),
            "worst_attack_absolute_peak_error_p95": max(row["absolute_peak_error_p95"] for row in by_blend.values()),
            "peak_ratio_p05": float(np.quantile(ratios, .05)), "peak_ratio_median": float(np.median(ratios)),
            "peak_ratio_p95": float(np.quantile(ratios, .95)), "by_blend": by_blend}


class FitPeakMiner:
    """Strict fit order, completed-update schedule, and per-file sign rotation.

    `signature` is the freshly verified cache signature; `prediction_sha256`
    comes from the verified p payload.  Rows must be the matching fit-only
    views produced from that cache and original signed corpus.  This helper
    does not replace the caller's content verification of those arrays.
    """
    def __init__(self, signature, prediction_sha256):
        if not isinstance(prediction_sha256, str) or len(prediction_sha256) != 64:
            raise ValueError("verified prediction content SHA required")
        self.source_sha256 = signature["source_sha256"]
        self.prediction_sha256 = prediction_sha256
        self.frames = int(signature["frames_per_recording"])
        if self.frames <= 1_024 or len(self.source_sha256) != 64:
            raise ValueError("invalid verified source/frame metadata")
        self.paths = tuple(row["path"] for row in signature["partitions"]["fit"])
        self.cal_paths = frozenset(row["path"] for row in signature["partitions"]["calibration"])
        if len(self.paths) != 234 or len(set(self.paths)) != 234 or len(self.cal_paths) != 54 or set(self.paths) & self.cal_paths:
            raise ValueError("exact disjoint fit234/cal54 manifest required")
        groups = Counter()
        for path in self.paths:
            blend, filter_value, take = _path_fields(path)
            if take % 5 == 0:
                raise ValueError("calibration take cannot enter fit mining")
            groups[(blend, filter_value)] += 1
        if len(groups) != 9 or set(groups.values()) != {26}:
            raise ValueError("nine fit corners with26 recordings each required")
        if any(_path_fields(path)[2] % 5 != 0 for path in self.cal_paths):
            raise ValueError("unexpected calibration partition")
        self._next_refresh = 0
        self._audited = False
        self._active = None
        self._events = {}
        self._visits = {path: 0 for path in self.paths}

    @property
    def active_readout_sha256(self):
        return None if self._active is None else self._active.readout_sha256

    def _validate_rows(self, rows):
        actual = tuple(str(row["path"]) for row in rows)
        if actual != self.paths:
            raise ValueError("mining rows must exactly match ordered signed fit manifest; no cal/reordering")
        for path, row in zip(self.paths, rows):
            x, p, wet, controls = row["dry"], row["original"], row["wet"], row["controls"]
            if (not isinstance(x, torch.Tensor) or not isinstance(p, torch.Tensor)
                    or not isinstance(wet, torch.Tensor)
                    or x.device.type != "cpu" or p.device.type != "cpu" or wet.device.type != "cpu"
                    or x.dtype != torch.float32 or p.dtype != torch.float32 or wet.dtype != torch.float32
                    or x.shape != (self.frames,) or p.shape != x.shape or wet.shape != x.shape):
                raise ValueError("complete matching float32 CPU Dry/frozen-p/Wet rows required")
            expected = torch.tensor(_path_fields(path)[:2], dtype=torch.float32) / 100
            if not isinstance(controls, torch.Tensor) or controls.shape != (2,) or not torch.equal(controls.cpu(), expected):
                raise ValueError("row controls disagree with audited filename")
            if not torch.isfinite(x).all() or not torch.isfinite(p).all() or not torch.isfinite(wet).all():
                raise ValueError("nonfinite fit input; never silently drop a failure")

    @torch.inference_mode()
    def _mine(self, rows, readout, optimizer, completed_updates, purpose, target, batch_size, block_frames):
        if not 1 <= batch_size <= MAX_BATCH_SIZE or not 1 <= block_frames <= MAX_BLOCK_FRAMES:
            raise ValueError("mining batch/block exceeds the fixed memory bound")
        self._validate_rows(rows)
        _check_optimizer_step(readout, optimizer, completed_updates)
        live_hash = module_sha256(readout)
        snapshot = copy.deepcopy(readout).eval().requires_grad_(False).to(target)
        weight_hash = module_sha256(snapshot)
        if weight_hash != live_hash:
            raise ValueError("snapshot did not preserve completed readout weights")
        bank, events = CausalTapBank(), []
        for offset in range(0, len(rows), batch_size):
            group = rows[offset:offset + batch_size]
            controls = torch.stack([row["controls"] for row in group]).to(target)
            count = len(group)
            highest = torch.full((count,), -torch.inf, device=target)
            lowest = torch.full((count,), torch.inf, device=target)
            highest_index = torch.zeros(count, dtype=torch.long, device=target)
            lowest_index = torch.zeros(count, dtype=torch.long, device=target)
            wet_highest, wet_lowest = torch.full((count,), -torch.inf), torch.full((count,), torch.inf)
            wet_highest_index, wet_lowest_index = torch.zeros(count, dtype=torch.long), torch.zeros(count, dtype=torch.long)
            error_sum, energy_sum = torch.zeros(count, dtype=torch.float64), torch.zeros(count, dtype=torch.float64)
            state = None
            for start in range(0, self.frames, block_frames):
                stop = min(start + block_frames, self.frames)
                dry = torch.stack([row["dry"][start:stop] for row in group]).to(target)
                original = torch.stack([row["original"][start:stop] for row in group]).to(target)
                features, state = bank(dry, original, state)
                current = original + snapshot.at_training_knots(features, controls)
                if not torch.isfinite(current).all():
                    raise ValueError("nonfinite current prediction during mining")
                scored_start = max(start, 1_024)
                if scored_start >= stop:
                    continue
                body = current[:, scored_start - start:]
                maximum, imax = body.max(1)
                minimum, imin = body.min(1)
                improve_max, improve_min = maximum > highest, minimum < lowest
                highest_index = torch.where(improve_max, imax + scored_start, highest_index)
                lowest_index = torch.where(improve_min, imin + scored_start, lowest_index)
                highest = torch.maximum(highest, maximum)
                lowest = torch.minimum(lowest, minimum)
                # Target only enters diagnostic scalar reductions after current
                # predictions/indices have been computed. No output is cached.
                wet = torch.stack([row["wet"][scored_start:stop] for row in group])
                current_cpu = body.cpu().double()
                wet_double = wet.double()
                error_sum += (current_cpu - wet_double).square().sum(1)
                energy_sum += wet_double.square().sum(1)
                wet_max, wet_imax = wet.max(1)
                wet_min, wet_imin = wet.min(1)
                wet_highest_index = torch.where(wet_max > wet_highest, wet_imax + scored_start, wet_highest_index)
                wet_lowest_index = torch.where(wet_min < wet_lowest, wet_imin + scored_start, wet_lowest_index)
                wet_highest, wet_lowest = torch.maximum(wet_highest, wet_max), torch.minimum(wet_lowest, wet_min)
            # Strict comparisons above preserve the earliest sample on ties.
            values = zip(highest_index.cpu().tolist(), highest.cpu().tolist(),
                         lowest_index.cpu().tolist(), lowest.cpu().tolist())
            for local, (row, (i, v, j, w)) in enumerate(zip(group, values)):
                frames = self.frames - 1_024
                energy, error = float(energy_sum[local]), float(error_sum[local])
                wet_max, wet_min = float(wet_highest[local]), float(wet_lowest[local])
                events.append(PredictedExtrema(str(row["path"]), int(i), float(v), int(j), float(w),
                    int(wet_highest_index[local]), wet_max, int(wet_lowest_index[local]), wet_min,
                    error, energy, frames, abs(max(abs(v), abs(w)) - max(abs(wet_max), abs(wet_min))),
                    error / max(energy, frames * 1e-8)))
        if module_sha256(readout) != live_hash:
            raise ValueError("live readout changed while its completed snapshot was mined")
        return PeakMiningResult(completed_updates, purpose, weight_hash, self.source_sha256,
                                self.prediction_sha256, self.frames, tuple(events), _fit_metrics(events))

    def refresh(self, rows, readout, optimizer, completed_updates, target=torch.device("cpu"),
                batch_size=MAX_BATCH_SIZE, block_frames=MAX_BLOCK_FRAMES):
        if self._next_refresh >= len(REFRESH_STEPS) or completed_updates != REFRESH_STEPS[self._next_refresh]:
            raise ValueError("refresh only in exact order0,250,...,3750;4000 is audit-only")
        result = self._mine(rows, readout, optimizer, completed_updates, "refresh", target, batch_size, block_frames)
        self._active = result
        self._events = {event.path: event for event in result.events}
        self._next_refresh += 1
        return result

    def audit(self, rows, readout, optimizer, completed_updates, target=torch.device("cpu"),
              batch_size=MAX_BATCH_SIZE, block_frames=MAX_BLOCK_FRAMES):
        if completed_updates != AUDIT_STEP or self._next_refresh != len(REFRESH_STEPS) or self._audited:
            raise ValueError("single4000 audit follows all scheduled refreshes and never creates training events")
        result = self._mine(rows, readout, optimizer, completed_updates, "audit", target, batch_size, block_frames)
        self._audited = True
        return result

    def next_event(self, path):
        path = str(path)
        if self._active is None or path not in self._events:
            raise ValueError("current fit event unavailable")
        if self._audited:
            raise ValueError("no further training events after terminal4000 audit")
        event, visit = self._events[path], self._visits[path]
        self._visits[path] += 1
        return ((event.positive_index, "positive") if visit % 2 == 0 else
                (event.negative_index, "negative"))


def mining_resource_bounds(frames=144_000, fit=234, calibration=54, width=32):
    """Logical tensor bounds, not an allocator/driver or wall-time guarantee."""
    parameters = 9 * (16 * width + width + width * width + width + width)
    batch, block = MAX_BATCH_SIZE, MAX_BLOCK_FRAMES
    return {
        "fit_dry_wet_frozen_p_bytes": fit * frames * 3 * 4,
        "fit_and_cal_existing_audio_p_bytes": (fit + calibration) * frames * 3 * 4,
        "new_full_prediction_cache_bytes": 0,
        "maximum_readout_snapshot_tensor_bytes": (parameters + 16) * 4,
        "max_raw_input_chunk_bytes": batch * block * 2 * 4,
        "max_two_input_fifo_bytes": batch * 64 * 2 * 4,
        "max_feature_tensor_bytes": batch * block * 16 * 4,
        "each_hidden_activation_tensor_bytes": batch * block * width * 4,
        "scalar_current_event_payload_bytes_without_python_or_paths": fit * (2 * 8 + 2 * 4),
        "scalar_event_and_metric_payload_bytes_without_python_or_paths": fit * (5 * 8 + 8 * 8),
        "max_cpu_float64_metric_workspace_estimate_bytes": batch * block * 8 * 4,
        "frames_per_refresh": fit * frames,
        "block_forward_calls_per_refresh": ((fit + batch - 1) // batch) * ((frames + block - 1) // block),
        "dense_multiply_accumulates_per_refresh_excluding_origin": fit * frames * (16 * width + width * width + width),
        "training_refresh_count": len(REFRESH_STEPS),
        "terminal_audit_count": 1,
        "note": "No autograd tape; tensor temporaries remain one batch/block. PyTorch/MPS allocator reserve is implementation-dependent and must be measured by the future training run."
    }
