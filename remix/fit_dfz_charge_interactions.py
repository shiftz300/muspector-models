"""Bounded CPU experiment: zero-origin charge/output quadratic residuals.

This is intentionally not a registered runtime/checkpoint schema. It reads the
existing signed feature cache, fits on the 234 fit recordings, chooses one
global hyperparameter pair on 54 calibration recordings, and only runs a full
CPU calibration when the sampled peak improves by at least five percent.
No official-eval audio, gain patch, limiter, or source-file mutation is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from .asrnn_effects import effect_files
from .fit_asrnn_effect_output import _partition
from .fit_dfz_charge import signature
from .stable_charge_residual import ChargeBank, ChargeReadout
from .stable_effect import load_stable_effect
from .train_asrnn_phase7 import _evaluate, _rows


NEIGHBORHOOD_WEIGHTS = (0.0, 1.0, 4.0)
RIDGES = (1.0e-4, 0.01, 1.0)
UNIFORM_SAMPLES = 2_048
WIDTH = 35
MINIMUM_SAMPLED_PEAK_IMPROVEMENT = 0.05


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interaction_features(charge: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
    """z=(six charge features, original); then z plus upper-triangle z_i*z_j."""
    if charge.shape[:-1] != original.shape or charge.shape[-1] != 6:
        raise ValueError("six charge features and matching original samples are required")
    z = torch.cat((charge, original[..., None]), dim=-1)
    left, right = torch.triu_indices(7, 7, device=z.device)
    return torch.cat((z, z[..., left] * z[..., right]), dim=-1)


class ChargeInteractionResidual(torch.nn.Module):
    """Temporary in-memory wrapper; the existing factory/schema is untouched."""

    def __init__(self, base, raw_weight: torch.Tensor):
        super().__init__()
        self.base = base.eval().requires_grad_(False)
        self.bank = ChargeBank().eval()
        self.readout = ChargeReadout(WIDTH)
        if raw_weight.shape != (9, WIDTH) or not torch.isfinite(raw_weight).all():
            raise ValueError("a finite 9x35 static readout is required")
        with torch.no_grad():
            self.readout.weight.copy_(raw_weight)
        self.control_count = self.base.control_count
        widths = getattr(base, "export_state_widths", [layer.hidden_size for layer in base.modules()
                                                       if isinstance(layer, torch.nn.LSTM)])
        self.export_state_widths = [*widths, self.bank.width]
        self.layers = len(self.export_state_widths)
        self.requires_grad_(False)

    def forward(self, dry, controls, state=None):
        if state is not None and len(state) != self.layers:
            raise ValueError("wrong quadratic charge state count")
        original, next_base = self.base(dry, controls, None if state is None else state[:-1])
        charge, next_charge = self.bank(dry, None if state is None else state[-1])
        value = original + self.readout(interaction_features(charge, original), controls)
        return value, (*next_base, next_charge)


def check_cached_partition(data: dict, expected_count: int, per_corner: int) -> None:
    required = {"features", "wet", "original", "controls", "peak", "energy"}
    if set(data) != required or data["features"].shape != (expected_count, 4_096, 6):
        raise ValueError("charge cache partition geometry differs")
    expected = {"wet": (expected_count, 4_096), "original": (expected_count, 4_096),
                "controls": (expected_count, 2), "peak": (expected_count,), "energy": (expected_count,)}
    if any(data[key].shape != shape for key, shape in expected.items()):
        raise ValueError("charge cache aligned tensor geometry differs")
    if any(value.dtype != torch.float32 or value.device.type != "cpu" or not torch.isfinite(value).all()
           for value in data.values()):
        raise ValueError("charge cache must be finite CPU float32")
    controls = data["controls"]
    if not torch.isin(controls, torch.tensor([0.0, 0.5, 1.0])).all():
        raise ValueError("charge cache control grid differs")
    corners = (controls * 2).long()
    counts = torch.bincount(corners[:, 0] * 3 + corners[:, 1], minlength=9)
    if not torch.equal(counts, torch.full((9,), per_corner)) or not (data["energy"] > 0).all():
        raise ValueError("charge cache nine-corner fit/calibration coverage differs")


@torch.inference_mode()
def fit_menu(data: dict) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """No calibration argument: all feature scaling and regression are fit-only."""
    features = interaction_features(data["features"], data["original"])
    # RMS only, never mean centering, preserves the zero-origin polynomial.
    rms = features.double().square().mean(dim=(0, 1)).sqrt().clamp_min(1.0e-8)
    corners = (data["controls"] * 2).long()
    corners = corners[:, 0] * 3 + corners[:, 1]
    equations = []
    for corner in range(9):
        mask = corners == corner
        x = features[mask].double() / rms
        y = (data["wet"][mask] - data["original"][mask]).double()
        factor = data["energy"][mask].double().rsqrt()[:, None]
        x, y = x * factor[..., None], y * factor
        pieces = []
        for first, last in ((0, UNIFORM_SAMPLES), (UNIFORM_SAMPLES, x.shape[1])):
            matrix = x[:, first:last].reshape(-1, WIDTH)
            target = y[:, first:last].reshape(-1)
            pieces.append((matrix.T @ matrix, matrix.T @ target))
        equations.append(pieces)
    candidates = {"unchanged": torch.zeros(9, WIDTH)}
    for neighborhood in NEIGHBORHOOD_WEIGHTS:
        for ridge in RIDGES:
            weights = []
            for (uniform_xx, uniform_xy), (neighbor_xx, neighbor_xy) in equations:
                xx = uniform_xx + neighborhood * neighbor_xx
                xy = uniform_xy + neighborhood * neighbor_xy
                regularization = max(float(xx.trace() / WIDTH) * ridge, 1.0e-10)
                standardized = torch.linalg.solve(xx + torch.eye(WIDTH, dtype=torch.float64) * regularization, xy)
                # Fold fixed fit-only scaling into static coefficients. Both
                # sampled scoring and the full wrapper use these exact floats.
                weights.append((standardized / rms).float())
            candidates[f"neighborhood-{neighborhood:g}-ridge-{ridge:g}"] = torch.stack(weights)
    return candidates, rms


@torch.inference_mode()
def sampled_quality(weight: torch.Tensor, data: dict) -> dict:
    features = interaction_features(data["features"], data["original"])
    controls = (data["controls"] * 2).long()
    corners = controls[:, 0] * 3 + controls[:, 1]
    predicted = data["original"] + (features * weight[corners, None]).sum(-1)
    peaks = (predicted.abs().amax(1) - data["peak"]).abs()
    esr = (predicted[:, :UNIFORM_SAMPLES] - data["wet"][:, :UNIFORM_SAMPLES]).square().mean(1) / data["energy"]
    return {
        "sampled_peak_error_p95": float(torch.quantile(peaks, 0.95)),
        "sampled_mean_esr": float(esr.mean()),
        "by_blend": {str(blend * 50): {"files": int((controls[:, 0] == blend).sum()),
                    "sampled_peak_error_p95": float(torch.quantile(peaks[controls[:, 0] == blend], 0.95))}
                     for blend in range(3)},
    }


def objective(value: dict) -> float:
    return value["sampled_peak_error_p95"] + max(0.0, value["sampled_mean_esr"] - 0.03)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("remix/runs/dfz-charge-ridge-phase9/fit-features.pt"))
    parser.add_argument("--source", type=Path, default=Path("remix/runs/dfz-nonlinear-readout-phase9/candidate.pt"))
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("experiment output exists; preserve prior evidence")
    torch.set_num_threads(2)
    started = time.monotonic()
    fit, cal = _partition(effect_files(Path("data/corpus/asrnn-physical-effects"), "dfz", "train"))
    if len(fit) != 234 or len(cal) != 54:
        raise ValueError("frozen DFZ 234/54 partition differs")
    expected = signature(args.source, fit, cal)
    cache_hash = file_sha256(args.cache)
    cached = torch.load(args.cache, map_location="cpu", weights_only=True)
    if cached["signature"] != expected:
        raise ValueError("source/protocol/fit/calibration signature differs from the existing cache")
    train, validation = cached["fit"], cached["calibration"]
    check_cached_partition(train, 234, 26)
    check_cached_partition(validation, 54, 6)
    candidates, rms = fit_menu(train)
    scores = {name: sampled_quality(weight, validation) for name, weight in candidates.items()}
    selected = min(scores, key=lambda name: objective(scores[name]))
    improved = (selected != "unchanged" and scores[selected]["sampled_mean_esr"] <= 0.03
                and scores[selected]["sampled_peak_error_p95"] <= scores["unchanged"]["sampled_peak_error_p95"]
                * (1 - MINIMUM_SAMPLED_PEAK_IMPROVEMENT))
    report = {
        "schema": 1, "experiment": "fit-only-charge-output-quadratic35", "selected_global_menu": selected,
        "sampled_calibration": scores, "full_calibration": None, "full_calibration_started": bool(improved),
        "passes_full_calibration_gate": False, "admitted": False,
        "protocol": {"neighborhood_weights": list(NEIGHBORHOOD_WEIGHTS), "ridges": list(RIDGES),
                     "uniform_samples": UNIFORM_SAMPLES, "neighbor_samples": 2_048,
                     "feature_order": "z=[charge0..5, original], followed by z_i*z_j in torch.triu_indices(7,7) order",
                     "feature_width": WIDTH, "bias": False, "feature_scaling": "fit-only RMS folded into fixed weights; no centering",
                     "fit_target": "signed wet-minus-original, weighted by fit Wet RMS",
                     "calibration_selection": "one global neighborhood/ridge setting for all nine corners; never per-clip or per-corner selection",
                     "selection_score": "sampled peak P95 + max(0, sampled mean ESR - 0.03)",
                     "minimum_sampled_peak_improvement_for_full_calibration": MINIMUM_SAMPLED_PEAK_IMPROVEMENT},
        "fit_examples": 234, "calibration_examples": 54,
        "cache_sha256": cache_hash, "cache_signature_sha256": hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest(),
        "source_sha256": expected["source_sha256"], "experiment_script_sha256": file_sha256(Path(__file__)),
        "fit_feature_rms": rms.tolist(), "selected_raw_weights": candidates[selected].tolist(),
        "fit_compute": "cpu-float64", "sampled_readout_compute": "cpu-float32", "full_calibration_compute": "cpu",
        "cached_original_compute": "mps", "cached_charge_compute": "cpu",
        "official_eval_opened": False, "source_audio_modified": False, "physical_audio_devices_used": False,
        "automatic_normalization": False, "automatic_limiting": False, "per_clip_gain_patch": False,
        "runtime_factory_modified": False, "new_checkpoint_or_cache_saved": False,
    }
    args.output.mkdir(parents=True)
    report_path = args.output / "metrics.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"stage": "sampled-calibration", "selected": selected, "metrics": scores[selected],
                      "full_calibration_required": bool(improved)}), flush=True)
    if improved:
        base, _ = load_stable_effect(args.source)
        model = ChargeInteractionResidual(base.cpu(), candidates[selected]).eval()
        report["full_calibration"] = _evaluate(model, _rows(cal, "dfz"), torch.device("cpu"), 8)
        report["passes_full_calibration_gate"] = report["full_calibration"]["passes_selection_gate"]
    if signature(args.source, fit, cal) != expected or file_sha256(args.cache) != cache_hash:
        raise ValueError("source or existing cache changed while running this experiment")
    report["status"] = ("passed-calibration-only-needs-schema-export" if report["passes_full_calibration_gate"]
                        else "rejected-full-calibration" if improved else "rejected-sampled-calibration")
    report["elapsed_seconds"] = time.monotonic() - started
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "selected_global_menu", "full_calibration", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
