"""Fit-only anchored readout screen for the immutable standalone DFZ core.

Uses the complete learned32-channel audio features, not original model output
or oracle inputs. Fixed six ridge strengths, shared versus nine-corner linear
projections. No bias, clip-specific gain, normalized audio or cached features.
Calibration remains a provisional screen until a complete model is replayed.
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
from .precheck_multirate_fuzz import sha256
from .train_dfz_core_tcn import selection_key


RIDGES = (1e-5, 1e-4, 1e-3, 1e-2, .1, 1.)


def anchored_solution(gram, rhs, source, ridge):
    """Normal equations for delta weights, anchored to the immutable head."""
    penalty = np.diag(np.maximum(np.diag(gram), 1e-8)) * ridge
    return source + np.linalg.solve(gram + penalty, rhs)


def summarize(predictions, targets, controls):
    energy = sum(float(np.sum(target.astype(np.float64) ** 2)) for target in targets)
    errors, esrs, peaks, ratios, groups = [], [], [], [], {str(a): [] for a in (0, 50, 100)}
    for predicted, target, control in zip(predictions, targets, controls):
        error = float(np.sum((predicted.astype(np.float64) - target) ** 2))
        target_energy = float(np.sum(target.astype(np.float64) ** 2))
        errors.append(error)
        esrs.append(error / max(target_energy, 1e-8 * len(target)))
        peak = abs(float(np.max(np.abs(predicted))) - float(np.max(np.abs(target))))
        peaks.append(peak)
        ratios.append(float(np.max(np.abs(predicted))) / float(np.max(np.abs(target))))
        groups[str(control)].append((esrs[-1], peak))
    result = {"examples": len(targets), "global_esr": sum(errors) / energy, "mean_per_file_esr": float(np.mean(esrs)),
              "p95_per_file_esr": float(np.quantile(esrs, .95)), "absolute_peak_error_p95": float(np.quantile(peaks, .95)),
              "peak_ratio_median": float(np.median(ratios)), "peak_ratio_p95": float(np.quantile(ratios, .95)),
              "by_attack": {key: {"files": len(values), "mean_esr": float(np.mean([v[0] for v in values])),
                                  "absolute_peak_error_p95": float(np.quantile([v[1] for v in values], .95))}
                            for key, values in groups.items()}}
    result["worst_attack_absolute_peak_error_p95"] = max(value["absolute_peak_error_p95"] for value in result["by_attack"].values())
    result["passes_selection_gate"] = (result["global_esr"] <= .05 and result["mean_per_file_esr"] <= .1
        and result["p95_per_file_esr"] <= .25 and result["absolute_peak_error_p95"] <= .02
        and result["worst_attack_absolute_peak_error_p95"] <= .03 and .75 <= result["peak_ratio_median"] <= 1.25
        and result["peak_ratio_p95"] <= 1.35)
    return result


@torch.inference_mode()
def fit(source, output):
    if output.exists():
        raise ValueError("new readout screen directory required")
    model, payload, parent = load_candidate(source)
    if model.audio.width != 32 or model.audio.output.weight.shape != (1, 32, 1):
        raise ValueError("immutable32-feature scalar readout required")
    source_hash, parent_hash = sha256(source), sha256(source.with_name("metrics.json"))
    names = (Path(__file__).name, "evaluate_multirate_fuzz.py", "train_dfz_core_tcn.py", "asrnn_effects.py")
    sources = {**parent["source_sha256"], **{name: sha256(Path(__file__).parent / name) for name in names}}
    provenance = parent["audio_provenance"]
    if set(provenance) != {"fit", "calibration"} or len(provenance["fit"]) != 234 or len(provenance["calibration"]) != 54:
        raise ValueError("original234/54 partition required")

    def unchanged():
        if sha256(source) != source_hash or sha256(source.with_name("metrics.json")) != parent_hash:
            raise ValueError("immutable parent changed")
        if any(sha256(Path(__file__).parent / name) != digest for name, digest in sources.items()):
            raise ValueError("screen implementation changed")
        if any(sha256(row["path"]) != row["sha256"] for records in provenance.values() for row in records):
            raise ValueError("source audio changed")

    def read(record, split):
        path = Path(record["path"])
        if "train" not in path.parts or (int(path.stem.split(",")[-1]) % 5 == 0) != (split == "calibration"):
            raise ValueError("readout screen cannot open eval or change partitions")
        dry, wet, knobs = read_effect_pair(path, "dfz")
        captured = []
        handle = model.audio.output.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
        try:
            model(torch.from_numpy(dry)[None], torch.from_numpy(knobs)[None])
        finally:
            handle.remove()
        if len(captured) != 1 or captured[0].shape != (1, 32, len(dry)):
            raise ValueError("unexpected complete causal feature geometry")
        a, b = (knobs * 2).astype(int)
        return captured[0][0, :, 1024:].T.contiguous().numpy(), wet[1024:], int(a * 3 + b), int(a * 50)

    unchanged()
    started = time.perf_counter()
    gram, rhs, counts = np.zeros((9, 32, 32)), np.zeros((9, 32)), np.zeros(9, dtype=int)
    source_weight = model.audio.output.weight.detach().numpy().reshape(32).astype(np.float64)
    for index, record in enumerate(provenance["fit"]):
        features, target, corner, _ = read(record, "fit")
        x, y = features.astype(np.float64), target.astype(np.float64)
        weights = 1 + 4 * (np.abs(y) / max(float(np.max(np.abs(y))), 1e-5)) ** 4
        weight = weights / (len(y) * max(float(np.mean(y*y)), 1e-5))
        residual = y - x @ source_weight
        gram[corner] += x.T @ (x * weight[:, None])
        rhs[corner] += x.T @ (residual * weight)
        xp, yp = x[1:] - .95*x[:-1], y[1:] - .95*y[:-1]
        pre_weight = .1 / (len(yp) * max(float(np.mean(yp*yp)), 1e-5))
        gram[corner] += pre_weight * (xp.T @ xp)
        rhs[corner] += pre_weight * (xp.T @ (yp - xp @ source_weight))
        counts[corner] += 1
        if (index+1) % 26 == 0:
            print(json.dumps({"fit_files": index+1, "elapsed_seconds": time.perf_counter()-started}), flush=True)
    if counts.tolist() != [26]*9:
        raise ValueError("fit control coverage differs")
    menus = [("unchanged", None, np.tile(source_weight.astype(np.float32), (9, 1)))]
    for mode in ("shared", "nine-corner"):
        for ridge in RIDGES:
            weights = (np.tile(anchored_solution(gram.sum(0), rhs.sum(0), source_weight, ridge), (9, 1)) if mode == "shared"
                       else np.stack([anchored_solution(a, b, source_weight, ridge) for a, b in zip(gram, rhs)]))
            if not np.isfinite(weights).all():
                raise ValueError("nonfinite anchored fit")
            menus.append((mode, ridge, weights.astype(np.float32)))
    targets, controls, predictions = [], [], [[] for _ in menus]
    for index, record in enumerate(provenance["calibration"]):
        features, target, corner, first_control = read(record, "calibration")
        values = features @ np.stack([weights[corner] for _, _, weights in menus], 1)
        targets.append(target)
        controls.append(first_control)
        for column, values_for_menu in enumerate(predictions):
            values_for_menu.append(values[:, column].copy())
        if (index+1) % 9 == 0:
            print(json.dumps({"calibration_files": index+1}), flush=True)
    history = [{"mode": mode, "ridge": ridge, "provisional_calibration": summarize(values, targets, controls)}
               for (mode, ridge, _), values in zip(menus, predictions)]
    selected = min(range(len(history)), key=lambda i: selection_key(history[i]["provisional_calibration"]))
    unchanged()
    output.mkdir(parents=True)
    if selected:
        torch.save({"experimental_schema": 1, "purpose": "unadmitted-conditional-readout-only", "parent_checkpoint_sha256": source_hash,
                    "weight": torch.from_numpy(menus[selected][2]), "bias": False, "mode": menus[selected][0]}, output / "candidate-readout.pt")
    report = {"schema": 1, "purpose": "anchored-multichannel-readout-screen-not-model-admission", "history": history,
              "selected": selected, "selection": history[selected], "fit_files_per_corner": counts.tolist(),
              "parent_checkpoint_sha256": source_hash, "parent_report_sha256": parent_hash, "parent_architecture": payload["architecture"],
              "source_sha256": sources, "audio_provenance": provenance, "source_and_audio_reverified": True,
              "full_model_replay_required": True, "features_cached_to_disk": False, "clip_specific_parameters": False,
              "loss": "complete-file signed weighted MSE +.1pre95, anchored delta ridge; no peak-maximum objective",
              "admitted": False, "official_eval_opened": False, "physical_audio_devices_used": False,
              "source_audio_modified": False, "automatic_gain_or_normalization": False, "elapsed_seconds": time.perf_counter()-started}
    if selected:
        report["readout_sha256"] = sha256(output / "candidate-readout.pt")
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"selection": report["selection"], "selected": selected}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    fit(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
