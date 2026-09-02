"""Run the frozen independent-family Blind stack as a physical-chain veto."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .blind2 import LABELS, BlindFamilyPresence, BlindPresence
from .evaluate_guitar_chain_presence import _device, _probabilities, _windows
from .physical_chain_data import discover, inventory
from .train_blind2 import _metrics


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _component(stack_path: Path, stack: dict, name: str) -> tuple[Path, dict]:
    component = stack["components"].get(name)
    if not isinstance(component, dict):
        raise ValueError(f"stack is missing component {name}")
    path = (stack_path.parent / component["path"]).resolve()
    actual = _sha256(path)
    if actual != component.get("sha256"):
        raise ValueError(f"stack component hash mismatch for {name}")
    return path, torch.load(path, map_location="cpu", weights_only=False)


def _wilson_upper(positives: int, total: int, z: float = 1.96) -> float:
    if total < 1 or not 0 <= positives <= total:
        raise ValueError("invalid binomial counts")
    proportion = positives / total
    denominator = 1.0 + z * z / total
    center = proportion + z * z / (2.0 * total)
    margin = z * np.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    )
    return float((center + margin) / denominator)


def _acceptance(overall: dict, hardware: dict[str, dict], coverage: dict) -> dict:
    clean_examples = int(overall["clean_examples"])
    clean_rate = overall["clean_false_positive_rate"]
    if clean_examples and clean_rate is not None:
        clean_false_positives = int(round(float(clean_rate) * clean_examples))
        clean_wilson_upper = _wilson_upper(clean_false_positives, clean_examples)
    else:
        clean_false_positives = 0
        clean_wilson_upper = 1.0
    performance_gates = {
        "micro_f1": overall["micro_f1"] >= 0.80,
        "macro_f1": overall["macro_f1"] >= 0.80,
        "each_label_recall": all(
            row["recall"] >= 0.80 for row in overall["per_label"].values()
        ),
        "clean_false_positive_rate": clean_rate is not None and clean_rate <= 0.05,
        "clean_false_positive_wilson_95_upper": clean_wilson_upper <= 0.05,
        "each_hardware_macro_f1": bool(hardware)
        and all(row["macro_f1"] >= 0.70 for row in hardware.values()),
    }
    coverage_gates = coverage["coverage_gates"]
    passed = all(performance_gates.values()) and all(coverage_gates.values())
    return {
        "physical_hardware_veto_passed": passed,
        "performance_gates": performance_gates,
        "coverage_gates": coverage_gates,
        "clean_false_positives": clean_false_positives,
        "clean_false_positive_wilson_95_upper": clean_wilson_upper,
        "product_promotion_allowed": False,
        "remaining_blocker": (
            "listening and then a separately sealed locked-final remain"
            if passed
            else "the physical multi-effect hardware veto did not pass"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    stack_path = args.stack.resolve()
    stack = json.loads(stack_path.read_text())
    if stack.get("schema") != 1 or stack.get("usable_model") is not None:
        raise ValueError("expected the frozen development stack with usable_model null")
    boundaries = stack.get("boundaries", {})
    if boundaries.get("order_input") is not False or boundaries.get("order_output") is not False:
        raise ValueError("physical veto requires an order-free stack")

    base_path, base_checkpoint = _component(stack_path, stack, "shared_presence")
    nonlinear_path, nonlinear_checkpoint = _component(
        stack_path, stack, "nonlinear_expert"
    )
    unknown_path, unknown_checkpoint = _component(stack_path, stack, "unknown_expert")
    gate_path, gate_checkpoint = _component(stack_path, stack, "any_effect_gate")
    if nonlinear_checkpoint["manifest"]["family"] != "nonlinear":
        raise ValueError("nonlinear expert family mismatch")
    if unknown_checkpoint["manifest"]["family"] != "unknown":
        raise ValueError("unknown expert family mismatch")
    if gate_checkpoint["manifest"]["family"] != "any":
        raise ValueError("any-effect gate family mismatch")

    target_device = _device(args.device)
    base = BlindPresence().to(target_device)
    base.load_state_dict(base_checkpoint["model"])
    nonlinear = BlindFamilyPresence("nonlinear").to(target_device)
    nonlinear.load_state_dict(nonlinear_checkpoint["model"])
    unknown = BlindFamilyPresence("unknown").to(target_device)
    unknown.load_state_dict(unknown_checkpoint["model"])
    gate = BlindFamilyPresence("any").to(target_device)
    gate.load_state_dict(gate_checkpoint["model"])
    for model in (base, nonlinear, unknown, gate):
        model.eval()

    manifest, rows = discover(args.manifest)
    probabilities = []
    for row in rows:
        windows = _windows(row.path)
        base_probabilities = _probabilities(
            base, windows, base_checkpoint["calibration"], target_device
        )
        result = base_probabilities.copy()
        result[LABELS.index("nonlinear")] = _probabilities(
            nonlinear,
            windows,
            nonlinear_checkpoint["calibration"],
            target_device,
            nonlinear_checkpoint.get("file_aggregation", "top-two"),
        )[0]
        result[LABELS.index("unknown")] = _probabilities(
            unknown,
            windows,
            unknown_checkpoint["calibration"],
            target_device,
            unknown_checkpoint.get("file_aggregation", "top-two"),
        )[0]
        gate_probability = _probabilities(
            gate,
            windows,
            gate_checkpoint["calibration"],
            target_device,
            gate_checkpoint.get("file_aggregation", "top-two"),
        )[0]
        gate_active = gate_probability >= float(gate_checkpoint["threshold"])
        if "base_fallback_threshold" in gate_checkpoint:
            gate_active = gate_active or float(base_probabilities.max()) >= float(
                gate_checkpoint["base_fallback_threshold"]
            )
        if "family_fallback" in gate_checkpoint:
            fallback = gate_checkpoint["family_fallback"]
            gate_active = gate_active or base_probabilities[
                LABELS.index(fallback["family"])
            ] >= float(fallback["threshold"])
        if not gate_active:
            result[:] = 0.0
        probabilities.append(result)

    probabilities_array = np.stack(probabilities)
    expected = np.stack([row.target for row in rows])
    thresholds = [float(value) for value in base_checkpoint["thresholds"]]
    thresholds[LABELS.index("nonlinear")] = float(nonlinear_checkpoint["threshold"])
    thresholds[LABELS.index("unknown")] = float(unknown_checkpoint["threshold"])
    sources = [f"physical:{row.hardware_group}" for row in rows]
    overall = _metrics(probabilities_array, expected, thresholds, sources)
    hardware = {
        name: _metrics(
            probabilities_array[indices],
            expected[indices],
            thresholds,
            [sources[index] for index in indices],
        )
        for name in sorted({row.hardware_group for row in rows if row.target.any()})
        if (
            indices := [
                index
                for index, row in enumerate(rows)
                if row.hardware_group == name and row.target.any()
            ]
        )
    }
    coverage = inventory(rows)
    acceptance = _acceptance(overall, hardware, coverage)
    report = {
        "schema": 1,
        "status": (
            "physical-hardware-veto-passed-not-promoted"
            if acceptance["physical_hardware_veto_passed"]
            else "physical-hardware-veto-failed"
        ),
        "device": str(target_device),
        "source_id": manifest["source_id"],
        "rights_basis": manifest["rights_basis"],
        "manifest": {
            "path": str(args.manifest.resolve()),
            "sha256": _sha256(args.manifest.resolve()),
        },
        "stack": {"path": str(stack_path), "sha256": _sha256(stack_path)},
        "components": {
            "shared_presence": {"path": str(base_path), "sha256": _sha256(base_path)},
            "nonlinear_expert": {"path": str(nonlinear_path), "sha256": _sha256(nonlinear_path)},
            "unknown_expert": {"path": str(unknown_path), "sha256": _sha256(unknown_path)},
            "any_effect_gate": {"path": str(gate_path), "sha256": _sha256(gate_path)},
        },
        "thresholds": dict(zip(LABELS, thresholds, strict=True)),
        "overall": overall,
        "hardware_groups": hardware,
        "inventory": coverage,
        "acceptance": acceptance,
        "usable_model": None,
        "boundaries": {
            "veto_only": True,
            "weights_changed": False,
            "thresholds_changed": False,
            "model_selection_allowed": False,
            "calibration_allowed": False,
            "presence_labels_only": True,
            "order_metadata_accepted": False,
            "family_decisions_independent": True,
            "locked_final_opened": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
