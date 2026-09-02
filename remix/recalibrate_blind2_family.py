"""Recalibrate one Blind family threshold with internal and external calibration splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from .blind2 import LABELS, BlindFamilyPresence, BlindPresence
from .blind_data2 import BlindPresenceData, inventory as blind_inventory
from .evaluate_guitar_chain_presence import _device, _probabilities as file_probabilities, _windows
from .guitar_chain_data import discover, inventory as chain_inventory
from .train_blind2 import SEED, _collect as collect_base, _loader, _probabilities
from .train_blind2_family import _binary_metrics, _collect


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


def _select_threshold(
    internal_probabilities: np.ndarray,
    internal_targets: np.ndarray,
    external_probabilities: np.ndarray,
    external_targets: np.ndarray,
    external_clean: np.ndarray,
) -> tuple[float, dict]:
    internal_expected = internal_targets[:, 0].astype(bool)
    external_expected = external_targets[:, 0].astype(bool)
    candidates = sorted(set(np.linspace(0.05, 0.995, 190).tolist()))
    best = None
    for threshold in candidates:
        internal_predicted = internal_probabilities[:, 0] >= threshold
        external_predicted = external_probabilities[:, 0] >= threshold
        internal_negative = ~internal_expected
        internal_false_positive_rate = float(
            np.mean(internal_predicted[internal_negative])
        )
        external_clean_false_positives = int(np.sum(external_predicted[external_clean]))
        external_clean_examples = int(np.sum(external_clean))
        external_clean_false_positive_rate = (
            external_clean_false_positives / external_clean_examples
        )
        external_clean_wilson_upper = _wilson_upper(
            external_clean_false_positives, external_clean_examples
        )
        if internal_false_positive_rate > 0.05 or external_clean_wilson_upper > 0.05:
            continue
        internal_recall = float(
            np.mean(internal_predicted[internal_expected])
        )
        external_recall = float(
            np.mean(external_predicted[external_expected])
        )
        internal_f1 = float(f1_score(internal_expected, internal_predicted, zero_division=0))
        external_f1 = float(f1_score(external_expected, external_predicted, zero_division=0))
        metrics = {
            "internal_negative_false_positive_rate": internal_false_positive_rate,
            "external_clean_false_positive_rate": external_clean_false_positive_rate,
            "external_clean_false_positive_wilson_95_upper": external_clean_wilson_upper,
            "internal_recall": internal_recall,
            "external_recall": external_recall,
            "internal_f1": internal_f1,
            "external_f1": external_f1,
        }
        objective = (
            min(internal_recall, external_recall),
            (internal_f1 + external_f1) / 2.0,
            -threshold,
        )
        if best is None or objective > best[0]:
            best = (objective, float(threshold), metrics)
    if best is None:
        raise ValueError("no family threshold satisfies both calibration false-positive gates")
    return best[1], best[2]


def _select_blend(
    internal_base: np.ndarray,
    internal_expert: np.ndarray,
    internal_targets: np.ndarray,
    external_base: np.ndarray,
    external_expert: np.ndarray,
    external_targets: np.ndarray,
    external_clean: np.ndarray,
) -> tuple[float, float, dict]:
    best = None
    for expert_weight in np.linspace(0.0, 1.0, 21):
        internal = (1.0 - expert_weight) * internal_base + expert_weight * internal_expert
        external = (1.0 - expert_weight) * external_base + expert_weight * external_expert
        try:
            threshold, metrics = _select_threshold(
                internal,
                internal_targets,
                external,
                external_targets,
                external_clean,
            )
        except ValueError:
            continue
        objective = (
            min(metrics["internal_recall"], metrics["external_recall"]),
            (metrics["internal_f1"] + metrics["external_f1"]) / 2.0,
            -metrics["internal_negative_false_positive_rate"],
            -metrics["external_clean_false_positive_rate"],
        )
        if best is None or objective > best[0]:
            best = (objective, float(expert_weight), threshold, metrics)
    if best is None:
        raise ValueError("no base/expert blend satisfies both calibration false-positive gates")
    return best[1], best[2], best[3]


def _select_or_gate(
    internal_base: np.ndarray,
    internal_gate: np.ndarray,
    internal_targets: np.ndarray,
    external_base: np.ndarray,
    external_gate: np.ndarray,
    external_targets: np.ndarray,
    external_clean: np.ndarray,
) -> tuple[float, float, dict]:
    internal_expected = internal_targets[:, 0].astype(bool)
    external_expected = external_targets[:, 0].astype(bool)
    candidates = [*np.linspace(0.05, 0.995, 190).tolist(), 1.01]
    best = None
    for gate_threshold in candidates:
        internal_gate_active = internal_gate[:, 0] >= gate_threshold
        external_gate_active = external_gate[:, 0] >= gate_threshold
        for base_fallback_threshold in candidates:
            internal_predicted = internal_gate_active | (
                internal_base[:, 0] >= base_fallback_threshold
            )
            external_predicted = external_gate_active | (
                external_base[:, 0] >= base_fallback_threshold
            )
            internal_negative = ~internal_expected
            internal_false_positive_rate = float(
                np.mean(internal_predicted[internal_negative])
            )
            external_clean_false_positives = int(
                np.sum(external_predicted[external_clean])
            )
            external_clean_examples = int(np.sum(external_clean))
            external_clean_false_positive_rate = (
                external_clean_false_positives / external_clean_examples
            )
            external_clean_wilson_upper = _wilson_upper(
                external_clean_false_positives, external_clean_examples
            )
            if internal_false_positive_rate > 0.05 or external_clean_wilson_upper > 0.05:
                continue
            internal_recall = float(np.mean(internal_predicted[internal_expected]))
            external_recall = float(np.mean(external_predicted[external_expected]))
            internal_f1 = float(
                f1_score(internal_expected, internal_predicted, zero_division=0)
            )
            external_f1 = float(
                f1_score(external_expected, external_predicted, zero_division=0)
            )
            metrics = {
                "internal_negative_false_positive_rate": internal_false_positive_rate,
                "external_clean_false_positive_rate": external_clean_false_positive_rate,
                "external_clean_false_positive_wilson_95_upper": external_clean_wilson_upper,
                "internal_recall": internal_recall,
                "external_recall": external_recall,
                "internal_f1": internal_f1,
                "external_f1": external_f1,
            }
            objective = (
                min(internal_recall, external_recall),
                (internal_f1 + external_f1) / 2.0,
                -internal_false_positive_rate,
            )
            if best is None or objective > best[0]:
                best = (
                    objective,
                    float(gate_threshold),
                    float(base_fallback_threshold),
                    metrics,
                )
    if best is None:
        raise ValueError("no OR gate satisfies both calibration false-positive gates")
    return best[1], best[2], best[3]


def _select_family_fallback(
    internal_active: np.ndarray,
    external_active: np.ndarray,
    internal_family_probabilities: np.ndarray,
    external_family_probabilities: np.ndarray,
    internal_targets: np.ndarray,
    external_targets: np.ndarray,
    external_clean: np.ndarray,
) -> tuple[str | None, float | None, dict]:
    internal_expected = internal_targets[:, 0].astype(bool)
    external_expected = external_targets[:, 0].astype(bool)
    best = None
    candidates = np.linspace(0.05, 0.995, 190)
    for family_index, family in enumerate(LABELS):
        for threshold in candidates:
            internal_predicted = internal_active | (
                internal_family_probabilities[:, family_index] >= threshold
            )
            external_predicted = external_active | (
                external_family_probabilities[:, family_index] >= threshold
            )
            internal_negative = ~internal_expected
            internal_false_positive_rate = float(
                np.mean(internal_predicted[internal_negative])
            )
            external_clean_false_positives = int(
                np.sum(external_predicted[external_clean])
            )
            external_clean_examples = int(np.sum(external_clean))
            external_clean_false_positive_rate = (
                external_clean_false_positives / external_clean_examples
            )
            external_clean_wilson_upper = _wilson_upper(
                external_clean_false_positives, external_clean_examples
            )
            if internal_false_positive_rate > 0.05 or external_clean_wilson_upper > 0.05:
                continue
            metrics = {
                "internal_negative_false_positive_rate": internal_false_positive_rate,
                "external_clean_false_positive_rate": external_clean_false_positive_rate,
                "external_clean_false_positive_wilson_95_upper": external_clean_wilson_upper,
                "internal_recall": float(np.mean(internal_predicted[internal_expected])),
                "external_recall": float(np.mean(external_predicted[external_expected])),
                "internal_f1": float(
                    f1_score(internal_expected, internal_predicted, zero_division=0)
                ),
                "external_f1": float(
                    f1_score(external_expected, external_predicted, zero_division=0)
                ),
            }
            objective = (
                min(metrics["internal_recall"], metrics["external_recall"]),
                (metrics["internal_f1"] + metrics["external_f1"]) / 2.0,
                -internal_false_positive_rate,
            )
            if best is None or objective > best[0]:
                best = (objective, family, float(threshold), metrics)
    if best is None:
        return None, None, {}
    return best[1], best[2], best[3]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus/guitar-effects-chains"))
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument(
        "--expert-file-aggregation", choices=("top-two", "mean"), default="mean"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.expert, map_location="cpu", weights_only=False)
    family = checkpoint["manifest"]["family"]
    family_index = None if family == "any" else LABELS.index(family)
    target_device = _device(args.device)
    model = BlindFamilyPresence(family).to(target_device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    base_checkpoint = torch.load(args.base, map_location="cpu", weights_only=False)
    base_model = BlindPresence().to(target_device)
    base_model.load_state_dict(base_checkpoint["model"])
    base_model.eval()

    internal_data = BlindPresenceData(Path.cwd(), "calibration", 480, SEED + 23)
    internal_logits, internal_targets, internal_sources = _collect(
        model, _loader(internal_data, 16, False), target_device, family
    )
    internal_probabilities = _probabilities(internal_logits, checkpoint["calibration"])
    internal_base_logits, internal_base_targets, _ = collect_base(
        base_model, _loader(internal_data, 16, False), target_device
    )
    base_target = (
        internal_base_targets.max(axis=1, keepdims=True)
        if family_index is None
        else internal_base_targets[:, family_index : family_index + 1]
    )
    np.testing.assert_array_equal(base_target, internal_targets)
    internal_all_base_probabilities = _probabilities(
        internal_base_logits, base_checkpoint["calibration"]
    )
    internal_base_probabilities = (
        internal_all_base_probabilities.max(axis=1, keepdims=True)
        if family_index is None
        else internal_all_base_probabilities[:, family_index : family_index + 1]
    )

    external_rows = discover(args.corpus, "calibration")
    external_probabilities = np.asarray([
        file_probabilities(
            model,
            _windows(row.path),
            checkpoint["calibration"],
            target_device,
            args.expert_file_aggregation,
        )[0]
        for row in external_rows
    ])[:, None]
    external_all_base_probabilities = np.stack([
        file_probabilities(
            base_model,
            _windows(row.path),
            base_checkpoint["calibration"],
            target_device,
        )
        for row in external_rows
    ])
    external_base_probabilities = (
        external_all_base_probabilities.max(axis=1, keepdims=True)
        if family_index is None
        else external_all_base_probabilities[:, family_index : family_index + 1]
    )
    external_full_targets = np.stack([row.target for row in external_rows])
    external_targets = (
        external_full_targets.max(axis=1, keepdims=True)
        if family_index is None
        else external_full_targets[:, family_index : family_index + 1]
    )
    external_clean = ~external_full_targets.astype(bool).any(axis=1)
    base_fallback_threshold = None
    family_fallback = None
    if family == "any":
        threshold, base_fallback_threshold, selection = _select_or_gate(
            internal_base_probabilities,
            internal_probabilities,
            internal_targets,
            external_base_probabilities,
            external_probabilities,
            external_targets,
            external_clean,
        )
        expert_weight = 1.0
        internal_active = (
            (internal_probabilities[:, 0] >= threshold)
            | (internal_base_probabilities[:, 0] >= base_fallback_threshold)
        )
        external_active = (
            (external_probabilities[:, 0] >= threshold)
            | (external_base_probabilities[:, 0] >= base_fallback_threshold)
        )
        fallback_family, fallback_threshold, fallback_selection = _select_family_fallback(
            internal_active,
            external_active,
            internal_all_base_probabilities,
            external_all_base_probabilities,
            internal_targets,
            external_targets,
            external_clean,
        )
        if fallback_family is not None and fallback_threshold is not None:
            fallback_index = LABELS.index(fallback_family)
            internal_active |= (
                internal_all_base_probabilities[:, fallback_index] >= fallback_threshold
            )
            external_active |= (
                external_all_base_probabilities[:, fallback_index] >= fallback_threshold
            )
            family_fallback = {"family": fallback_family, "threshold": fallback_threshold}
            selection = {
                "base_or_gate": selection,
                "family_fallback": fallback_selection,
            }
        blended_internal = internal_active.astype(np.float64)[:, None]
        blended_external = external_active.astype(np.float64)[:, None]
        metrics_threshold = 0.5
    else:
        expert_weight, threshold, selection = _select_blend(
            internal_base_probabilities,
            internal_probabilities,
            internal_targets,
            external_base_probabilities,
            external_probabilities,
            external_targets,
            external_clean,
        )
        blended_internal = (
            (1.0 - expert_weight) * internal_base_probabilities
            + expert_weight * internal_probabilities
        )
        blended_external = (
            (1.0 - expert_weight) * external_base_probabilities
            + expert_weight * external_probabilities
        )
        metrics_threshold = threshold
    internal_metrics = _binary_metrics(
        blended_internal, internal_targets, metrics_threshold, internal_sources
    )
    external_sources = [f"guitar:{row.guitar}" for row in external_rows]
    external_metrics = _binary_metrics(
        blended_external, external_targets, metrics_threshold, external_sources
    )
    external_predicted = blended_external[:, 0] >= metrics_threshold
    external_clean_false_positive_rate = float(np.mean(external_predicted[external_clean]))

    args.output.mkdir(parents=True, exist_ok=True)
    updated = dict(checkpoint)
    updated["threshold"] = threshold
    updated["base_blend_weight"] = 1.0 - expert_weight
    updated["expert_blend_weight"] = expert_weight
    if base_fallback_threshold is not None:
        updated["base_fallback_threshold"] = base_fallback_threshold
    if family_fallback is not None:
        updated["family_fallback"] = family_fallback
    updated["base_checkpoint_sha256"] = hashlib.sha256(args.base.read_bytes()).hexdigest()
    updated["file_aggregation"] = args.expert_file_aggregation
    updated["threshold_recalibration"] = {
        "internal_split": "calibration",
        "external_source": "dafx25-guitar-effects-chains",
        "external_split": "calibration",
        "selection": selection,
    }
    artifact = args.output / "model.pt"
    torch.save(updated, artifact)
    report = {
        "schema": 1,
        "status": "calibration-only-not-promoted",
        "family": family,
        "device": str(target_device),
        "old_threshold": checkpoint["threshold"],
        "threshold": threshold,
        "base_blend_weight": 1.0 - expert_weight,
        "expert_blend_weight": expert_weight,
        "base_fallback_threshold": base_fallback_threshold,
        "family_fallback": family_fallback,
        "file_aggregation": args.expert_file_aggregation,
        "selection": selection,
        "internal_calibration": internal_metrics,
        "external_calibration": {
            **external_metrics,
            "clean_examples": int(np.sum(external_clean)),
            "clean_false_positive_rate": external_clean_false_positive_rate,
        },
        "inventory": {
            "blind": blind_inventory(Path.cwd()),
            "external": chain_inventory(args.corpus),
        },
        "boundaries": {
            "weights_changed": False,
            "development_opened": False,
            "locked_final_opened": False,
            "order_labels_used": False,
            "controls_used": False,
        },
        "artifact": {
            "path": str(artifact.resolve()),
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        },
        "base": {
            "path": str(args.base.resolve()),
            "sha256": hashlib.sha256(args.base.read_bytes()).hexdigest(),
        },
    }
    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
