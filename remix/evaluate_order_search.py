#!/usr/bin/env python3
"""Calibrate and validate classifier/search hybrid topology recovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .data import DAFxOrderDataset, SyntheticControlDataset, dry_sources
from .delay_model import DelayControlEstimator, apply_delay_residual
from .drive_model import DriveControlEstimator, apply_drive_estimate
from .model import PairedEstimator
from .order_search import rank_topologies, search_margin
from .pedalboard_data import PedalboardControlDataset
from .physics import apply_delay_hints, deconvolution_context
from .quality import contract_manifest
from .reverb_model import ReverbControlEstimator, apply_reverb_estimate, reverb_features
from .spec import KINDS, order_targets, ranked_topologies
from .train import CORPUS, RUN, device


def evenly_spaced(dataset: Dataset, count: int) -> Dataset:
    if count >= len(dataset):
        return dataset
    indices = np.linspace(0, len(dataset) - 1, count).round().astype(int).tolist()
    return Subset(dataset, indices)


@torch.no_grad()
def predictions(
    dataset: Dataset,
    main: PairedEstimator,
    drive: DriveControlEstimator,
    delay: DelayControlEstimator,
    reverb: ReverbControlEstimator,
    target_device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    order_rows, control_rows, records = [], [], []
    for batch in DataLoader(dataset, batch_size=8):
        dry, wet = batch["dry"].to(target_device), batch["wet"].to(target_device)
        raw = main(dry, wet)
        context = deconvolution_context(dry, wet)
        estimate = apply_delay_hints(raw, dry, wet, context)
        impulse = reverb_features(dry, wet, context)
        estimate = apply_delay_residual(
            estimate, dry, wet, delay, impulse, context
        )
        estimate = apply_drive_estimate(estimate, dry, wet, drive)
        estimate = apply_reverb_estimate(estimate, dry, wet, reverb, impulse)
        order_rows.append(estimate.order_logits.cpu())
        control_rows.append(torch.sigmoid(estimate.control_logits).cpu())
        for index in range(len(batch["dry"])):
            records.append(
                {
                    "dry": batch["dry"][index].numpy(),
                    "wet": batch["wet"][index].numpy(),
                    "topology": batch["topology"][index].tolist(),
                    "order": batch["order"][index].numpy(),
                    "order_mask": batch["order_mask"][index].numpy(),
                }
            )
    return torch.cat(order_rows).numpy(), torch.cat(control_rows).numpy(), records


def examples(
    records: list[dict],
    order_logits: np.ndarray,
    controls: np.ndarray,
) -> list[dict]:
    result = []
    for index, item in enumerate(records):
        encoded = item["topology"]
        topology = tuple(KINDS[value] for value in encoded if value >= 0)
        classifier = ranked_topologies(topology, order_logits[index])[0][0]
        if len(topology) >= 2:
            ranked = rank_topologies(
                item["dry"],
                item["wet"],
                topology,
                controls[index],
            )
            searched = ranked[0][0]
            margin = search_margin(ranked)
            errors = dict(ranked)
            classifier_error = errors[classifier]
            search_error = ranked[0][1]
        else:
            searched, margin = classifier, float("inf")
            classifier_error = search_error = 0.0
        result.append(
            {
                "truth": np.asarray(item["order"]) >= 0.5,
                "mask": np.asarray(item["order_mask"]) >= 0.5,
                "classifier": np.asarray(order_targets(classifier)[0]) >= 0.5,
                "search": np.asarray(order_targets(searched)[0]) >= 0.5,
                "margin": margin,
                "classifier_reconstruction_error": classifier_error,
                "search_reconstruction_error": search_error,
            }
        )
    return result


def metrics(rows: list[dict], threshold: float) -> dict[str, float | int]:
    correct = count = exact = eligible = searched = 0
    reconstruction_error = 0.0
    for row in rows:
        use_search = row["margin"] >= threshold
        prediction = row["search"] if use_search else row["classifier"]
        mask = row["mask"]
        if not mask.any():
            continue
        matches = prediction == row["truth"]
        correct += int((matches & mask).sum())
        count += int(mask.sum())
        exact += int((matches | ~mask).all())
        eligible += 1
        searched += int(use_search)
        reconstruction_error += row[
            "search_reconstruction_error"
            if use_search
            else "classifier_reconstruction_error"
        ]
    return {
        "pairwise": correct / max(count, 1),
        "exact": exact / max(eligible, 1),
        "relations": count,
        "examples": eligible,
        "search_coverage": searched / max(eligible, 1),
        "mean_gain_aligned_reconstruction_error": reconstruction_error
        / max(eligible, 1),
    }


def domain_metrics(domains: dict[str, list[dict]], threshold: float) -> dict:
    return {name: metrics(rows, threshold) for name, rows in domains.items()}


def score(values: dict[str, dict]) -> float:
    return sum(value["exact"] + 0.25 * value["pairwise"] for value in values.values()) / len(values)


def admissible(current: dict[str, dict], baseline: dict[str, dict]) -> bool:
    return all(
        current[name]["exact"] >= value["exact"] - 0.01
        and current[name]["pairwise"] >= value["pairwise"] - 0.015
        for name, value in baseline.items()
    )


def make_domains(corpus: Path, split: str, samples: int, equivalence_db: float) -> dict[str, Dataset]:
    seed = 20260901 if split == "calibrate" else 20260831
    domains: dict[str, Dataset] = {
        "real": evenly_spaced(DAFxOrderDataset(corpus, split), samples * 2)
    }
    for renderer in ("reference", "alternate", "stress"):
        domains[renderer] = SyntheticControlDataset(
            dry_sources(corpus, split),
            samples,
            seed=seed,
            renderers=(renderer,),
            order_equivalence_db=equivalence_db,
        )
    domains["pedalboard"] = PedalboardControlDataset(
        dry_sources(corpus, split),
        samples,
        seed=20260911 if split == "calibrate" else 20260904,
        order_equivalence_db=equivalence_db,
    )
    return domains


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--output", type=Path, default=RUN / "order-search-metrics.json")
    parser.add_argument("--samples", type=int, default=160)
    parser.add_argument("--order-equivalence-db", type=float, default=-30.0)
    args = parser.parse_args()

    target_device = device()
    main_model = PairedEstimator().to(target_device)
    drive_model = DriveControlEstimator().to(target_device)
    delay_model = DelayControlEstimator().to(target_device)
    reverb_model = ReverbControlEstimator().to(target_device)
    for model, filename in (
        (main_model, "paired-estimator.pt"),
        (drive_model, "drive-estimator.pt"),
        (delay_model, "delay-estimator.pt"),
        (reverb_model, "reverb-estimator.pt"),
    ):
        model.load_state_dict(
            torch.load(RUN / filename, map_location=target_device, weights_only=True)
        )
        model.eval()

    evaluated = {}
    for split in ("calibrate", "valid"):
        evaluated[split] = {}
        for name, dataset in make_domains(
            args.corpus, split, args.samples, args.order_equivalence_db
        ).items():
            order, controls, records = predictions(
                dataset,
                main_model,
                drive_model,
                delay_model,
                reverb_model,
                target_device,
            )
            evaluated[split][name] = examples(records, order, controls)

    thresholds = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, float("inf"))
    baseline = domain_metrics(evaluated["calibrate"], float("inf"))
    selected_threshold, selected = float("inf"), baseline
    selected_score = score(baseline)
    sweep = {}
    for threshold in thresholds:
        current = domain_metrics(evaluated["calibrate"], threshold)
        sweep[str(threshold)] = current
        if admissible(current, baseline) and score(current) > selected_score:
            selected_threshold, selected = threshold, current
            selected_score = score(current)

    report = {
        "schema": 1,
        "audio_quality": contract_manifest(),
        "order_equivalence_db": args.order_equivalence_db,
        "samples_per_synthetic_domain": args.samples,
        "search_renderers": ["reference", "alternate", "stress"],
        "selected_margin_threshold": selected_threshold,
        "calibration_classifier": baseline,
        "calibration_selected": selected,
        "validation_classifier": domain_metrics(evaluated["valid"], float("inf")),
        "validation_selected": domain_metrics(evaluated["valid"], selected_threshold),
        "calibration_sweep": sweep,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
