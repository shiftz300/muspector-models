#!/usr/bin/env python3
"""Evaluate a frozen EGDB-PG Amp+cab checkpoint on fresh sealed profiles."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .egdb_pg_amp_data import EgdbPgAmpPairs, RATE, _profiles
from .egdb_pg_amp_model5 import WetToneComplexAmpCabInverse
from .egdb_pg_amp_model6 import WetToneComplexDynamicsAmpCabInverse
from .egdb_pg_amp_model7 import WetToneComplexTemporalAmpCabInverse
from .egdb_pg_amp_model8 import (
    WetToneComplexUNetAmpCabInverse,
    WetToneComplexUNetJointAmpCabInverse,
)
from .egdb_pg_amp_model9 import WetToneComplexDemucsJointAmpCabInverse
from .egdb_pg_amp_model10 import (
    WetToneGrayBoxAmpCabInverse,
    WetTonePhaseGrayBoxAmpCabInverse,
)
from .egdb_pg_amp_model11 import WetTonePhaseGrayBoxDynamicsAmpCabInverse
from .egdb_pg_amp_model12 import WetTonePhaseGrayBoxMultibandDynamicsAmpCabInverse
from .open_riff_box_stage_model import (
    OpenRiffBoxStagewiseGrayBoxInverse,
    OpenRiffBoxStagewiseTransientGrayBoxInverse,
)
from .train_egdb_pg_amp import SEED, _quality, _runtime


def evaluate(args: argparse.Namespace) -> dict:
    workspace = args.workspace.resolve()
    run = args.run.resolve()
    checkpoint = run / "model.pt"
    training_metrics_path = run / "metrics.json"
    output = run / f"{args.split}_metrics.json"
    if output.exists():
        raise FileExistsError(f"refusing to replace fresh validation report: {output}")

    training = json.loads(training_metrics_path.read_text())
    digest_before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest_before != training["model"]["sha256"]:
        raise PermissionError("checkpoint hash differs from sealed training report")
    if training["status"] != "trained-awaiting-fresh-validation":
        raise PermissionError("checkpoint was not trained under the sealed-validation protocol")
    if training["data"].get("development_audio_opened") is not False:
        raise PermissionError("training report says development audio was opened")

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    architecture = payload["architecture"]
    model_classes = {
        5: WetToneComplexAmpCabInverse,
        6: WetToneComplexDynamicsAmpCabInverse,
        7: WetToneComplexTemporalAmpCabInverse,
        8: WetToneComplexUNetAmpCabInverse,
        9: WetToneComplexUNetJointAmpCabInverse,
        10: WetToneComplexDemucsJointAmpCabInverse,
        11: WetToneGrayBoxAmpCabInverse,
        12: WetTonePhaseGrayBoxAmpCabInverse,
        13: WetTonePhaseGrayBoxDynamicsAmpCabInverse,
        14: WetTonePhaseGrayBoxMultibandDynamicsAmpCabInverse,
        15: OpenRiffBoxStagewiseGrayBoxInverse,
        16: OpenRiffBoxStagewiseTransientGrayBoxInverse,
    }
    if architecture.get("schema") not in model_classes:
        raise ValueError("fresh evaluator admits only sealed model schemas 5 through 16")
    model = model_classes[architecture["schema"]](
        architecture["channels"],
        architecture["depth"],
        architecture["condition_size"],
    )
    model.load_state_dict(payload["state_dict"])
    if architecture["schema"] in {5, 11, 12}:
        model.tone_encoder_sha256 = architecture["tone_encoder_sha256"]
    elif architecture["schema"] in {6, 7, 8, 9, 10, 13, 14}:
        model.base_checkpoint_sha256 = architecture["base_checkpoint_sha256"]
        model.base.tone_encoder_sha256 = architecture["tone_encoder_sha256"]
    model.eval()

    dataset = EgdbPgAmpPairs(
        workspace,
        args.split,
        args.samples,
        args.target_frames,
        args.context_frames,
        SEED + 4,
        args.tone_reference_frames,
    )
    quality = _quality(model, dataset)
    runtime = _runtime(model)
    contract = dataset.contract
    fit_profiles = {profile for _, profile in _profiles(contract, "fit")}
    fresh_profiles = {profile for _, profile in dataset.profile_rows}
    split_contract_key = {
        "fresh_validation": "fresh_validation_not_downloaded",
        "fresh_validation_v2": "fresh_validation_v2_not_downloaded",
    }[args.split]
    nonfresh_tracks = set().union(*(
        set(values)
        for name, values in contract["tracks"].items()
        if name != split_contract_key
    ))
    manifest = model.manifest()
    digest_after = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    gates = {
        "checkpoint_unchanged": digest_after == digest_before,
        "training_validation_sealed": training["data"]["development_audio_opened"] is False,
        "fresh_audio_audited": bool(
            dataset.audit.get("passed")
            and dataset.audit.get(f"{args.split}_downloaded") is True
        ),
        "locked_final_unopened": dataset.audit.get("locked_final_downloaded") is False,
        "profile_disjoint_fresh_validation": not bool(fit_profiles & fresh_profiles),
        "track_disjoint_fresh_validation": not bool(
            nonfresh_tracks & set(dataset.tracks)
        ),
        "wet_only_order_independent": all(
            manifest[name] is False
            for name in (
                "profile_id_input",
                "gain_category_input",
                "graph_order_input",
                "neighbor_effect_input",
                "recurrent_state_input",
                "clean_or_oracle_input",
            )
        ),
        "aggregate_quality": bool(quality["accepted"]),
        "individual_pass_fraction": quality["pass_fraction"] >= 0.80,
        "all_gain_categories": bool(quality["all_categories_accepted"]),
        "all_unseen_profiles": bool(quality["all_unseen_profiles_accepted"]),
        "cpu_budget": runtime["realtime_factor"] <= 0.35,
    }
    accepted = all(gates.values())
    report = {
        "schema": 1,
        "status": (
            f"accepted-{args.split.replace('_', '-')}"
            if accepted else f"rejected-{args.split.replace('_', '-')}"
        ),
        "accepted": accepted,
        "product_candidate": accepted,
        "usable_model": None,
        "mechanism": "amp",
        "model": {
            **manifest,
            "checkpoint": str(checkpoint),
            "sha256_before_validation": digest_before,
            "sha256_after_validation": digest_after,
        },
        args.split: quality,
        "runtime": runtime,
        "gates": gates,
        "data": {
            "split": args.split,
            "samples": len(dataset),
            "profile_count": len(dataset.profile_rows),
            "track_count": len(dataset.tracks),
            "profiles_seen_in_fit": sorted(fit_profiles & fresh_profiles),
            "tracks_seen_outside_fresh_validation": sorted(
                nonfresh_tracks & set(dataset.tracks)
            ),
            "audit": str((workspace / "data/corpus/egdb-pg-subset-v1/audio_audit.json").resolve()),
        },
        "quality": {
            "metric_schema": 2,
            "generated_audio_written": False,
            "demo_generated": False,
            "source_audio_modified": False,
            "locked_final_audio_opened": False,
        },
        "limitations": [
            "software Amp+cab evidence, not physical amplifier evidence",
            "objective fresh-profile acceptance still requires listening acceptance before product promotion",
            "locked-final audio remains unopened",
        ],
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"],
        "accepted": accepted,
        "pass_fraction": quality["pass_fraction"],
        "improvements": {
            name: row["median_reduction"]
            for name, row in quality["metrics"].items()
        },
        "gates": gates,
        "runtime": runtime,
        "checkpoint_sha256": digest_after,
    }, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("fresh_validation", "fresh_validation_v2"),
        default="fresh_validation",
    )
    parser.add_argument("--samples", type=int, default=72)
    parser.add_argument("--target-frames", type=int, default=16384)
    parser.add_argument("--context-frames", type=int, default=4096)
    parser.add_argument("--tone-reference-frames", type=int, default=3 * RATE)
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
