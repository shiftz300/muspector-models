#!/usr/bin/env python3
"""Audit frozen Reverb v3 on the calibration-only per-example safety contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch

from .ambience4 import AmbiencePairsV4MixedFrequencyProfileBank
from .ambience_model4 import AmbienceFrequencyProfileBankExpert
from .reverb_quality2 import measure, summarize
from .train_ambience4_frequency_profile import SEED


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "runs/foundation/product4-reverb-frequency-profile-v3-mixed-formal/"
            "ambience/model.pt"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "runs/foundation/product4-reverb-frequency-profile-v3-"
            "acceptance2-calibration-v1.json"
        ),
    )
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--samples", type=int, default=192)
    args = parser.parse_args()
    if args.samples < 24:
        raise ValueError("acceptance2 audit needs at least 24 calibration examples")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace acceptance2 audit: {output}")

    checkpoint = args.checkpoint.resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    architecture = payload["architecture"]
    model = AmbienceFrequencyProfileBankExpert(
        channels=architecture["channels"], depth=architecture["depth"],
        n_fft=architecture["n_fft"], hop=architecture["hop"],
    ).to(args.device).eval()
    model.load_state_dict(payload["state_dict"])
    dataset = AmbiencePairsV4MixedFrequencyProfileBank(
        args.workspace.resolve(), "calibration", args.samples, 65_536, SEED
    )
    rows = []
    results = []
    groups = defaultdict(list)
    with torch.inference_mode():
        for index in range(len(dataset)):
            item = dataset[index]
            restored, _ = model(
                item["wet"][None].to(args.device),
                item["controls"][None].to(args.device),
                item["late_base"][None].to(args.device),
            )
            start = int(item["target_start"])
            result = measure(
                item["wet"][start:].numpy(),
                restored[0, start:].cpu().numpy(),
                item["clean"][start:].numpy(),
            )
            decay = dataset.decay_stratum(
                float(item["control_values"]["decay_p999_seconds"])
            )
            room = dataset.room_group(item)
            results.append(result)
            groups[f"source:{item['source_id']}"].append(result)
            groups[f"room:{room}"].append(result)
            groups[f"decay:{decay}"].append(result)
            rows.append({
                "index": index,
                "source_id": item["source_id"],
                "rir_source_id": item["rir_source_id"],
                "room_group": room,
                "decay_stratum": decay,
                "profile_mode": item["profile_mode"],
                "decision": result["decision"],
                "passed": result["passed"],
                "evidence_measurable": result["evidence_measurable"],
                "hard_bypass_max_absolute_error": result["hard_bypass_max_absolute_error"],
                "tail_passed": result["tail"].get("passed"),
                "active_reduction": result["active_foreground"].get("reduction"),
                "universal_passed": result["universal"]["passed"],
            })
    aggregate = summarize(results)
    group_reports = {name: summarize(values) for name, values in sorted(groups.items())}
    accepted = bool(aggregate["accepted"] and all(row["accepted"] for row in group_reports.values()))
    report = {
        "schema": 1,
        "status": "accepted-calibration-contract" if accepted else "rejected-calibration-contract",
        "accepted": accepted,
        "partition": "calibration",
        "development_or_listening_rows_used_for_threshold_selection": False,
        "checkpoint": {
            "path": str(checkpoint.relative_to(args.workspace.resolve())),
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        },
        "accelerator": args.device,
        "aggregate": aggregate,
        "groups": group_reports,
        "rows": rows,
        "locked_final_accessed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"],
        "accepted": accepted,
        "aggregate": aggregate,
        "failed_groups": [name for name, row in group_reports.items() if not row["accepted"]],
        "output": str(output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
