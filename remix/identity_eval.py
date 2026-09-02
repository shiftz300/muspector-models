#!/usr/bin/env python3
"""Lock and replay shadow device identity behind the canonical chain."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .adapter import analysis
from .asrnn_effects import effect_files, read_effect_pair
from .chain import ChainRuntime
from .identity import IdentityRuntime, Router
from .packages import digest
from .route import RouteRuntime


GATES = {
    "rat_recall_total": 0.50,
    "rat_recall_eligible": 0.60,
    "dfz_false_route_total": 0.10,
    "dfz_false_route_eligible": 0.15,
    "automatic_deliveries": 0,
    "input_mutations": 0,
    "artifact_mutations": 0,
}


def spread(paths: list[Path], count: int) -> list[Path]:
    if len(paths) < count:
        raise ValueError(f"not enough files: {len(paths)} < {count}")
    indices = np.linspace(0, len(paths) - 1, count).round().astype(int)
    return [paths[int(index)] for index in indices]


def lock(args) -> dict:
    if args.output.exists():
        raise FileExistsError(f"identity lock already exists: {args.output}")
    rat = json.loads(args.rat_lock.read_text())
    records = [
        {
            "device": "rat",
            "path": row["path"],
            "sha256": row["sha256"],
            "expected_route": "rat",
        }
        for row in rat["records"]
    ]
    for path in spread(effect_files(args.corpus, "dfz", "eval"), args.negatives):
        records.append(
            {
                "device": "dfz",
                "path": path.relative_to(args.corpus).as_posix(),
                "sha256": digest(path),
                "expected_route": None,
            }
        )
    report = {
        "schema": 1,
        "status": "locked-development-replay",
        "source": "ASRNN official RAT and DFZ eval reused for identity development only",
        "record": "https://zenodo.org/records/20406285",
        "license": "CC-BY-NC-4.0",
        "selection": "existing 32-file RAT development lock plus evenly spaced sorted DFZ eval files before identity inference",
        "records": records,
        "scope": "RAT candidate recall and same-family DFZ false-route audit; not sealed or release evidence",
        "source_audio_modified": False,
        "physical_audio_devices_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _rate(rows: list[dict], key: str, denominator: str) -> float:
    selected = rows if denominator == "all" else [row for row in rows if row["eligible"]]
    return sum(bool(row[key]) for row in selected) / max(len(selected), 1)


def replay(args) -> dict:
    if args.output.exists():
        raise FileExistsError(f"identity report already exists: {args.output}")
    lock_bytes = args.lock.read_bytes()
    locked = json.loads(lock_bytes)
    if locked.get("status") != "locked-development-replay":
        raise ValueError("identity development inputs were not locked")
    chain = ChainRuntime(
        args.family, args.manifest, args.gate, args.order, args.bundle, args.evidence
    )
    identity = IdentityRuntime(args.identity)
    candidate = RouteRuntime(args.route, identity) if args.route else identity
    router = Router(candidate)
    rows = []
    for index, record in enumerate(locked["records"]):
        path = args.corpus / record["path"]
        if digest(path) != record["sha256"]:
            raise ValueError("locked identity source changed")
        dry, wet, _ = read_effect_pair(path, record["device"])
        before = hashlib.sha256(dry.tobytes() + wet.tobytes()).hexdigest()
        canonical = chain.infer(
            analysis(dry, 48_000, 44_100, 220_500),
            analysis(wet, 48_000, 44_100, 220_500),
        )
        eligible = canonical["decision"] == "accepted" and canonical["active"] == ["drive"]
        route = router.infer(canonical, dry, wet, 48_000)
        identity_report = route["identity"]
        rows.append(
            {
                "index": index,
                "device": record["device"],
                "path": record["path"],
                "chain_decision": canonical["decision"],
                "chain_active": canonical["active"],
                "eligible": eligible,
                "identity_decision": identity_report["decision"] if identity_report else None,
                "identity_candidate": identity_report["candidate"] if identity_report else None,
                "identity_label": identity_report["label"] if identity_report else None,
                "identity_score": identity_report["score"] if identity_report else None,
                "route": route["device"],
                "rat_route": route["device"] == "rat",
                "automatic_delivery": route["automatic_delivery"],
                "inputs_unchanged": before
                == hashlib.sha256(dry.tobytes() + wet.tobytes()).hexdigest(),
            }
        )
        if (index + 1) % 8 == 0 or index + 1 == len(locked["records"]):
            print(json.dumps({"stage": "identity", "done": index + 1, "total": len(locked["records"])}), flush=True)
    chain.assert_artifacts_unchanged()
    candidate.assert_artifacts_unchanged()
    for record in locked["records"]:
        if digest(args.corpus / record["path"]) != record["sha256"]:
            raise ValueError("identity source changed during replay")
    groups = {device: [row for row in rows if row["device"] == device] for device in ("rat", "dfz")}
    metrics = {
        "examples": len(rows),
        "rat": {
            "examples": len(groups["rat"]),
            "eligible": sum(row["eligible"] for row in groups["rat"]),
            "routes": sum(row["rat_route"] for row in groups["rat"]),
            "recall_total": _rate(groups["rat"], "rat_route", "all"),
            "recall_eligible": _rate(groups["rat"], "rat_route", "eligible"),
        },
        "dfz": {
            "examples": len(groups["dfz"]),
            "eligible": sum(row["eligible"] for row in groups["dfz"]),
            "false_routes": sum(row["rat_route"] for row in groups["dfz"]),
            "false_route_total": _rate(groups["dfz"], "rat_route", "all"),
            "false_route_eligible": _rate(groups["dfz"], "rat_route", "eligible"),
        },
        "automatic_deliveries": sum(row["automatic_delivery"] for row in rows),
        "input_mutations": sum(not row["inputs_unchanged"] for row in rows),
    }
    failures = []
    if metrics["rat"]["recall_total"] < GATES["rat_recall_total"]:
        failures.append("rat-recall-total")
    if metrics["rat"]["recall_eligible"] < GATES["rat_recall_eligible"]:
        failures.append("rat-recall-eligible")
    if metrics["dfz"]["false_route_total"] > GATES["dfz_false_route_total"]:
        failures.append("dfz-false-route-total")
    if metrics["dfz"]["false_route_eligible"] > GATES["dfz_false_route_eligible"]:
        failures.append("dfz-false-route-eligible")
    if metrics["automatic_deliveries"] > GATES["automatic_deliveries"]:
        failures.append("automatic-delivery")
    if metrics["input_mutations"] > GATES["input_mutations"]:
        failures.append("input-mutation")
    report = {
        "schema": 1,
        "status": "accepted-shadow-development" if not failures else "rejected",
        "accepted": not failures,
        "source": locked["source"],
        "scope": locked["scope"],
        "lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
        "gates": GATES,
        "metrics": metrics,
        "failures": failures,
        "rows": rows,
        "artifacts_unchanged": True,
        "route_sha256": digest(args.route) if args.route else None,
        "source_audio_modified": False,
        "physical_audio_devices_used": False,
        "automatic_normalization": False,
        "automatic_limiting": False,
        "lossy_reencoding": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    locking = commands.add_parser("lock")
    locking.add_argument("--rat-lock", type=Path, required=True)
    locking.add_argument("--corpus", type=Path, required=True)
    locking.add_argument("--negatives", type=int, default=32)
    locking.add_argument("--output", type=Path, required=True)
    running = commands.add_parser("replay")
    running.add_argument("--lock", type=Path, required=True)
    running.add_argument("--corpus", type=Path, required=True)
    running.add_argument("--identity", type=Path, required=True)
    running.add_argument("--route", type=Path)
    running.add_argument("--family", type=Path, required=True)
    running.add_argument("--manifest", type=Path, required=True)
    running.add_argument("--gate", type=Path, required=True)
    running.add_argument("--order", type=Path, required=True)
    running.add_argument("--bundle", type=Path, required=True)
    running.add_argument("--evidence", type=Path, required=True)
    running.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = lock(args) if args.mode == "lock" else replay(args)
    print(json.dumps({"status": report["status"], "accepted": report.get("accepted"), "failures": report.get("failures", []), "metrics": report.get("metrics")}))
    if args.mode == "replay" and not report["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
