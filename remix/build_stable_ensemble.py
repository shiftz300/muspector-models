#!/usr/bin/env python3
"""Build a fixed causal ensemble and admit it on the unchanged calibration split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .asrnn_effects import effect_files
from .stable_effect import StableEffectEnsemble, load_stable_effect
from .train_asrnn_phase7 import _evaluate, _partition, _rows


def build(args) -> dict:
    if args.output.exists():
        raise ValueError(f"ensemble output already exists: {args.output}")
    imported = [load_stable_effect(path) for path in args.members]
    if any(payload["schema"] != 2 or payload["device"] != "cs3" for _, payload in imported):
        raise ValueError("this experiment requires standard CS-3 members")
    model = StableEffectEnsemble([item[0] for item in imported], args.weights).eval()
    sources = [{"path": str(path), "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "source_sha256": payload["source_checkpoint_sha256"], "weight": weight}
               for path, (_, payload), weight in zip(args.members, imported, args.weights)]
    definition_hash = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    payload = {"schema": 3, "architecture": "stable-conditioned-lstm-ensemble", "sample_rate": 48000,
               "device": "cs3", "control_count": 1, "members": [item[1] for item in imported],
               "weights": args.weights, "sources": sources, "license": "CC-BY-NC-4.0",
               "local_gradient_updates": 0}
    args.output.mkdir(parents=True)
    checkpoint = args.output / "candidate.pt"
    torch.save(payload, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    # Evaluate the reloaded bytes, not only the in-memory construction.
    model, _ = load_stable_effect(checkpoint)
    audit = json.loads(args.group_audit.read_text())
    _, calibration = _partition(effect_files(args.corpus, "cs3", "train"), audit)
    metrics = _evaluate(model, _rows(calibration, "cs3"), torch.device("cpu"), 16)
    norms = [float(layer.weight_hh_l0.detach()[2 * member.hidden_size:3 * member.hidden_size].abs().sum(1).max())
             for member in model.members for layer in member.rnn_layers]
    row = {"member": "fixed-convex-ensemble", "source_sha256": definition_hash,
           "checkpoint_sha256": digest, "architecture": payload["architecture"], "sources": sources,
           "parameters": sum(parameter.numel() for parameter in model.parameters()), "calibration": metrics,
           "import_runtime": {"candidate_recurrent_infinity_norms": norms}}
    report = {"schema": 1, "phase": "phase-8-fixed-ensemble", "screen_complete": True, "expected_candidates": 1,
              "archive_sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest(),
              "group_audit_sha256": hashlib.sha256(args.group_audit.read_bytes()).hexdigest(),
              "calibration_caveat": "public pretrained weights may have seen official train; selection only, not independent final evidence",
              "results": [row], "selected": row, "source_audio_modified": False, "physical_audio_devices_used": False,
              "weights_fixed_before_challenge": True, "automatic_normalization": False, "automatic_limiting": False}
    (args.output / "screen.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--members", type=Path, nargs="+", required=True)
    parser.add_argument("--weights", type=float, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus/asrnn-physical-effects"))
    parser.add_argument("--group-audit", type=Path, default=Path("remix/runs/asrnn-cs3-phase7-group-audit.json"))
    parser.add_argument("--archive", type=Path, default=Path("data/downloads/asrnn-results-20406285.zip"))
    args = parser.parse_args()
    report = build(args)
    print(json.dumps(report["selected"]["calibration"], indent=2))
    if not report["selected"]["calibration"]["passes_selection_gate"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
