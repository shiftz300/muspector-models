#!/usr/bin/env python3
"""Convert a verified public ASRNN stable RAT result into the local runtime schema."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .asrnn_data import LICENSE, RECORD_URL
from .stable_rat import StableRatRenderer


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def import_checkpoint(source: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError(f"stable RAT output already exists: {output}")
    payload = torch.load(source, map_location="cpu", weights_only=True)
    args = payload.get("model_args", {})
    if (
        args.get("type") != "StableLSTM_inf"
        or args.get("input_size") != 1
        or args.get("output_size") != 1
        or args.get("condition_size") != 3
    ):
        raise ValueError("source is not a supported ASRNN stable RAT checkpoint")
    hidden_size = int(args["hidden_size"])
    layers = int(args["num_layers"])
    input_coef = float(args["input_coef"])
    source_state = payload["model_state"]
    model = StableRatRenderer(hidden_size, layers, input_coef)
    local_state = model.state_dict()
    for index in range(layers):
        for name in ("weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0"):
            local_state[f"rnn_layers.{index}.{name}"] = source_state[
                f"rnn_layers.{index}.lstm.{name}"
            ].clone()
    local_state["output_layer.weight"] = source_state["output_layer.weight"].clone()
    model.load_state_dict(local_state, strict=True)

    candidate_norms = []
    for layer in model.rnn_layers:
        candidate = layer.weight_hh_l0[2 * hidden_size : 3 * hidden_size]
        candidate_norms.append(
            float(torch.linalg.matrix_norm(candidate, ord=float("inf")).detach())
        )
    with torch.inference_mode():
        silence = torch.zeros(3, 4_096)
        static = torch.tensor(((0.0, 1.0, 0.0), (0.5, 0.5, 0.5), (1.0, 0.0, 1.0)))
        static_output, _ = model(silence, static)
        time = torch.linspace(0.0, 1.0, 4_096)
        dynamic = torch.stack(
            (
                torch.sin(time * 7.0).mul(0.5).add(0.5),
                torch.cos(time * 11.0).mul(0.5).add(0.5),
                torch.sin(time * 5.0).mul(0.5).add(0.5),
            ),
            dim=1,
        ).unsqueeze(0)
        dynamic_output, _ = model(torch.zeros(1, 4_096), dynamic)
    if float(static_output.abs().max()) != 0.0 or float(dynamic_output.abs().max()) != 0.0:
        raise ValueError("source stable RAT checkpoint does not preserve exact silence")
    if max(candidate_norms) >= 1.0:
        raise ValueError("source stable RAT candidate recurrent norm is not contractive")

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": 1,
            "sample_rate": 48_000,
            "architecture": "stable-conditioned-lstm",
            "hidden_size": hidden_size,
            "layers": layers,
            "input_coef": input_coef,
            "state_dict": model.state_dict(),
            "source_checkpoint_sha256": _sha256(source),
            "dataset_record": RECORD_URL,
            "dataset_license": LICENSE,
            "scope": "internal non-commercial real-hardware pilot",
        },
        output,
    )
    report = {
        "schema": 1,
        "source": str(source),
        "source_sha256": _sha256(source),
        "output": str(output),
        "output_sha256": _sha256(output),
        "candidate_recurrent_infinity_norms": candidate_norms,
        "static_silence_max_absolute_output": float(static_output.abs().max()),
        "dynamic_control_silence_max_absolute_output": float(dynamic_output.abs().max()),
        "license": LICENSE,
        "official_gpl_runtime_embedded": False,
        "physical_audio_devices_used": False,
    }
    (output.parent / "import-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(import_checkpoint(args.source, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
