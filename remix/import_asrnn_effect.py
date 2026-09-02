#!/usr/bin/env python3
"""Convert a verified public ASRNN stable effect checkpoint to the local schema."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .asrnn_data import LICENSE, RECORD_URL
from .asrnn_effects import effect_spec
from .stable_effect import StableEffectRenderer


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def import_checkpoint(source: Path, output: Path, device: str, *, allow_spectral_regularization: bool = False) -> dict:
    if output.exists():
        raise ValueError(f"stable effect output already exists: {output}")
    spec = effect_spec(device)
    payload = torch.load(source, map_location="cpu", weights_only=True)
    args = payload.get("model_args", {})
    if (
        args.get("type") not in (("StableLSTM_inf", "StableLSTM_2") if allow_spectral_regularization else ("StableLSTM_inf",))
        or args.get("input_size") != 1
        or args.get("output_size") != 1
        or args.get("condition_size") != len(spec.control_names)
    ):
        raise ValueError(
            f"source is not a supported stable {spec.display_name} checkpoint"
        )
    hidden_size = int(args["hidden_size"])
    layers = int(args["num_layers"])
    input_coef = float(args["input_coef"])
    source_state = payload["model_state"]
    model = StableEffectRenderer(
        len(spec.control_names),
        hidden_size,
        layers,
        input_coef,
        spec.inverted_controls,
    )
    local_state = model.state_dict()
    for index in range(layers):
        for name in ("weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0"):
            local_state[f"rnn_layers.{index}.{name}"] = source_state[
                f"rnn_layers.{index}.lstm.{name}"
            ].clone()
    local_state["output_layer.weight"] = source_state["output_layer.weight"].clone()
    model.load_state_dict(local_state, strict=True)

    candidate_norms = []
    spectral_norms = []
    for layer in model.rnn_layers:
        candidate = layer.weight_hh_l0[2 * hidden_size : 3 * hidden_size]
        candidate_norms.append(
            float(torch.linalg.matrix_norm(candidate, ord=float("inf")).detach())
        )
        spectral_norms.append(float(torch.linalg.matrix_norm(candidate, ord=2).detach()))
    with torch.inference_mode():
        silence = torch.zeros(3, 4_096)
        static = torch.stack(
            (
                torch.zeros(model.control_count),
                torch.full((model.control_count,), 0.5),
                torch.ones(model.control_count),
            )
        )
        static_output, _ = model(silence, static)
        time = torch.linspace(0.0, 1.0, 4_096)
        dynamic = torch.stack(
            [
                torch.sin(time * (7.0 + 2.0 * index)).mul(0.5).add(0.5)
                for index in range(model.control_count)
            ],
            dim=1,
        ).unsqueeze(0)
        dynamic_output, _ = model(torch.zeros(1, 4_096), dynamic)
    if float(static_output.abs().max()) != 0.0 or float(dynamic_output.abs().max()) != 0.0:
        raise ValueError("source stable effect checkpoint does not preserve exact silence")
    norms = spectral_norms if args['type'] == 'StableLSTM_2' else candidate_norms
    if max(norms) >= 1.0:
        raise ValueError("source stable effect candidate recurrent norm is not contractive")

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": 2,
            "sample_rate": 48_000,
            "architecture": "stable-conditioned-lstm",
            "device": spec.key,
            "device_name": spec.display_name,
            "control_count": model.control_count,
            "control_names": list(spec.control_names),
            "inverted_controls": list(spec.inverted_controls),
            "hidden_size": hidden_size,
            "layers": layers,
            "input_coef": input_coef,
            "state_dict": model.state_dict(),
            "source_checkpoint_sha256": sha256(source),
            "source_regularization": args['type'],
            "formal_infinity_norm_stability": args['type'] == 'StableLSTM_inf',
            "dataset_record": RECORD_URL,
            "dataset_license": LICENSE,
            "scope": "internal non-commercial real-hardware pilot",
        },
        output,
    )
    return {
        "schema": 1,
        "device": spec.key,
        "source": str(source),
        "source_sha256": sha256(source),
        "output": str(output),
        "output_sha256": sha256(output),
        "candidate_recurrent_infinity_norms": candidate_norms,
        "candidate_recurrent_spectral_norms": spectral_norms,
        "formal_infinity_norm_stability": args['type'] == 'StableLSTM_inf',
        "static_silence_max_absolute_output": float(static_output.abs().max()),
        "dynamic_control_silence_max_absolute_output": float(dynamic_output.abs().max()),
        "license": LICENSE,
        "official_gpl_runtime_embedded": False,
        "physical_audio_devices_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("rat", "dfz", "cs3"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = import_checkpoint(args.source, args.output, args.device)
    report_path = args.output.parent / "import-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
