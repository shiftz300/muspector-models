#!/usr/bin/env python3
"""True per-bin convex-hull oracle for the Product4 frequency-profile bank."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from .ambience4 import AmbiencePairsV4MixedFrequencyProfileBank
from .reverb_quality2 import measure, summarize
from .train_ambience4_frequency_profile import SEED


def _oracle(
    wet: torch.Tensor,
    bank: torch.Tensor,
    clean: torch.Tensor,
    *,
    n_fft: int = 1_024,
    hop: int = 256,
) -> torch.Tensor:
    window = torch.hann_window(n_fft, device=wet.device, dtype=wet.dtype)
    waveforms = torch.cat((wet[None], bank), dim=0)
    spectra = torch.stft(
        waveforms, n_fft, hop, window=window, center=True,
        pad_mode="constant", return_complex=True,
    )
    clean_spectrum = torch.stft(
        clean, n_fft, hop, window=window, center=True,
        pad_mode="constant", return_complex=True,
    )
    wet_spectrum = spectra[0]
    directions = spectra[1:] - wet_spectrum[None]
    target = clean_spectrum - wet_spectrum
    fractions = (
        target[None].real * directions.real + target[None].imag * directions.imag
    ) / directions.abs().square().add(1.0e-8)
    fractions = fractions.clamp(0.0, 1.0)
    projected = wet_spectrum[None] + fractions * directions
    errors = (projected - clean_spectrum[None]).abs().square()
    indices = errors.argmin(dim=0)
    selected_fraction = torch.gather(fractions, 0, indices[None])[0]
    selected_direction = torch.complex(
        torch.gather(directions.real, 0, indices[None])[0],
        torch.gather(directions.imag, 0, indices[None])[0],
    )
    selected_error = torch.gather(errors, 0, indices[None])[0]
    useful = selected_error < (wet_spectrum - clean_spectrum).abs().square()
    estimate = wet_spectrum + selected_fraction * useful * selected_direction
    return torch.istft(
        estimate, n_fft, hop, window=window, center=True, length=wet.shape[0]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output", type=Path,
        default=Path(
            "runs/foundation/product4-reverb-frequency-profile-"
            "acceptance2-tf-oracle-calibration-v1.json"
        ),
    )
    parser.add_argument("--samples", type=int, default=192)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    args = parser.parse_args()
    if args.samples < 24:
        raise ValueError("time-frequency oracle needs at least 24 calibration examples")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace time-frequency oracle: {output}")
    dataset = AmbiencePairsV4MixedFrequencyProfileBank(
        args.workspace.resolve(), "calibration", args.samples, 65_536, SEED
    )
    selected_results = []
    rows = []
    groups = defaultdict(list)
    with torch.inference_mode():
        for index in range(len(dataset)):
            item = dataset[index]
            restored = _oracle(
                item["wet"].to(args.device),
                item["late_base"].to(args.device),
                item["clean"].to(args.device),
            ).cpu().numpy()
            start = int(item["target_start"])
            wet = item["wet"][start:].numpy()
            clean = item["clean"][start:].numpy()
            candidate = restored[start:]
            result = measure(wet, candidate, clean)
            if result["decision"] != "effective-restoration":
                candidate = wet.copy()
                result = measure(wet, candidate, clean)
            selected_results.append(result)
            decay = dataset.decay_stratum(
                float(item["control_values"]["decay_p999_seconds"])
            )
            room = dataset.room_group(item)
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
                "oracle_decision": result["decision"],
                "evidence_measurable": result["evidence_measurable"],
            })
    aggregate = summarize(selected_results)
    group_reports = {name: summarize(values) for name, values in sorted(groups.items())}
    accepted = bool(aggregate["accepted"] and all(row["accepted"] for row in group_reports.values()))
    report = {
        "schema": 1,
        "status": (
            "accepted-training-only-time-frequency-clean-oracle"
            if accepted else "rejected-training-only-time-frequency-clean-oracle"
        ),
        "accepted": accepted,
        "partition": "calibration",
        "oracle": "per-bin best convex interpolation between Wet and one physical candidate",
        "fallback": "exact Wet hard bypass when the oracle output fails the per-example contract",
        "deployable": False,
        "clean_input_at_runtime": False,
        "development_or_listening_rows_used": False,
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
