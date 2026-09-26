#!/usr/bin/env python3
"""External OpenSLR 26 audit of the frozen Product4 frequency-profile model."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .ambience2 import HISTORY_FRAMES, AmbienceAbstention, _controls, _render, restore_known_ambience_profile
from .ambience4 import (
    DECAY_DOMAIN_SECONDS,
    PRODUCT4_CLEAN_SOURCE_IDS,
    TARGET_FRAMES_MINIMUM,
    AmbiencePairsV4,
    release_event_frames,
)
from .ambience4_frequency_shaping import profile_frequency_shortening_inverse
from .ambience_model4 import AmbienceFrequencyProfileBankExpert
from .audit_ambience4_shaping import _evaluate
from .openslr26_rir_data import SOURCE_ID, discover_openslr26_rirs, load_openslr26_rir
from .product2 import _condition_clean
from .product_data import Clean, _read, discover_clean


SEED = 20260910
QUALITY_CONTRACT = "tail-removal-with-global-nonregression"
EXPECTED_ARCHITECTURE = "known-profile-frequency-decay-bank-selector"


def _clean_buckets(workspace: Path) -> dict[str, tuple[Clean, ...]]:
    buckets: dict[str, list[Clean]] = defaultdict(list)
    for item in discover_clean(workspace):
        if item.split == "development" and item.source_id in PRODUCT4_CLEAN_SOURCE_IDS:
            buckets[item.source_id].append(item)
    result = {name: tuple(rows) for name, rows in sorted(buckets.items()) if rows}
    if len(result) < 2:
        raise ValueError("OpenSLR 26 audit requires at least two development Clean sources")
    return result


def _event_clean(rows, source: str, total_frames: int, seed: int):
    start = seed % len(rows)
    best = -1
    searched = min(len(rows), 64)
    for offset in range(searched):
        selected = rows[(start + offset) % len(rows)]
        for crop in range(8):
            value = _read(
                selected, total_frames,
                seed + offset * 32_452_843 + crop * 49_979_687,
            )
            score = release_event_frames(value[HISTORY_FRAMES:])
            best = max(best, score)
            if score >= 4:
                return selected, value
    raise ValueError(
        f"OpenSLR 26 audit found no release event for {source}; "
        f"searched_files={searched} best_frames={best}"
    )


def _load_frozen_model(workspace: Path, device: torch.device):
    metrics_path = (workspace / "runs/foundation/product4-reverb-frequency-profile-v2-formal/ambience/metrics.json").resolve()
    report = json.loads(metrics_path.read_text())
    manifest = report["model"]
    selected = next(
        row for row in report["training"]["calibration_quality_candidates"]
        if row["epoch"] == report["training"]["selected_epoch"]
    )
    if manifest["architecture"] != EXPECTED_ARCHITECTURE or report["training"]["selected_epoch"] != 2:
        raise ValueError("frequency-profile checkpoint differs from frozen candidate")
    if not all((selected["all_sources_accepted"], selected["all_rooms_accepted"], selected["all_decay_strata_accepted"])):
        raise ValueError("frequency-profile checkpoint did not pass calibration groups")
    checkpoint_path = Path(manifest["checkpoint"]).resolve()
    payload = checkpoint_path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != manifest["sha256"]:
        raise ValueError("frequency-profile checkpoint digest differs")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    expected = {key: value for key, value in manifest.items() if key not in {"checkpoint", "sha256"}}
    if checkpoint["architecture"] != expected:
        raise ValueError("frequency-profile checkpoint manifest differs from metrics")
    model = AmbienceFrequencyProfileBankExpert(
        channels=manifest["channels"], depth=manifest["depth"],
        n_fft=manifest["n_fft"], hop=manifest["hop"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval().to(device)
    return model, {
        "metrics_path": str(metrics_path),
        "metrics_sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": digest,
        "selected_epoch": 2,
        "candidate_variants": manifest["candidate_variants"],
    }


def _profile_bank(wet, impulse, controls, variants):
    try:
        exact, _ = restore_known_ambience_profile(wet, impulse, controls)
        return np.stack([exact] * len(variants)).astype(np.float32), "exact"
    except AmbienceAbstention:
        rows = []
        for variant in variants:
            parameters = {key: value for key, value in variant.items() if key != "id"}
            try:
                restored, _ = profile_frequency_shortening_inverse(wet, impulse, controls, **parameters)
            except AmbienceAbstention:
                restored = wet.copy()
            rows.append(restored)
        return np.stack(rows).astype(np.float32), "shaping"


def _materialize(workspace: Path, model, seal: dict, device: torch.device):
    root = (workspace / "data/corpus/openslr26-simulated-rir").resolve()
    rirs, inventory = discover_openslr26_rirs(root)
    clean_buckets = _clean_buckets(workspace)
    total_frames = HISTORY_FRAMES + TARGET_FRAMES_MINIMUM
    rows = []
    elapsed = 0.0
    for rir_index, record in enumerate(rirs):
        impulse = load_openslr26_rir(record.path)
        decay = AmbiencePairsV4.decay_stratum(record.decay_p999_seconds)
        for source_index, (source, clean_rows) in enumerate(clean_buckets.items()):
            seed = SEED + rir_index * 15_485_863 + source_index * 32_452_843
            rng = random.Random(seed)
            selected, clean = _event_clean(clean_rows, source, total_frames, rng.getrandbits(63))
            clean, _ = _condition_clean(clean, rng)
            wet, control_values = _render(clean, impulse, rng)
            try:
                bank, mode = _profile_bank(wet, impulse, control_values, seal["candidate_variants"])
                wet_tensor = torch.from_numpy(wet)[None].to(device)
                bank_tensor = torch.from_numpy(bank)[None].to(device)
                controls_tensor = torch.from_numpy(_controls(control_values, DECAY_DOMAIN_SECONDS))[None].to(device)
                if device.type == "mps":
                    torch.mps.synchronize()
                started = time.perf_counter()
                with torch.inference_mode():
                    restored, _ = model(wet_tensor, controls_tensor, bank_tensor)
                if device.type == "mps":
                    torch.mps.synchronize()
                elapsed += time.perf_counter() - started
                restored = restored[0].cpu().numpy().astype(np.float32)
            except (AmbienceAbstention, ValueError) as error:
                rows.append({
                    "source": source, "room": record.room, "decay": decay,
                    "clean_group": selected.group, "error": str(error),
                })
                continue
            rows.append({
                "wet": wet[HISTORY_FRAMES:], "restored": restored[HISTORY_FRAMES:],
                "clean": clean[HISTORY_FRAMES:], "source": source,
                "room": record.room, "decay": decay,
                "clean_group": selected.group, "mode": mode,
            })
    return rows, elapsed, {
        **inventory,
        "development_clean_sources": list(clean_buckets),
        "crops_per_source_per_profile": 1,
        "attempted_examples": len(rows),
    }


def audit(workspace: Path, device: torch.device) -> dict:
    model, seal = _load_frozen_model(workspace, device)
    rows, elapsed, inventory = _materialize(workspace, model, seal, device)
    evaluation = _evaluate(rows, QUALITY_CONTRACT)
    fallback_tail = evaluation["shaping_fallback"].get("tail", {})
    safety_gates = {
        "all_standard_external_gates": bool(evaluation["accepted"]),
        "fallback_pass_fraction": fallback_tail.get("pass_fraction", 0.0) >= 0.75,
        "fallback_median_tail_reduction": fallback_tail.get("median_tail_excess_reduction", 0.0) >= 0.30,
        "fallback_never_adds_reverb": fallback_tail.get("added_reverb_fraction", 1.0) == 0.0,
    }
    accepted = all(safety_gates.values())
    return {
        "schema": 1,
        "status": "accepted-frequency-profile-external-simulation" if accepted else "rejected-frequency-profile-external-simulation",
        "accepted": accepted,
        "promotable": False,
        "mechanism": "ambience",
        "quality_contract": QUALITY_CONTRACT,
        "frozen_safety_gates": safety_gates,
        "external_validation": evaluation,
        "inventory": inventory,
        "model_seal": seal,
        "selection_contract": {
            "dataset": SOURCE_ID,
            "external_data_used_for_parameter_or_checkpoint_selection": False,
            "checkpoint_frozen_before_external_evaluation": True,
            "evaluation_count": 1,
        },
        "profile_contract": {
            "profile_required": True,
            "profile_inputs": ["rir", "mix", "room_gain_db"],
            "clean_input": False,
            "unknown_profile_decision": "abstain",
            "chain_order_input": False,
            "graph_order_input": False,
            "neighbor_effect_input": False,
        },
        "runtime": {
            "accelerator": device.type,
            "network_total_seconds": elapsed,
            "network_mean_seconds_per_attempt": elapsed / len(rows),
            "physical_candidate_generation_included": False,
            "audio_callback": False,
        },
        "provenance": {
            "rir_source": SOURCE_ID,
            "locked_final_opened": False,
            "generated_audio_written": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument(
        "--output", type=Path,
        default=Path("runs/foundation/product4-reverb-frequency-profile-openslr26-v1/metrics.json"),
    )
    args = parser.parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace OpenSLR 26 model audit: {output}")
    report = audit(args.workspace.resolve(), torch.device(args.device))
    output.parent.mkdir(parents=True, exist_ok=False)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "metrics": str(output), "status": report["status"],
        "accepted": report["accepted"],
        "attempted_examples": report["inventory"]["attempted_examples"],
        "modes": report["external_validation"]["modes"],
        "gates": report["external_validation"]["gates"],
        "frozen_safety_gates": report["frozen_safety_gates"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
