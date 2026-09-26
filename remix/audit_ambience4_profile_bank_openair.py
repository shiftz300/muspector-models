#!/usr/bin/env python3
"""One-shot OpenAIR audit of the frozen Product4 profile-bank gray box."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .ambience2 import HISTORY_FRAMES, AmbienceAbstention, _controls, _render, restore_known_ambience_profile
from .ambience4 import DECAY_DOMAIN_SECONDS, TARGET_FRAMES_MINIMUM, AmbiencePairsV4
from .ambience4_shaping import profile_shortening_inverse
from .ambience_model4 import AmbienceProfileBankExpert
from .audit_ambience4_shaping import _evaluate
from .audit_ambience4_shaping_ok5 import _clean_buckets, _event_clean
from .openair_rir_data import SOURCE_ID, discover_openair_rirs, load_openair_rir
from .product2 import _condition_clean


SEED = 20260909
QUALITY_CONTRACT = "tail-removal-with-global-nonregression"
EXPECTED_ARCHITECTURE = "known-profile-response-shortening-bank-selector"


def _load_frozen_model(workspace: Path, device: torch.device) -> tuple[AmbienceProfileBankExpert, dict]:
    metrics_path = (
        workspace
        / "runs/foundation/product4-reverb-profile-bank-graybox-v17/ambience/metrics.json"
    ).resolve()
    report = json.loads(metrics_path.read_text())
    manifest = report["model"]
    if manifest["architecture"] != EXPECTED_ARCHITECTURE:
        raise ValueError("profile-bank architecture differs from frozen screen")
    if not report["development"]["accepted"] or report["training"]["selected_epoch"] != 4:
        raise ValueError("profile-bank checkpoint is not the frozen internal candidate")
    checkpoint_path = Path(manifest["checkpoint"]).resolve()
    payload = checkpoint_path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != manifest["sha256"]:
        raise ValueError("profile-bank checkpoint digest differs")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint["architecture"] != {key: value for key, value in manifest.items() if key not in {"checkpoint", "sha256"}}:
        raise ValueError("profile-bank checkpoint manifest differs from metrics")
    model = AmbienceProfileBankExpert(
        channels=manifest["channels"],
        depth=manifest["depth"],
        n_fft=manifest["n_fft"],
        hop=manifest["hop"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval().to(device)
    return model, {
        "metrics_path": str(metrics_path),
        "metrics_sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": digest,
        "selected_epoch": report["training"]["selected_epoch"],
        "candidate_variants": manifest["candidate_variants"],
    }


def _profile_bank(wet: np.ndarray, impulse: np.ndarray, controls: dict, variants: list[dict]) -> tuple[np.ndarray, str]:
    try:
        exact, _ = restore_known_ambience_profile(wet, impulse, controls)
        return np.stack([exact] * len(variants)).astype(np.float32), "exact"
    except AmbienceAbstention:
        rows = []
        for variant in variants:
            rows.append(profile_shortening_inverse(
                wet,
                impulse,
                controls,
                early_ms=variant["early_ms"],
                target_rt60_ms=variant["target_rt60_ms"],
                maximum_gain=variant["maximum_gain"],
                low_band_strength=variant["low_band_strength"],
                high_band_strength=variant["high_band_strength"],
                low_band_end_hz=variant["transition_band_hz"][0],
                high_band_start_hz=variant["transition_band_hz"][1],
            )[0])
        return np.stack(rows).astype(np.float32), "shaping"


def _materialize(
    workspace: Path,
    model: AmbienceProfileBankExpert,
    seal: dict,
    device: torch.device,
    crops_per_source: int,
) -> tuple[list[dict], float, dict]:
    rir_root = (workspace / "data/corpus/openair-external-v1").resolve()
    rirs, inventory = discover_openair_rirs(rir_root)
    clean_buckets = _clean_buckets(workspace)
    total_frames = HISTORY_FRAMES + TARGET_FRAMES_MINIMUM
    rows = []
    elapsed = 0.0
    for rir_index, record in enumerate(rirs):
        impulse = load_openair_rir(record.path)
        decay = AmbiencePairsV4.decay_stratum(record.decay_p999_seconds)
        for source_index, (source, clean_rows) in enumerate(clean_buckets.items()):
            for crop in range(crops_per_source):
                seed = SEED + rir_index * 15_485_863 + source_index * 32_452_843 + crop * 49_979_687
                rng = random.Random(seed)
                selected, clean = _event_clean(
                    clean_rows, source, total_frames, rng.getrandbits(63)
                )
                clean, _ = _condition_clean(clean, rng)
                wet, control_values = _render(clean, impulse, rng)
                try:
                    bank, mode = _profile_bank(
                        wet, impulse, control_values, seal["candidate_variants"]
                    )
                    wet_tensor = torch.from_numpy(wet)[None].to(device)
                    bank_tensor = torch.from_numpy(bank)[None].to(device)
                    controls = torch.from_numpy(
                        _controls(control_values, DECAY_DOMAIN_SECONDS)
                    )[None].to(device)
                    if device.type == "mps":
                        torch.mps.synchronize()
                    started = time.perf_counter()
                    with torch.inference_mode():
                        restored, _ = model(wet_tensor, controls, bank_tensor)
                    if device.type == "mps":
                        torch.mps.synchronize()
                    elapsed += time.perf_counter() - started
                    restored = restored[0].cpu().numpy().astype(np.float32)
                except (AmbienceAbstention, ValueError) as error:
                    rows.append({
                        "source": source,
                        "room": record.room,
                        "decay": decay,
                        "clean_group": selected.group,
                        "error": str(error),
                    })
                    continue
                rows.append({
                    "wet": wet[HISTORY_FRAMES:],
                    "restored": restored[HISTORY_FRAMES:],
                    "clean": clean[HISTORY_FRAMES:],
                    "source": source,
                    "room": record.room,
                    "decay": decay,
                    "clean_group": selected.group,
                    "mode": mode,
                })
    return rows, elapsed, {
        **inventory,
        "development_clean_sources": list(clean_buckets),
        "crops_per_source_per_measurement": crops_per_source,
        "attempted_examples": len(rows),
    }


def audit(workspace: Path, crops_per_source: int, device: torch.device) -> dict:
    if crops_per_source < 2:
        raise ValueError("external audit requires at least two crops per source and RIR")
    model, seal = _load_frozen_model(workspace, device)
    rows, elapsed, inventory = _materialize(
        workspace, model, seal, device, crops_per_source
    )
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
        "status": (
            "accepted-profile-bank-graybox-external-rooms"
            if accepted else "rejected-profile-bank-graybox-external-rooms"
        ),
        "accepted": accepted,
        "promotable": False,
        "promotion_blockers": [
            "partitioned profile-candidate runtime equivalence and timing are not sealed",
            "listening acceptance and locked-final remain unopened",
        ],
        "mechanism": "ambience",
        "quality_contract": QUALITY_CONTRACT,
        "frozen_safety_gates": safety_gates,
        "external_validation": evaluation,
        "inventory": inventory,
        "model_seal": seal,
        "selection_contract": {
            "dataset": SOURCE_ID,
            "external_data_used_for_parameter_or_checkpoint_selection": False,
            "checkpoint_frozen_before_external_data_evaluation": True,
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
    parser.add_argument("--crops-per-source", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument(
        "--output", type=Path,
        default=Path("runs/foundation/product4-reverb-profile-bank-openair-external-v1/metrics.json"),
    )
    args = parser.parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace OpenAIR external audit: {output}")
    report = audit(args.workspace.resolve(), args.crops_per_source, torch.device(args.device))
    output.parent.mkdir(parents=True, exist_ok=False)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "metrics": str(output),
        "status": report["status"],
        "accepted": report["accepted"],
        "attempted_examples": report["inventory"]["attempted_examples"],
        "modes": report["external_validation"]["modes"],
        "gates": report["external_validation"]["gates"],
        "frozen_safety_gates": report["frozen_safety_gates"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
