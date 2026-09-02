#!/usr/bin/env python3
"""Fit the canonical Reverb device profile from offline calibration renders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .forward_reverb import fit_reverb_profile, profile_manifest, save_reverb_profile


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "remix/runs/reverb-forward-phase1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="reference")
    parser.add_argument("--output", type=Path, default=RUN)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    profile = fit_reverb_profile(args.domain)
    path = args.output / "reverb-device-profile.npz"
    save_reverb_profile(profile, path)
    report = {
        **profile_manifest(profile, path),
        "status": "fitted-candidate",
        "policy": {
            "source_files_read_only": True,
            "challenge_renderer_opened": False,
            "locked_tele_test_opened": False,
            "physical_audio_devices_used": False,
        },
    }
    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
