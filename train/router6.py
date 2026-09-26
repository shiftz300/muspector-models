#!/usr/bin/env python3
"""Train router model6 after consuming the rejected model5 A2 seal."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from remix.packages import digest
from train import router2, router4


P3 = (
    "data/corpus/guitar-techs/P3_music/P3_music/audio/directinput/directinput_01.wav",
    "data/corpus/guitar-techs/P3_music/P3_music/audio/directinput/directinput_05.wav",
    "data/corpus/guitar-techs/P3_music/P3_music/audio/directinput/directinput_10.wav",
    "data/corpus/guitar-techs/P3_music/P3_music/audio/directinput/directinput_12.wav",
)
EXPANDED = (
    ("fit", True, "turbo15", "turbo15.nam", "https://www.tone3000.com/tones/turbo-rat-2411", "T3K"),
    ("valid", True, "turbo100", "turbo100.nam", "https://www.tone3000.com/tones/turbo-rat-2411", "T3K"),
    ("fit", True, "deuce5", "deuce5.nam", "https://www.tone3000.com/tones/drps-proco-deucetone-turbo-rat--63240", "T3K"),
    ("valid", True, "deuce4", "deuce4.nam", "https://www.tone3000.com/tones/drps-proco-deucetone-turbo-rat--63240", "T3K"),
    ("fit", False, "fab1", "fab1.nam", "https://www.tone3000.com/tones/fab-distortion-pedal-1985", "T3K"),
    ("valid", False, "fab2", "fab2.nam", "https://www.tone3000.com/tones/fab-distortion-pedal-1985", "T3K"),
    ("fit", False, "pogo1", "pogo1.nam", "https://www.tone3000.com/tones/pogolab-distortion-pedal-4748", "T3K"),
    ("valid", False, "pogo2", "pogo2.nam", "https://www.tone3000.com/tones/pogolab-distortion-pedal-4748", "T3K"),
)


def output_arg() -> Path:
    return Path(sys.argv[sys.argv.index("--output") + 1])


def annotate(output: Path) -> None:
    seal = output.parent / "model5" / "seal.json"
    lineage = {
        "source": "model5 rejected seal converted to development-only data",
        "seal": seal.as_posix(), "seal_sha256": digest(seal),
        "seal_status": json.loads(seal.read_text())["status"],
        "future_requirement": "a completely new untouched multi-author capture-model seal",
    }
    for name in ("fit.json", "valid.json", "scores.json"):
        path = output / name
        if path.exists():
            document = json.loads(path.read_text())
            document["lineage"] = lineage
            if name == "valid.json":
                document["scope"] = "expanded development validation only; model5 seal is consumed"
            path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def main() -> None:
    router2.CLIPS = router2.CLIPS + router4.P2 + P3
    router2.MODELS = router2.MODELS + router4.EXPANDED + EXPANDED
    output = output_arg()
    try:
        router2.main()
    finally:
        annotate(output)


if __name__ == "__main__":
    main()
