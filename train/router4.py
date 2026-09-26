#!/usr/bin/env python3
"""Train router model4 after consuming the rejected model3 A2 seal.

The underlying trainer remains generic router2.  This round expands it with
the now-development-only multi-author A2 models and independent P2 DI clips.
Any promotion still requires a completely new untouched seal.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from remix.packages import digest
from train import router2


P2 = (
    "data/corpus/guitar-techs/P2_chords/P2_chords/audio/directinput/directinput_Set1_aug.wav",
    "data/corpus/guitar-techs/P2_chords/P2_chords/audio/directinput/directinput_Set2_min.wav",
    "data/corpus/guitar-techs/P2_chords/P2_chords/audio/directinput/directinput_Set3_maj.wav",
    "data/corpus/guitar-techs/P2_chords/P2_chords/audio/directinput/directinput_Set4_dim.wav",
)
EXPANDED = (
    ("fit", True, "rat09", "rat09.nam", "https://www.tone3000.com/tones/proco-rat-vintage-reissue-2298", "T3K"),
    ("valid", True, "rat11", "rat11.nam", "https://www.tone3000.com/tones/proco-rat-vintage-reissue-2298", "T3K"),
    ("fit", True, "rat2d2", "rat2d2.nam", "https://www.tone3000.com/tones/proco-rat-2-82634", "T3K"),
    ("valid", True, "rat2d6", "rat2d6.nam", "https://www.tone3000.com/tones/proco-rat-2-82634", "T3K"),
    ("fit", False, "diy", "diy.nam", "https://www.tone3000.com/tones/diy-drive-pedal-bjt-silicon-celestion-eight-15-ir-5700", "CC0-1.0"),
    ("fit", False, "grind1", "grind1.nam", "https://www.tone3000.com/tones/dna-klirrton-grindstein-2620", "CC0-1.0"),
    ("fit", False, "grind2", "grind2.nam", "https://www.tone3000.com/tones/dna-klirrton-grindstein-2620", "CC0-1.0"),
    ("valid", False, "grind3", "grind3.nam", "https://www.tone3000.com/tones/dna-klirrton-grindstein-2620", "CC0-1.0"),
)


def output_arg() -> Path:
    index = sys.argv.index("--output")
    return Path(sys.argv[index + 1])


def annotate(output: Path) -> None:
    seal = output.parent / "model3" / "seal.json"
    lineage = {
        "source": "model3 rejected seal converted to development-only data",
        "seal": seal.as_posix(), "seal_sha256": digest(seal),
        "seal_status": json.loads(seal.read_text())["status"],
        "future_requirement": "a completely new untouched multi-author capture-model seal",
    }
    for name in ("fit.json", "valid.json"):
        path = output / name
        if path.exists():
            document = json.loads(path.read_text())
            document["lineage"] = lineage
            if name == "valid.json":
                document["scope"] = "expanded development validation only; model3 seal is consumed"
            path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def main() -> None:
    router2.CLIPS = router2.CLIPS + P2
    router2.MODELS = router2.MODELS + EXPANDED
    output = output_arg()
    try:
        router2.main()
    finally:
        annotate(output)


if __name__ == "__main__":
    main()
