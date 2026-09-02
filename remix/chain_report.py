"""Report the accepted chain across clean-source domains without training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .data import dry_sources
from .forward_chain import ForwardChainRuntime
from .stages import APPLE, examples, evaluate
from .net import SpectralNet


def load_model(path: Path) -> SpectralNet:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = SpectralNet(
        payload["channels"], payload["n_fft"], payload["hop"],
        tuple(tuple(value) for value in payload.get("dilations", ())),
    )
    model.load_state_dict(payload["state_dict"])
    return model


def source_domains(corpus: Path, split: str) -> dict[str, list[Path]]:
    domains = {"dafx": dry_sources(corpus, split)}
    payload = json.loads(APPLE.read_text())["captures"]
    for row in payload:
        if row["split"] != split:
            continue
        path = Path(row["source_path"])
        if path.is_file():
            domains.setdefault(row["source_domain"], []).append(path)
    return {name: sorted(set(paths)) for name, paths in sorted(domains.items()) if paths}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus/guitar-effects-chains"))
    parser.add_argument("--drive", type=Path, default=Path("runs/chain/model7/drive.pt"))
    parser.add_argument("--reverb", type=Path, default=Path("runs/chain/model7/reverb.pt"))
    parser.add_argument("--output", type=Path, default=Path("runs/chain/model7/domains.json"))
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--frames", type=int, default=16_384)
    args = parser.parse_args()
    drive, reverb = load_model(args.drive), load_model(args.reverb)
    target = torch.device("cpu")
    runtime = ForwardChainRuntime()
    rows = {}
    for split in ("valid", "calibrate", "test"):
        for domain, paths in source_domains(args.corpus, split).items():
            if not paths:
                continue
            sample = examples(runtime, paths, args.examples, args.frames, 20260920 + len(rows) * 17)
            rows[f"{split}:{domain}"] = {
                "split": split,
                "domain": domain,
                "source_files": len(paths),
                "report": evaluate(drive, reverb, sample, target, 1.0, 1.0),
            }
    result = {
        "schema": 1,
        "model": {"drive": str(args.drive), "reverb": str(args.reverb)},
        "sampling": {"examples_per_domain": args.examples, "frames": args.frames, "synthetic_forward_only": True},
        "domains": rows,
        "quality": {"source_audio_modified": False, "physical_audio_devices_used": False, "automatic_normalization": False, "lossy_reencoding": False},
        "limitation": "This is a domain balance report using controllable forward effects; it does not replace the locked real DAFx seal.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"domains": len(rows), "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
