"""Import a declared local model into an immutable Muspector repository.

The command only reads files named by ``model.json`` and writes a verified
package.  It never inspects audio inputs or infers a manufacturer/model name.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .packages import install, load


def validate_model(path: Path) -> dict:
    if path.name != "model.json":
        raise ValueError("external model manifest must be named model.json")
    model = load(path)
    if "identity" not in model or "provenance" not in model:
        raise ValueError("external model requires declared identity and provenance")

    backend = model["runtime"]["backend"]
    if backend == "neural-amp-modeler":
        if model["kind"] != "forward":
            raise ValueError("NAM models provide forward rendering only")
        artifacts = [row for row in model["artifacts"] if row["role"] == "model"]
        if len(artifacts) != 1 or Path(artifacts[0]["path"]).suffix.lower() != ".nam":
            raise ValueError("NAM manifest requires exactly one .nam model artifact")
    return model


def ingest(source: Path, destination: Path) -> Path:
    source = source.resolve()
    manifest = source / "model.json" if source.is_dir() else source
    validate_model(manifest)
    return install(manifest, destination, manifest.parent)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="model directory or model.json")
    parser.add_argument("destination", type=Path, help="local model repository")
    args = parser.parse_args()
    package = ingest(args.source, args.destination)
    print(json.dumps({"schema": 1, "package": str(package)}, indent=2))


if __name__ == "__main__":
    main()
