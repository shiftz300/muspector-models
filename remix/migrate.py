"""Materialize accepted research artifacts as canonical model packages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .packages import index, install, materialize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, help="models directory to create")
    parser.add_argument(
        "--set", choices=("base", "development", "remix", "all"), default="base"
    )
    parser.add_argument(
        "--sources",
        type=Path,
        default=Path("models/catalog/development/sources.json"),
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("../muspector"),
        help="source checkout containing accepted research artifacts",
    )
    parser.add_argument(
        "--remix",
        type=Path,
        default=Path("models/catalog/remix/sources.json"),
        help="locally owned Remixer package sources",
    )
    parser.add_argument(
        "--local",
        type=Path,
        default=Path("."),
        help="model workspace containing locally owned Remixer artifacts",
    )
    parser.add_argument(
        "--assets",
        type=Path,
        default=Path("../muspector/models/inspector"),
        help="source directory containing embedded base assets",
    )
    parser.add_argument("--url", help="static repository base URL; writes index.json")
    args = parser.parse_args()
    written = []
    if args.set in {"base", "all"}:
        written.extend(
            install(path, args.destination, args.assets)
            for path in (
                Path("models/inspector/family.json"),
                Path("models/inspector/identity.json"),
            )
        )
    if args.set in {"development", "all"}:
        written.extend(materialize(args.sources, args.destination, args.workspace))
    if args.set in {"remix", "all"}:
        written.extend(materialize(args.remix, args.destination, args.local))
    repository = index(args.destination, args.url) if args.url else None
    print(json.dumps({"schema": 1, "packages": [str(path) for path in written], "index": str(repository) if repository else None}, indent=2))


if __name__ == "__main__":
    main()
