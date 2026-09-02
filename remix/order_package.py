"""Load the sealed Order 2 runtime from a materialized model package."""

from __future__ import annotations

from pathlib import Path

from . import order_transfer_runtime as _transfer
from .order import OrderRuntime
from .packages import verify


class PackageRuntime(OrderRuntime):
    """Bind the immutable research runtime to package-local artifacts."""

    def __init__(self, root: Path):
        root = Path(root).resolve()
        package = verify(root)
        if package["id"] != "order" or package["version"] != "2.0.0":
            raise ValueError("unsupported order package")
        if package["runtime"]["entrypoint"] != "python.remix.order_package.v2":
            raise ValueError("order package entrypoint differs")
        artifacts = root / "artifacts"
        original = _transfer.BUNDLE
        try:
            _transfer.BUNDLE = artifacts / "remixer.pt"
            super().__init__(artifacts)
        finally:
            _transfer.BUNDLE = original
