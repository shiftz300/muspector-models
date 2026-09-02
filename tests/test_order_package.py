import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from remix import order_transfer_runtime
from remix.order_package import PackageRuntime


class OrderPackageTests(unittest.TestCase):
    def test_loader_binds_package_local_bundle_and_restores_global(self):
        package = {
            "id": "order",
            "version": "2.0.0",
            "runtime": {"entrypoint": "python.remix.order_package.v2"},
        }
        original = order_transfer_runtime.BUNDLE
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            observed = []

            def construct(run):
                observed.append((Path(run), order_transfer_runtime.BUNDLE))
                return object()

            with patch("remix.order_package.verify", return_value=package), patch(
                "remix.order.TransferRemixerRuntime", side_effect=construct
            ):
                runtime = PackageRuntime(root)

            self.assertIsNotNone(runtime.base)
            self.assertEqual(observed[0][0], root.resolve() / "artifacts")
            self.assertEqual(observed[0][1], root.resolve() / "artifacts/remixer.pt")
            self.assertEqual(order_transfer_runtime.BUNDLE, original)

    def test_loader_rejects_wrong_package(self):
        package = {
            "id": "knobs",
            "version": "1.0.0",
            "runtime": {"entrypoint": "python.remix.knobs.v1"},
        }
        with tempfile.TemporaryDirectory() as temporary, patch(
            "remix.order_package.verify", return_value=package
        ):
            with self.assertRaisesRegex(ValueError, "unsupported order package"):
                PackageRuntime(Path(temporary))


if __name__ == "__main__":
    unittest.main()
