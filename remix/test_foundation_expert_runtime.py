from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np

from .audit_foundation_packages import MAXIMUM_ABSOLUTE_ERROR, audit
from .audit_foundation_drive_package import audit as audit_drive
from .foundation_expert_runtime import FoundationChainRuntime, FoundationExpertRuntime
from .packages import materialize, validate_collection


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "runs/foundation/product3-spectral/spectral/model.pt"
DRIVE_CHECKPOINT = ROOT / "runs/foundation/product3-safe/nonlinear/model.pt"


class FoundationExpertRuntimeTests(unittest.TestCase):
    def test_all_forward_orders_pass_bounded_chain_acceptance(self) -> None:
        report = audit(CHECKPOINT)
        self.assertEqual(report["status"], "development-chain-pass")
        self.assertLessEqual(report["maximum_absolute_error"], MAXIMUM_ABSOLUTE_ERROR)
        self.assertEqual(len(report["orders"]), 6)

    def test_expert_api_has_no_order_or_neighbor_input(self) -> None:
        parameters = set(inspect.signature(FoundationExpertRuntime.restore).parameters)
        self.assertEqual(parameters, {"self", "wet", "controls", "state"})
        self.assertFalse(parameters & {"order", "graph", "neighbor", "previous", "next"})

    def test_drive_passes_all_24_known_control_chain_orders(self) -> None:
        report = audit_drive(DRIVE_CHECKPOINT, CHECKPOINT)
        self.assertEqual(report["status"], "development-chain-pass")
        self.assertEqual(len(report["orders"]), 24)

    def test_chain_rejects_duplicate_stage_identity(self) -> None:
        experts = {
            "dynamics": FoundationExpertRuntime("dynamics"),
            "echo": FoundationExpertRuntime("echo"),
            "spectral": FoundationExpertRuntime("spectral", CHECKPOINT),
        }
        runtime = FoundationChainRuntime(experts)
        stages = [
            {"instance_id": "same", "mechanism": "echo", "controls": {}},
            {"instance_id": "same", "mechanism": "echo", "controls": {}},
        ]
        with self.assertRaisesRegex(ValueError, "unique"):
            runtime.restore(np.zeros(4096, dtype=np.float32), stages)

    def test_four_packages_materialize_and_load_independently(self) -> None:
        collection = validate_collection(ROOT / "models/catalog/foundation/collection.json")
        self.assertFalse(collection["default_install"])
        with tempfile.TemporaryDirectory(prefix="foundation-experts-") as temporary:
            written = materialize(
                ROOT / "models/catalog/foundation/sources.json",
                Path(temporary),
                ROOT,
            )
            runtimes = {path.parent.name: FoundationExpertRuntime.package(path) for path in written}
            self.assertEqual(
                set(runtimes),
                {
                    "foundation-drive",
                    "foundation-dynamics",
                    "foundation-echo",
                    "foundation-spectral",
                },
            )
            self.assertEqual(
                {runtime.mechanism for runtime in runtimes.values()},
                {"nonlinear", "dynamics", "echo", "spectral"},
            )


if __name__ == "__main__":
    unittest.main()
