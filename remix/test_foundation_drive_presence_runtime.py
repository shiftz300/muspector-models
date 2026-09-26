from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np

from .foundation_drive_presence_runtime import FoundationDrivePresenceRuntime, _windows
from .packages import materialize, validate_collection


ROOT = Path(__file__).resolve().parents[1]
NONLINEAR = ROOT / "runs/foundation/product2/blind2-chainpresence-nonlinear-expert/model.pt"
GATE = ROOT / "runs/foundation/product2/blind2-chainpresence-any-gate-robust/model.pt"


class FoundationDrivePresenceRuntimeTests(unittest.TestCase):
    def test_api_has_no_order_controls_or_neighbor_input(self) -> None:
        parameters = set(inspect.signature(FoundationDrivePresenceRuntime.analyze).parameters)
        self.assertEqual(parameters, {"self", "wet"})

    def test_audio_windowing_is_finite_and_non_mutating(self) -> None:
        audio = np.linspace(-0.2, 0.2, 12_000, dtype=np.float32)
        before = audio.copy()
        windows = _windows(audio)
        self.assertEqual(tuple(windows.shape), (1, 220_500))
        self.assertTrue(np.array_equal(audio, before))
        with self.assertRaises(ValueError):
            _windows(np.asarray([np.nan], dtype=np.float32))

    def test_checkpoint_contract_loads_without_shared_reverb_head(self) -> None:
        runtime = FoundationDrivePresenceRuntime(NONLINEAR, GATE)
        self.assertEqual(runtime.labels, ("nonlinear",))
        self.assertFalse(hasattr(runtime, "ambience"))

    def test_analysis_is_finite_and_preserves_wet(self) -> None:
        runtime = FoundationDrivePresenceRuntime(NONLINEAR, GATE)
        time = np.arange(220_500, dtype=np.float32) / 44_100.0
        wet = (0.08 * np.sin(2.0 * np.pi * 173.0 * time)).astype(np.float32)
        before = wet.copy()
        result = runtime.analyze(wet)
        self.assertTrue(np.array_equal(wet, before))
        self.assertEqual(result["labels"], ["nonlinear"])
        self.assertTrue(0.0 <= result["probabilities"]["nonlinear"] <= 1.0)
        self.assertTrue(0.0 <= result["any_effect_probability"] <= 1.0)
        self.assertFalse(result["order_output"])
        self.assertFalse(result["controls_output"])

    def test_package_materializes_and_loads(self) -> None:
        collection = validate_collection(
            ROOT / "models/catalog/foundation-presence/collection.json"
        )
        self.assertFalse(collection["default_install"])
        with tempfile.TemporaryDirectory(prefix="foundation-drive-presence-") as temporary:
            written = materialize(
                ROOT / "models/catalog/foundation-presence/sources.json",
                Path(temporary),
                ROOT,
            )
            self.assertEqual(len(written), 1)
            runtime = FoundationDrivePresenceRuntime.package(written[0])
            runtime.assert_artifacts_unchanged()


if __name__ == "__main__":
    unittest.main()
