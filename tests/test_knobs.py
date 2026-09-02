import json
import tempfile
import unittest
from pathlib import Path

from remix.knobs import KnobRuntime, verify_evidence


class KnobTests(unittest.TestCase):
    def test_evidence_requires_accepted_knob_capability_and_matching_bundle(self):
        report = {
            "capabilities": {"knob": {"accepted": True, "failures": []}},
            "artifacts": {"bundle": "a" * 64},
            "artifacts_unchanged": True,
            "physical_audio_devices_used": False,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "seal.json"
            path.write_text(json.dumps(report))
            self.assertEqual(verify_evidence(path, "a" * 64), report)
            with self.assertRaises(ValueError):
                verify_evidence(path, "b" * 64)

    def test_evidence_rejects_joint_failure_only_when_knob_failed(self):
        report = {
            "capabilities": {"knob": {"accepted": False, "failures": ["knob.macro_p95"]}},
            "artifacts": {"bundle": "a" * 64},
            "artifacts_unchanged": True,
            "physical_audio_devices_used": False,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "seal.json"
            path.write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                verify_evidence(path, "a" * 64)

    def test_controls_can_reuse_a_matching_inverse_report(self):
        runtime = object.__new__(KnobRuntime)
        runtime.bundle_hash = "a" * 64
        runtime.evidence_hash = "b" * 64
        report = {
            "active_effects": ["drive"],
            "normalized_controls": [0.5] * 9,
            "bundle_sha256": "a" * 64,
        }
        decoded = runtime.controls_from_report(report)
        self.assertEqual(decoded["active"], ["drive"])
        self.assertEqual(
            set(decoded["controls"]),
            {"drive.gain_db", "drive.tone", "drive.level_db"},
        )
        self.assertFalse(decoded["order_used"])

        report["bundle_sha256"] = "c" * 64
        with self.assertRaises(ValueError):
            runtime.controls_from_report(report)

    def test_active_families_use_canonical_order(self):
        runtime = object.__new__(KnobRuntime)
        runtime.bundle_hash = "a" * 64
        runtime.evidence_hash = "b" * 64
        decoded = runtime.controls_from_report({
            "active_effects": ["reverb", "drive", "delay"],
            "normalized_controls": [0.5] * 9,
            "bundle_sha256": "a" * 64,
        })
        self.assertEqual(decoded["active"], ["drive", "delay", "reverb"])


if __name__ == "__main__":
    unittest.main()
