from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from .audit_centered_multirate_fuzz import audit
from .centered_multirate_fuzz import CenteredMultirateFuzz
from .evaluate_multirate_fuzz import CENTERED_ARCHITECTURE, model_from_payload, validate_calibration_entry
from .multirate_admission import validate_evidence, validate_structure
from .multirate_fuzz import MultirateFuzz
from .test_multirate_admission import complete_evidence
from .widen_multirate_fuzz import WIDE_GEOMETRY


def centered_evidence():
    evidence = list(complete_evidence())
    for report in evidence[:4]:
        report["architecture"] = CENTERED_ARCHITECTURE
    evidence[3]["purpose"] = "trained-centered-core-audit"
    return evidence


class CenteredAdmissionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.model = CenteredMultirateFuzz(**WIDE_GEOMETRY).eval()
        self.payload = {"experimental_schema": 1, "architecture": CENTERED_ARCHITECTURE,
                        "geometry": WIDE_GEOMETRY.copy(), "device": "dfz", "sample_rate": 48000,
                        "state_dict": self.model.state_dict()}

    def test_factory_preserves_exact_class_and_all_weights(self):
        loaded = model_from_payload(self.payload)
        self.assertIs(type(loaded), CenteredMultirateFuzz)
        self.assertTrue(all(torch.equal(value, loaded.state_dict()[name]) for name, value in self.payload["state_dict"].items()))

    def test_mislabeled_unsupported_or_cast_weights_rejected(self):
        for change in (lambda p: p.update(architecture="causal-multirate-gcn"),
                       lambda p: p.update(architecture="invented"),
                       lambda p: p["state_dict"].update({"audio.output.weight": p["state_dict"]["audio.output.weight"].double()})):
            payload = deepcopy(self.payload)
            change(payload)
            with self.assertRaises((ValueError, RuntimeError)):
                model_from_payload(payload)

    def test_centered_bound_is_not_plain_bound(self):
        base = MultirateFuzz(**WIDE_GEOMETRY)
        base.audio.output.weight.data.fill_(.05)
        centered = CenteredMultirateFuzz.from_base(base)
        original = validate_structure(base, {**self.payload, "architecture": "causal-multirate-gcn"})
        changed = validate_structure(centered, self.payload)
        self.assertEqual(original["fast_update_bound"], 1.5)
        self.assertEqual(changed["fast_update_bound"], 3.)
        self.assertGreater(changed["conservative_output_bound_for_abs_dry_le1"], original["conservative_output_bound_for_abs_dry_le1"])
        self.assertEqual(changed["state_tensor_bytes_mono"], 393160)
        self.assertEqual(changed["zero_audio_tail_expiry_frames"], 2046)

    def test_threshold_or_slow_gate_geometry_rejected(self):
        for change in (lambda m: setattr(m.audio.activation_slow[0], "bias", torch.nn.Parameter(torch.zeros(32))),
                       lambda m: setattr(m.audio.activation_current[0], "out_features", 31),
                       lambda m: m.audio.activation_current.__delitem__(0),
                       lambda m: m.controller.gate_bias.__setitem__(0, torch.nn.Parameter(torch.zeros(15)))):
            model = deepcopy(self.model)
            change(model)
            with self.assertRaises(ValueError):
                validate_structure(model, self.payload)

    def test_centered_evidence_requires_its_own_trained_audit(self):
        evidence = centered_evidence()
        before = deepcopy(evidence)
        self.assertTrue(all(validate_evidence(*evidence).values()))
        self.assertEqual(evidence, before)
        evidence[3]["purpose"] = "trained-wide-core-audit"
        with self.assertRaises(ValueError):
            validate_evidence(*evidence)

    def test_cross_architecture_evidence_never_mixed(self):
        for index in range(4):
            evidence = centered_evidence()
            evidence[index]["architecture"] = "causal-multirate-gcn"
            with self.assertRaises(ValueError):
                validate_evidence(*evidence)
        evidence = centered_evidence()
        with self.assertRaises(ValueError):
            validate_calibration_entry(evidence[0], evidence[4], "causal-multirate-gcn")

    def test_centered_quality_and_parity_gates_are_unchanged(self):
        for change in (lambda e: e[0]["calibration"].update(absolute_peak_error_p95=.0200001),
                       lambda e: e[2]["official_eval"]["by_first_control"]["0"].update(absolute_peak_error_p95=.030001),
                       lambda e: e[3]["real_audio_cases"][0].update(parity_max=2.000001e-6),
                       lambda e: e[3]["safety"]["onnx"].update(expired_tail_peak=1e-12)):
            evidence = centered_evidence()
            change(evidence)
            with self.assertRaises(ValueError):
                validate_evidence(*evidence)

    def test_existing_audit_directory_refused_before_source_access(self):
        with tempfile.TemporaryDirectory() as temporary, patch("remix.audit_centered_multirate_fuzz.load_candidate") as load:
            with self.assertRaisesRegex(ValueError, "new trained audit"):
                audit(Path("not-opened"), Path(temporary))
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
