import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from .precheck_multirate_fuzz import sha256
from .promote_multirate_fuzz import EVIDENCE_NAMES, RUNTIME_SOURCES, _read_card, promote
from .multirate_admission import validate_evidence
from .test_multirate_admission import complete_evidence
from .widen_multirate_fuzz import WIDE_GEOMETRY


class MultiratePackageTests(unittest.TestCase):
    def test_existing_destination_stops_before_reads_or_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            marker = output / "user-owned"
            marker.write_text("preserve me")
            with self.assertRaisesRegex(ValueError, "no overwrite"):
                promote(SimpleNamespace(output=output))
            self.assertEqual(marker.read_text(), "preserve me")
            self.assertEqual([path.name for path in output.iterdir()], ["user-owned"])

    def synthetic_files(self, directory):
        """Card binding fixture only; these are deliberately not model weights."""
        cal, cpu, deployed, runtime, checkpoint, graph, _ = complete_evidence()
        (directory / "model.pt").write_bytes(b"checkpoint")
        (directory / "model.onnx").write_bytes(b"graph")
        training_path = directory / "training-evidence.json"
        training_path.write_text(json.dumps({"checkpoint_sha256": checkpoint, "source_and_audio_reverified": True}))
        for report in (cal, cpu, deployed):
            report["training_report_sha256"] = sha256(training_path)
        runtime["parent_report_sha256"] = sha256(training_path)
        cal_path = directory / "calibration-evidence.json"
        cal_path.write_text(json.dumps(cal))
        entry = sha256(cal_path)
        cpu["calibration_entry_sha256"] = deployed["calibration_entry_sha256"] = entry
        for name, report in (("pytorch", cpu), ("onnx", deployed), ("runtime", runtime)):
            (directory / f"{name}-evidence.json").write_text(json.dumps(report))
        files = {name: sha256(directory / name) for name in ("model.pt", "model.onnx", "training-evidence.json",
                                                             *[name + "-evidence.json" for name in EVIDENCE_NAMES])}
        card = {"schema": 1, "accepted": True, "status": "accepted-internal-noncommercial-development-pilot",
                "architecture": "causal-multirate-gcn", "geometry": WIDE_GEOMETRY, "device": "dfz", "sample_rate": 48000,
                "control_names": ["blend", "filter"], "ui_integration_allowed": False, "commercial_release_allowed": False,
                "physical_audio_devices_used": False, "new_locked_final_audio_opened": False,
                "unseen_control_interpolation_evaluated": False, "source_audio_modified": False,
                "dataset_license": "CC-BY-NC-4.0", "development_control_grid": [[0., .5, 1.], [0., .5, 1.]],
                "runtime_source_sha256": {name: sha256(Path(__file__).parent / name) for name in RUNTIME_SOURCES},
                "files": files, "quality_contract": deployed["quality_policy"],
                "checks": validate_evidence(cal, cpu, deployed, runtime, checkpoint, graph, entry)}
        (directory / "model-card.json").write_text(json.dumps(card))
        return card

    def test_card_binding_does_not_load_fake_weights_or_audio(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            card = self.synthetic_files(root)
            self.assertEqual(_read_card(root), card)

    def test_modified_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.synthetic_files(root)
            (root / "onnx-evidence.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                _read_card(root)

    def test_scope_or_source_edits_are_rejected(self):
        for change in (lambda c: c.update(commercial_release_allowed=True), lambda c: c.update(status="production"),
                       lambda c: c.update(quality_contract={}), lambda c: c.update(runtime_source_sha256={}),
                       lambda c: c["files"].update({"../outside": "0" * 64})):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                card = self.synthetic_files(root)
                change(card)
                (root / "model-card.json").write_text(json.dumps(card))
                with self.assertRaises(ValueError):
                    _read_card(root)

    def test_partial_copy_without_card_cannot_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "model.pt").write_bytes(b"not activated")
            with self.assertRaises(FileNotFoundError):
                _read_card(root)


if __name__ == "__main__":
    unittest.main()
