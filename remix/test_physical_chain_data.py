import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from .blind2 import LABELS
from .evaluate_physical_chain_presence import _acceptance
from .physical_chain_data import PhysicalChainPresence, discover, inventory


class PhysicalChainDataTests(unittest.TestCase):
    def _write_manifest(self, root: Path, rows: list[dict], **extra: object) -> Path:
        for row in rows:
            path = root / str(row["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        manifest = {
            "schema": 1,
            "scope": "physical-multi-effect-veto",
            "source_id": "self-capture-v1",
            "physical_hardware": True,
            "product_evaluation_authorized": True,
            "rights_basis": "self-recorded audio authorized for product evaluation",
            "rows": rows,
            **extra,
        }
        path = root / "manifest.json"
        path.write_text(json.dumps(manifest))
        return path

    def test_discovers_clean_and_unordered_family_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._write_manifest(
                root,
                [
                    {
                        "path": "audio/clean.wav",
                        "group": "phrase-1",
                        "hardware_group": "bypass-a",
                        "capture_variant_id": "v000",
                        "families": [],
                    },
                    {
                        "path": "audio/wet.wav",
                        "group": "phrase-1",
                        "hardware_group": "board-a",
                        "capture_variant_id": "v001",
                        "families": ["ambience", "nonlinear"],
                    },
                ],
            )
            _, rows = discover(path)
            np.testing.assert_array_equal(rows[0].target, np.zeros(len(LABELS)))
            np.testing.assert_array_equal(
                rows[1].target,
                np.asarray([1.0, 0.0, 1.0, 0.0], dtype=np.float32),
            )

    def test_rejects_any_order_metadata_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._write_manifest(
                root,
                [
                    {
                        "path": "audio/wet.wav",
                        "group": "phrase-1",
                        "hardware_group": "board-a",
                        "capture_variant_id": "v001",
                        "families": ["nonlinear", "echo"],
                        "pedal_order": ["drive", "delay"],
                    }
                ],
            )
            with self.assertRaisesRegex(ValueError, "order metadata is forbidden"):
                discover(path)

    def test_rejects_single_effect_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._write_manifest(
                root,
                [
                    {
                        "path": "audio/wet.wav",
                        "group": "phrase-1",
                        "hardware_group": "board-a",
                        "capture_variant_id": "v001",
                        "families": ["echo"],
                    }
                ],
            )
            with self.assertRaisesRegex(ValueError, "single-effect"):
                discover(path)

    def test_rejects_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {
                "schema": 1,
                "scope": "physical-multi-effect-veto",
                "source_id": "self-capture-v1",
                "physical_hardware": True,
                "product_evaluation_authorized": True,
                "rights_basis": "self-recorded audio authorized for product evaluation",
                "rows": [
                    {
                        "path": "../outside.wav",
                        "group": "phrase-1",
                        "hardware_group": "board-a",
                        "capture_variant_id": "v001",
                        "families": ["nonlinear", "echo"],
                    }
                ],
            }
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "escapes the manifest directory"):
                discover(path)

    def test_rejects_missing_product_evaluation_authorization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._write_manifest(
                root,
                [
                    {
                        "path": "audio/wet.wav",
                        "group": "phrase-1",
                        "hardware_group": "board-a",
                        "capture_variant_id": "v001",
                        "families": ["nonlinear", "echo"],
                    }
                ],
                product_evaluation_authorized=False,
            )
            with self.assertRaisesRegex(ValueError, "product_evaluation_authorized"):
                discover(path)

    def test_inventory_requires_enough_independent_evidence(self) -> None:
        rows = [
            PhysicalChainPresence(
                Path(f"clean-{index}.wav"),
                f"phrase-{index % 12}",
                "bypass-a",
                f"clean-{index}",
                np.zeros(len(LABELS), dtype=np.float32),
            )
            for index in range(73)
        ]
        rows.extend(
            PhysicalChainPresence(
                Path(f"wet-{index}.wav"),
                f"phrase-{index % 12}",
                f"board-{index % 2}",
                f"wet-{index}",
                np.ones(len(LABELS), dtype=np.float32),
            )
            for index in range(50)
        )
        self.assertTrue(inventory(rows)["coverage_passed"])

    def test_acceptance_requires_clean_evidence_and_wilson_bound(self) -> None:
        coverage = {"coverage_gates": {"enough": True}}
        hardware = {"board-a": {"macro_f1": 0.9}}
        overall = {
            "micro_f1": 0.9,
            "macro_f1": 0.9,
            "clean_examples": 0,
            "clean_false_positive_rate": None,
            "per_label": {label: {"recall": 0.9} for label in LABELS},
        }
        self.assertFalse(
            _acceptance(overall, hardware, coverage)["physical_hardware_veto_passed"]
        )
        overall["clean_examples"] = 73
        overall["clean_false_positive_rate"] = 0.0
        self.assertTrue(
            _acceptance(overall, hardware, coverage)["physical_hardware_veto_passed"]
        )


if __name__ == "__main__":
    unittest.main()
