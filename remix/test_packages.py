from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from .packages import QUALITY, index, install, materialize, validate, validate_catalog, validate_collection, verify


ROOT = Path(__file__).resolve().parents[1]
CLIENT = Path(os.environ.get("MUSPECTOR_CLIENT", ROOT.parent / "muspector")).resolve()


class PackageTests(unittest.TestCase):
    def test_base_metadata_separates_capabilities(self) -> None:
        collection = validate_collection(ROOT / "models/catalog/base/collection.json")
        catalog = validate_catalog(ROOT / "models/catalog/base/catalog.json")
        self.assertTrue(collection["default_install"])
        self.assertGreaterEqual(len(catalog["devices"]), 9)
        dfz = next(device for device in catalog["devices"] if device["id"] == "dfz")
        self.assertEqual(dfz["capabilities"]["forward"], [])
        self.assertEqual(dfz["capabilities"]["inverse"][0]["package_id"], "dfzknobs")
        base = {
            json.loads((ROOT / "models/inspector/family.json").read_text())["id"],
            json.loads((ROOT / "models/inspector/identity.json").read_text())["id"],
        }
        development = {
            entry["package"]["id"]
            for entry in json.loads(
                (ROOT / "models/catalog/development/sources.json").read_text()
            )["packages"]
        }
        remix = {
            entry["package"]["id"]
            for entry in json.loads(
                (ROOT / "models/catalog/remix/sources.json").read_text()
            )["packages"]
        }
        references = {
            reference["package_id"]
            for device in catalog["devices"]
            for rows in device["capabilities"].values()
            for reference in rows
        }
        references.update(
            reference["package_id"]
            for rows in catalog["chain_capabilities"].values()
            for reference in rows
        )
        self.assertEqual({member["package_id"] for member in collection["packages"]}, base)
        self.assertLessEqual(references, base | development | remix)

    def test_forward_quality_cannot_be_borrowed_by_inverse(self) -> None:
        package = {
            "schema": 1, "id": "test", "version": "1", "display_name": "test",
            "kind": "forward", "quality": "development", "device": "drive",
            "runtime": {"backend": "native", "entrypoint": "test", "minimum_client_version": "0.1"},
            "sample_rate": 48000, "channels": 1,
            "controls": [{"id": "gain", "label": "Gain", "unit": "dB", "minimum": 0, "maximum": 1, "default": 0.5}],
            "audio_behavior": "loss_preserving_render", "audio_quality": QUALITY,
            "artifacts": [{"path": "weights", "role": "weights", "bytes": 1, "sha256": "0" * 64}],
            "license": {"id": "test", "commercial_use": False, "redistribution": False, "notice": None},
            "evidence": [], "limitations": [],
        }
        validate(package)
        package["kind"] = "inverse"
        with self.assertRaises(ValueError):
            validate(package)

    def test_fixed_forward_snapshot_may_have_no_controls(self) -> None:
        package = {
            "schema": 1, "id": "fixed", "version": "1", "display_name": "fixed",
            "kind": "forward", "quality": "experimental", "device": "rat",
            "runtime": {"backend": "neural-amp-modeler", "entrypoint": "native.nam.v1", "minimum_client_version": "0.1"},
            "sample_rate": 48000, "channels": 1, "controls": [],
            "audio_behavior": "loss_preserving_render", "audio_quality": QUALITY,
            "artifacts": [{"path": "fixed.nam", "role": "model", "bytes": 1, "sha256": "0" * 64}],
            "license": {"id": "test", "commercial_use": False, "redistribution": False, "notice": None},
            "evidence": [], "limitations": [],
        }
        self.assertEqual(validate(package)["controls"], [])

    def test_audio_restoration_inverse_has_its_own_contract(self) -> None:
        package = {
            "schema": 1, "id": "restore", "version": "1", "display_name": "restore",
            "kind": "inverse", "quality": "development", "device": "generic-eq",
            "runtime": {"backend": "python", "entrypoint": "test", "minimum_client_version": "0.1"},
            "sample_rate": 48000, "channels": 1,
            "controls": [{"id": "gain", "label": "Gain", "unit": "dB", "minimum": -1, "maximum": 1, "default": 0}],
            "audio_behavior": "loss_preserving_restore", "audio_quality": QUALITY,
            "artifacts": [{"path": "weights", "role": "weights", "bytes": 1, "sha256": "0" * 64}],
            "license": {"id": "test", "commercial_use": False, "redistribution": False, "notice": None},
            "evidence": [], "limitations": [],
        }
        validate(package)
        package["audio_behavior"] = "loss_preserving_render"
        with self.assertRaisesRegex(ValueError, "invalid audio behavior"):
            validate(package)

    def test_all_research_sources_materialize_without_touching_runs(self) -> None:
        sources = ROOT / "models/catalog/development/sources.json"
        before = sources.read_bytes()
        with tempfile.TemporaryDirectory(prefix="muspector-packages-") as temporary:
            written = materialize(sources, Path(temporary), CLIENT)
            self.assertEqual(len(written), 8)
            kinds = [verify(path)["kind"] for path in written]
            self.assertEqual(kinds.count("forward"), 5)
            self.assertEqual(kinds.count("inverse"), 3)
            self.assertEqual(kinds.count("order"), 0)
        self.assertEqual(sources.read_bytes(), before)

    def test_sealed_remix_sources_materialize_from_model_workspace(self) -> None:
        sources = ROOT / "models/catalog/remix/sources.json"
        collection = validate_collection(ROOT / "models/catalog/remix/collection.json")
        self.assertFalse(collection["default_install"])
        with tempfile.TemporaryDirectory(prefix="muspector-remix-") as temporary:
            written = materialize(sources, Path(temporary), ROOT)
            packages = {package["id"]: package["kind"] for package in map(verify, written)}
            self.assertEqual(packages, {"knobs": "inverse", "order": "order", "chain": "classifier", "route": "classifier"})

    def test_base_builds_a_static_repository(self) -> None:
        manifests = (
            ROOT / "models/inspector/family.json",
            ROOT / "models/inspector/identity.json",
        )
        with tempfile.TemporaryDirectory(prefix="muspector-base-") as temporary:
            destination = Path(temporary)
            written = [
                install(manifest, destination, CLIENT / "models/inspector")
                for manifest in manifests
            ]
            self.assertEqual([verify(path)["id"] for path in written], ["family", "identity"])
            repository = json.loads(index(destination, "https://example.invalid/models").read_text())
            self.assertEqual(len(repository["packages"]), 2)
            self.assertTrue(all(row["manifest_url"].startswith("https://") for row in repository["packages"]))


if __name__ == "__main__":
    unittest.main()
