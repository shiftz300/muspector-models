from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from .importer import ingest, validate_model
from .packages import QUALITY, verify


class ImporterTests(unittest.TestCase):
    def manifest(self, root: Path, payload: bytes = b"declared-nam") -> Path:
        artifact = root / "rat.nam"
        artifact.write_bytes(payload)
        document = {
            "schema": 1,
            "id": "rat-demo",
            "version": "1.0.0",
            "display_name": "Declared RAT capture",
            "kind": "forward",
            "quality": "experimental",
            "device": "rat",
            "identity": {
                "manufacturer": "Pro Co",
                "model": "RAT",
                "family": "drive",
                "declared": True,
            },
            "provenance": {
                "author": "Example",
                "url": "https://example.invalid/rat",
                "capture": "fixed setting",
            },
            "runtime": {
                "backend": "neural-amp-modeler",
                "entrypoint": "native.nam.v1",
                "minimum_client_version": "0.1.0",
            },
            "sample_rate": 48000,
            "channels": 1,
            "controls": [],
            "audio_behavior": "loss_preserving_render",
            "audio_quality": QUALITY,
            "artifacts": [{
                "path": "rat.nam",
                "role": "model",
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }],
            "license": {
                "id": "declared-local",
                "commercial_use": False,
                "redistribution": False,
                "notice": None,
            },
            "evidence": [],
            "limitations": ["Identity is publisher-declared, not inferred from audio."],
        }
        path = root / "model.json"
        path.write_text(json.dumps(document))
        return path

    def test_imports_hash_verified_fixed_nam_without_knobs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="muspector-model-") as source_name, tempfile.TemporaryDirectory(prefix="muspector-store-") as store_name:
            source, store = Path(source_name), Path(store_name)
            original = (source / "rat.nam")
            manifest = self.manifest(source)
            target = ingest(manifest, store)
            package = verify(target)
            self.assertEqual(package["identity"]["model"], "RAT")
            self.assertEqual(package["controls"], [])
            self.assertEqual((target / "rat.nam").read_bytes(), original.read_bytes())

    def test_rejects_tampering_without_partial_package(self) -> None:
        with tempfile.TemporaryDirectory(prefix="muspector-model-") as source_name, tempfile.TemporaryDirectory(prefix="muspector-store-") as store_name:
            source, store = Path(source_name), Path(store_name)
            manifest = self.manifest(source)
            (source / "rat.nam").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                ingest(manifest, store)
            self.assertFalse((store / "packages").exists())
            self.assertEqual(list((store / ".staging").glob("*")), [])

    def test_rejects_inferred_identity_and_wrong_nam_shape(self) -> None:
        with tempfile.TemporaryDirectory(prefix="muspector-model-") as source_name:
            source = Path(source_name)
            manifest = self.manifest(source)
            document = json.loads(manifest.read_text())
            document["identity"]["declared"] = False
            manifest.write_text(json.dumps(document))
            with self.assertRaises(ValueError):
                validate_model(manifest)

            document["identity"]["declared"] = True
            document["artifacts"][0]["path"] = "rat.bin"
            manifest.write_text(json.dumps(document))
            with self.assertRaises(ValueError):
                validate_model(manifest)


if __name__ == "__main__":
    unittest.main()
