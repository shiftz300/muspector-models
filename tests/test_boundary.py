import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class BoundaryTests(unittest.TestCase):
    def test_repository_contains_no_generated_model_or_audio_artifacts(self):
        forbidden = {".bin", ".joblib", ".npz", ".onnx", ".pt", ".pth", ".wav", ".flac", ".mp3"}
        ignored_roots = {".git", "data", "runs"}
        paths = [
            path
            for path in ROOT.rglob("*")
            if path.is_file()
            and path.suffix.lower() in forbidden
            and path.relative_to(ROOT).parts[0] not in ignored_roots
            and "runs" not in path.relative_to(ROOT).parts
            and "cache" not in path.relative_to(ROOT).parts
        ]
        self.assertEqual(paths, [])

    def test_cycles_explicitly_forbid_physical_audio_devices(self):
        cycles = list((ROOT / "cycles").glob("*.json"))
        self.assertTrue(cycles)
        for path in cycles:
            document = json.loads(path.read_text())
            self.assertFalse(document["data"]["physical_audio_devices_used"])
            self.assertTrue(document["data"]["source_read_only"])


if __name__ == "__main__":
    unittest.main()
