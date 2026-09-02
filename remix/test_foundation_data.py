import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile

from .foundation_data import Pair, _assert_group_disjoint, apple_pairs, read_pair


class FoundationDataTests(unittest.TestCase):
    def test_group_leak_is_rejected(self):
        common = dict(
            id="a",
            mechanism="nonlinear",
            family="drive",
            device="test",
            group="same-performance",
            source="dry.wav",
            wet="wet.wav",
            origin="test",
            gradient_scope="internal-research-only",
            weight_scope="blocked",
        )
        with self.assertRaisesRegex(ValueError, "crosses foundation splits"):
            _assert_group_disjoint(
                [Pair(split="fit", **common), Pair(split="development", **{**common, "id": "b"})]
            )

    def test_apple_manifest_maps_whole_groups_and_reads_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            corpus = workspace / "corpus"
            root = corpus / "apple-au"
            wet_path = root / "train/drive/wet.wav"
            source_path = corpus / "clean/source.wav"
            wet_path.parent.mkdir(parents=True)
            source_path.parent.mkdir(parents=True)
            clean = np.sin(2 * np.pi * 220 * np.arange(88_200) / 44_100).astype(np.float32) * 0.1
            wet = np.tanh(clean[:44_100] * 3.0)
            soundfile.write(source_path, clean, 44_100, subtype="FLOAT")
            soundfile.write(wet_path, np.stack((wet, wet), axis=1), 44_100, subtype="FLOAT")
            manifest = {
                "schema": 1,
                "role": "real plugin wet captures; split follows the original performance group",
                "renderer": {"format": "Audio Unit v2 offline render"},
                "captures": [
                    {
                        "label": "drive",
                        "split": "train",
                        "path": "train/drive/wet.wav",
                        "source_path": str(source_path),
                        "source_offset": 0.0,
                        "source_group": "phrase-a",
                        "component": {"name": "AUDistortion"},
                    }
                ],
            }
            (root / "manifest.json").write_text(json.dumps(manifest))
            pairs, report = apple_pairs(root, corpus)
            self.assertEqual(pairs[0].split, "fit")
            self.assertEqual(report["counts"], {"fit/drive": 1})
            paired_wet, paired_clean = read_pair(pairs[0])
            self.assertEqual(len(paired_wet), 48_000)
            self.assertEqual(paired_wet.shape, paired_clean.shape)

    def test_apple_manifest_rejects_group_leak(self):
        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory) / "corpus"
            root = corpus / "apple-au"
            source = corpus / "clean.wav"
            root.mkdir(parents=True)
            soundfile.write(source, np.zeros(96_000, dtype=np.float32), 48_000)
            captures = []
            for split in ("train", "valid"):
                wet = root / split / "drive/wet.wav"
                wet.parent.mkdir(parents=True)
                soundfile.write(wet, np.zeros((48_000, 2), dtype=np.float32), 48_000)
                captures.append(
                    {
                        "label": "drive",
                        "split": split,
                        "path": str(wet.relative_to(root)),
                        "source_path": str(source),
                        "source_offset": 0.0,
                        "source_group": "leaked",
                        "component": {"name": "AUDistortion"},
                    }
                )
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema": 1,
                        "role": "real plugin wet captures; split follows the original performance group",
                        "renderer": {"format": "Audio Unit v2 offline render"},
                        "captures": captures,
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "source group crosses splits"):
                apple_pairs(root, corpus)


if __name__ == "__main__":
    unittest.main()
