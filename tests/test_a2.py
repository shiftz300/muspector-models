import numpy as np
import unittest

from remix.a2 import full


def document():
    return {
        "architecture": "SlimmableContainer",
        "sample_rate": 48_000.0,
        "config": {"submodels": [
            {"model": {"architecture": "WaveNet", "weights": [1.0]}},
            {"model": {"architecture": "WaveNet", "weights": [1.0, 2.0]}},
        ]},
    }


class A2Tests(unittest.TestCase):
    def test_selects_full_capacity_without_mutating_document(self):
        source = document()
        selected = full(source)
        self.assertEqual(selected["weights"], [1.0, 2.0])
        self.assertEqual(selected["sample_rate"], 48_000.0)
        self.assertNotIn("sample_rate", source["config"]["submodels"][1]["model"])

    def test_rejects_wrong_container_contract(self):
        for field, value in (("architecture", "WaveNet"), ("sample_rate", 44_100)):
            with self.subTest(field=field):
                source = document()
                source[field] = value
                with self.assertRaises(ValueError):
                    full(source)

    def test_rejects_non_wavenet_member(self):
        source = document()
        source["config"]["submodels"][0]["model"]["architecture"] = "LSTM"
        with self.assertRaises(ValueError):
            full(source)


if __name__ == "__main__":
    unittest.main()
