import json
import unittest
from pathlib import Path

from remix.adapter_eval import GATES


ROOT = Path(__file__).resolve().parents[1]


class AdapterEvalTests(unittest.TestCase):
    def test_physical_rat_control_gates_are_not_generic_knob_gates(self):
        self.assertEqual(set(GATES["controls"]), {"distortion", "filter", "volume"})
        self.assertLessEqual(GATES["controls"]["volume"]["mae"], GATES["macro_mae"])

    def test_cycle_keeps_development_and_hardware_boundaries_explicit(self):
        cycle = json.loads((ROOT / "cycles/adapter.json").read_text())
        self.assertEqual(cycle["status"], "complete-development")
        self.assertEqual(cycle["model"]["selection"], "explicit-only")
        self.assertFalse(cycle["data"]["physical_audio_devices_used"])
        self.assertTrue(cycle["data"]["source_read_only"])
        self.assertFalse(cycle["data"]["source_audio_modified"])
        self.assertIn("not new sealed evidence", cycle["development"]["source"])

    def test_chain_package_owns_adapter_contract_and_replay(self):
        source = json.loads((ROOT / "models/catalog/remix/sources.json").read_text())
        chain = next(row["package"] for row in source["packages"] if row["package"]["id"] == "chain")
        artifacts = {row["path"] for row in chain["artifacts"]}
        self.assertEqual(chain["version"], "1.1.0")
        self.assertEqual(chain["runtime"]["entrypoint"], "python.remix.adapter.v1")
        self.assertIn("contract/adapter.py", artifacts)
        self.assertIn("evidence/rat.json", artifacts)
        self.assertEqual(chain["quality"], "development")


if __name__ == "__main__":
    unittest.main()
