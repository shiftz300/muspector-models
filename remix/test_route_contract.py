import inspect
import json
import unittest
from pathlib import Path

from .ambience_model2 import AmbienceExpert
from .ambience_model3 import AmbienceProfileExpert
from .ambience_model4 import (
    AmbienceDecayBankExpert,
    AmbienceGrayboxExpert,
    AmbienceSparseMaskExpert,
)
from .amp_cab_model import AmpCabInverseExpert
from .amp_cab_model2 import AmpCabInverseExpert2
from .amp_cab_model3 import AmpCabInverseExpert3
from .amp_model import AmpInverseExpert
from .amp_model2 import AmpDynamicsInverseExpert
from .amp_model3 import AmpFrameDynamicsInverseExpert
from .amp_model4 import AmpStructuredInverseExpert
from .amp_model5 import AmpStageSupervisedInverseExpert
from .chorus_model3 import ChorusTrajectoryV3
from .flanger3 import manifest as flanger_manifest
from .foundation_model import Expert
from .inverse3 import NonlinearInverseV3
from .modulation_model3 import TremoloInverseV3
from .phaser3 import DEFAULT_MARGIN_THRESHOLD, manifest as phaser_manifest
from .spectral_model3 import SpectralInverseV3


ROOT = Path(__file__).resolve().parents[1]


class RouteContractTests(unittest.TestCase):
    def _model_manifests(self):
        models = (
            Expert("nonlinear"),
            Expert("dynamics"),
            Expert("temporal"),
            NonlinearInverseV3(),
            AmpInverseExpert(),
            AmpDynamicsInverseExpert(),
            AmpFrameDynamicsInverseExpert(),
            AmpStructuredInverseExpert(),
            AmpStageSupervisedInverseExpert(),
            AmbienceExpert(),
            AmbienceProfileExpert(),
            AmbienceGrayboxExpert(),
            AmbienceSparseMaskExpert(),
            AmbienceDecayBankExpert(),
            SpectralInverseV3(),
            TremoloInverseV3(),
            ChorusTrajectoryV3(),
            AmpCabInverseExpert(),
            AmpCabInverseExpert2(),
            AmpCabInverseExpert3(),
        )
        return [(type(model).__name__, model.manifest(), model.forward) for model in models]

    def test_every_inverse_expert_is_order_independent(self):
        manifests = self._model_manifests()
        manifests.extend((
            ("phaser", phaser_manifest(DEFAULT_MARGIN_THRESHOLD), None),
            ("flanger", flanger_manifest(), None),
        ))
        forbidden = ("order", "topology", "neighbor", "neighbour", "previous", "next", "graph")
        for name, manifest, forward in manifests:
            with self.subTest(expert=name):
                self.assertIs(manifest.get("graph_order_input"), False)
                neighbor = manifest.get(
                    "neighbor_effect_input", manifest.get("neighbouring_effect_input")
                )
                self.assertIs(neighbor, False)
                if forward is not None:
                    inputs = tuple(inspect.signature(forward).parameters)
                    self.assertFalse(
                        any(token in value.lower() for value in inputs for token in forbidden),
                        inputs,
                    )

    def test_training_modules_cannot_use_weight_only_authorization(self):
        offenders = []
        for path in (ROOT / "remix").glob("*.py"):
            if path.name in {"license_gate.py", "test_route_contract.py"}:
                continue
            if "require_product_weights" in path.read_text():
                offenders.append(path.name)
        self.assertEqual(offenders, [])

    def test_current_product_status_is_not_promoted(self):
        for name in ("foundation.json", "restore.json"):
            cycle = json.loads((ROOT / "cycles" / name).read_text())
            self.assertIsNone(cycle["usable_model"])
            self.assertIn("not-promoted", cycle["status"])


if __name__ == "__main__":
    unittest.main()
