import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from .blind2 import LABELS, WINDOW
from .guitar_chain_data import ChainPresence
from .blind_data2 import (
    DEFAULT_SOURCE_CYCLE_WEIGHT,
    MULTIMODAL_SOURCE_ID,
    NONLINEAR_IMPLEMENTATIONS,
    SOURCE_CYCLE_WEIGHT,
    BlindPresenceData,
    _echo,
    _nonlinear,
    _nuisance,
    _unknown,
)


class BlindData2Test(unittest.TestCase):
    def test_renderers_preserve_geometry_and_are_finite(self) -> None:
        import random

        value = np.random.default_rng(7).normal(0.0, 0.08, 240_000).astype(np.float32)
        for renderer in (_nonlinear, _echo, _unknown):
            wet = renderer(value, random.Random(4))
            self.assertEqual(wet.shape, value.shape)
            self.assertTrue(np.isfinite(wet).all())

    def test_nonlinear_diagnostics_do_not_change_render(self) -> None:
        import random

        value = np.random.default_rng(9).normal(0.0, 0.08, 240_000).astype(np.float32)
        details = {}
        reference = _nonlinear(value, random.Random(14))
        diagnostic = _nonlinear(value, random.Random(14), details)
        np.testing.assert_array_equal(diagnostic, reference)
        self.assertIn(details["implementation"], NONLINEAR_IMPLEMENTATIONS)
        self.assertGreaterEqual(details["drive"], 1.4)
        self.assertLessEqual(details["drive"], 10.0)
        self.assertTrue(np.isfinite(details["scale_invariant_residual_db"]))

    def test_nuisance_cannot_inject_unlabelled_nonlinearity(self) -> None:
        import random

        value = np.linspace(-4.0, 4.0, 240_000, dtype=np.float32)
        full = _nuisance(value, random.Random(17))
        scaled = _nuisance(value * 0.25, random.Random(17))
        np.testing.assert_allclose(scaled, full * 0.25, atol=2.0e-6, rtol=2.0e-6)

    def test_dataset_contract_has_presence_but_no_order(self) -> None:
        fake = object.__new__(BlindPresenceData)
        fake.workspace = Path(".")
        fake.split = "fit"
        fake.samples = 1
        fake.seed = 5
        fake.epoch = 0
        fake.clean = [type("Clean", (), {"source_id": "egfxset", "path": Path("x"), "group": "g", "split": "fit"})()]
        fake.clean_by_source = {"egfxset": fake.clean}
        fake.rirs = []
        fake.real = {label: [Path(label)] for label in LABELS}
        fake.random_position_chains = []
        with patch("remix.blind_data2._read_real", return_value=np.zeros(WINDOW, dtype=np.float32)):
            row = fake[0]
        self.assertEqual(tuple(row["audio"].shape), (WINDOW,))
        self.assertEqual(tuple(row["target"].shape), (len(LABELS),))
        self.assertEqual(tuple(row["target_mask"].shape), (len(LABELS),))
        np.testing.assert_array_equal(row["target_mask"].numpy(), np.ones(len(LABELS)))
        self.assertNotIn("order", row)
        self.assertNotIn("controls", row)

    def test_amplifier_baseline_supervises_added_effects_only(self) -> None:
        fake = object.__new__(BlindPresenceData)
        fake.workspace = Path(".")
        fake.split = "fit"
        fake.samples = 4
        fake.seed = 5
        fake.epoch = 0
        clean = type(
            "Clean",
            (),
            {"source_id": MULTIMODAL_SOURCE_ID, "path": Path("x"), "group": "p01", "split": "fit"},
        )()
        fake.clean = [clean]
        fake.clean_by_source = {MULTIMODAL_SOURCE_ID: fake.clean}
        fake.source_schedule = (MULTIMODAL_SOURCE_ID,)
        fake.rirs = []
        fake.real = {label: [Path(label)] for label in LABELS}
        fake.random_position_chains = []
        with patch("remix.blind_data2._read", return_value=np.zeros(240_000, dtype=np.float32)):
            row = fake[3]
        target = row["target"].numpy()
        mask = row["target_mask"].numpy()
        np.testing.assert_array_equal(mask, target)
        self.assertGreaterEqual(int(target.sum()), 2)
        self.assertTrue(row["source"].startswith("synthetic-product-render:"))
        self.assertNotIn("order", row)

    def test_random_position_chain_uses_presence_without_order(self) -> None:
        fake = object.__new__(BlindPresenceData)
        fake.workspace = Path(".")
        fake.split = "fit"
        fake.samples = 2
        fake.seed = 5
        fake.epoch = 0
        fake.clean = []
        fake.clean_by_source = {}
        fake.rirs = []
        fake.real = {label: [Path(label)] for label in LABELS}
        target = np.asarray([1.0, 0.0, 1.0, 1.0], dtype=np.float32)
        fake.random_position_chains = [
            ChainPresence(Path("chain.wav"), "prs_neck_pick01", "prs", "fit", target, "10101")
        ]
        with patch("remix.blind_data2._read_real", return_value=np.zeros(WINDOW, dtype=np.float32)):
            row = fake[1]
        np.testing.assert_array_equal(row["target"].numpy(), target)
        np.testing.assert_array_equal(row["target_mask"].numpy(), np.ones(len(LABELS)))
        self.assertEqual(row["source"], "dafx-random-position-presence")
        self.assertNotIn("order", row)

    def test_epoch_changes_realization_seed(self) -> None:
        fake = object.__new__(BlindPresenceData)
        fake.epoch = 0
        fake.set_epoch(3)
        self.assertEqual(fake.epoch, 3)
        with self.assertRaises(ValueError):
            fake.set_epoch(-1)

    def test_isolated_note_sources_are_downweighted(self) -> None:
        self.assertLess(
            SOURCE_CYCLE_WEIGHT["longitudinal-guitar-string-ageing"],
            DEFAULT_SOURCE_CYCLE_WEIGHT,
        )
        self.assertGreater(
            SOURCE_CYCLE_WEIGHT["longitudinal-guitar-string-ageing"],
            SOURCE_CYCLE_WEIGHT["eg-ipt"],
        )
        self.assertLess(SOURCE_CYCLE_WEIGHT["eg-ipt"], DEFAULT_SOURCE_CYCLE_WEIGHT)
        self.assertLess(
            SOURCE_CYCLE_WEIGHT["freepats-electric-guitar-direct"],
            SOURCE_CYCLE_WEIGHT["eg-ipt"],
        )
        self.assertEqual(
            SOURCE_CYCLE_WEIGHT["karoryfer-emilyguitar"],
            SOURCE_CYCLE_WEIGHT["freepats-electric-guitar-direct"],
        )
        self.assertEqual(
            SOURCE_CYCLE_WEIGHT[MULTIMODAL_SOURCE_ID],
            SOURCE_CYCLE_WEIGHT["freepats-electric-guitar-direct"],
        )


if __name__ == "__main__":
    unittest.main()
