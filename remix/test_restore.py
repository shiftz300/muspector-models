import unittest
from unittest.mock import Mock

import torch

from .restore import Bank, Graph, Profile, Stage, plan


class RestoreTests(unittest.TestCase):
    def test_plan_reverses_each_request_without_fixed_stage_order(self):
        self.assertEqual(plan(("reverb", "drive")), ("drive", "reverb"))
        self.assertEqual(plan(("drive", "reverb")), ("reverb", "drive"))

    def test_bank_executes_the_selected_reverse_order(self):
        calls = []

        def stage(name):
            model = Mock()
            model.side_effect = lambda value, strength: calls.append((name, strength)) or value
            return model

        bank = Bank({"drive": stage("drive"), "reverb": stage("reverb")})
        result = bank.run(torch.zeros(1, 16), ("reverb", "drive"))
        self.assertEqual(result.restore, ("drive", "reverb"))
        self.assertEqual(calls, [("drive", 1.0), ("reverb", 1.0)])

    def test_plan_rejects_repeated_or_unknown_stages(self):
        for value in ((), ("drive", "drive"), ("delay",)):
            with self.subTest(value=value), self.assertRaises(ValueError):
                plan(value)

    def test_graph_supports_repeated_families_and_reverse_instance_order(self):
        calls = []

        class Runtime:
            def restore(self, audio, stage, profile):
                calls.append((stage.slot, stage.family, stage.device, profile.id))
                return audio

        graph = Graph({"drive": Runtime(), "reverb": Runtime()})
        forward = (
            Stage("verb", "reverb", "reverb"),
            Stage("boost", "drive", "drive", "ts9"),
            Stage("dist", "drive", "drive", "rat"),
        )
        result = graph.run(torch.zeros(1, 64), forward)
        self.assertEqual(tuple(stage.slot for stage in result.restore), ("dist", "boost", "verb"))
        self.assertEqual(calls, [("dist", "drive", "rat", "original"), ("boost", "drive", "ts9", "original"), ("verb", "reverb", None, "original")])

    def test_graph_carries_explicit_reference_profile(self):
        seen = []

        class Runtime:
            def restore(self, audio, stage, profile):
                seen.append(profile)
                return audio

        reference = torch.ones(1, 64)
        Graph({"compress": Runtime()}).run(
            torch.zeros(1, 64),
            (Stage("comp", "compressor", "compress"),),
            Profile("my-clean", "reference", reference),
        )
        self.assertTrue(torch.equal(seen[0].reference, reference))

    def test_graph_rejects_missing_runtime_and_geometry_change(self):
        stage = Stage("mod", "chorus", "chorus")
        with self.assertRaises(ValueError):
            Graph({"drive": Mock()}).run(torch.zeros(1, 64), (stage,))

        class Broken:
            def restore(self, audio, stage, profile):
                return audio[:, :-1]

        with self.assertRaises(ValueError):
            Graph({"chorus": Broken()}).run(torch.zeros(1, 64), (stage,))


if __name__ == "__main__":
    unittest.main()
