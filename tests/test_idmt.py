import unittest

from remix.idmt import select


class IdmtTests(unittest.TestCase):
    def test_selection_is_balanced_and_group_unique_per_effect(self):
        rows = []
        effects = {
            "drive": ("Distortion", "Overdrive"),
            "delay": ("FeedbackDelay", "SlapbackDelay"),
            "reverb": ("Reverb",),
        }
        for target, names in effects.items():
            for effect in names:
                for index in range(12):
                    rows.append({
                        "split": "test", "target": target, "effect": effect,
                        "group": f"g-{target}-{effect}-{index}",
                        "dry": f"dry-{target}-{effect}-{index}.wav",
                        "wet": f"wet-{target}-{effect}-{index}.wav",
                        "instrument_setting": 9,
                    })
        result = select(rows, 8, "test")
        self.assertEqual(len(result), 32)
        self.assertEqual({row["target"] for row in result}, {"clean", "drive", "delay", "reverb"})
        for target in ("clean", "drive", "delay", "reverb"):
            chosen = [row for row in result if row["target"] == target]
            self.assertEqual(len(chosen), 8)
            self.assertEqual(len({row["group"] for row in chosen}), 8)


if __name__ == "__main__":
    unittest.main()
