import unittest

from remix.family_eval import metrics, select_margin


def row(margin, exact, truth=None, predicted=None):
    truth = ["drive"] if truth is None else truth
    predicted = truth if predicted is None else predicted
    return {"margin": margin, "exact": exact, "truth": truth, "predicted": predicted}


class FamilyEvalTests(unittest.TestCase):
    def test_selection_excludes_highest_observed_error(self):
        rows = [row(0.01, False, ["drive"], []), row(0.02, True), row(0.03, True)]
        threshold = select_margin(rows)
        self.assertGreater(threshold, 0.01)
        result = metrics(rows, threshold)
        self.assertEqual(result["accepted_errors"], 0)
        self.assertEqual(result["accepted"], 2)

    def test_no_errors_needs_no_extra_margin(self):
        self.assertEqual(select_margin([row(0.01, True)]), 0.0)


if __name__ == "__main__":
    unittest.main()
