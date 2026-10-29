import unittest

import numpy as np

from experiment.poincare_pure_mechanism.analyze import error_transitions


class ErrorTransitionTest(unittest.TestCase):
    def test_rates_apply_the_same_mask_to_prediction_and_label(self):
        labels = np.asarray([0, 1, 2, 3, 4, 5])
        baseline = {
            "sample_ids": [("d", index) for index in range(6)],
            "labels": labels,
            "predictions": np.asarray([0, 2, 2, 5, 3, 5]),
        }
        variant = {
            "sample_ids": list(baseline["sample_ids"]),
            "labels": labels.copy(),
            "predictions": np.asarray([1, 1, 2, 3, 3, 5]),
        }
        steps = np.abs(np.subtract.outer(np.arange(6), np.arange(6)))
        distances = np.minimum(steps, 6 - steps) / 3.0
        rows = error_transitions(
            baseline, variant, "P-0", "B-P", 2024, distances)
        adjacent = next(
            row for row in rows if row["baseline_error_tier"] == "adjacent")
        same = next(row for row in rows if row["baseline_error_tier"] == "same")
        self.assertEqual(adjacent["baseline_wrong"], 2)
        self.assertEqual(adjacent["wrong_to_correct"], 1)
        self.assertEqual(adjacent["correction_rate"], 0.5)
        self.assertEqual(same["baseline_correct"], 3)
        self.assertEqual(same["correct_to_wrong"], 1)
        self.assertAlmostEqual(same["harm_rate"], 1.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
