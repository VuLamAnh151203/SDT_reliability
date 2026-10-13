from __future__ import annotations

import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
SCRIPTS = HERE / "scripts"
for path in (HERE, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import classification_metrics, distance_bin, margin_bin
from e1_mixed_contribution import aggregate_pairs


class ResearchGapValidationTests(unittest.TestCase):
    def test_target_modality_mixed_is_not_pair_weighted(self):
        baseline = {("c", 2): {
            "baseline_margin": 0.3, "baseline_correct": 1, "target_emotion": 2,
        }}
        rows = [
            {"conversation_id": "c", "target_idx": "2", "source_idx": "0",
             "modality": "text", "replacement_strategy": "train_mean",
             "signed_impact": "0.02", "conversation_length": "3"},
            {"conversation_id": "c", "target_idx": "2", "source_idx": "1",
             "modality": "text", "replacement_strategy": "train_mean",
             "signed_impact": "-0.03", "conversation_length": "3"},
        ]
        result = aggregate_pairs(rows, baseline, 1e-4)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["mixed"], 1)
        self.assertEqual(result[0]["n_beneficial"], 1)
        self.assertEqual(result[0]["n_harmful"], 1)
        self.assertAlmostEqual(result[0]["mixed_span"], 0.05)

    def test_standard_metrics_count_each_target_once(self):
        metrics = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1])
        self.assertAlmostEqual(metrics["accuracy"], 75.0)

    def test_pre_registered_bins_are_deterministic(self):
        self.assertEqual(distance_bin(5), "01-05")
        self.assertEqual(distance_bin(6), "06-10")
        self.assertEqual(margin_bin(0.049), "low_<0.05")
        self.assertEqual(margin_bin(0.2), "high_>=0.20")


if __name__ == "__main__":
    unittest.main()
