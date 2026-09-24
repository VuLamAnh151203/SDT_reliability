import unittest

import numpy as np

from geometry_analysis.analyze_geometry import (
    balanced_indices, class_balanced_pair_values, confidence_bin_rows,
    knn_rows, pearson,
    rankdata, silhouette_values, spearman, tier_name, wheel_steps)
from geometry_analysis.summarize_causal import causal_id, paired_deltas


class GeometryAnalysisTest(unittest.TestCase):
    def test_correlations_and_tied_ranks(self):
        values = np.asarray([3.0, 1.0, 1.0, 2.0])
        np.testing.assert_allclose(rankdata(values), [4.0, 1.5, 1.5, 3.0])
        self.assertAlmostEqual(pearson(values, values), 1.0)
        self.assertAlmostEqual(spearman(values, values), 1.0)
        self.assertAlmostEqual(spearman(values, -values), -1.0)

    def test_balanced_subsample_retains_every_class(self):
        labels = np.asarray([0] * 20 + [1] * 5 + [2] * 3)
        selected = balanced_indices(labels, 12, seed=7)
        self.assertEqual(selected.size, 12)
        self.assertEqual(set(labels[selected]), {0, 1, 2})

    def test_class_pair_balancing_uses_equal_counts(self):
        labels = np.asarray([0, 0, 0, 1, 1, 1])
        matrix = np.abs(
            np.arange(6)[:, None] - np.arange(6)[None, :]).astype(float)
        semantic = np.asarray([[0.0, 1.0], [1.0, 0.0]])
        geometric, target = class_balanced_pair_values(
            matrix, labels, semantic, cap=2, seed=3)
        unique, counts = np.unique(target, return_counts=True)
        self.assertEqual(dict(zip(unique, counts)), {0.0: 4, 1.0: 2})
        self.assertEqual(geometric.size, 6)

    def test_tiers_silhouette_and_knn(self):
        class_distance = np.asarray([0.0, 1 / 3, 2 / 3, 1.0])
        np.testing.assert_array_equal(
            wheel_steps(class_distance, 6), [0, 1, 2, 3])
        self.assertEqual(
            [tier_name(step, 3) for step in range(4)],
            ["same", "adjacent", "middle", "far"])
        labels = np.asarray([0, 0, 1, 1])
        points = np.asarray([0.0, 0.1, 2.0, 2.1])
        matrix = np.abs(points[:, None] - points[None, :])
        self.assertTrue((silhouette_values(matrix, labels) > 0.8).all())
        rows = knn_rows(matrix, labels, "t", (1,), n_classes=2)
        self.assertEqual(rows[0]["accuracy"], 100.0)

    def test_causal_ablation_ids_and_paired_deltas(self):
        base = {
            "wheel_geometry": "poincare", "hyperbolic_dim": 16,
            "wheel_radius_mode": "free", "wheel_zero_residual": False,
        }
        self.assertEqual(causal_id(base), "P-FREE")
        self.assertEqual(
            causal_id({**base, "wheel_radius_mode": "fixed"}), "P-FIXED")
        self.assertEqual(
            causal_id({**base, "hyperbolic_dim": 2}), "P-2D")
        self.assertEqual(
            causal_id({**base, "wheel_zero_residual": True}), "P-ZERORES")
        self.assertEqual(
            causal_id({**base, "wheel_geometry": "euclidean"}), "E-FREE")

        rows = []
        scores = {
            "P-FREE": 75.0, "P-FIXED": 74.0, "P-2D": 73.0,
            "P-ZERORES": 72.0, "E-FREE": 71.0,
        }
        for run_id, score in scores.items():
            rows.append({
                "id": run_id, "seed": 2024, "weighted_f1": score,
                "macro_f1": score - 1, "accuracy": score + 1,
            })
        deltas = paired_deltas(rows)
        self.assertEqual(len(deltas), 4)
        self.assertEqual(
            [row["delta_weighted_f1"] for row in deltas],
            [1.0, 2.0, 3.0, 4.0])
        self.assertTrue(all(row["supports_hypothesis"] for row in deltas))

    def test_confidence_bins_include_one_and_report_statistics(self):
        rows = confidence_bin_rows(
            np.asarray([0.05, 0.15, 0.85, 1.0]),
            np.asarray([1.0, 2.0, 3.0, 5.0]),
            np.asarray([0, 1, 1, 1]), "t", n_bins=2)
        self.assertEqual([row["count"] for row in rows], [2, 2])
        self.assertEqual(rows[1]["confidence_high"], 1.0)
        self.assertEqual(rows[1]["radius_mean"], 4.0)
        self.assertEqual(rows[1]["radius_median"], 4.0)
        self.assertEqual(rows[1]["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
