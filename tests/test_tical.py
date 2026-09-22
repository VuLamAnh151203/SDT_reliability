import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from losses import SDTCOLDLoss, masked_kl_per_utterance
from model import MODALITIES, Transformer_Based_Model
from tical import (AnchorBank, HyperbolicProjector, compute_consistency,
                   blend_typicality, circular_class_distance_matrix,
                   emotion_wheel_angles, emotion_wheel_prototypes,
                   hyp_cpcc_loss, hyperbolic_prototype_outputs,
                   poincare_distance, wheel_label_discrepancy)


torch.set_num_threads(1)


class TiCALTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.config = dict(dataset="IEMOCAP", temp=2.0, D_text=5, D_visual=3,
                           D_audio=7, n_head=2, n_classes=6, hidden_dim=8,
                           n_speakers=2, dropout=0.0, fusion_variant="sdt")
        self.mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.float32)
        speakers = F.one_hot(torch.tensor([[0, 1, 0], [1, 0, 0]]), 2).float()
        self.inputs = (torch.randn(3, 2, 5), torch.randn(3, 2, 3),
                       torch.randn(3, 2, 7), self.mask, speakers, [3, 2])

    def test_projection_and_distance_stay_inside_poincare_ball(self):
        projector = HyperbolicProjector(8, 4)
        projected = projector(torch.randn(3, 5, 8) * 100)
        self.assertTrue((projected.norm(dim=-1) < 1).all())
        distance = poincare_distance(projected[:, :1], projected[:, 1:])
        self.assertTrue(torch.isfinite(distance).all())
        self.assertTrue((distance > 0).all())

    def test_anchor_bank_is_fifo_and_checkpoint_loads_dynamic_buffers(self):
        bank = AnchorBank(2, n_classes=3, max_size=3)
        features = torch.tensor([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0]])
        labels = torch.tensor([0, 1, 2, 1])
        bank.update(features, labels)
        self.assertEqual(bank.size, 3)
        torch.testing.assert_close(bank.labels, torch.tensor([1, 2, 1]))
        loaded = AnchorBank(2, n_classes=3, max_size=3)
        loaded.load_state_dict(bank.state_dict())
        distance, nearest = loaded.query(torch.tensor([[0.3, 0.0]]))
        self.assertEqual(nearest.item(), 1)
        self.assertTrue(torch.isfinite(distance).all())

    def test_consistency_orders_guideline_sanity_cases(self):
        high = compute_consistency(*(torch.tensor([0.9]) for _ in range(3)),
                                   torch.tensor([0.0]))
        disagreement = compute_consistency(*(torch.tensor([0.9]) for _ in range(3)),
                                           torch.tensor([1.0]))
        weak_modality = compute_consistency(torch.tensor([0.9]), torch.tensor([0.1]),
                                           torch.tensor([0.9]), torch.tensor([0.0]))
        self.assertGreater(high.item(), disagreement.item())
        self.assertGreater(high.item(), weak_modality.item())

    def test_emotion_wheel_prototypes_follow_iemocap_order(self):
        angles = emotion_wheel_angles("IEMOCAP", 6)
        expected_order = (0, 4, 3, 5, 1, 2)
        step = 2.0 * torch.pi / 6.0
        for position, class_id in enumerate(expected_order):
            torch.testing.assert_close(
                angles[class_id], angles.new_tensor(position) * step)
        prototypes = emotion_wheel_prototypes(
            angles, feature_dim=4, radius=0.75)
        torch.testing.assert_close(
            prototypes.norm(dim=-1), torch.full((6,), 0.75))
        self.assertTrue((prototypes[:, 2:] == 0).all())
        distances = circular_class_distance_matrix(angles)
        torch.testing.assert_close(distances, distances.t())
        torch.testing.assert_close(
            distances.diag(), torch.zeros(6), atol=1e-4, rtol=0)
        self.assertLess(distances[0, 4], distances[0, 5])

    def test_prototype_prediction_blending_and_wheel_disagreement(self):
        angles = emotion_wheel_angles("IEMOCAP", 6)
        prototypes = emotion_wheel_prototypes(angles, 2, radius=0.7)
        _, logits, prediction, typicality = hyperbolic_prototype_outputs(
            prototypes, prototypes, temperature=0.5)
        torch.testing.assert_close(prediction, torch.arange(6))
        self.assertTrue((typicality > 0.98).all())
        self.assertTrue(torch.equal(logits.argmax(dim=-1), torch.arange(6)))

        anchor = torch.tensor([0.25, 0.81])
        proto = torch.tensor([1.0, 0.25])
        torch.testing.assert_close(blend_typicality(anchor, proto, 0.0), anchor)
        torch.testing.assert_close(blend_typicality(anchor, proto, 1.0), proto)
        torch.testing.assert_close(
            blend_typicality(anchor, proto, 0.5),
            torch.sqrt(anchor * proto))

        matrix = circular_class_distance_matrix(angles)
        close = wheel_label_discrepancy(
            torch.tensor([0]), torch.tensor([4]), torch.tensor([0]), matrix)
        far = wheel_label_discrepancy(
            torch.tensor([0]), torch.tensor([5]), torch.tensor([0]), matrix)
        self.assertLess(close.item(), far.item())

    def test_wheel_cpcc_and_integrated_losses_backpropagate(self):
        angles = emotion_wheel_angles("IEMOCAP", 6)
        class_distances = circular_class_distance_matrix(angles)
        standalone = HyperbolicProjector(5, 2)(torch.randn(8, 5))
        standalone_loss = hyp_cpcc_loss(
            standalone,
            torch.tensor([0, 4, 3, 5, 1, 2, 0, 3]),
            class_distance_matrix=class_distances)
        self.assertTrue(torch.isfinite(standalone_loss))

        model = Transformer_Based_Model(
            **self.config, use_tical=True, tical_mode="full",
            tical_warmup_epochs=1, anchor_conf_threshold=0.0,
            anchor_size=100, hyperbolic_dim=2,
            use_emotion_wheel=True, wheel_prototype_radius=0.7,
            wheel_temperature=0.5, wheel_anchor_mix=0.5)
        model.set_tical_epoch(1)
        warmup = model(*self.inputs)
        anchor_labels = warmup["logits"].argmax(dim=-1)
        model.update_tical_anchors(warmup, anchor_labels, self.mask.bool())
        model.set_tical_epoch(2)
        output = model(*self.inputs)
        self.assertTrue(output["tical"]["ready"])
        self.assertTrue(output["tical"]["wheel_enabled"])
        self.assertIsNotNone(output["tical"]["prototype_tau"])
        labels = torch.tensor([[0, 4, 3], [5, 1, 0]])
        criterion = SDTCOLDLoss(
            gamma_1=0, gamma_2=0, gamma_3=0,
            lambda_co=0, lambda_reg=0, tical_mode="full",
            lambda_hyp=0, lambda_wheel_proto=1.0,
            lambda_wheel_cpcc=0.5)
        loss, parts, _ = criterion(output, labels, self.mask.bool())
        self.assertGreater(parts["wheel_proto"].item(), 0.0)
        self.assertTrue(torch.isfinite(parts["wheel_cpcc"]))
        torch.testing.assert_close(
            loss,
            parts["weighted_wheel_proto"]
            + parts["weighted_wheel_cpcc"])
        loss.backward()
        projector_gradient = model.tical.projectors["t"].linear.weight.grad
        self.assertIsNotNone(projector_gradient)
        self.assertGreater(projector_gradient.abs().sum().item(), 0.0)

    def test_tical_queries_old_bank_and_ca_kd_matches_formula(self):
        model = Transformer_Based_Model(
            **self.config, use_tical=True, tical_mode="kd",
            tical_warmup_epochs=1, anchor_conf_threshold=0.0,
            anchor_size=100, hyperbolic_dim=4)
        model.set_tical_epoch(1)
        warmup = model(*self.inputs)
        self.assertFalse(warmup["tical"]["ready"])
        labels = warmup["logits"].argmax(dim=-1)
        added = model.update_tical_anchors(warmup, labels, self.mask.bool())
        self.assertEqual(added, int(self.mask.sum().item()))
        sizes_before = [model.tical.anchor_banks[m].size for m in MODALITIES]

        model.set_tical_epoch(2)
        output = model(*self.inputs)
        self.assertTrue(output["tical"]["ready"])
        self.assertFalse(output["tical"]["wheel_enabled"])
        for name in MODALITIES:
            torch.testing.assert_close(
                output["tical"]["tau"][name],
                output["tical"]["anchor_tau"][name])
        torch.testing.assert_close(
            output["tical"]["label_discrepancy"],
            output["tical"]["categorical_discrepancy"])
        self.assertEqual(sizes_before,
                         [model.tical.anchor_banks[m].size for m in MODALITIES])
        criterion = SDTCOLDLoss(lambda_co=0, lambda_reg=0,
                                tical_mode="kd", lambda_hyp=0)
        _, parts, _ = criterion(output, labels, self.mask.bool())
        per_item = sum(masked_kl_per_utterance(
            output["student_kl_log_prob"][m], output["teacher_kl_prob"], self.mask)
            for m in MODALITIES)
        kappa = output["tical"]["kappa"]
        expected = (kappa * per_item).sum() / kappa.sum().clamp_min(1e-8)
        torch.testing.assert_close(parts["ca_distillation"], expected)
        self.assertEqual(parts["cold"].item(), 0.0)
        model.eval()
        with self.assertRaises(RuntimeError):
            model.update_tical_anchors(output, labels, self.mask.bool())

    def test_observation_warmup_is_exact_sdt_prediction_and_loss(self):
        baseline = Transformer_Based_Model(**self.config).eval()
        tical = Transformer_Based_Model(
            **self.config, use_tical=True, tical_mode="observe",
            tical_warmup_epochs=5, hyperbolic_dim=4).eval()
        result = tical.load_state_dict(baseline.state_dict(), strict=False)
        self.assertFalse(result.unexpected_keys)
        expected, actual = baseline(*self.inputs), tical(*self.inputs)
        torch.testing.assert_close(actual["logits"], expected["logits"], rtol=0, atol=0)
        baseline_loss, _, _ = SDTCOLDLoss(lambda_co=0, lambda_reg=0)(
            expected, torch.zeros_like(self.mask, dtype=torch.long), self.mask)
        observed_loss, _, _ = SDTCOLDLoss(
            lambda_co=0, lambda_reg=0, tical_mode="observe")(
                actual, torch.zeros_like(self.mask, dtype=torch.long), self.mask)
        torch.testing.assert_close(observed_loss, baseline_loss, rtol=0, atol=0)

    def test_hyp_cpcc_backpropagates_and_cold_cannot_mix_with_tical(self):
        features = HyperbolicProjector(5, 3)(torch.randn(6, 5))
        loss = hyp_cpcc_loss(features, torch.tensor([0, 0, 1, 1, 2, 2]))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        with self.assertRaises(ValueError):
            Transformer_Based_Model(**{**self.config, "fusion_variant": "guided"},
                                    use_tical=True)


if __name__ == "__main__":
    unittest.main()
