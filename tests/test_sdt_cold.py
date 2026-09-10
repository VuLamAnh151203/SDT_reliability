import importlib.util
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataloader import collate_dialogues, split_dialogues
from losses import COLDLoss, SDTCOLDLoss, symmetric_kl
from model import DistributionHead, MODALITIES, Transformer_Based_Model, cold_fusion
from sdt_backbone import MaskedKLDivLoss, MaskedNLLLoss, Multimodal_GatedFusion


torch.set_num_threads(1)


class SDTCOLDTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.config = dict(dataset="IEMOCAP", temp=2.0, D_text=5, D_visual=3,
                           D_audio=7, n_head=2, n_classes=6, hidden_dim=8,
                           n_speakers=2, dropout=0.0)
        self.mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.float32)
        self.labels = torch.tensor([[0, 2, 1, 5], [3, 4, -100, -100]])
        speakers = F.one_hot(torch.tensor([[0, 1, 0, 1], [1, 0, 0, 0]]), 2).float()
        self.inputs = (torch.randn(4, 2, 5), torch.randn(4, 2, 3), torch.randn(4, 2, 7),
                       self.mask, speakers, [4, 2])

    def model(self, variant="guided"):
        return Transformer_Based_Model(**self.config, fusion_variant=variant)

    def test_distribution_sampling_eval_and_bounds(self):
        head = DistributionHead(8, -4.0, 4.0)
        features = torch.randn(2, 4, 8)
        train1, train2 = head(features), head(features)
        self.assertFalse(torch.equal(train1["z"], train2["z"]))
        head.eval()
        result = head(features)
        self.assertTrue(torch.equal(result["z"], result["mu"]))
        with torch.no_grad():
            head.logvar.bias.fill_(1000.0)
        result = head(features)
        self.assertTrue(torch.isfinite(result["variance"]).all())
        self.assertLessEqual(result["logvar"].max().item(), 4.0)

    def test_sdt_preserving_distribution_initialization(self):
        model = Transformer_Based_Model(
            **self.config, fusion_variant="guided",
            distribution_init="sdt-preserving", initial_logvar=-6.0).eval()
        output = model(*self.inputs)
        identity = torch.eye(self.config["hidden_dim"])
        expected_variance = torch.full_like(
            output["distributions"]["t"]["variance"], torch.exp(torch.tensor(-6.0)))
        for index, name in enumerate(MODALITIES):
            head = model.distribution_heads[name]
            torch.testing.assert_close(head.mu.weight, identity)
            torch.testing.assert_close(head.mu.bias, torch.zeros_like(head.mu.bias))
            torch.testing.assert_close(head.logvar.weight, torch.zeros_like(head.logvar.weight))
            torch.testing.assert_close(head.logvar.bias, torch.full_like(head.logvar.bias, -6.0))
            torch.testing.assert_close(output["distributions"][name]["mu"],
                                       output["enhanced"][:, :, index, :])
            torch.testing.assert_close(output["distributions"][name]["variance"], expected_variance)
        torch.testing.assert_close(output["reliability"],
                                   torch.full_like(output["reliability"], 1.0 / 3.0))
        torch.testing.assert_close(output["fusion_weights"], output["sdt_weights"])
        expected_fused = (output["sdt_weights"] * output["enhanced"]).sum(dim=-2)
        torch.testing.assert_close(output["fused"], expected_fused)

        model.train()
        sampled = model(*self.inputs)
        self.assertFalse(torch.equal(sampled["latents"], sampled["enhanced"]))
        self.assertLess((sampled["latents"] - sampled["enhanced"]).std().item(), 0.07)

    def test_sdt_preserving_logvar_must_fit_clamp(self):
        with self.assertRaises(ValueError):
            DistributionHead(8, -4.0, 4.0, "sdt-preserving", -6.0)

    def test_both_gates_match_requested_equations(self):
        features, latents = torch.randn(2, 4, 3, 8), torch.randn(2, 4, 3, 8)
        norms = torch.tensor([3.0, 1.0, 2.0]).expand(2, 4, 3)
        reliability = norms / norms.sum(dim=-1, keepdim=True)
        gate = Multimodal_GatedFusion(8)
        for variant in ("replace", "guided"):
            fused, weights, sdt_weights = cold_fusion(features, latents, reliability, gate, variant)
            expected = reliability.unsqueeze(-1).expand_as(latents)
            if variant == "guided":
                expected = torch.softmax(gate.fc(features), dim=-2) * expected
                expected = expected / expected.sum(dim=-2, keepdim=True)
            torch.testing.assert_close(weights, expected)
            torch.testing.assert_close(weights.sum(dim=-2), torch.ones(2, 4, 8))
            torch.testing.assert_close(fused, (expected * latents).sum(dim=-2))
        uniform = torch.ones_like(reliability) / 3
        _, weights, sdt = cold_fusion(features, latents, uniform, gate, "guided")
        torch.testing.assert_close(weights, sdt)

    def test_cold_kl_matches_manual_probability_formula(self):
        a, b = torch.tensor([0.1, 1.3, 0.6]), torch.tensor([0.2, 1.1, 0.9])
        p, q = a.softmax(0), b.softmax(0)
        expected = (p * (p / q).log()).sum() + (q * (q / p).log()).sum()
        torch.testing.assert_close(symmetric_kl(a, b), expected)
        torch.testing.assert_close(symmetric_kl(a, a), torch.tensor(0.0))

    def test_cold_uses_global_tav_concat_and_valid_utterances(self):
        output = self.model().eval()(*self.inputs)
        valid = self.mask.bool()
        parts, errors = COLDLoss()(output, self.labels, valid)
        all_qualities, all_scores = [], []
        for index, name in enumerate(MODALITIES):
            error = F.cross_entropy(output["student_logits"][name][valid], self.labels[valid], reduction="none")
            quality = -error.detach()
            score = output["reliability_logits"][..., index][valid]
            torch.testing.assert_close(errors[name], error)
            torch.testing.assert_close(parts["cold_" + name], symmetric_kl(quality, score))
            all_qualities.append(quality)
            all_scores.append(score)
        torch.testing.assert_close(parts["cold_tav"], symmetric_kl(torch.cat(all_qualities), torch.cat(all_scores)))
        torch.testing.assert_close(parts["cold"], sum(parts["cold_" + m] for m in (*MODALITIES, "tav")))
        self.assertEqual(len(errors["t"]), 6)

    def test_padding_values_and_labels_cannot_change_any_loss(self):
        output = self.model().eval()(*self.inputs)
        loss_fn = SDTCOLDLoss()
        expected, parts, _ = loss_fn(output, self.labels, self.mask)
        changed_labels = self.labels.clone()
        changed_labels[~self.mask.bool()] = 900
        def corrupt(value):
            if isinstance(value, dict):
                return {key: corrupt(item) for key, item in value.items()}
            if torch.is_tensor(value) and value.shape[:2] == self.mask.shape:
                value = value.clone()
                value[~self.mask.bool()] = float("nan")
            return value
        actual, changed_parts, _ = loss_fn(corrupt(output), changed_labels, self.mask)
        torch.testing.assert_close(actual, expected)
        for name in parts:
            torch.testing.assert_close(changed_parts[name], parts[name])

    def test_gradients_reach_all_heads_classifiers_and_encoder(self):
        for variant in ("guided", "replace"):
            model = self.model(variant)
            output = model(*self.inputs)
            total, parts, _ = SDTCOLDLoss()(output, self.labels, self.mask)
            self.assertTrue(torch.isfinite(total))
            total.backward()
            for name in MODALITIES:
                for parameter in model.distribution_heads[name].parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    self.assertGreater(parameter.grad.abs().sum().item(), 0)
                classifier = getattr(model, name + "_output_layer")
                self.assertGreater(classifier[-1].weight.grad.abs().sum().item(), 0)
            self.assertGreater(model.t_t.transformer_inter[0].self_attn.linear_q.weight.grad.abs().sum().item(), 0)
            if variant == "guided":
                self.assertGreater(model.last_gate.fc.weight.grad.abs().sum().item(), 0)
            else:
                self.assertIsNone(model.last_gate.fc.weight.grad)

    def test_error_detach_only_affects_cold_supervision(self):
        for detach in (True, False):
            model = self.model()
            output = model(*self.inputs)
            parts, _ = COLDLoss(detach_errors=detach)(output, self.labels, self.mask)
            parts["cold"].backward()
            gradient = model.t_output_layer[-1].weight.grad
            if detach:
                self.assertIsNone(gradient)
            else:
                self.assertGreater(gradient.abs().sum().item(), 0)
            self.assertGreater(model.distribution_heads["t"].logvar.weight.grad.abs().sum().item(), 0)

    def test_sdt_distillation_still_backpropagates_to_teacher(self):
        model = self.model()
        output = model(*self.inputs)
        loss, _, _ = SDTCOLDLoss(gamma_1=0, gamma_2=0, gamma_3=1,
                                 lambda_co=0, lambda_reg=0)(output, self.labels, self.mask)
        loss.backward()
        self.assertGreater(model.all_output_layer.weight.grad.abs().sum().item(), 0)

    def test_eval_determinism_and_variance_reliability(self):
        model = self.model().eval()
        first, second = model(*self.inputs), model(*self.inputs)
        torch.testing.assert_close(first["logits"], second["logits"], rtol=0, atol=0)
        norms = first["variance_norm"]
        confidence = 1 / (norms + 1e-8)
        torch.testing.assert_close(first["confidence"], confidence)
        torch.testing.assert_close(first["error_scores"], confidence)
        torch.testing.assert_close(first["reliability_logits"], -(norms + 1e-8).log())
        torch.testing.assert_close(first["reliability"], confidence / confidence.sum(dim=-1, keepdim=True))
        largest_variance = norms.argmax(dim=-1)
        smallest_reliability = first["reliability"].argmin(dim=-1)
        torch.testing.assert_close(largest_variance, smallest_reliability)

    def test_empty_masks_and_singleton_cross_modal_loss(self):
        output = self.model()(*self.inputs)
        loss, parts, _ = SDTCOLDLoss()(output, self.labels, torch.zeros_like(self.mask))
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        single = torch.zeros_like(self.mask)
        single[0, 0] = 1
        parts, _ = COLDLoss()(output, self.labels, single)
        for name in MODALITIES:
            self.assertEqual(parts["cold_" + name].item(), 0.0)
        self.assertGreater(parts["cold_tav"].item(), 0.0)

    def test_regularizer_is_gaussian_kl(self):
        output = self.model().eval()(*self.inputs)
        valid = self.mask.bool()
        parts, _ = COLDLoss()(output, self.labels, valid)
        expected = sum(torch.distributions.kl_divergence(
            torch.distributions.Normal(output["distributions"][m]["mu"][valid],
                                       (0.5 * output["distributions"][m]["logvar"][valid]).exp()),
            torch.distributions.Normal(0.0, 1.0)).mean() for m in MODALITIES)
        torch.testing.assert_close(parts["reg"], expected)

    def test_baseline_and_encoder_match_original_sdt(self):
        original_path = ROOT.parent / "SDT" / "model.py"
        if not original_path.exists():
            self.skipTest("optional comparison requires the original SDT/model.py")
        spec = importlib.util.spec_from_file_location("original_sdt_for_test", original_path)
        original_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original_module)
        original = original_module.Transformer_Based_Model(**self.config).eval()
        baseline = self.model("sdt").eval()
        baseline.load_state_dict(original.state_dict(), strict=True)
        original_output = original(*self.inputs)
        output = baseline(*self.inputs)
        corresponding = ([output["student_log_prob"][m] for m in MODALITIES]
                         + [output["log_prob"], output["prob"]]
                         + [output["student_kl_log_prob"][m] for m in MODALITIES]
                         + [output["teacher_kl_prob"]])
        for expected, actual in zip(original_output, corresponding):
            torch.testing.assert_close(actual, expected)
        cold_model = self.model().eval()
        result = cold_model.load_state_dict(original.state_dict(), strict=False)
        self.assertFalse(result.unexpected_keys)
        self.assertTrue(all(key.startswith("distribution_heads.") for key in result.missing_keys))
        enhanced = cold_model(*self.inputs)["enhanced"]
        torch.testing.assert_close(enhanced, output["enhanced"], rtol=0, atol=0)
        weights = torch.tensor([1., 2., 3., 4., 5., 6.])
        ce, kd = MaskedNLLLoss(weights), MaskedKLDivLoss()
        expected_loss = sum(ce(original_output[i].reshape(-1, 6), self.labels.reshape(-1), self.mask)
                            for i in (0, 1, 2, 3))
        expected_loss += sum(kd(original_output[i].reshape(-1, 6), original_output[8].reshape(-1, 6), self.mask)
                             for i in (5, 6, 7))
        actual_loss, parts, _ = SDTCOLDLoss(weights)(output, self.labels, self.mask)
        torch.testing.assert_close(actual_loss, expected_loss)
        self.assertEqual(parts["cold"].item(), 0)

    def test_meld_forward_and_backward(self):
        config = {**self.config, "dataset": "MELD", "n_speakers": 9, "n_classes": 7}
        model = Transformer_Based_Model(**config)
        inputs = list(self.inputs)
        inputs[4] = F.one_hot(torch.tensor([[0, 8, 2, 3], [7, 4, 0, 0]]), 9).float()
        loss, _, _ = SDTCOLDLoss()(model(*inputs), self.labels, self.mask)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertFalse(model.t_t_gate.fc.weight.requires_grad)

    def test_collation_and_original_vs_validation_splits(self):
        items = []
        for index, length in enumerate((4, 2)):
            items.append((torch.ones(length, 5), torch.ones(length, 3), torch.ones(length, 7),
                          torch.ones(length, 2), torch.ones(length), torch.zeros(length, dtype=torch.long), str(index)))
        batch = collate_dialogues(items)
        self.assertEqual(batch[0].shape, (4, 2, 5))
        self.assertEqual(batch[4].shape, (2, 4))
        self.assertEqual(batch[5][1, 3].item(), -100)
        class Dataset:
            trainVid = list(range(20))
            testVid = [20, 21]
        original = split_dialogues(Dataset(), "test")
        validated = split_dialogues(Dataset(), "validation", 0.1)
        self.assertEqual(original["train"], list(range(20)))
        self.assertEqual(original["valid"], [])
        self.assertEqual(validated["valid"], [0, 1])
        self.assertFalse(set(validated["train"]) & set(validated["valid"]))


if __name__ == "__main__":
    unittest.main()
