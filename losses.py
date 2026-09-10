"""Masked SDT objective plus unimodal and concatenated TAV COLD losses."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import MODALITIES
from sdt_backbone import MaskedKLDivLoss, MaskedNLLLoss


def symmetric_kl(first_scores, second_scores):
    """One categorical distribution over all entries, NOT over classes/batches."""
    if first_scores.ndim != 1 or first_scores.shape != second_scores.shape:
        raise ValueError("symmetric_kl expects equally sized 1-D score vectors")
    if first_scores.numel() == 0:
        return first_scores.sum() + second_scores.sum()
    log_p = F.log_softmax(first_scores, dim=0)
    log_q = F.log_softmax(second_scores, dim=0)
    return ((log_p.exp() - log_q.exp()) * (log_p - log_q)).sum()


def masked_weighted_nll(log_prob, target, mask, class_weights=None,
                        example_weights=None):
    """NLL with SDT-compatible normalization and optional example weights."""
    valid = mask.reshape(-1).bool()
    if valid.sum().item() == 0:
        return log_prob.sum() * 0.0
    selected_target = target.reshape(-1)[valid]
    selected_log_prob = log_prob.reshape(-1, log_prob.size(-1))[valid]
    weights = torch.ones_like(selected_target, dtype=log_prob.dtype)
    if example_weights is not None:
        weights = example_weights.reshape(-1)[valid].type_as(log_prob)
    per_example = F.nll_loss(selected_log_prob, selected_target,
                             weight=class_weights, reduction="none")
    if class_weights is None:
        denominator = weights.sum()
    else:
        denominator = (weights * class_weights[selected_target]).sum()
    if denominator.item() <= 0:
        return selected_log_prob.sum() * 0.0
    return (per_example * weights).sum() / denominator


def masked_weighted_kl(log_pred, target, mask, example_weights=None):
    """KL rows averaged with optional example weights."""
    valid = mask.reshape(-1).bool()
    if valid.sum().item() == 0:
        return log_pred.sum() * 0.0
    log_pred = log_pred.reshape(-1, log_pred.size(-1))[valid]
    target = target.reshape(-1, target.size(-1))[valid]
    weights = torch.ones(log_pred.size(0), device=log_pred.device,
                         dtype=log_pred.dtype)
    if example_weights is not None:
        weights = example_weights.reshape(-1)[valid].type_as(log_pred)
    denominator = weights.sum()
    if denominator.item() <= 0:
        return log_pred.sum() * 0.0
    per_example = F.kl_div(log_pred, target, reduction="none").sum(dim=-1)
    return (per_example * weights).sum() / denominator


def reliability_kl(predicted_logits, target_probabilities, mask):
    """KL(target || prediction), averaged over valid utterances."""
    valid = mask.bool()
    if valid.sum().item() == 0:
        return predicted_logits.sum() * 0.0
    targets = target_probabilities[valid].type_as(predicted_logits)
    if targets.shape[-1] != len(MODALITIES):
        raise ValueError("reliability targets must have T/A/V probabilities")
    if not torch.isfinite(targets).all() or (targets < 0).any():
        raise ValueError("reliability targets must be finite and nonnegative")
    targets = targets / targets.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    log_targets = targets.clamp_min(1e-12).log()
    log_predictions = F.log_softmax(predicted_logits[valid], dim=-1)
    return (targets * (log_targets - log_predictions)).sum(dim=-1).mean()


class COLDLoss(nn.Module):
    def __init__(self, detach_errors=True):
        super().__init__()
        self.detach_errors = detach_errors

    def forward(self, outputs, labels, valid_mask):
        valid = valid_mask.bool()
        targets = labels[valid]
        zero = outputs["logits"][valid].sum() * 0.0
        parts = {"cold_" + name: zero for name in (*MODALITIES, "tav")}
        errors = {}
        scores = {}
        regularizers = []
        if outputs["distributions"]:
            for index, name in enumerate(MODALITIES):
                dist = outputs["distributions"][name]
                # Mask BEFORE CE, including padded labels such as -100.
                errors[name] = F.cross_entropy(
                    outputs["student_logits"][name][valid], targets, reduction="none")
                target_errors = errors[name].detach() if self.detach_errors else errors[name]
                # Quality and reliability have the same direction:
                # lower CE -> larger -CE; lower variance -> larger reliability.
                quality = -target_errors
                scores[name] = outputs["reliability_logits"][..., index][valid]
                parts["cold_" + name] = symmetric_kl(quality, scores[name])
                # Assumption for the unspecified L_reg: KL(q || N(0,I)),
                # mean valid utterances AND latent dimensions, sum modalities.
                mu = dist["mu"][valid]
                logvar = dist["logvar"][valid]
                elementwise = mu.square() + logvar.exp() - 1.0 - logvar
                regularizers.append(0.5 * elementwise.sum() / max(1, elementwise.numel()))
            tav_errors = torch.cat([errors[m] for m in MODALITIES], dim=0)
            if self.detach_errors:
                tav_errors = tav_errors.detach()
            parts["cold_tav"] = symmetric_kl(
                -tav_errors, torch.cat([scores[m] for m in MODALITIES], dim=0))
        parts["cold"] = sum(parts.values())
        parts["reg"] = sum(regularizers, zero)
        return parts, errors


class SDTCOLDLoss(nn.Module):
    def __init__(self, class_weights=None, gamma_1=1.0, gamma_2=1.0,
                 gamma_3=1.0, lambda_co=0.1, lambda_reg=0.1,
                 detach_errors=True, lambda_reliability=1.0):
        super().__init__()
        weights = (gamma_1, gamma_2, gamma_3, lambda_co, lambda_reg,
                   lambda_reliability)
        if any(not math.isfinite(w) or w < 0 for w in weights):
            raise ValueError("loss weights must be finite and nonnegative")
        self.gamma_1, self.gamma_2, self.gamma_3 = weights[:3]
        self.lambda_co, self.lambda_reg = weights[3:5]
        self.lambda_reliability = weights[5]
        self.ce = MaskedNLLLoss(class_weights)
        self.kl = MaskedKLDivLoss()
        self.cold = COLDLoss(detach_errors=detach_errors)

    def forward(self, outputs, labels, valid_mask, reliability_targets=None,
                sample_keep_mask=None, modality_keep_mask=None):
        targets = labels.reshape(-1)
        n_classes = outputs["logits"].size(-1)
        if sample_keep_mask is None and modality_keep_mask is None:
            task = self.ce(outputs["log_prob"].reshape(-1, n_classes), targets, valid_mask)
            students = sum(self.ce(outputs["student_log_prob"][m].reshape(-1, n_classes),
                                   targets, valid_mask) for m in MODALITIES)
        else:
            sample_weights = (torch.ones_like(valid_mask) if sample_keep_mask is None
                              else sample_keep_mask).type_as(outputs["logits"])
            if modality_keep_mask is None:
                modality_keep_mask = torch.ones(
                    *valid_mask.shape, len(MODALITIES),
                    device=valid_mask.device, dtype=sample_weights.dtype)
            task = masked_weighted_nll(
                outputs["log_prob"], labels, valid_mask, self.ce.weight,
                sample_weights)
            students = sum(masked_weighted_nll(
                outputs["student_log_prob"][m], labels, valid_mask,
                self.ce.weight, sample_weights * modality_keep_mask[..., index])
                for index, m in enumerate(MODALITIES))
        # Preserve SDT source semantics: teacher is NOT detached and no T^2 factor.
        teacher = outputs["teacher_kl_prob"].reshape(-1, n_classes)
        if sample_keep_mask is None and modality_keep_mask is None:
            kd = sum(self.kl(outputs["student_kl_log_prob"][m].reshape(-1, n_classes),
                             teacher, valid_mask) for m in MODALITIES)
        else:
            kd = sum(masked_weighted_kl(
                outputs["student_kl_log_prob"][m], outputs["teacher_kl_prob"],
                valid_mask, sample_weights * modality_keep_mask[..., index])
                for index, m in enumerate(MODALITIES))
        sdt = self.gamma_1 * task + self.gamma_2 * students + self.gamma_3 * kd
        cold_parts, errors = self.cold(outputs, labels, valid_mask)
        weighted_cold = self.lambda_co * cold_parts["cold"]
        weighted_reg = self.lambda_reg * cold_parts["reg"]
        zero = outputs["logits"][valid_mask.bool()].sum() * 0.0
        reliability = zero
        if reliability_targets is not None:
            if outputs["reliability_logits"] is None:
                raise ValueError("model does not expose reliability logits")
            rel_mask = valid_mask if sample_keep_mask is None else (
                valid_mask.bool() & sample_keep_mask.bool())
            reliability = reliability_kl(
                outputs["reliability_logits"], reliability_targets, rel_mask)
        weighted_reliability = self.lambda_reliability * reliability
        total = sdt + weighted_cold + weighted_reg + weighted_reliability
        valid = valid_mask.bool()
        sample_keep_rate = (torch.ones((), device=total.device) if sample_keep_mask is None
                            else sample_keep_mask[valid].float().mean())
        modality_keep_rate = (torch.ones((), device=total.device) if modality_keep_mask is None
                              else modality_keep_mask[valid].float().mean())
        parts = {"total": total, "sdt": sdt, "task": task, "student_ce": students,
                 "distillation": kd, **cold_parts,
                 "weighted_cold": weighted_cold, "weighted_reg": weighted_reg,
                 "reliability_loss": reliability,
                 "weighted_reliability": weighted_reliability,
                 "sample_keep_rate": sample_keep_rate,
                 "modality_keep_rate": modality_keep_rate}
        return total, parts, errors
