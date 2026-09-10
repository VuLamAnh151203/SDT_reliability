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
                scores[name] = outputs["error_scores"][..., index][valid]
                parts["cold_" + name] = symmetric_kl(target_errors, scores[name])
                # Assumption for the unspecified L_reg: KL(q || N(0,I)),
                # sum latent dimensions, mean valid utterances, sum modalities.
                mu = dist["mu"][valid]
                logvar = dist["logvar"][valid]
                elementwise = mu.square() + logvar.exp() - 1.0 - logvar
                regularizers.append(0.5 * elementwise.sum() / max(1, targets.numel()))
            tav_errors = torch.cat([errors[m] for m in MODALITIES], dim=0)
            if self.detach_errors:
                tav_errors = tav_errors.detach()
            parts["cold_tav"] = symmetric_kl(
                tav_errors, torch.cat([scores[m] for m in MODALITIES], dim=0))
        parts["cold"] = sum(parts.values())
        parts["reg"] = sum(regularizers, zero)
        return parts, errors


class SDTCOLDLoss(nn.Module):
    def __init__(self, class_weights=None, gamma_1=1.0, gamma_2=1.0,
                 gamma_3=1.0, lambda_co=0.1, lambda_reg=1e-4,
                 detach_errors=True):
        super().__init__()
        weights = (gamma_1, gamma_2, gamma_3, lambda_co, lambda_reg)
        if any(not math.isfinite(w) or w < 0 for w in weights):
            raise ValueError("loss weights must be finite and nonnegative")
        self.gamma_1, self.gamma_2, self.gamma_3 = weights[:3]
        self.lambda_co, self.lambda_reg = weights[3:]
        self.ce = MaskedNLLLoss(class_weights)
        self.kl = MaskedKLDivLoss()
        self.cold = COLDLoss(detach_errors=detach_errors)

    def forward(self, outputs, labels, valid_mask):
        targets = labels.reshape(-1)
        n_classes = outputs["logits"].size(-1)
        task = self.ce(outputs["log_prob"].reshape(-1, n_classes), targets, valid_mask)
        students = sum(self.ce(outputs["student_log_prob"][m].reshape(-1, n_classes),
                               targets, valid_mask) for m in MODALITIES)
        # Preserve SDT source semantics: teacher is NOT detached and no T^2 factor.
        teacher = outputs["teacher_kl_prob"].reshape(-1, n_classes)
        kd = sum(self.kl(outputs["student_kl_log_prob"][m].reshape(-1, n_classes),
                         teacher, valid_mask) for m in MODALITIES)
        sdt = self.gamma_1 * task + self.gamma_2 * students + self.gamma_3 * kd
        cold_parts, errors = self.cold(outputs, labels, valid_mask)
        total = sdt + self.lambda_co * cold_parts["cold"] + self.lambda_reg * cold_parts["reg"]
        parts = {"total": total, "sdt": sdt, "task": task, "student_ce": students,
                 "distillation": kd, **cold_parts}
        return total, parts, errors
