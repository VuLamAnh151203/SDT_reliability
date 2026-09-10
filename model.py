"""SDT with distribution heads and the requested T/A/V COLD fusion."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from sdt_backbone import SDTBackbone


MODALITIES = ("t", "a", "v")
FUSION_VARIANTS = ("guided", "replace", "oof-guided", "sdt")
DISTRIBUTION_INIT_MODES = ("random", "sdt-preserving")


class DistributionHead(nn.Module):
    def __init__(self, hidden_dim, logvar_min=-8.0, logvar_max=8.0,
                 init_mode="random", initial_logvar=-6.0):
        super().__init__()
        if not (-20.0 <= logvar_min < logvar_max <= 20.0):
            raise ValueError("require -20 <= logvar_min < logvar_max <= 20")
        if init_mode not in DISTRIBUTION_INIT_MODES:
            raise ValueError("unknown distribution init mode: {}".format(init_mode))
        if not math.isfinite(initial_logvar):
            raise ValueError("initial_logvar must be finite")
        if init_mode == "sdt-preserving" and not logvar_min <= initial_logvar <= logvar_max:
            raise ValueError("initial_logvar must be inside the logvar clamp range")
        self.mu = nn.Linear(hidden_dim, hidden_dim)
        self.logvar = nn.Linear(hidden_dim, hidden_dim)
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        self.init_mode = init_mode
        self.initial_logvar = initial_logvar
        if init_mode == "sdt-preserving":
            # At initialization: mu == H' and sigma == exp(initial_logvar / 2).
            nn.init.eye_(self.mu.weight)
            nn.init.zeros_(self.mu.bias)
            nn.init.zeros_(self.logvar.weight)
            nn.init.constant_(self.logvar.bias, initial_logvar)

    def forward(self, features):
        mu = self.mu(features)
        logvar = self.logvar(features).clamp(self.logvar_min, self.logvar_max)
        variance = logvar.exp()
        latent = mu + (0.5 * logvar).exp() * torch.randn_like(mu) if self.training else mu
        return {"mu": mu, "logvar": logvar, "variance": variance, "z": latent}


def cold_fusion(features, latents, reliability, gate, variant):
    """Retain SDT's per-feature gate; r is a scalar per utterance/modality.

    features/latents: [B,L,3,D]; reliability: [B,L,3].
    The learned gate reads H', while the fused values are z.
    """
    if variant == "replace":
        weights = reliability.unsqueeze(-1).expand_as(latents)
        sdt_weights = None
    elif variant == "guided":
        gate_logits = gate.fc(features)
        sdt_weights = torch.softmax(gate_logits, dim=-2)
        # Equivalent to normalize(g_sdt * r), computed stably in log space.
        weights = torch.softmax(gate_logits + reliability.log().unsqueeze(-1), dim=-2)
    else:
        raise ValueError("COLD fusion requires 'guided' or 'replace'")
    return (weights * latents).sum(dim=-2), weights, sdt_weights


class Transformer_Based_Model(SDTBackbone):
    def __init__(self, dataset, temp, D_text, D_visual, D_audio, n_head,
                 n_classes, hidden_dim, n_speakers, dropout,
                 fusion_variant="guided", logvar_min=-8.0, logvar_max=8.0,
                 cold_eps=1e-8, distribution_init="random",
                 initial_logvar=-6.0):
        if fusion_variant not in FUSION_VARIANTS:
            raise ValueError("unknown fusion_variant: {}".format(fusion_variant))
        if not math.isfinite(temp) or temp <= 0:
            raise ValueError("temperature must be finite and positive")
        if not math.isfinite(cold_eps) or cold_eps <= 0:
            raise ValueError("cold_eps must be finite and positive")
        if hidden_dim < 2 or hidden_dim % 2 or n_head < 1 or hidden_dim % n_head:
            raise ValueError("hidden_dim must be positive, even, and divisible by n_head")
        super().__init__(dataset, temp, D_text, D_visual, D_audio, n_head,
                         n_classes, hidden_dim, n_speakers, dropout)
        self.fusion_variant = fusion_variant
        self.cold_eps = cold_eps
        self.distribution_init = distribution_init
        self.initial_logvar = initial_logvar
        if fusion_variant in ("guided", "replace"):
            self.distribution_heads = nn.ModuleDict({
                name: DistributionHead(
                    hidden_dim, logvar_min, logvar_max,
                    distribution_init, initial_logvar)
                for name in MODALITIES
            })
        elif fusion_variant == "oof-guided":
            self.reliability_heads = nn.ModuleDict({
                name: nn.Linear(hidden_dim, 1) for name in MODALITIES
            })
            # Uniform predicted reliability at initialization makes guided
            # fusion exactly equal to SDT's learned gate.
            for head in self.reliability_heads.values():
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
        if fusion_variant == "replace":
            self.last_gate.requires_grad_(False)

    def forward(self, textf, visuf, acouf, u_mask, qmask, dia_len):
        """Label-free forward. qmask is [B,L,S]; inputs are [L,B,D_m]."""
        enhanced = self.encode_modalities(textf, visuf, acouf, u_mask, qmask, dia_len)
        features = torch.stack(enhanced, dim=-2)
        distributions = {}
        variance_norm = confidence = reliability_logits = reliability = None
        if self.fusion_variant == "sdt":
            latents = features
            sdt_weights = torch.softmax(self.last_gate.fc(features), dim=-2)
            fusion_weights = sdt_weights
            fused = (fusion_weights * features).sum(dim=-2)
        elif self.fusion_variant == "oof-guided":
            # No Gaussian bottleneck and no sampling: classifiers and fusion
            # retain SDT's enhanced representations H'.
            latents = features
            reliability_logits = torch.stack([
                self.reliability_heads[name](feature).squeeze(-1)
                for name, feature in zip(MODALITIES, enhanced)
            ], dim=-1)
            reliability = torch.softmax(reliability_logits, dim=-1)
            confidence = reliability
            fused, fusion_weights, sdt_weights = cold_fusion(
                features, latents, reliability, self.last_gate, "guided")
        else:
            distributions = {
                name: self.distribution_heads[name](feature)
                for name, feature in zip(MODALITIES, enhanced)
            }
            latents = torch.stack([distributions[m]["z"] for m in MODALITIES], dim=-2)
            # Keep these scalar scores in float32 for stable norms/reciprocals.
            variance_norm = torch.stack([
                distributions[m]["variance"].float().norm(p=2, dim=-1)
                for m in MODALITIES
            ], dim=-1)
            # A large variance norm means high uncertainty and therefore low
            # confidence/reliability. reliability_logits is used by COLD so
            # softmax produces probabilities proportional to inverse variance.
            confidence = (variance_norm + self.cold_eps).reciprocal()
            reliability_logits = -(variance_norm + self.cold_eps).log()
            reliability = confidence / confidence.sum(dim=-1, keepdim=True)
            fused, fusion_weights, sdt_weights = cold_fusion(
                features, latents, reliability, self.last_gate, self.fusion_variant)

        student_logits = {
            name: classifier(latents[:, :, index, :])
            for index, (name, classifier) in enumerate(zip(MODALITIES, (
                self.t_output_layer, self.a_output_layer, self.v_output_layer)))
        }
        logits = self.all_output_layer(fused)
        return {
            "logits": logits,
            "log_prob": F.log_softmax(logits, dim=-1),
            "prob": F.softmax(logits, dim=-1),
            "student_logits": student_logits,
            "student_log_prob": {m: F.log_softmax(student_logits[m], dim=-1) for m in MODALITIES},
            "student_kl_log_prob": {m: F.log_softmax(student_logits[m] / self.temp, dim=-1) for m in MODALITIES},
            "teacher_kl_prob": F.softmax(logits / self.temp, dim=-1),
            "enhanced": features,
            "distributions": distributions,
            "latents": latents,
            "variance_norm": variance_norm,
            "confidence": confidence,
            "reliability_logits": reliability_logits,
            # Backward-compatible alias for checkpoints/analysis written by
            # the first implementation. New code should use confidence.
            "error_scores": confidence,
            "reliability": reliability,
            "sdt_weights": sdt_weights,
            "fusion_weights": fusion_weights,
            "fused": fused,
        }
