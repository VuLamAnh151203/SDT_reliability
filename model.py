"""SDT with optional COLD variants or an isolated TiCAL integration."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from sdt_backbone import SDTBackbone
from tical import MODALITIES as TICAL_MODALITIES, TiCALModule


MODALITIES = ("t", "a", "v")
FUSION_VARIANTS = ("guided", "replace", "oof-guided", "sdt")
FUSION_FEATURE_SOURCES = ("enhanced", "pure")
WHEEL_FEATURE_SOURCES = ("pure", "enhanced")
DISTRIBUTION_INIT_MODES = ("random", "sdt-preserving")
WHEEL_UNCERTAINTY_MODES = ("none", "gate", "supervise", "full")
PROTOTYPE_EVIDENCE_FUSION_MODES = (
    "none", "uniform", "class-wise", "calibrated")


def poincare_log_map_zero(points, eps=1e-5):
    """Map unit-ball points to the tangent space at the origin."""
    norm = points.norm(p=2, dim=-1, keepdim=True)
    safe_norm = norm.clamp_min(eps)
    radius = norm.clamp(max=1.0 - eps)
    tangent = torch.atanh(radius) * points / safe_norm
    return torch.where(norm > eps, tangent, torch.zeros_like(tangent))


class WheelUncertaintyHead(nn.Module):
    """Predict a scalar wheel-error proxy in [0, 1] for one modality."""

    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        if input_dim < 1 or hidden_dim < 1:
            raise ValueError("uncertainty head dimensions must be positive")
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        # sigmoid(0)=0.5 for all modalities, hence uniform fusion weights and
        # an exact SDT fusion baseline before the warm-up finishes.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, tangent_features):
        return torch.sigmoid(self.network(tangent_features).squeeze(-1))


class PrototypeEvidenceGate(nn.Module):
    """Predict utterance- and class-specific modality gate scores."""

    def __init__(self, input_dim, hidden_dim, n_classes, dropout):
        super().__init__()
        if min(input_dim, hidden_dim, n_classes) < 1:
            raise ValueError("prototype evidence gate dimensions must be positive")
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )
        # Every modality starts with score zero for every class. Softmax over
        # modalities is therefore exactly uniform before the gate learns.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, projected_features):
        return self.network(projected_features)


class PrototypeEvidenceSelector(nn.Module):
    """Predict whether prototype evidence should modify one SDT prediction."""

    def __init__(self, input_dim, hidden_dim, dropout, initial_alpha):
        super().__init__()
        if min(input_dim, hidden_dim) < 1:
            raise ValueError("prototype evidence selector dimensions must be positive")
        if not 0.0 < initial_alpha < 1.0:
            raise ValueError("initial prototype evidence alpha must be in (0, 1)")
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.constant_(
            self.network[-1].bias,
            math.log(initial_alpha / (1.0 - initial_alpha)))

    def forward(self, reliability_features):
        return torch.sigmoid(
            self.network(reliability_features).squeeze(-1))


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
                 initial_logvar=-6.0, use_tical=False, tical_mode="kd",
                 tical_warmup_epochs=5, anchor_size=2048,
                 anchor_conf_threshold=0.8, hyperbolic_dim=128,
                 hyp_eps=1e-5, typicality_eps=1e-8,
                 consistency_t=0.2, consistency_k=0.5,
                 detach_tau=True, detach_kappa=True, beta_gate=1.0,
                 use_emotion_wheel=False, wheel_prototype_radius=0.75,
                 wheel_temperature=1.0, wheel_anchor_mix=0.5,
                 anchor_balance="none", anchor_min_per_class=0,
                 anchor_admission="teacher", wheel_geometry="poincare",
                 wheel_radius_mode="free", wheel_fixed_radius=0.75,
                 wheel_zero_residual=False,
                 wheel_uncertainty_mode="none",
                 wheel_uncertainty_hidden_dim=32,
                 wheel_uncertainty_temperature=1.0,
                 wheel_uncertainty_gate_strength=0.5,
                 wheel_uncertainty_warmup_epochs=10,
                 wheel_uncertainty_ramp_epochs=5,
                 fusion_feature_source="enhanced",
                 wheel_feature_source="pure",
                 wheel_semantic_control="correct",
                 wheel_permutation_seed=1729,
                 wheel_detach_features=False,
                  wheel_prototype_layout="fixed-wheel",
                 wheel_prototype_radius_mode="fixed-shared",
                 wheel_prototype_radius_min=0.20,
                  wheel_prototype_radius_max=0.90,
                  wheel_free_prototype_max_radius=0.95,
                 prototype_evidence_fusion="none",
                 prototype_evidence_weight=0.0,
                 prototype_evidence_hidden_dim=32,
                 prototype_evidence_warmup_epochs=5,
                 prototype_evidence_ramp_epochs=5,
                 prototype_evidence_initial_alpha=0.05,
                 prototype_evidence_entropy_strength=1.0,
                 prototype_evidence_gate_temperature=1.0,
                 wheel_prototypes_per_class=1,
                 wheel_multi_prototype_temperature=0.2,
                 wheel_multi_prototype_init_std=0.02,
                 wheel_prototype_diversity_margin=0.5,
                 prototype_loss_warmup_epochs=0,
                 prototype_loss_ramp_epochs=0,
                 wheel_prototype_structure_mode="pairwise-mse",
                 wheel_ordinal_margin=0.2,
                 wheel_free_prototype_init="wheel",
                 wheel_private_residual_init_std=0.02,
                 use_data_prototypes=False,
                 data_prototype_warmup_epochs=5,
                 data_prototype_ramp_epochs=10,
                 data_prototype_temperature=1.0):
        if fusion_variant not in FUSION_VARIANTS:
            raise ValueError("unknown fusion_variant: {}".format(fusion_variant))
        if fusion_feature_source not in FUSION_FEATURE_SOURCES:
            raise ValueError("unknown fusion feature source: {}".format(
                fusion_feature_source))
        if fusion_feature_source == "pure" and fusion_variant != "sdt":
            raise ValueError("pure fusion requires --fusion-variant sdt")
        if wheel_feature_source not in WHEEL_FEATURE_SOURCES:
            raise ValueError("unknown Wheel feature source: {}".format(
                wheel_feature_source))
        if wheel_feature_source != "pure" and not use_emotion_wheel:
            raise ValueError(
                "enhanced Wheel features require emotion-wheel mode")
        if not math.isfinite(temp) or temp <= 0:
            raise ValueError("temperature must be finite and positive")
        if not math.isfinite(cold_eps) or cold_eps <= 0:
            raise ValueError("cold_eps must be finite and positive")
        if hidden_dim < 2 or hidden_dim % 2 or n_head < 1 or hidden_dim % n_head:
            raise ValueError("hidden_dim must be positive, even, and divisible by n_head")
        if tical_mode not in ("observe", "kd", "hyp", "fusion", "full"):
            raise ValueError("unknown tical_mode: {}".format(tical_mode))
        if use_tical and fusion_variant != "sdt":
            raise ValueError("TiCAL must use --fusion-variant sdt; COLD cannot run with TiCAL")
        if use_emotion_wheel and not use_tical:
            raise ValueError("emotion wheel requires TiCAL")
        if wheel_uncertainty_mode not in WHEEL_UNCERTAINTY_MODES:
            raise ValueError("unknown wheel uncertainty mode: {}".format(
                wheel_uncertainty_mode))
        if prototype_evidence_fusion not in PROTOTYPE_EVIDENCE_FUSION_MODES:
            raise ValueError("unknown prototype evidence fusion mode: {}".format(
                prototype_evidence_fusion))
        if (not math.isfinite(prototype_evidence_weight)
                or prototype_evidence_weight < 0):
            raise ValueError("prototype evidence weight must be finite and nonnegative")
        if (prototype_evidence_hidden_dim < 1
                or prototype_evidence_warmup_epochs < 0
                or prototype_evidence_ramp_epochs < 0):
            raise ValueError("invalid prototype evidence fusion schedule")
        if not 0.0 < prototype_evidence_initial_alpha < 1.0:
            raise ValueError("prototype evidence initial alpha must be in (0,1)")
        if (not math.isfinite(prototype_evidence_entropy_strength)
                or prototype_evidence_entropy_strength < 0):
            raise ValueError(
                "prototype evidence entropy strength must be finite and nonnegative")
        if (not math.isfinite(prototype_evidence_gate_temperature)
                or prototype_evidence_gate_temperature <= 0):
            raise ValueError(
                "prototype evidence gate temperature must be finite and positive")
        if prototype_loss_warmup_epochs < 0 or prototype_loss_ramp_epochs < 0:
            raise ValueError("prototype loss schedule must be nonnegative")
        if prototype_evidence_fusion != "none":
            if not use_emotion_wheel:
                raise ValueError(
                    "prototype evidence fusion requires emotion-wheel prototypes")
            if prototype_evidence_weight <= 0:
                raise ValueError(
                    "prototype evidence fusion requires a positive weight")
        elif prototype_evidence_weight != 0:
            raise ValueError(
                "prototype evidence weight requires an enabled fusion mode")
        if wheel_uncertainty_mode != "none":
            if not use_emotion_wheel or wheel_geometry != "poincare":
                raise ValueError(
                    "wheel uncertainty requires Poincare emotion-wheel mode")
            if tical_mode != "observe":
                raise ValueError(
                    "wheel uncertainty is isolated with --tical-mode observe")
        if (not math.isfinite(wheel_uncertainty_temperature)
                or wheel_uncertainty_temperature <= 0):
            raise ValueError("wheel uncertainty temperature must be positive")
        if (not math.isfinite(wheel_uncertainty_gate_strength)
                or not 0.0 <= wheel_uncertainty_gate_strength <= 1.0):
            raise ValueError("wheel uncertainty gate strength must be in [0,1]")
        if (wheel_uncertainty_hidden_dim < 1
                or wheel_uncertainty_warmup_epochs < 0
                or wheel_uncertainty_ramp_epochs < 0):
            raise ValueError("invalid wheel uncertainty dimensions/schedule")
        if tical_warmup_epochs < 0 or not math.isfinite(beta_gate) or beta_gate < 0:
            raise ValueError("TiCAL warmup and beta_gate must be nonnegative")
        if TICAL_MODALITIES != MODALITIES:
            raise RuntimeError("TiCAL modality order differs from SDT")
        super().__init__(dataset, temp, D_text, D_visual, D_audio, n_head,
                         n_classes, hidden_dim, n_speakers, dropout)
        self.fusion_variant = fusion_variant
        self.fusion_feature_source = fusion_feature_source
        self.cold_eps = cold_eps
        self.distribution_init = distribution_init
        self.initial_logvar = initial_logvar
        self.use_tical = use_tical
        self.tical_mode = tical_mode
        self.tical_warmup_epochs = tical_warmup_epochs
        self.beta_gate = beta_gate
        self.use_emotion_wheel = bool(use_emotion_wheel)
        self.wheel_feature_source = wheel_feature_source
        self.wheel_geometry = wheel_geometry
        self.wheel_radius_mode = wheel_radius_mode
        self.wheel_fixed_radius = wheel_fixed_radius
        self.wheel_zero_residual = bool(wheel_zero_residual)
        self.wheel_semantic_control = wheel_semantic_control
        self.wheel_permutation_seed = int(wheel_permutation_seed)
        self.wheel_detach_features = bool(wheel_detach_features)
        self.wheel_prototype_layout = wheel_prototype_layout
        self.wheel_prototype_radius_mode = wheel_prototype_radius_mode
        self.wheel_uncertainty_mode = wheel_uncertainty_mode
        self.wheel_uncertainty_temperature = float(
            wheel_uncertainty_temperature)
        self.wheel_uncertainty_gate_strength = float(
            wheel_uncertainty_gate_strength)
        self.wheel_uncertainty_warmup_epochs = int(
            wheel_uncertainty_warmup_epochs)
        self.wheel_uncertainty_ramp_epochs = int(
            wheel_uncertainty_ramp_epochs)
        self.prototype_evidence_fusion = prototype_evidence_fusion
        self.prototype_evidence_weight = float(prototype_evidence_weight)
        self.prototype_evidence_warmup_epochs = int(
            prototype_evidence_warmup_epochs)
        self.prototype_evidence_ramp_epochs = int(
            prototype_evidence_ramp_epochs)
        self.prototype_evidence_initial_alpha = float(
            prototype_evidence_initial_alpha)
        self.prototype_evidence_entropy_strength = float(
            prototype_evidence_entropy_strength)
        self.prototype_evidence_gate_temperature = float(
            prototype_evidence_gate_temperature)
        self.prototype_loss_warmup_epochs = int(
            prototype_loss_warmup_epochs)
        self.prototype_loss_ramp_epochs = int(prototype_loss_ramp_epochs)
        self.use_data_prototypes = bool(use_data_prototypes)
        self.tical_epoch = 0
        if use_tical:
            self.tical = TiCALModule(
                hidden_dim, hyperbolic_dim, n_classes, anchor_size,
                anchor_conf_threshold, hyp_eps, typicality_eps,
                consistency_t, consistency_k, detach_tau, detach_kappa,
                dataset, use_emotion_wheel, wheel_prototype_radius,
                wheel_temperature, wheel_anchor_mix, anchor_balance,
                anchor_min_per_class, anchor_admission, wheel_geometry,
                wheel_radius_mode, wheel_fixed_radius,
                wheel_zero_residual, wheel_semantic_control,
                wheel_permutation_seed, wheel_detach_features,
                 wheel_prototype_layout,
                wheel_prototype_radius_mode,
                wheel_prototype_radius_min,
                 wheel_prototype_radius_max,
                 wheel_free_prototype_max_radius,
                 wheel_prototypes_per_class,
                 wheel_multi_prototype_temperature,
                 wheel_multi_prototype_init_std,
                 wheel_prototype_diversity_margin,
                 wheel_prototype_structure_mode,
                 wheel_ordinal_margin,
                 wheel_free_prototype_init,
                 wheel_private_residual_init_std,
                 use_data_prototypes,
                 data_prototype_warmup_epochs,
                 data_prototype_ramp_epochs,
                 data_prototype_temperature)
        if wheel_uncertainty_mode != "none":
            self.wheel_uncertainty_heads = nn.ModuleDict({
                name: WheelUncertaintyHead(
                    hyperbolic_dim, wheel_uncertainty_hidden_dim, dropout)
                for name in MODALITIES
            })
        if prototype_evidence_fusion == "class-wise":
            self.prototype_evidence_gates = nn.ModuleDict({
                name: PrototypeEvidenceGate(
                    hyperbolic_dim, prototype_evidence_hidden_dim,
                    n_classes, dropout)
                for name in MODALITIES
            })
        elif prototype_evidence_fusion == "calibrated":
            # Positive modality temperatures calibrate the three hyperbolic
            # distance classifiers before their evidence is compared.
            self.prototype_evidence_log_temperatures = nn.Parameter(
                torch.zeros(len(MODALITIES)))
            # Only C x M parameters encode stable class/modality preferences.
            self.prototype_evidence_class_bias = nn.Parameter(
                torch.zeros(n_classes, len(MODALITIES)))
            # Features: entropy and maximum probability for every modality,
            # followed by the same two confidence statistics from SDT.
            self.prototype_evidence_selector = PrototypeEvidenceSelector(
                2 * len(MODALITIES) + 2,
                prototype_evidence_hidden_dim, dropout,
                prototype_evidence_initial_alpha)
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
        if (self.use_tical or self.fusion_feature_source == "pure"
                or getattr(self, "capture_pure", False)):
            enhanced, pure = self.encode_modalities(
                textf, visuf, acouf, u_mask, qmask, dia_len, return_pure=True)
        else:
            enhanced = self.encode_modalities(textf, visuf, acouf, u_mask, qmask, dia_len)
            pure = None
        features = torch.stack(enhanced, dim=-2)
        pure_features = (torch.stack(pure, dim=-2)
                         if pure is not None else None)
        distributions = {}
        variance_norm = confidence = reliability_logits = reliability = None
        tical_output = None
        wheel_uncertainty = None
        wheel_uncertainty_weights = None
        wheel_uncertainty_scales = None
        wheel_uncertainty_progress = self.wheel_uncertainty_progress()
        prototype_evidence_modality_logits = None
        prototype_evidence_weights = None
        prototype_evidence_logits = None
        prototype_evidence_alpha = None
        prototype_evidence_temperatures = None
        prototype_evidence_modality_entropy = None
        prototype_evidence_progress = self.prototype_evidence_progress()
        prototype_loss_progress = self.prototype_loss_progress()
        if self.fusion_variant == "sdt":
            fusion_features = (pure_features
                               if self.fusion_feature_source == "pure"
                               else features)
            latents = fusion_features
            sdt_weights = torch.softmax(
                self.last_gate.fc(fusion_features), dim=-2)
            fusion_weights = sdt_weights
            if self.use_tical:
                # TiCAL originally regularized the pure self-modal features
                # H_TT/H_AA/H_VV.  The enhanced option attaches the same
                # projector and objectives to H'_T/H'_A/H'_V while leaving
                # the ordinary SDT fusion path unchanged.
                tical_features = (pure if self.wheel_feature_source == "pure"
                                  else enhanced)
                tical_output = self.tical(
                    tical_features, u_mask.bool(),
                    query_enabled=self.tical_epoch > self.tical_warmup_epochs)
                tical_output["feature_source"] = self.wheel_feature_source
                tical_output["prototype_loss_progress"] = (
                    prototype_loss_progress)
                if self.wheel_uncertainty_mode != "none":
                    wheel_uncertainty = {
                        name: self.wheel_uncertainty_heads[name](
                            poincare_log_map_zero(
                                tical_output["projected"][name],
                                tical_output["hyp_eps"]))
                        for name in MODALITIES
                    }
                    uncertainty_stack = torch.stack(
                        [wheel_uncertainty[name] for name in MODALITIES],
                        dim=-1)
                    wheel_uncertainty_weights = torch.softmax(
                        -uncertainty_stack
                        / self.wheel_uncertainty_temperature,
                        dim=-1)
                if self.tical_mode in ("fusion", "full") and tical_output["ready"]:
                    valid = u_mask.bool()
                    tau_full = fusion_features.new_ones(
                        *u_mask.shape, len(MODALITIES))
                    kappa_full = fusion_features.new_ones(*u_mask.shape)
                    tau_full[valid] = torch.stack(
                        [tical_output["tau"][name] for name in MODALITIES], dim=-1)
                    kappa_full[valid] = tical_output["kappa"]
                    strength = self.beta_gate * (1.0 - kappa_full)
                    factor = (tau_full + self.tical.typicality_eps).pow(
                        strength.unsqueeze(-1)).unsqueeze(-1)
                    fusion_weights = sdt_weights * factor
                    fusion_weights = fusion_weights / fusion_weights.sum(
                        dim=-2, keepdim=True).clamp_min(self.tical.typicality_eps)
            if (self.wheel_uncertainty_mode in ("gate", "full")
                    and wheel_uncertainty_weights is not None
                    and wheel_uncertainty_progress > 0.0):
                gate_strength = (self.wheel_uncertainty_gate_strength
                                 * wheel_uncertainty_progress)
                wheel_uncertainty_scales = (
                    (1.0 - gate_strength)
                    + gate_strength * len(MODALITIES)
                    * wheel_uncertainty_weights)
                fusion_weights = (
                    fusion_weights
                    * wheel_uncertainty_scales.unsqueeze(-1))
                fusion_weights = fusion_weights / fusion_weights.sum(
                    dim=-2, keepdim=True).clamp_min(
                        self.tical.typicality_eps)
            fused = (fusion_weights * fusion_features).sum(dim=-2)
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
        sdt_logits = self.all_output_layer(fused)
        logits = sdt_logits
        if (self.prototype_evidence_fusion != "none"
                and tical_output is not None
                and tical_output.get("wheel_logits") is not None):
            valid = u_mask.bool()
            raw_valid_modality_logits = torch.stack([
                tical_output["wheel_logits"][name]
                for name in MODALITIES
            ], dim=-1)
            prototype_evidence_modality_logits = sdt_logits.new_zeros(
                *sdt_logits.shape, len(MODALITIES))
            if self.prototype_evidence_fusion == "calibrated":
                prototype_evidence_temperatures = (
                    self.prototype_evidence_log_temperatures.exp()
                    .clamp(0.05, 20.0))
                calibrated_valid_logits = (
                    raw_valid_modality_logits
                    / prototype_evidence_temperatures.view(1, 1, -1))
                valid_modality_logits = F.log_softmax(
                    calibrated_valid_logits, dim=1)
                valid_probabilities = valid_modality_logits.exp()
                valid_entropy = -(
                    valid_probabilities * valid_modality_logits
                ).sum(dim=1) / math.log(self.n_classes)
                prototype_evidence_modality_entropy = (
                    sdt_logits.new_zeros(
                        *sdt_logits.shape[:-1], len(MODALITIES)))
                prototype_evidence_modality_entropy[valid] = valid_entropy

                # Confidence statistics are detached so the selector cannot
                # make a modality artificially overconfident merely to raise
                # its own residual strength.
                detached_entropy = valid_entropy.detach()
                detached_maximum = valid_probabilities.max(
                    dim=1).values.detach()
                valid_sdt_probabilities = F.softmax(
                    sdt_logits[valid], dim=-1)
                valid_sdt_log_probabilities = valid_sdt_probabilities.clamp_min(
                    1e-12).log()
                valid_sdt_entropy = -(
                    valid_sdt_probabilities * valid_sdt_log_probabilities
                ).sum(dim=-1, keepdim=True) / math.log(self.n_classes)
                valid_sdt_maximum = valid_sdt_probabilities.max(
                    dim=-1, keepdim=True).values
                selector_features = torch.cat((
                    detached_entropy,
                    detached_maximum,
                    valid_sdt_entropy.detach(),
                    valid_sdt_maximum.detach(),
                ), dim=-1)
                valid_alpha = self.prototype_evidence_selector(
                    selector_features)
                prototype_evidence_alpha = sdt_logits.new_zeros(
                    *sdt_logits.shape[:-1])
                prototype_evidence_alpha[valid] = valid_alpha

                gate_scores = (
                    valid_modality_logits
                    - self.prototype_evidence_entropy_strength
                    * detached_entropy.unsqueeze(1)
                    + self.prototype_evidence_class_bias.unsqueeze(0))
                valid_weights = torch.softmax(
                    gate_scores
                    / self.prototype_evidence_gate_temperature,
                    dim=-1)
                prototype_evidence_weights = sdt_logits.new_full(
                    (*sdt_logits.shape, len(MODALITIES)),
                    1.0 / len(MODALITIES))
                prototype_evidence_weights[valid] = valid_weights
            else:
                valid_modality_logits = raw_valid_modality_logits
                prototype_evidence_alpha = sdt_logits.new_ones(
                    *sdt_logits.shape[:-1])
            prototype_evidence_modality_logits[valid] = valid_modality_logits
            if self.prototype_evidence_fusion == "uniform":
                prototype_evidence_weights = sdt_logits.new_full(
                    (*sdt_logits.shape, len(MODALITIES)),
                    1.0 / len(MODALITIES))
            elif self.prototype_evidence_fusion == "class-wise":
                gate_scores = torch.stack([
                    self.prototype_evidence_gates[name](
                        tical_output["projected"][name])
                    for name in MODALITIES
                ], dim=-1)
                prototype_evidence_weights = torch.softmax(
                    gate_scores, dim=-1)
            prototype_evidence_logits = (
                prototype_evidence_weights
                * prototype_evidence_modality_logits).sum(dim=-1)
            # Per-utterance shifts do not change a softmax, but centering keeps
            # the additive residual numerically comparable to ordinary logits.
            prototype_evidence_logits = (
                prototype_evidence_logits
                - prototype_evidence_logits.mean(dim=-1, keepdim=True))
            logits = (
                sdt_logits
                + self.prototype_evidence_weight
                * prototype_evidence_progress
                * prototype_evidence_alpha.unsqueeze(-1)
                * prototype_evidence_logits)
        return {
            "logits": logits,
            "sdt_logits": sdt_logits,
            "log_prob": F.log_softmax(logits, dim=-1),
            "prob": F.softmax(logits, dim=-1),
            "student_logits": student_logits,
            "student_log_prob": {m: F.log_softmax(student_logits[m], dim=-1) for m in MODALITIES},
            "student_kl_log_prob": {m: F.log_softmax(student_logits[m] / self.temp, dim=-1) for m in MODALITIES},
            "teacher_kl_prob": F.softmax(logits / self.temp, dim=-1),
            "enhanced": features,
            "pure": pure_features,
            # Keep the exact graph nodes consumed by both the SDT gates and
            # the Wheel projector.  Mechanism diagnostics use these tensors
            # to measure task/Wheel gradient alignment at H_TT/H_AA/H_VV.
            "pure_modalities": (
                {name: feature for name, feature in zip(MODALITIES, pure)}
                if pure is not None else None),
            "fusion_feature_source": self.fusion_feature_source,
            "wheel_feature_source": self.wheel_feature_source,
            "fusion_features": latents,
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
            "tical": tical_output,
            "wheel_uncertainty": wheel_uncertainty,
            "wheel_uncertainty_weights": wheel_uncertainty_weights,
            "wheel_uncertainty_scales": wheel_uncertainty_scales,
            "wheel_uncertainty_progress": wheel_uncertainty_progress,
            "prototype_evidence_fusion": self.prototype_evidence_fusion,
            "prototype_evidence_modality_logits": (
                prototype_evidence_modality_logits),
            "prototype_evidence_weights": prototype_evidence_weights,
            "prototype_evidence_logits": prototype_evidence_logits,
            "prototype_evidence_alpha": prototype_evidence_alpha,
            "prototype_evidence_temperatures": (
                prototype_evidence_temperatures),
            "prototype_evidence_modality_entropy": (
                prototype_evidence_modality_entropy),
            "prototype_evidence_progress": prototype_evidence_progress,
            "prototype_evidence_effective_weight": (
                self.prototype_evidence_weight
                * prototype_evidence_progress),
            "prototype_loss_progress": prototype_loss_progress,
        }

    def set_tical_epoch(self, epoch):
        self.tical_epoch = int(epoch)
        if self.use_tical:
            self.tical.set_epoch(epoch)

    def wheel_uncertainty_progress(self):
        if self.wheel_uncertainty_mode == "none":
            return 0.0
        elapsed = self.tical_epoch - self.wheel_uncertainty_warmup_epochs
        if elapsed <= 0:
            return 0.0
        if self.wheel_uncertainty_ramp_epochs == 0:
            return 1.0
        return min(1.0, elapsed / self.wheel_uncertainty_ramp_epochs)

    def prototype_loss_progress(self):
        if not self.use_emotion_wheel:
            return 0.0
        # Preserve the pre-schedule behavior exactly for every existing
        # configuration and unit test.
        if (self.prototype_loss_warmup_epochs == 0
                and self.prototype_loss_ramp_epochs == 0):
            return 1.0
        elapsed = self.tical_epoch - self.prototype_loss_warmup_epochs
        if elapsed <= 0:
            return 0.0
        if self.prototype_loss_ramp_epochs == 0:
            return 1.0
        return min(1.0, elapsed / self.prototype_loss_ramp_epochs)

    def prototype_evidence_progress(self):
        if self.prototype_evidence_fusion == "none":
            return 0.0
        elapsed = self.tical_epoch - self.prototype_evidence_warmup_epochs
        if elapsed <= 0:
            return 0.0
        if self.prototype_evidence_ramp_epochs == 0:
            return 1.0
        return min(1.0, elapsed / self.prototype_evidence_ramp_epochs)

    @torch.no_grad()
    def update_tical_anchors(self, outputs, labels, valid_mask):
        """Update after optimizer.step(); never call this during validation/test."""
        if not self.use_tical or outputs["tical"] is None:
            return 0
        if not self.training:
            raise RuntimeError("cannot update TiCAL anchors outside training")
        return self.tical.update(
            outputs["tical"]["projected"], outputs["logits"],
            outputs["student_logits"], labels, valid_mask)

    def tical_anchor_summary(self):
        return self.tical.summary() if self.use_tical else None
