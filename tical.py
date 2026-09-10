"""TiCAL-inspired hyperbolic consistency estimation for SDT.

The module owns only TiCAL-specific state.  Labels are used solely by the
post-optimizer anchor update performed by the training loop; forward/query is
label-free and therefore safe for validation and test inference.
"""

import math

import torch
import torch.nn as nn


MODALITIES = ("t", "a", "v")


class HyperbolicProjector(nn.Module):
    """Learned Euclidean projection followed by the exp-map at the ball origin."""

    def __init__(self, input_dim, output_dim, eps=1e-5):
        super().__init__()
        if input_dim < 1 or output_dim < 1:
            raise ValueError("projection dimensions must be positive")
        if not 0 < eps < 0.1:
            raise ValueError("hyp_eps must be in (0, 0.1)")
        self.linear = nn.Linear(input_dim, output_dim)
        # Default Linear initialization pushes high-dimensional vectors close
        # to the unit-ball boundary.  Start near the origin for stable distance.
        nn.init.xavier_uniform_(self.linear.weight, gain=0.01)
        nn.init.zeros_(self.linear.bias)
        self.eps = eps

    def forward(self, features):
        tangent = self.linear(features)
        norm = tangent.norm(p=2, dim=-1, keepdim=True).clamp_min(self.eps)
        projected = torch.tanh(norm) * tangent / norm
        projected_norm = projected.norm(p=2, dim=-1, keepdim=True)
        scale = ((1.0 - self.eps) / projected_norm.clamp_min(self.eps)).clamp(max=1.0)
        projected = projected * scale
        if not torch.isfinite(projected).all():
            raise FloatingPointError("nonfinite Poincare projection")
        return projected


def poincare_distance(first, second, eps=1e-5):
    """Poincare-ball distance with broadcasting over all leading dimensions."""
    if first.size(-1) != second.size(-1):
        raise ValueError("Poincare vectors must have the same final dimension")
    first_sq = (first * first).sum(dim=-1).clamp(max=1.0 - eps)
    second_sq = (second * second).sum(dim=-1).clamp(max=1.0 - eps)
    difference_sq = ((first - second) ** 2).sum(dim=-1)
    denominator = ((1.0 - first_sq) * (1.0 - second_sq)).clamp_min(eps)
    argument = (1.0 + 2.0 * difference_sq / denominator).clamp_min(1.0 + eps)
    distance = torch.acosh(argument)
    if not torch.isfinite(distance).all():
        raise FloatingPointError("nonfinite hyperbolic distance")
    return distance


class AnchorBank(nn.Module):
    """A bounded FIFO bank of detached hyperbolic features and class labels."""

    def __init__(self, feature_dim, n_classes, max_size, eps=1e-5,
                 query_chunk_size=256):
        super().__init__()
        if max_size < 1 or query_chunk_size < 1:
            raise ValueError("anchor size and query chunk size must be positive")
        self.feature_dim = feature_dim
        self.n_classes = n_classes
        self.max_size = max_size
        self.eps = eps
        self.query_chunk_size = query_chunk_size
        self.register_buffer("features", torch.empty(0, feature_dim))
        self.register_buffer("labels", torch.empty(0, dtype=torch.long))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Dynamic FIFO buffers have checkpoint-dependent first dimensions.
        feature_key, label_key = prefix + "features", prefix + "labels"
        if feature_key in state_dict:
            self.features = torch.empty_like(state_dict[feature_key])
        if label_key in state_dict:
            self.labels = torch.empty_like(state_dict[label_key])
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    @property
    def size(self):
        return int(self.labels.numel())

    @torch.no_grad()
    def update(self, features, labels):
        features = features.detach()
        labels = labels.detach().long()
        if features.ndim != 2 or features.size(-1) != self.feature_dim:
            raise ValueError("anchor features have the wrong shape")
        if labels.shape != (features.size(0),):
            raise ValueError("anchor labels have the wrong shape")
        if not features.numel():
            return
        if not torch.isfinite(features).all():
            raise ValueError("anchor features must be finite")
        if (labels < 0).any() or (labels >= self.n_classes).any():
            raise ValueError("anchor labels are outside the class range")
        self.features = torch.cat((self.features, features), dim=0)[-self.max_size:]
        self.labels = torch.cat((self.labels, labels), dim=0)[-self.max_size:]

    def query(self, features):
        if self.size == 0:
            raise RuntimeError("cannot query an empty anchor bank")
        if features.ndim != 2 or features.size(-1) != self.feature_dim:
            raise ValueError("query features have the wrong shape")
        distances, nearest_labels = [], []
        for chunk in features.split(self.query_chunk_size, dim=0):
            pairwise = poincare_distance(
                chunk.unsqueeze(1), self.features.unsqueeze(0), self.eps)
            nearest_distance, nearest_index = pairwise.min(dim=1)
            distances.append(nearest_distance)
            nearest_labels.append(self.labels[nearest_index])
        return torch.cat(distances), torch.cat(nearest_labels)

    def class_counts(self):
        return torch.bincount(self.labels.detach().cpu(), minlength=self.n_classes)


def compute_typicality(distances, eps=1e-8):
    if distances.ndim != 1:
        raise ValueError("typicality expects a 1-D distance vector")
    if distances.numel() == 0:
        return distances
    minimum, maximum = distances.min(), distances.max()
    typicality = (maximum - distances) / (maximum - minimum + eps)
    typicality = typicality.clamp(0.0, 1.0)
    if not torch.isfinite(typicality).all():
        raise FloatingPointError("nonfinite typicality")
    return typicality


def categorical_label_discrepancy(pseudo_t, pseudo_a, pseudo_v):
    return ((pseudo_t != pseudo_a).float()
            + (pseudo_t != pseudo_v).float()
            + (pseudo_a != pseudo_v).float()) / 3.0


def compute_consistency(tau_t, tau_a, tau_v, label_discrepancy,
                        exponent=0.2, label_scale=0.5, eps=1e-8):
    if exponent < 0 or label_scale < 0:
        raise ValueError("consistency exponent and label scale must be nonnegative")
    typicality = (tau_t.clamp_min(eps) * tau_a.clamp_min(eps)
                  * tau_v.clamp_min(eps)).pow(exponent)
    result = (typicality * torch.exp(-label_scale * label_discrepancy)).clamp(0.0, 1.0)
    if not torch.isfinite(result).all():
        raise FloatingPointError("nonfinite consistency")
    return result


def hyp_cpcc_loss(features, pseudo_labels, eps=1e-8, hyp_eps=1e-5):
    """Minimize 1 - Pearson correlation of geometry and categorical distance."""
    if features.ndim != 2 or pseudo_labels.ndim != 1:
        raise ValueError("HypCPCC expects [N,D] features and [N] labels")
    if features.size(0) < 2:
        return features.sum() * 0.0
    row, column = torch.triu_indices(
        features.size(0), features.size(0), offset=1, device=features.device)
    geometric = poincare_distance(features[row], features[column], hyp_eps)
    categorical = (pseudo_labels[row] != pseudo_labels[column]).type_as(geometric)
    geometric = geometric - geometric.mean()
    categorical = categorical - categorical.mean()
    denominator = geometric.square().sum().sqrt() * categorical.square().sum().sqrt()
    if denominator.detach().item() <= eps:
        return geometric.sum() * 0.0
    correlation = (geometric * categorical).sum() / denominator.clamp_min(eps)
    return 1.0 - correlation.clamp(-1.0, 1.0)


class TiCALModule(nn.Module):
    """Project pure modalities, query three HASLs, and estimate consistency."""

    def __init__(self, hidden_dim, hyperbolic_dim, n_classes, anchor_size=2048,
                 anchor_conf_threshold=0.8, hyp_eps=1e-5,
                 typicality_eps=1e-8, consistency_t=0.2,
                 consistency_k=0.5, detach_tau=True, detach_kappa=True):
        super().__init__()
        values = (anchor_conf_threshold, hyp_eps, typicality_eps,
                  consistency_t, consistency_k)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("TiCAL scalar settings must be finite")
        if not 0 <= anchor_conf_threshold <= 1:
            raise ValueError("anchor confidence threshold must be in [0,1]")
        if hyp_eps <= 0 or typicality_eps <= 0:
            raise ValueError("TiCAL eps values must be positive")
        if consistency_t < 0 or consistency_k < 0:
            raise ValueError("TiCAL consistency settings must be nonnegative")
        self.anchor_conf_threshold = anchor_conf_threshold
        self.hyp_eps = hyp_eps
        self.typicality_eps = typicality_eps
        self.consistency_t = consistency_t
        self.consistency_k = consistency_k
        self.detach_tau = detach_tau
        self.detach_kappa = detach_kappa
        self.projectors = nn.ModuleDict({
            name: HyperbolicProjector(hidden_dim, hyperbolic_dim, hyp_eps)
            for name in MODALITIES
        })
        self.anchor_banks = nn.ModuleDict({
            name: AnchorBank(hyperbolic_dim, n_classes, anchor_size, hyp_eps)
            for name in MODALITIES
        })

    def banks_ready(self):
        return all(self.anchor_banks[name].size > 0 for name in MODALITIES)

    def forward(self, pure_features, valid_mask, query_enabled=True):
        projected = {
            name: self.projectors[name](feature)
            for name, feature in zip(MODALITIES, pure_features)
        }
        result = {"ready": False, "projected": projected, "distance": None,
                  "pseudo_labels": None, "tau": None, "label_discrepancy": None,
                  "kappa": None, "hyp_eps": self.hyp_eps}
        valid = valid_mask.bool()
        if not query_enabled or not self.banks_ready() or not valid.any():
            return result

        distance, pseudo, tau = {}, {}, {}
        for name in MODALITIES:
            distance[name], pseudo[name] = self.anchor_banks[name].query(projected[name][valid])
            tau[name] = compute_typicality(distance[name], self.typicality_eps)
            if self.detach_tau:
                tau[name] = tau[name].detach()
            assert ((tau[name] >= 0) & (tau[name] <= 1)).all()
        discrepancy = categorical_label_discrepancy(
            pseudo["t"], pseudo["a"], pseudo["v"])
        kappa = compute_consistency(
            tau["t"], tau["a"], tau["v"], discrepancy,
            self.consistency_t, self.consistency_k, self.typicality_eps)
        if self.detach_kappa:
            kappa = kappa.detach()
        assert ((kappa >= 0) & (kappa <= 1)).all()
        assert torch.isfinite(kappa).all()
        result.update({"ready": True, "distance": distance,
                       "pseudo_labels": pseudo, "tau": tau,
                       "label_discrepancy": discrepancy, "kappa": kappa})
        return result

    @torch.no_grad()
    def update(self, projected, teacher_logits, labels, valid_mask):
        if not self.training:
            raise RuntimeError("TiCAL anchor banks are frozen during validation/test")
        probability = teacher_logits.softmax(dim=-1)
        confidence, prediction = probability.max(dim=-1)
        eligible = (valid_mask.bool() & prediction.eq(labels)
                    & confidence.gt(self.anchor_conf_threshold))
        for name in MODALITIES:
            self.anchor_banks[name].update(projected[name][eligible], labels[eligible])
        return int(eligible.sum().item())

    def summary(self):
        return {
            name: {"size": self.anchor_banks[name].size,
                   "class_counts": self.anchor_banks[name].class_counts().tolist()}
            for name in MODALITIES
        }
