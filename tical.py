"""TiCAL-inspired hyperbolic consistency estimation for SDT.

The module owns only TiCAL-specific state.  Labels are used solely by the
post-optimizer anchor update performed by the training loop; forward/query is
label-free and therefore safe for validation and test inference.
"""

import math

import torch
import torch.nn as nn


MODALITIES = ("t", "a", "v")
EMOTION_WHEEL_ORDERS = {
    "IEMOCAP": (0, 4, 3, 5, 1, 2),
    "MELD": (4, 1, 2, 6, 5, 3, 0),
}


def emotion_wheel_angles(dataset, n_classes, device=None, dtype=torch.float32):
    """Return class-indexed angles while preserving the semantic wheel order."""
    if dataset not in EMOTION_WHEEL_ORDERS:
        raise ValueError("emotion wheel is unavailable for {}".format(dataset))
    order = EMOTION_WHEEL_ORDERS[dataset]
    if len(order) != n_classes or sorted(order) != list(range(n_classes)):
        raise ValueError("emotion wheel order does not match the class count")
    ordered = torch.arange(n_classes, device=device, dtype=dtype)
    ordered = ordered * (2.0 * math.pi / n_classes)
    angles = torch.empty(n_classes, device=device, dtype=dtype)
    angles[torch.tensor(order, device=device, dtype=torch.long)] = ordered
    return angles


def circular_class_distance_matrix(class_angles):
    """Pairwise shortest angular distance normalized to [0, 1]."""
    if class_angles.ndim != 1 or class_angles.numel() < 2:
        raise ValueError("class_angles must be a 1-D tensor with at least 2 entries")
    difference = class_angles[:, None] - class_angles[None, :]
    cosine = torch.cos(difference).clamp(-1.0, 1.0)
    return torch.acos(cosine) / math.pi


def emotion_wheel_prototypes(class_angles, feature_dim, radius=0.75, eps=1e-5):
    """Embed fixed emotion-wheel prototypes in the first two ball dimensions."""
    if feature_dim < 2:
        raise ValueError("emotion-wheel prototypes require at least 2 dimensions")
    if not math.isfinite(radius) or not 0.0 < radius < 1.0 - eps:
        raise ValueError("prototype radius must be in (0, 1 - hyp_eps)")
    prototypes = class_angles.new_zeros(class_angles.numel(), feature_dim)
    prototypes[:, 0] = radius * torch.cos(class_angles)
    prototypes[:, 1] = radius * torch.sin(class_angles)
    return prototypes


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


def hyperbolic_prototype_outputs(features, prototypes, temperature=1.0,
                                 eps=1e-5):
    """Return class distances, logits, pseudo-labels, and absolute typicality."""
    if features.ndim != 2 or prototypes.ndim != 2:
        raise ValueError("features and prototypes must be two-dimensional")
    if features.size(-1) != prototypes.size(-1):
        raise ValueError("features and prototypes must share their final dimension")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("wheel temperature must be finite and positive")
    distances = poincare_distance(
        features.unsqueeze(1), prototypes.unsqueeze(0), eps)
    nearest_distance, pseudo_labels = distances.min(dim=-1)
    logits = -distances / temperature
    typicality = torch.exp(-nearest_distance / temperature).clamp(0.0, 1.0)
    return distances, logits, pseudo_labels, typicality


class AnchorBank(nn.Module):
    """A bounded FIFO bank of detached hyperbolic features and class labels."""

    def __init__(self, feature_dim, n_classes, max_size, eps=1e-5,
                 query_chunk_size=256, balance_mode="none"):
        super().__init__()
        if max_size < 1 or query_chunk_size < 1:
            raise ValueError("anchor size and query chunk size must be positive")
        if balance_mode not in ("none", "equal"):
            raise ValueError("anchor balance mode must be 'none' or 'equal'")
        if balance_mode == "equal" and max_size < n_classes:
            raise ValueError("equal anchor balance requires at least one slot per class")
        self.feature_dim = feature_dim
        self.n_classes = n_classes
        self.max_size = max_size
        self.eps = eps
        self.query_chunk_size = query_chunk_size
        self.balance_mode = balance_mode
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
        combined_features = torch.cat((self.features, features), dim=0)
        combined_labels = torch.cat((self.labels, labels), dim=0)
        if self.balance_mode == "none":
            self.features = combined_features[-self.max_size:]
            self.labels = combined_labels[-self.max_size:]
            return

        base, remainder = divmod(self.max_size, self.n_classes)
        retained_features, retained_labels = [], []
        for class_index in range(self.n_classes):
            capacity = base + int(class_index < remainder)
            class_indices = combined_labels.eq(class_index).nonzero(
                as_tuple=False).flatten()[-capacity:]
            if class_indices.numel():
                retained_features.append(combined_features[class_indices])
                retained_labels.append(combined_labels[class_indices])
        self.features = torch.cat(retained_features, dim=0)
        self.labels = torch.cat(retained_labels, dim=0)

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

    def has_class_coverage(self, minimum):
        """Return whether every class currently has at least ``minimum`` items."""
        if minimum < 0:
            raise ValueError("minimum class coverage must be nonnegative")
        if minimum == 0:
            return True
        if self.size < self.n_classes * minimum:
            return False
        counts = torch.bincount(self.labels, minlength=self.n_classes)
        return bool(counts.ge(minimum).all().item())


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


def blend_typicality(anchor, prototype, prototype_weight=0.5, eps=1e-8):
    """Geometrically blend data-driven and fixed-prototype typicality."""
    if anchor.shape != prototype.shape:
        raise ValueError("anchor and prototype typicality must have equal shapes")
    if not math.isfinite(prototype_weight) or not 0.0 <= prototype_weight <= 1.0:
        raise ValueError("prototype typicality weight must be in [0, 1]")
    if prototype_weight == 0.0:
        return anchor
    if prototype_weight == 1.0:
        return prototype
    return (anchor.clamp_min(eps).pow(1.0 - prototype_weight)
            * prototype.clamp_min(eps).pow(prototype_weight)).clamp(0.0, 1.0)


def categorical_label_discrepancy(pseudo_t, pseudo_a, pseudo_v):
    return ((pseudo_t != pseudo_a).float()
            + (pseudo_t != pseudo_v).float()
            + (pseudo_a != pseudo_v).float()) / 3.0


def wheel_label_discrepancy(pseudo_t, pseudo_a, pseudo_v,
                            class_distance_matrix):
    """Average semantic wheel distance across the three modality pairs."""
    if class_distance_matrix.ndim != 2 or (
            class_distance_matrix.size(0) != class_distance_matrix.size(1)):
        raise ValueError("class distance matrix must be square")
    return (class_distance_matrix[pseudo_t, pseudo_a]
            + class_distance_matrix[pseudo_t, pseudo_v]
            + class_distance_matrix[pseudo_a, pseudo_v]) / 3.0


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


def hyp_cpcc_loss(features, pseudo_labels, eps=1e-8, hyp_eps=1e-5,
                  class_distance_matrix=None):
    """Minimize 1 - correlation of geometry and categorical/wheel distance."""
    if features.ndim != 2 or pseudo_labels.ndim != 1:
        raise ValueError("HypCPCC expects [N,D] features and [N] labels")
    if features.size(0) < 2:
        return features.sum() * 0.0
    row, column = torch.triu_indices(
        features.size(0), features.size(0), offset=1, device=features.device)
    geometric = poincare_distance(features[row], features[column], hyp_eps)
    if class_distance_matrix is None:
        categorical = (pseudo_labels[row] != pseudo_labels[column]).type_as(geometric)
    else:
        if class_distance_matrix.ndim != 2 or (
                class_distance_matrix.size(0) != class_distance_matrix.size(1)):
            raise ValueError("class distance matrix must be square")
        if pseudo_labels.numel() and (
                pseudo_labels.min() < 0
                or pseudo_labels.max() >= class_distance_matrix.size(0)):
            raise ValueError("labels exceed the class distance matrix")
        categorical = class_distance_matrix[
            pseudo_labels[row], pseudo_labels[column]].type_as(geometric)
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
                 consistency_k=0.5, detach_tau=True, detach_kappa=True,
                 dataset="IEMOCAP", use_emotion_wheel=False,
                 wheel_prototype_radius=0.75, wheel_temperature=1.0,
                 wheel_anchor_mix=0.5, anchor_balance="none",
                 anchor_min_per_class=0, anchor_admission="teacher"):
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
        if anchor_min_per_class < 0:
            raise ValueError("minimum anchors per class must be nonnegative")
        if anchor_min_per_class * n_classes > anchor_size:
            raise ValueError(
                "minimum per-class coverage exceeds total anchor capacity")
        if anchor_admission not in ("teacher", "modality"):
            raise ValueError("anchor admission must be 'teacher' or 'modality'")
        self.anchor_conf_threshold = anchor_conf_threshold
        self.hyp_eps = hyp_eps
        self.typicality_eps = typicality_eps
        self.consistency_t = consistency_t
        self.consistency_k = consistency_k
        self.detach_tau = detach_tau
        self.detach_kappa = detach_kappa
        self.anchor_min_per_class = int(anchor_min_per_class)
        self.anchor_admission = anchor_admission
        self.use_emotion_wheel = bool(use_emotion_wheel)
        self.wheel_temperature = float(wheel_temperature)
        self.wheel_anchor_mix = float(wheel_anchor_mix)
        if self.use_emotion_wheel:
            if not math.isfinite(self.wheel_temperature) or self.wheel_temperature <= 0:
                raise ValueError("wheel temperature must be finite and positive")
            if (not math.isfinite(self.wheel_anchor_mix)
                    or not 0.0 <= self.wheel_anchor_mix <= 1.0):
                raise ValueError("wheel anchor mix must be in [0, 1]")
            wheel_angles = emotion_wheel_angles(dataset, n_classes)
            self.register_buffer("wheel_angles", wheel_angles)
            self.register_buffer(
                "wheel_prototypes",
                emotion_wheel_prototypes(
                    wheel_angles, hyperbolic_dim,
                    wheel_prototype_radius, hyp_eps))
            self.register_buffer(
                "wheel_class_distances",
                circular_class_distance_matrix(wheel_angles))
        self.projectors = nn.ModuleDict({
            name: HyperbolicProjector(hidden_dim, hyperbolic_dim, hyp_eps)
            for name in MODALITIES
        })
        self.anchor_banks = nn.ModuleDict({
            name: AnchorBank(
                hyperbolic_dim, n_classes, anchor_size, hyp_eps,
                balance_mode=anchor_balance)
            for name in MODALITIES
        })

    def banks_ready(self):
        if self.anchor_min_per_class == 0:
            return all(self.anchor_banks[name].size > 0 for name in MODALITIES)
        return all(
            self.anchor_banks[name].has_class_coverage(
                self.anchor_min_per_class)
            for name in MODALITIES
        )

    def forward(self, pure_features, valid_mask, query_enabled=True):
        projected = {
            name: self.projectors[name](feature)
            for name, feature in zip(MODALITIES, pure_features)
        }
        result = {"ready": False, "projected": projected, "distance": None,
                  "pseudo_labels": None, "anchor_tau": None, "tau": None,
                  "categorical_discrepancy": None,
                  "wheel_discrepancy": None, "label_discrepancy": None,
                  "kappa": None, "hyp_eps": self.hyp_eps,
                  "wheel_enabled": self.use_emotion_wheel,
                  "wheel_distances": None, "wheel_logits": None,
                  "wheel_pseudo_labels": None, "prototype_tau": None,
                  "wheel_class_distances": (
                      self.wheel_class_distances
                      if self.use_emotion_wheel else None)}
        valid = valid_mask.bool()
        if self.use_emotion_wheel and valid.any():
            wheel_distances, wheel_logits = {}, {}
            wheel_pseudo, prototype_tau = {}, {}
            for name in MODALITIES:
                (wheel_distances[name], wheel_logits[name],
                 wheel_pseudo[name], prototype_tau[name]) = (
                    hyperbolic_prototype_outputs(
                        projected[name][valid], self.wheel_prototypes,
                        self.wheel_temperature, self.hyp_eps))
                if self.detach_tau:
                    prototype_tau[name] = prototype_tau[name].detach()
            result.update({
                "wheel_distances": wheel_distances,
                "wheel_logits": wheel_logits,
                "wheel_pseudo_labels": wheel_pseudo,
                "prototype_tau": prototype_tau,
            })
        if not query_enabled or not self.banks_ready() or not valid.any():
            return result

        distance, pseudo, anchor_tau = {}, {}, {}
        for name in MODALITIES:
            distance[name], pseudo[name] = self.anchor_banks[name].query(projected[name][valid])
            anchor_tau[name] = compute_typicality(
                distance[name], self.typicality_eps)
            if self.detach_tau:
                anchor_tau[name] = anchor_tau[name].detach()
            assert ((anchor_tau[name] >= 0) & (anchor_tau[name] <= 1)).all()
        categorical_discrepancy = categorical_label_discrepancy(
            pseudo["t"], pseudo["a"], pseudo["v"])
        if self.use_emotion_wheel:
            tau = {
                name: blend_typicality(
                    anchor_tau[name], result["prototype_tau"][name],
                    self.wheel_anchor_mix, self.typicality_eps)
                for name in MODALITIES
            }
            wheel_discrepancy = wheel_label_discrepancy(
                pseudo["t"], pseudo["a"], pseudo["v"],
                self.wheel_class_distances)
            discrepancy = wheel_discrepancy
        else:
            tau = anchor_tau
            wheel_discrepancy = None
            discrepancy = categorical_discrepancy
        kappa = compute_consistency(
            tau["t"], tau["a"], tau["v"], discrepancy,
            self.consistency_t, self.consistency_k, self.typicality_eps)
        if self.detach_kappa:
            kappa = kappa.detach()
        assert ((kappa >= 0) & (kappa <= 1)).all()
        assert torch.isfinite(kappa).all()
        result.update({"ready": True, "distance": distance,
                       "pseudo_labels": pseudo, "anchor_tau": anchor_tau,
                       "tau": tau,
                       "categorical_discrepancy": categorical_discrepancy,
                       "wheel_discrepancy": wheel_discrepancy,
                       "label_discrepancy": discrepancy, "kappa": kappa})
        return result

    @torch.no_grad()
    def update(self, projected, teacher_logits, student_logits, labels,
               valid_mask):
        if not self.training:
            raise RuntimeError("TiCAL anchor banks are frozen during validation/test")
        probability = teacher_logits.softmax(dim=-1)
        confidence, prediction = probability.max(dim=-1)
        base_eligible = (valid_mask.bool() & prediction.eq(labels)
                         & confidence.gt(self.anchor_conf_threshold))
        admitted = []
        for name in MODALITIES:
            eligible = base_eligible
            if self.anchor_admission == "modality":
                eligible = (eligible
                            & student_logits[name].argmax(dim=-1).eq(labels))
            self.anchor_banks[name].update(projected[name][eligible], labels[eligible])
            admitted.append(eligible)
        # Keep the historical metric semantics: count samples which inserted
        # at least one anchor, rather than summing insertions across banks.
        contributed = torch.stack(admitted, dim=-1).any(dim=-1)
        return int(contributed.sum().item())

    def summary(self):
        return {
            name: {"size": self.anchor_banks[name].size,
                   "class_counts": self.anchor_banks[name].class_counts().tolist()}
            for name in MODALITIES
        }
