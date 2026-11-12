"""TiCAL-inspired hyperbolic consistency estimation for SDT.

The module owns only TiCAL-specific state.  Labels are used solely by the
post-optimizer anchor update performed by the training loop; forward/query is
label-free and therefore safe for validation and test inference.
"""

import math

import torch
import torch.nn as nn

from geometry_analysis.geometry import (GEOMETRIES, GeometryProjector,
                                        geometry_cpcc_loss,
                                        geometry_distance,
                                        geometry_prototype_outputs,
                                        wheel_prototypes)


MODALITIES = ("t", "a", "v")
WHEEL_SEMANTIC_CONTROLS = ("correct", "random")
WHEEL_PROTOTYPE_RADIUS_MODES = (
    "fixed-shared",
    "learnable-shared",
    "learnable-per-class",
    "per-class-regularized",
)
WHEEL_PROTOTYPE_LAYOUTS = (
    "fixed-wheel", "free", "modality-specific", "shared-private",
    "angular-residual")
WHEEL_PROTOTYPE_STRUCTURE_MODES = (
    "pairwise-mse", "ordinal", "relational-kl")
WHEEL_FREE_PROTOTYPE_INITS = ("wheel", "random")
DATA_PROTOTYPE_LAYOUTS = ("shared", "modality-specific")
DATA_PROTOTYPE_INITS = ("centroid", "random")
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


def controlled_emotion_wheel_angles(dataset, n_classes, control="correct",
                                    permutation_seed=1729, device=None,
                                    dtype=torch.float32):
    """Return class angles and the applied label-to-angle permutation.

    The random control uses an isolated CPU generator, so it does not consume
    the global training RNG. Dihedral permutations (rotations/reflections)
    are rejected because they preserve every pairwise Wheel distance.
    """
    if control not in WHEEL_SEMANTIC_CONTROLS:
        raise ValueError("unknown Wheel semantic control: {}".format(control))
    base = emotion_wheel_angles(dataset, n_classes, dtype=dtype)
    identity = torch.arange(n_classes, dtype=torch.long)
    if control == "correct":
        return base.to(device=device), identity.to(device=device)
    generator = torch.Generator(device="cpu").manual_seed(int(permutation_seed))
    base_distances = circular_class_distance_matrix(base)
    for _ in range(256):
        permutation = torch.randperm(n_classes, generator=generator)
        candidate = base[permutation]
        if not torch.allclose(
                circular_class_distance_matrix(candidate), base_distances,
                atol=1e-7, rtol=0.0):
            return candidate.to(device=device), permutation.to(device=device)
    raise RuntimeError("could not sample a non-isometric Wheel permutation")


def circular_class_distance_matrix(class_angles):
    """Pairwise shortest angular distance normalized to [0, 1]."""
    if class_angles.ndim != 1 or class_angles.numel() < 2:
        raise ValueError("class_angles must be a 1-D tensor with at least 2 entries")
    difference = class_angles[:, None] - class_angles[None, :]
    cosine = torch.cos(difference).clamp(-1.0, 1.0)
    return torch.acos(cosine) / math.pi


def emotion_wheel_prototypes(class_angles, feature_dim, radius=0.75, eps=1e-5):
    """Embed fixed emotion-wheel prototypes in the first two ball dimensions."""
    return wheel_prototypes(
        class_angles, feature_dim, "poincare", radius, eps)


class HyperbolicProjector(GeometryProjector):
    """Learned Euclidean projection followed by the exp-map at the ball origin."""

    def __init__(self, input_dim, output_dim, eps=1e-5):
        super().__init__(input_dim, output_dim, "poincare", eps)


def poincare_distance(first, second, eps=1e-5):
    """Poincare-ball distance with broadcasting over all leading dimensions."""
    return geometry_distance(first, second, "poincare", eps)


def poincare_mobius_add(first, second, eps=1e-5):
    """Stable Mobius addition for curvature -1 with broadcasting."""
    first_sq = (first * first).sum(dim=-1, keepdim=True)
    second_sq = (second * second).sum(dim=-1, keepdim=True)
    inner = (first * second).sum(dim=-1, keepdim=True)
    numerator = ((1.0 + 2.0 * inner + second_sq) * first
                 + (1.0 - first_sq) * second)
    denominator = (1.0 + 2.0 * inner
                   + first_sq * second_sq).clamp_min(eps)
    result = numerator / denominator
    norm = result.norm(p=2, dim=-1, keepdim=True)
    scale = ((1.0 - eps) / norm.clamp_min(eps)).clamp(max=1.0)
    return result * scale


def poincare_log_map(base, points, eps=1e-5):
    """Logarithmic map at an arbitrary Poincare-ball base point."""
    difference = poincare_mobius_add(-base, points, eps)
    norm = difference.norm(p=2, dim=-1, keepdim=True)
    direction = difference / norm.clamp_min(eps)
    base_sq = (base * base).sum(dim=-1, keepdim=True)
    conformal = 2.0 / (1.0 - base_sq).clamp_min(eps)
    magnitude = (2.0 / conformal) * torch.atanh(
        norm.clamp(max=1.0 - eps))
    tangent = magnitude * direction
    return torch.where(norm > eps, tangent, torch.zeros_like(tangent))


def poincare_exp_map(base, tangent, eps=1e-5):
    """Exponential map at an arbitrary Poincare-ball base point."""
    norm = tangent.norm(p=2, dim=-1, keepdim=True)
    direction = tangent / norm.clamp_min(eps)
    base_sq = (base * base).sum(dim=-1, keepdim=True)
    conformal = 2.0 / (1.0 - base_sq).clamp_min(eps)
    displacement = torch.tanh(conformal * norm / 2.0) * direction
    displacement = torch.where(
        norm > eps, displacement, torch.zeros_like(displacement))
    return poincare_mobius_add(base, displacement, eps)


def poincare_frechet_mean(points, eps=1e-5, iterations=5):
    """Differentiable equal-weight Karcher mean over the penultimate axis."""
    if points.ndim < 3 or points.size(-2) < 1:
        raise ValueError("Frechet mean expects [...,J,D] with J >= 1")
    if points.size(-2) == 1:
        return points.squeeze(-2)
    tangent_zero = poincare_log_map(
        torch.zeros_like(points), points, eps).mean(dim=-2)
    mean = poincare_exp_map(
        torch.zeros_like(tangent_zero), tangent_zero, eps)
    for _ in range(iterations):
        base = mean.unsqueeze(-2).expand_as(points)
        update = poincare_log_map(base, points, eps).mean(dim=-2)
        mean = poincare_exp_map(mean, update, eps)
    return mean


def hyperbolic_prototype_outputs(features, prototypes, temperature=1.0,
                                 eps=1e-5):
    """Return class distances, logits, pseudo-labels, and absolute typicality."""
    return geometry_prototype_outputs(
        features, prototypes, "poincare", temperature, eps)


class AnchorBank(nn.Module):
    """A bounded FIFO bank of detached hyperbolic features and class labels."""

    def __init__(self, feature_dim, n_classes, max_size, eps=1e-5,
                 query_chunk_size=256, balance_mode="none",
                 geometry="poincare"):
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
        if geometry not in GEOMETRIES:
            raise ValueError("unknown anchor geometry: {}".format(geometry))
        self.geometry = geometry
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
            pairwise = geometry_distance(
                chunk.unsqueeze(1), self.features.unsqueeze(0),
                self.geometry, self.eps)
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
                  class_distance_matrix=None, geometry="poincare"):
    """Minimize 1 - correlation of geometry and categorical/wheel distance."""
    return geometry_cpcc_loss(
        features, pseudo_labels, geometry, eps, hyp_eps,
        class_distance_matrix)


class TiCALModule(nn.Module):
    """Project pure modalities, query three HASLs, and estimate consistency."""

    def __init__(self, hidden_dim, hyperbolic_dim, n_classes, anchor_size=2048,
                 anchor_conf_threshold=0.8, hyp_eps=1e-5,
                 typicality_eps=1e-8, consistency_t=0.2,
                 consistency_k=0.5, detach_tau=True, detach_kappa=True,
                 dataset="IEMOCAP", use_emotion_wheel=False,
                 wheel_prototype_radius=0.75, wheel_temperature=1.0,
                 wheel_anchor_mix=0.5, anchor_balance="none",
                 anchor_min_per_class=0, anchor_admission="teacher",
                 wheel_geometry="poincare", wheel_radius_mode="free",
                 wheel_fixed_radius=0.75, wheel_zero_residual=False,
                 wheel_semantic_control="correct",
                 wheel_permutation_seed=1729,
                 wheel_detach_features=False,
                 wheel_prototype_layout="fixed-wheel",
                 wheel_prototype_radius_mode="fixed-shared",
                 wheel_prototype_radius_min=0.20,
                 wheel_prototype_radius_max=0.90,
                 wheel_free_prototype_max_radius=0.95,
                 wheel_prototypes_per_class=1,
                 wheel_multi_prototype_temperature=0.2,
                 wheel_multi_prototype_init_std=0.02,
                 wheel_prototype_diversity_margin=0.5,
                 wheel_prototype_structure_mode="pairwise-mse",
                 wheel_ordinal_margin=0.2,
                 wheel_free_prototype_init="wheel",
                 wheel_private_residual_init_std=0.02,
                 use_data_prototypes=False,
                 data_prototype_layout="shared",
                 data_prototype_init="centroid",
                 data_prototype_initial_radius=0.3,
                 replace_fixed_wheel_prototypes=False,
                 data_prototype_warmup_epochs=5,
                 data_prototype_ramp_epochs=10,
                 data_prototype_temperature=1.0,
                 wheel_angular_residual_scale=1.0,
                 wheel_angular_residual_init_std=0.02,
                 wheel_relation_target_temperature=1.0,
                 wheel_relation_prototype_temperature=1.0):
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
        if wheel_geometry not in GEOMETRIES:
            raise ValueError("unknown wheel geometry: {}".format(wheel_geometry))
        self.wheel_geometry = wheel_geometry
        self.wheel_radius_mode = wheel_radius_mode
        self.wheel_fixed_radius = float(wheel_fixed_radius)
        self.wheel_zero_residual = bool(wheel_zero_residual)
        if wheel_semantic_control not in WHEEL_SEMANTIC_CONTROLS:
            raise ValueError("unknown Wheel semantic control: {}".format(
                wheel_semantic_control))
        self.wheel_semantic_control = wheel_semantic_control
        self.wheel_permutation_seed = int(wheel_permutation_seed)
        self.wheel_detach_features = bool(wheel_detach_features)
        if wheel_prototype_layout not in WHEEL_PROTOTYPE_LAYOUTS:
            raise ValueError("unknown Wheel prototype layout: {}".format(
                wheel_prototype_layout))
        self.wheel_prototype_layout = wheel_prototype_layout
        self.wheel_temperature = float(wheel_temperature)
        self.wheel_anchor_mix = float(wheel_anchor_mix)
        if wheel_prototype_radius_mode not in WHEEL_PROTOTYPE_RADIUS_MODES:
            raise ValueError("unknown Wheel prototype radius mode: {}".format(
                wheel_prototype_radius_mode))
        self.wheel_prototype_radius_mode = wheel_prototype_radius_mode
        self.wheel_prototype_radius = float(wheel_prototype_radius)
        self.wheel_prototype_radius_min = float(wheel_prototype_radius_min)
        self.wheel_prototype_radius_max = float(wheel_prototype_radius_max)
        self.wheel_free_prototype_max_radius = float(
            wheel_free_prototype_max_radius)
        if wheel_prototype_structure_mode not in (
                WHEEL_PROTOTYPE_STRUCTURE_MODES):
            raise ValueError("unknown prototype structure mode: {}".format(
                wheel_prototype_structure_mode))
        if wheel_free_prototype_init not in WHEEL_FREE_PROTOTYPE_INITS:
            raise ValueError("unknown free prototype initialization: {}".format(
                wheel_free_prototype_init))
        if (not math.isfinite(wheel_ordinal_margin)
                or wheel_ordinal_margin < 0):
            raise ValueError("Wheel ordinal margin must be nonnegative")
        if (not math.isfinite(wheel_private_residual_init_std)
                or wheel_private_residual_init_std < 0):
            raise ValueError("private residual init std must be nonnegative")
        self.wheel_prototype_structure_mode = (
            wheel_prototype_structure_mode)
        self.wheel_ordinal_margin = float(wheel_ordinal_margin)
        self.wheel_free_prototype_init = wheel_free_prototype_init
        self.wheel_private_residual_init_std = float(
            wheel_private_residual_init_std)
        angular_values = (
            wheel_angular_residual_scale,
            wheel_angular_residual_init_std,
            wheel_relation_target_temperature,
            wheel_relation_prototype_temperature,
        )
        if any(not math.isfinite(value) for value in angular_values):
            raise ValueError("angular-residual settings must be finite")
        if wheel_angular_residual_scale < 0:
            raise ValueError("angular residual scale must be nonnegative")
        if wheel_angular_residual_init_std < 0:
            raise ValueError("angular residual init std must be nonnegative")
        if (wheel_relation_target_temperature <= 0
                or wheel_relation_prototype_temperature <= 0):
            raise ValueError("relational temperatures must be positive")
        self.wheel_angular_residual_scale = float(
            wheel_angular_residual_scale)
        self.wheel_angular_residual_init_std = float(
            wheel_angular_residual_init_std)
        self.wheel_relation_target_temperature = float(
            wheel_relation_target_temperature)
        self.wheel_relation_prototype_temperature = float(
            wheel_relation_prototype_temperature)
        if (data_prototype_warmup_epochs < 0
                or data_prototype_ramp_epochs < 0):
            raise ValueError("data prototype schedule must be nonnegative")
        if (not math.isfinite(data_prototype_temperature)
                or data_prototype_temperature <= 0):
            raise ValueError("data prototype temperature must be positive")
        self.use_data_prototypes = bool(use_data_prototypes)
        if self.use_data_prototypes and not self.use_emotion_wheel:
            raise ValueError("data prototypes require emotion-wheel mode")
        if data_prototype_layout not in DATA_PROTOTYPE_LAYOUTS:
            raise ValueError("unknown data prototype layout: {}".format(
                data_prototype_layout))
        self.data_prototype_layout = data_prototype_layout
        if data_prototype_init not in DATA_PROTOTYPE_INITS:
            raise ValueError("unknown data prototype initialization: {}".format(
                data_prototype_init))
        if (not math.isfinite(data_prototype_initial_radius)
                or not 0 < data_prototype_initial_radius < 1 - self.hyp_eps):
            raise ValueError(
                "data prototype initial radius must be in (0,1-eps)")
        self.data_prototype_init = data_prototype_init
        self.data_prototype_initial_radius = float(
            data_prototype_initial_radius)
        self.replace_fixed_wheel_prototypes = bool(
            replace_fixed_wheel_prototypes)
        if self.replace_fixed_wheel_prototypes:
            if not self.use_data_prototypes:
                raise ValueError(
                    "replacing fixed Wheel prototypes requires data prototypes")
            if self.data_prototype_init != "random":
                raise ValueError(
                    "train-from-start replacement requires random initialization")
        self.data_prototype_warmup_epochs = int(
            data_prototype_warmup_epochs)
        self.data_prototype_ramp_epochs = int(data_prototype_ramp_epochs)
        self.data_prototype_temperature = float(data_prototype_temperature)
        self.data_prototype_epoch = 0
        if wheel_prototypes_per_class < 1:
            raise ValueError("prototypes per class must be positive")
        if (wheel_prototypes_per_class > 1
                and wheel_prototype_layout != "modality-specific"):
            raise ValueError(
                "multiple prototypes require modality-specific layout")
        multi_values = (wheel_multi_prototype_temperature,
                        wheel_multi_prototype_init_std,
                        wheel_prototype_diversity_margin)
        if any(not math.isfinite(value) or value < 0 for value in multi_values):
            raise ValueError("multi-prototype settings must be finite and nonnegative")
        if wheel_multi_prototype_temperature <= 0:
            raise ValueError("multi-prototype temperature must be positive")
        self.wheel_prototypes_per_class = int(wheel_prototypes_per_class)
        self.wheel_multi_prototype_temperature = float(
            wheel_multi_prototype_temperature)
        self.wheel_multi_prototype_init_std = float(
            wheel_multi_prototype_init_std)
        self.wheel_prototype_diversity_margin = float(
            wheel_prototype_diversity_margin)
        self.register_parameter("wheel_prototype_radius_logits", None)
        self.register_parameter("wheel_free_prototype_tangent", None)
        self.register_parameter("wheel_shared_center_tangent", None)
        self.register_parameter("wheel_angular_subspace", None)
        self.register_parameter("wheel_angular_residual", None)
        self.wheel_modality_prototype_tangents = nn.ParameterDict()
        self.wheel_modality_private_tangents = nn.ParameterDict()
        self.register_parameter("data_prototype_tangent", None)
        data_bank_count = (len(MODALITIES)
                           if self.data_prototype_layout == "modality-specific"
                           else None)
        centroid_shape = ((data_bank_count, n_classes, hyperbolic_dim)
                          if data_bank_count is not None
                          else (n_classes, hyperbolic_dim))
        count_shape = ((data_bank_count, n_classes)
                       if data_bank_count is not None else (n_classes,))
        self.register_buffer(
            "data_prototype_centroid_sum",
            torch.zeros(*centroid_shape),
            persistent=self.use_data_prototypes)
        self.register_buffer(
            "data_prototype_centroid_count",
            torch.zeros(*count_shape), persistent=self.use_data_prototypes)
        self.register_buffer(
            "data_prototype_initialized",
            torch.tensor(False), persistent=self.use_data_prototypes)
        if self.use_emotion_wheel:
            if not math.isfinite(self.wheel_temperature) or self.wheel_temperature <= 0:
                raise ValueError("wheel temperature must be finite and positive")
            if (not math.isfinite(self.wheel_anchor_mix)
                    or not 0.0 <= self.wheel_anchor_mix <= 1.0):
                raise ValueError("wheel anchor mix must be in [0, 1]")
            radius_values = (
                self.wheel_prototype_radius,
                self.wheel_prototype_radius_min,
                self.wheel_prototype_radius_max,
            )
            if any(not math.isfinite(value) for value in radius_values):
                raise ValueError("Wheel prototype radii must be finite")
            if not (0.0 < self.wheel_prototype_radius_min
                    < self.wheel_prototype_radius
                    < self.wheel_prototype_radius_max
                    < 1.0 - self.hyp_eps):
                raise ValueError(
                    "prototype radii must satisfy 0 < min < initial < max < 1-eps")
            if (not math.isfinite(self.wheel_free_prototype_max_radius)
                    or not 0.0 < self.wheel_free_prototype_max_radius
                    < 1.0 - self.hyp_eps):
                raise ValueError(
                    "free prototype maximum radius must be in (0, 1-eps)")
            if (self.wheel_prototype_layout in (
                    "free", "modality-specific", "shared-private",
                    "angular-residual")
                    and self.wheel_geometry != "poincare"):
                raise ValueError(
                    "free Wheel prototype layouts require Poincare geometry")
            if (self.wheel_prototype_layout in (
                    "free", "modality-specific", "shared-private",
                    "angular-residual")
                    and self.wheel_prototype_radius_mode != "fixed-shared"):
                raise ValueError(
                    "free Wheel prototype layouts are incompatible with learned radius modes")
            if (self.wheel_prototype_radius_mode != "fixed-shared"
                    and self.wheel_geometry != "poincare"):
                raise ValueError(
                    "learnable prototype radii require Poincare geometry")
            wheel_angles, wheel_permutation = controlled_emotion_wheel_angles(
                dataset, n_classes, wheel_semantic_control,
                wheel_permutation_seed)
            self.register_buffer("wheel_angles", wheel_angles)
            # Recreated deterministically from config; keeping this
            # non-persistent preserves compatibility with older checkpoints.
            self.register_buffer(
                "wheel_label_permutation", wheel_permutation,
                persistent=False)
            self.register_buffer(
                "wheel_prototypes",
                wheel_prototypes(
                    wheel_angles, hyperbolic_dim, wheel_geometry,
                    wheel_prototype_radius, hyp_eps))
            if self.use_data_prototypes:
                if self.wheel_geometry != "poincare":
                    raise ValueError(
                        "data prototypes require Poincare Wheel geometry")
                if self.wheel_prototype_layout != "fixed-wheel":
                    raise ValueError(
                        "dual data prototypes require fixed-wheel anchors")
                # This parameter is present before optimizer construction but
                # receives no gradient until centroid initialization finishes.
                self.data_prototype_tangent = nn.Parameter(
                    torch.zeros(*centroid_shape))
                if self.data_prototype_init == "random":
                    random_direction = torch.randn(*centroid_shape)
                    random_direction = random_direction / random_direction.norm(
                        p=2, dim=-1, keepdim=True).clamp_min(self.hyp_eps)
                    tangent_radius = math.atanh(
                        self.data_prototype_initial_radius)
                    with torch.no_grad():
                        self.data_prototype_tangent.copy_(
                            tangent_radius * random_direction)
                        self.data_prototype_initialized.fill_(True)
            initial_norm = self.wheel_prototypes.norm(
                p=2, dim=-1, keepdim=True).clamp_min(self.hyp_eps)
            wheel_initial_tangent = (
                torch.atanh(initial_norm.clamp(max=1.0 - self.hyp_eps))
                * self.wheel_prototypes / initial_norm)
            if self.wheel_free_prototype_init == "random":
                random_direction = torch.randn_like(wheel_initial_tangent)
                random_direction = random_direction / random_direction.norm(
                    p=2, dim=-1, keepdim=True).clamp_min(self.hyp_eps)
                initial_tangent = (
                    torch.atanh(initial_norm) * random_direction)
            else:
                initial_tangent = wheel_initial_tangent
            if self.wheel_prototype_layout == "free":
                # Optimize unconstrained class vectors in the tangent space at
                # the origin.  Exp_0 maps them into the Poincare ball on every
                # forward pass.  Initialization recovers the fixed semantic
                # wheel exactly, but every coordinate is subsequently free.
                self.wheel_free_prototype_tangent = nn.Parameter(
                    initial_tangent.clone())
            elif self.wheel_prototype_layout == "angular-residual":
                # U is optimized in R^{D x 2}; a reduced QR factorization in
                # every forward pass enforces U^T U = I exactly.  The raw
                # class residuals are projected onto span(U)^perp before use.
                subspace = torch.zeros(hyperbolic_dim, 2)
                subspace[:2, :2] = torch.eye(2)
                self.wheel_angular_subspace = nn.Parameter(subspace)
                residual = self.wheel_angular_residual_init_std * torch.randn(
                    n_classes, hyperbolic_dim)
                residual[:, :2] = 0.0
                self.wheel_angular_residual = nn.Parameter(residual)
            elif self.wheel_prototype_layout == "modality-specific":
                # q_c: the shared semantic centers constrained by the Wheel.
                self.wheel_shared_center_tangent = nn.Parameter(
                    initial_tangent.clone())
                # p_c^T, p_c^A, p_c^V: independently learned prototypes used
                # by the corresponding modality-distance classifiers.
                if self.wheel_prototypes_per_class == 1:
                    modality_initial = initial_tangent
                else:
                    modality_initial = initial_tangent.unsqueeze(1).expand(
                        -1, self.wheel_prototypes_per_class, -1).clone()
                    noise = torch.randn_like(modality_initial)
                    noise = noise - noise.mean(dim=1, keepdim=True)
                    modality_initial = (
                        modality_initial
                        + self.wheel_multi_prototype_init_std * noise)
                self.wheel_modality_prototype_tangents = nn.ParameterDict({
                    name: nn.Parameter(modality_initial.clone())
                    for name in MODALITIES
                })
            elif self.wheel_prototype_layout == "shared-private":
                # s_c is a shared, fully learned class tangent. Each modality
                # adds a centered private residual r_c^m before Exp_0. The
                # zero-mean constraint makes s_c the exact tangent barycenter
                # while leaving modality-specific evidence unconstrained.
                self.wheel_shared_center_tangent = nn.Parameter(
                    initial_tangent.clone())
                residual_stack = (
                    self.wheel_private_residual_init_std
                    * torch.randn(
                        len(MODALITIES), *initial_tangent.shape,
                        device=initial_tangent.device,
                        dtype=initial_tangent.dtype))
                residual_stack = residual_stack - residual_stack.mean(
                    dim=0, keepdim=True)
                self.wheel_modality_private_tangents = nn.ParameterDict({
                    name: nn.Parameter(residual_stack[index].clone())
                    for index, name in enumerate(MODALITIES)
                })
            elif self.wheel_prototype_radius_mode != "fixed-shared":
                fraction = ((self.wheel_prototype_radius
                             - self.wheel_prototype_radius_min)
                            / (self.wheel_prototype_radius_max
                               - self.wheel_prototype_radius_min))
                initial_logit = math.log(fraction / (1.0 - fraction))
                parameter_count = (1 if self.wheel_prototype_radius_mode
                                   == "learnable-shared" else n_classes)
                self.wheel_prototype_radius_logits = nn.Parameter(
                    torch.full((parameter_count,), initial_logit))
            self.register_buffer(
                "wheel_class_distances",
                circular_class_distance_matrix(wheel_angles))
        self.projectors = nn.ModuleDict({
            name: GeometryProjector(
                hidden_dim, hyperbolic_dim, wheel_geometry, hyp_eps,
                wheel_radius_mode, wheel_fixed_radius,
                wheel_zero_residual)
            for name in MODALITIES
        })
        self.anchor_banks = nn.ModuleDict({
            name: AnchorBank(
                hyperbolic_dim, n_classes, anchor_size, hyp_eps,
                balance_mode=anchor_balance, geometry=wheel_geometry)
            for name in MODALITIES
        })

    def set_epoch(self, epoch):
        """Advance the centroid-initialization and auxiliary-loss schedule."""
        epoch = int(epoch)
        if (self.use_data_prototypes
                and not bool(self.data_prototype_initialized)
                and epoch != self.data_prototype_epoch):
            if 0 < epoch <= self.data_prototype_warmup_epochs:
                # Keep only the most recent warm-up epoch. Its centroids match
                # the latest projector better than an average over old models.
                self.data_prototype_centroid_sum.zero_()
                self.data_prototype_centroid_count.zero_()
            elif epoch > self.data_prototype_warmup_epochs:
                self._initialize_data_prototypes()
        self.data_prototype_epoch = epoch

    def data_prototype_progress(self):
        if (not self.use_data_prototypes
                or not bool(self.data_prototype_initialized)):
            return 0.0
        elapsed = (self.data_prototype_epoch
                   - self.data_prototype_warmup_epochs)
        if elapsed <= 0:
            return 0.0
        if self.data_prototype_ramp_epochs == 0:
            return 1.0
        return min(1.0, elapsed / self.data_prototype_ramp_epochs)

    @torch.no_grad()
    def _initialize_data_prototypes(self):
        """Copy class centroids into the learnable full-D tangent bank once."""
        if (not self.use_data_prototypes
                or bool(self.data_prototype_initialized)):
            return False
        if not (self.data_prototype_centroid_count > 0).all():
            return False
        centroid = (self.data_prototype_centroid_sum
                    / self.data_prototype_centroid_count.unsqueeze(-1))
        if not torch.isfinite(centroid).all():
            raise FloatingPointError("nonfinite data prototype centroid")
        self.data_prototype_tangent.copy_(centroid)
        self.data_prototype_initialized.fill_(True)
        return True

    def current_data_prototypes(self, modality=None):
        if (not self.use_data_prototypes
                or not bool(self.data_prototype_initialized)):
            return None
        tangent = self.data_prototype_tangent
        if self.data_prototype_layout == "modality-specific":
            if modality not in MODALITIES:
                raise ValueError(
                    "modality-specific data prototypes require t/a/v")
            tangent = tangent[MODALITIES.index(modality)]
        return self._exp_map_free_prototypes(tangent)

    def data_prototype_cpcc_loss(self):
        """Correlate learned prototype distances with semantic Wheel order."""
        if not bool(self.data_prototype_initialized):
            return self.wheel_prototypes.sum() * 0.0
        modalities = (MODALITIES if self.data_prototype_layout
                      == "modality-specific" else (None,))
        losses = []
        for name in modalities:
            prototypes = self.current_data_prototypes(name)
            labels = torch.arange(
                prototypes.size(0), device=prototypes.device)
            losses.append(geometry_cpcc_loss(
                prototypes, labels, "poincare", self.typicality_eps,
                self.hyp_eps, self.wheel_class_distances))
        return sum(losses) / len(losses)

    def _exp_map_free_prototypes(self, tangent):
        """Map arbitrary tangent parameters safely into the Poincare ball."""
        norm = tangent.norm(p=2, dim=-1, keepdim=True)
        direction = tangent / norm.clamp_min(self.hyp_eps)
        prototypes = torch.tanh(norm) * direction
        prototypes = torch.where(
            norm > self.hyp_eps, prototypes, torch.zeros_like(prototypes))
        prototype_norm = prototypes.norm(p=2, dim=-1, keepdim=True)
        scale = ((1.0 - self.hyp_eps)
                 / prototype_norm.clamp_min(self.hyp_eps)).clamp(max=1.0)
        return prototypes * scale

    def current_wheel_angular_subspace(self):
        """Return the learnable orthonormal D x 2 angular embedding."""
        if self.wheel_prototype_layout != "angular-residual":
            return None
        basis, triangular = torch.linalg.qr(
            self.wheel_angular_subspace, mode="reduced")
        # Fix the otherwise arbitrary QR column signs.  Detaching the sign
        # leaves the useful QR gradient unchanged and avoids sign-flip noise.
        diagonal = torch.diagonal(triangular)
        signs = torch.where(
            diagonal >= 0, torch.ones_like(diagonal),
            -torch.ones_like(diagonal)).detach()
        return basis * signs.unsqueeze(0)

    def current_wheel_angular_residuals(self):
        """Project class residuals onto the complement of span(U)."""
        if self.wheel_prototype_layout != "angular-residual":
            return None
        subspace = self.current_wheel_angular_subspace()
        in_plane = (self.wheel_angular_residual @ subspace) @ subspace.T
        return self.wheel_angular_residual - in_plane

    def current_wheel_angular_tangents(self):
        """Construct U q_c + alpha r_c^perp in the tangent space at zero."""
        if self.wheel_prototype_layout != "angular-residual":
            return None
        subspace = self.current_wheel_angular_subspace()
        reference_radius = math.atanh(self.wheel_prototype_radius)
        references = reference_radius * torch.stack((
            torch.cos(self.wheel_angles), torch.sin(self.wheel_angles)),
            dim=-1)
        anchors = references @ subspace.T
        return (anchors + self.wheel_angular_residual_scale
                * self.current_wheel_angular_residuals())

    def wheel_anchor_regularization(self):
        """Penalize only the orthogonal residual of angular prototypes."""
        residuals = self.current_wheel_angular_residuals()
        if residuals is None:
            if self.use_emotion_wheel:
                return self.wheel_prototypes.sum() * 0.0
            return torch.tensor(0.0)
        return residuals.square().sum(dim=-1).mean()

    def current_wheel_prototype_radii(self, modality=None):
        """Return the bounded Euclidean ball radius of every class prototype."""
        if not self.use_emotion_wheel:
            return None
        if self.wheel_prototype_layout in (
                "modality-specific", "shared-private", "angular-residual"):
            return self.current_wheel_prototypes(modality).norm(p=2, dim=-1)
        if self.wheel_free_prototype_tangent is not None:
            return self.current_wheel_prototypes().norm(p=2, dim=-1)
        if self.wheel_prototype_radius_logits is None:
            return self.wheel_angles.new_full(
                (self.wheel_angles.numel(),), self.wheel_prototype_radius)
        radii = self.wheel_prototype_radius_min + (
            self.wheel_prototype_radius_max
            - self.wheel_prototype_radius_min
        ) * torch.sigmoid(self.wheel_prototype_radius_logits)
        if radii.numel() == 1:
            radii = radii.expand(self.wheel_angles.numel())
        return radii

    def current_wheel_prototypes(self, modality=None):
        """Build shared centers or the requested modality prototypes.

        With a modality-aware layout, ``modality=None`` returns shared q_c.
        Passing t/a/v returns the corresponding full modality prototypes.
        Other layouts ignore ``modality`` and share one prototype bank.
        """
        if not self.use_emotion_wheel:
            return None
        if self.wheel_prototype_layout == "angular-residual":
            return self._exp_map_free_prototypes(
                self.current_wheel_angular_tangents())
        if self.wheel_prototype_layout == "modality-specific":
            if modality is None:
                tangent = self.wheel_shared_center_tangent
            else:
                if modality not in MODALITIES:
                    raise ValueError("unknown prototype modality: {}".format(
                        modality))
                tangent = self.wheel_modality_prototype_tangents[modality]
            return self._exp_map_free_prototypes(tangent)
        if self.wheel_prototype_layout == "shared-private":
            if modality is None:
                tangent = self.wheel_shared_center_tangent
            else:
                if modality not in MODALITIES:
                    raise ValueError("unknown prototype modality: {}".format(
                        modality))
                residuals = self.current_wheel_private_residuals()
                tangent = (self.wheel_shared_center_tangent
                           + residuals[modality])
            return self._exp_map_free_prototypes(tangent)
        if self.wheel_free_prototype_tangent is not None:
            return self._exp_map_free_prototypes(
                self.wheel_free_prototype_tangent)
        if self.wheel_prototype_radius_logits is None:
            return self.wheel_prototypes
        directions = torch.stack(
            (torch.cos(self.wheel_angles), torch.sin(self.wheel_angles)),
            dim=-1)
        if self.wheel_prototypes.size(-1) > 2:
            directions = torch.cat((
                directions,
                directions.new_zeros(
                    directions.size(0), self.wheel_prototypes.size(-1) - 2),
            ), dim=-1)
        return directions * self.current_wheel_prototype_radii().unsqueeze(-1)

    def current_wheel_private_residuals(self):
        """Return zero-mean modality-private tangent residuals."""
        if self.wheel_prototype_layout != "shared-private":
            return None
        stacked = torch.stack([
            self.wheel_modality_private_tangents[name]
            for name in MODALITIES
        ], dim=0)
        centered = stacked - stacked.mean(dim=0, keepdim=True)
        return {
            name: centered[index]
            for index, name in enumerate(MODALITIES)
        }

    def wheel_private_residual_regularization(self):
        """Keep private evidence residuals bounded without removing them."""
        residuals = self.current_wheel_private_residuals()
        if residuals is None:
            if self.use_emotion_wheel:
                return self.wheel_prototypes.sum() * 0.0
            return torch.tensor(0.0)
        return sum(
            residual.square().mean() for residual in residuals.values()
        ) / len(MODALITIES)

    def current_wheel_prototype_barycenters(self, modality):
        """Return one Poincare barycenter per class for a modality bank."""
        prototypes = self.current_wheel_prototypes(modality)
        if prototypes.ndim == 2:
            return prototypes
        return poincare_frechet_mean(prototypes, self.hyp_eps)

    def wheel_prototype_outputs(self, features, prototypes):
        """Classify by one prototype or a soft-min pool over J prototypes."""
        if prototypes.ndim == 2:
            distance, logits, pseudo, typicality = (
                geometry_prototype_outputs(
                    features, prototypes, self.wheel_geometry,
                    self.wheel_temperature, self.hyp_eps))
            return distance, logits, pseudo, typicality, None
        if prototypes.ndim != 3:
            raise ValueError("prototypes must have shape [C,D] or [C,J,D]")
        subprototype_distance = geometry_distance(
            features[:, None, None, :], prototypes[None, :, :, :],
            self.wheel_geometry, self.hyp_eps)
        pool_temperature = self.wheel_multi_prototype_temperature
        class_distance = -pool_temperature * (
            torch.logsumexp(
                -subprototype_distance / pool_temperature, dim=-1)
            - math.log(prototypes.size(1)))
        logits = -class_distance / self.wheel_temperature
        nearest, pseudo = class_distance.min(dim=-1)
        typicality = torch.exp(
            -nearest / self.wheel_temperature).clamp(0.0, 1.0)
        return (class_distance, logits, pseudo, typicality,
                subprototype_distance)

    def wheel_radius_regularization(self):
        """Penalize only the explicitly regularized per-class radius mode."""
        if (not self.use_emotion_wheel
                or self.wheel_prototype_radius_mode
                != "per-class-regularized"):
            if self.use_emotion_wheel:
                return self.wheel_prototypes.sum() * 0.0
            return torch.tensor(0.0)
        radii = self.current_wheel_prototype_radii()
        return ((radii - self.wheel_prototype_radius) ** 2).mean()

    def wheel_prototype_structure_loss(self):
        """Match only relative prototype distances to the semantic Wheel.

        Mean normalization intentionally removes absolute scale: classification
        is free to learn the useful global spread, orientation, radius, and all
        residual coordinates, while this term preserves the Wheel's pairwise
        relational structure.
        """
        if (not self.use_emotion_wheel
                or self.wheel_prototype_layout not in (
                    "free", "modality-specific", "shared-private",
                    "angular-residual")):
            if self.use_emotion_wheel:
                return self.wheel_prototypes.sum() * 0.0
            return torch.tensor(0.0)
        prototypes = self.current_wheel_prototypes()
        learned_matrix = geometry_distance(
            prototypes[:, None, :], prototypes[None, :, :],
            "poincare", self.hyp_eps)
        target_matrix = self.wheel_class_distances.type_as(learned_matrix)
        if self.wheel_prototype_structure_mode == "relational-kl":
            count = prototypes.size(0)
            diagonal = torch.eye(
                count, device=prototypes.device, dtype=torch.bool)
            target_logits = (
                -target_matrix / self.wheel_relation_target_temperature
            ).masked_fill(diagonal, float("-inf"))
            learned_logits = (
                -learned_matrix / self.wheel_relation_prototype_temperature
            ).masked_fill(diagonal, float("-inf"))
            target_probability = torch.softmax(target_logits, dim=-1)
            learned_log_probability = torch.log_softmax(
                learned_logits, dim=-1)
            off_diagonal = ~diagonal
            target_off = target_probability[off_diagonal]
            learned_log_off = learned_log_probability[off_diagonal]
            target_log_off = target_off.clamp_min(
                self.typicality_eps).log()
            return (target_off * (
                target_log_off - learned_log_off)).sum() / count
        if self.wheel_prototype_structure_mode == "ordinal":
            count = prototypes.size(0)
            class_index = torch.arange(count, device=prototypes.device)
            anchor = class_index[:, None, None]
            near = class_index[None, :, None]
            far = class_index[None, None, :]
            ordered = (
                target_matrix[:, :, None] + self.typicality_eps
                < target_matrix[:, None, :])
            valid = ordered & (near != anchor) & (far != anchor)
            violations = torch.relu(
                self.wheel_ordinal_margin
                + learned_matrix[:, :, None]
                - learned_matrix[:, None, :])
            return violations[valid].mean()
        row, column = torch.triu_indices(
            prototypes.size(0), prototypes.size(0), offset=1,
            device=prototypes.device)
        learned = learned_matrix[row, column]
        target = target_matrix[row, column]
        learned = learned / learned.mean().clamp_min(self.typicality_eps)
        target = target / target.mean().clamp_min(self.typicality_eps)
        return (learned - target).square().mean()

    def wheel_prototype_alignment_loss(self):
        """Align each modality's class barycenter with semantic center q_c."""
        if (not self.use_emotion_wheel
                or self.wheel_prototype_layout != "modality-specific"):
            if self.use_emotion_wheel:
                return self.wheel_prototypes.sum() * 0.0
            return torch.tensor(0.0)
        centers = self.current_wheel_prototypes()
        return sum(
            geometry_distance(
                self.current_wheel_prototype_barycenters(name), centers,
                "poincare", self.hyp_eps).square().mean()
            for name in MODALITIES
        ) / len(MODALITIES)

    def wheel_prototype_diversity_loss(self):
        """Keep same-class subprototypes from collapsing onto one point."""
        if (not self.use_emotion_wheel
                or self.wheel_prototype_layout != "modality-specific"
                or self.wheel_prototypes_per_class == 1):
            if self.use_emotion_wheel:
                return self.wheel_prototypes.sum() * 0.0
            return torch.tensor(0.0)
        losses = []
        for name in MODALITIES:
            prototypes = self.current_wheel_prototypes(name)
            row, column = torch.triu_indices(
                prototypes.size(1), prototypes.size(1), offset=1,
                device=prototypes.device)
            pairwise = geometry_distance(
                prototypes[:, row, :], prototypes[:, column, :],
                "poincare", self.hyp_eps)
            losses.append(torch.relu(
                self.wheel_prototype_diversity_margin - pairwise
            ).square().mean())
        return sum(losses) / len(losses)

    def wheel_prototype_boundary_loss(self):
        """Softly keep free prototypes away from the unstable ball boundary."""
        if (not self.use_emotion_wheel
                or self.wheel_prototype_layout not in (
                    "free", "modality-specific", "shared-private",
                    "angular-residual")):
            if self.use_emotion_wheel:
                return self.wheel_prototypes.sum() * 0.0
            return torch.tensor(0.0)
        # Penalize in tangent space rather than after tanh.  This represents
        # the same ball-radius threshold but avoids vanishing gradients when a
        # prototype has already moved close to radius one.
        if self.wheel_prototype_layout == "free":
            tangents = [self.wheel_free_prototype_tangent]
        elif self.wheel_prototype_layout == "angular-residual":
            tangents = [self.current_wheel_angular_tangents()]
        elif self.wheel_prototype_layout == "modality-specific":
            tangents = [self.wheel_shared_center_tangent]
            tangents.extend(
                self.wheel_modality_prototype_tangents[name]
                for name in MODALITIES)
        else:
            tangents = [self.wheel_shared_center_tangent]
            residuals = self.current_wheel_private_residuals()
            tangents.extend(
                self.wheel_shared_center_tangent + residuals[name]
                for name in MODALITIES)
        tangent_radii = torch.cat([
            tangent.norm(p=2, dim=-1).reshape(-1) for tangent in tangents])
        maximum = tangent_radii.new_tensor(
            self.wheel_free_prototype_max_radius)
        maximum = torch.atanh(maximum)
        return torch.relu(tangent_radii - maximum).square().mean()

    def banks_ready(self):
        if self.anchor_min_per_class == 0:
            return all(self.anchor_banks[name].size > 0 for name in MODALITIES)
        return all(
            self.anchor_banks[name].has_class_coverage(
                self.anchor_min_per_class)
            for name in MODALITIES
        )

    def forward(self, pure_features, valid_mask, query_enabled=True):
        projector_inputs = tuple(
            feature.detach() if self.wheel_detach_features else feature
            for feature in pure_features)
        projected = {
            name: self.projectors[name](feature)
            for name, feature in zip(MODALITIES, projector_inputs)
        }
        result = {"ready": False, "projected": projected, "distance": None,
                  "pseudo_labels": None, "anchor_tau": None, "tau": None,
                  "categorical_discrepancy": None,
                  "wheel_discrepancy": None, "label_discrepancy": None,
                  "kappa": None, "hyp_eps": self.hyp_eps,
                  "wheel_enabled": self.use_emotion_wheel,
                  "wheel_geometry": self.wheel_geometry,
                  "wheel_radius_mode": self.wheel_radius_mode,
                  "wheel_prototype_layout": self.wheel_prototype_layout,
                  "wheel_prototype_radius_mode": (
                      self.wheel_prototype_radius_mode),
                  "wheel_prototype_radii": (
                      self.current_wheel_prototype_radii()),
                  "wheel_radius_regularization": (
                      self.wheel_radius_regularization()),
                  "wheel_prototype_structure": (
                      self.wheel_prototype_structure_loss()),
                  "wheel_prototype_structure_mode": (
                      self.wheel_prototype_structure_mode),
                  "wheel_ordinal_margin": self.wheel_ordinal_margin,
                  "wheel_prototype_alignment": (
                      self.wheel_prototype_alignment_loss()),
                  "wheel_private_residual": (
                      self.wheel_private_residual_regularization()),
                  "wheel_anchor_regularization": (
                      self.wheel_anchor_regularization()),
                  "wheel_angular_residual_scale": (
                      self.wheel_angular_residual_scale),
                  "wheel_relation_target_temperature": (
                      self.wheel_relation_target_temperature),
                  "wheel_relation_prototype_temperature": (
                      self.wheel_relation_prototype_temperature),
                  "wheel_angular_subspace": (
                      self.current_wheel_angular_subspace()),
                  "wheel_free_prototype_init": (
                      self.wheel_free_prototype_init),
                  "wheel_prototype_diversity": (
                      self.wheel_prototype_diversity_loss()),
                  "wheel_prototype_boundary": (
                      self.wheel_prototype_boundary_loss()),
                  "wheel_prototypes_per_class": (
                      self.wheel_prototypes_per_class),
                  "wheel_multi_prototype_temperature": (
                      self.wheel_multi_prototype_temperature),
                  "wheel_free_prototype_max_radius": (
                      self.wheel_free_prototype_max_radius),
                  "wheel_fixed_radius": self.wheel_fixed_radius,
                  "wheel_zero_residual": self.wheel_zero_residual,
                  "wheel_semantic_control": self.wheel_semantic_control,
                  "wheel_label_permutation": (
                      self.wheel_label_permutation
                      if self.use_emotion_wheel else None),
                  "wheel_detach_features": self.wheel_detach_features,
                  "wheel_distances": None, "wheel_logits": None,
                  "wheel_subprototype_distances": None,
                  "wheel_pseudo_labels": None, "prototype_tau": None,
                  "data_prototype_enabled": self.use_data_prototypes,
                  "data_prototype_layout": self.data_prototype_layout,
                  "data_prototype_init": self.data_prototype_init,
                  "replace_fixed_wheel_prototypes": (
                      self.replace_fixed_wheel_prototypes),
                  "data_prototype_ready": bool(
                      self.data_prototype_initialized),
                  "data_prototype_progress": (
                      self.data_prototype_progress()),
                  "data_prototype_logits": None,
                  "data_prototype_distances": None,
                  "data_prototype_cpcc": (
                      self.data_prototype_cpcc_loss()
                      if self.use_data_prototypes
                      else next(iter(projected.values())).sum() * 0.0),
                  "data_prototype_counts": (
                      self.data_prototype_centroid_count
                      if self.use_data_prototypes else None),
                  "wheel_class_distances": (
                      self.wheel_class_distances
                      if self.use_emotion_wheel else None)}
        valid = valid_mask.bool()
        if self.use_emotion_wheel and valid.any():
            wheel_distances, wheel_logits = {}, {}
            wheel_pseudo, prototype_tau = {}, {}
            wheel_subprototype_distances = {}
            for name in MODALITIES:
                primary_prototypes = (
                    self.current_data_prototypes(name)
                    if self.replace_fixed_wheel_prototypes
                    else self.current_wheel_prototypes(name))
                (wheel_distances[name], wheel_logits[name],
                 wheel_pseudo[name], prototype_tau[name],
                 wheel_subprototype_distances[name]) = (
                    self.wheel_prototype_outputs(
                        projected[name][valid],
                        primary_prototypes))
                if self.detach_tau:
                    prototype_tau[name] = prototype_tau[name].detach()
            result.update({
                "wheel_distances": wheel_distances,
                "wheel_logits": wheel_logits,
                "wheel_subprototype_distances": (
                    wheel_subprototype_distances),
                "wheel_pseudo_labels": wheel_pseudo,
                "prototype_tau": prototype_tau,
            })
            if self.replace_fixed_wheel_prototypes:
                result["data_prototype_distances"] = wheel_distances
                result["data_prototype_logits"] = wheel_logits
            elif bool(self.data_prototype_initialized):
                data_distances, data_logits = {}, {}
                for name in MODALITIES:
                    data_prototypes = self.current_data_prototypes(name)
                    (data_distances[name], data_logits[name], _, _, _) = (
                        self.wheel_prototype_outputs(
                            projected[name][valid], data_prototypes))
                    # Keep a separate temperature without changing the fixed
                    # Wheel classifier used by the original 75.31 setup.
                    data_logits[name] = (
                        -data_distances[name]
                        / self.data_prototype_temperature)
                result["data_prototype_distances"] = data_distances
                result["data_prototype_logits"] = data_logits
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
        if (self.use_data_prototypes
                and not bool(self.data_prototype_initialized)):
            valid_labels = labels[valid_mask.bool()]
            for modality_index, name in enumerate(MODALITIES):
                valid_projected = projected[name][valid_mask.bool()].detach()
                tangent = poincare_log_map(
                    torch.zeros_like(valid_projected), valid_projected,
                    self.hyp_eps)
                if self.data_prototype_layout == "modality-specific":
                    centroid_sum = self.data_prototype_centroid_sum[
                        modality_index]
                    centroid_count = self.data_prototype_centroid_count[
                        modality_index]
                else:
                    centroid_sum = self.data_prototype_centroid_sum
                    centroid_count = self.data_prototype_centroid_count
                centroid_sum.index_add_(0, valid_labels, tangent)
                centroid_count.index_add_(
                    0, valid_labels,
                    torch.ones_like(
                        valid_labels, dtype=centroid_count.dtype))
            if self.data_prototype_epoch > self.data_prototype_warmup_epochs:
                self._initialize_data_prototypes()
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
