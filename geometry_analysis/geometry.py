"""Shared Euclidean, spherical, and Poincare operations.

Every geometry uses the same learned affine projection.  Only the final
manifold map and distance change.  This is important for a controlled
ablation: parameter count and pre-geometry representation remain identical.
"""

import math

import torch
import torch.nn as nn


GEOMETRIES = ("euclidean", "spherical", "poincare")


def _validate_geometry(geometry):
    if geometry not in GEOMETRIES:
        raise ValueError(
            "geometry must be one of {}; got {!r}".format(GEOMETRIES, geometry))


class GeometryProjector(nn.Module):
    """Common affine head followed by a geometry-specific map."""

    def __init__(self, input_dim, output_dim, geometry="poincare", eps=1e-5):
        super().__init__()
        if input_dim < 1 or output_dim < 1:
            raise ValueError("projection dimensions must be positive")
        if not 0.0 < eps < 0.1:
            raise ValueError("geometry eps must be in (0, 0.1)")
        _validate_geometry(geometry)
        self.geometry = geometry
        self.eps = float(eps)
        self.linear = nn.Linear(input_dim, output_dim)
        # Keep the affine initialization identical for all geometries.
        nn.init.xavier_uniform_(self.linear.weight, gain=0.01)
        nn.init.zeros_(self.linear.bias)

    def forward(self, features):
        tangent = self.linear(features)
        if self.geometry == "euclidean":
            projected = tangent
        elif self.geometry == "spherical":
            norm = tangent.norm(p=2, dim=-1, keepdim=True).clamp_min(self.eps)
            projected = tangent / norm
        else:
            # Exponential map at the origin of the unit-curvature Poincare ball.
            norm = tangent.norm(p=2, dim=-1, keepdim=True).clamp_min(self.eps)
            projected = torch.tanh(norm) * tangent / norm
            projected_norm = projected.norm(p=2, dim=-1, keepdim=True)
            scale = ((1.0 - self.eps)
                     / projected_norm.clamp_min(self.eps)).clamp(max=1.0)
            projected = projected * scale
        if not torch.isfinite(projected).all():
            raise FloatingPointError(
                "nonfinite {} projection".format(self.geometry))
        return projected


def wheel_prototypes(class_angles, feature_dim, geometry="poincare",
                     radius=0.75, eps=1e-5):
    """Embed a fixed semantic wheel in a two-dimensional subspace of R^D.

    Spherical prototypes must have unit norm.  Euclidean and Poincare
    prototypes use ``radius``; Poincare additionally requires radius < 1.
    The remaining dimensions are an orthogonal residual subspace available to
    projected samples, while the semantic prototype plane stays identical.
    """
    _validate_geometry(geometry)
    if class_angles.ndim != 1 or class_angles.numel() < 2:
        raise ValueError("class_angles must contain at least two classes")
    if feature_dim < 2:
        raise ValueError("emotion-wheel prototypes require at least 2 dimensions")
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("prototype radius must be finite and positive")
    if geometry == "poincare" and radius >= 1.0 - eps:
        raise ValueError("Poincare prototype radius must be in (0, 1-eps)")
    effective_radius = 1.0 if geometry == "spherical" else float(radius)
    prototypes = class_angles.new_zeros(class_angles.numel(), feature_dim)
    prototypes[:, 0] = effective_radius * torch.cos(class_angles)
    prototypes[:, 1] = effective_radius * torch.sin(class_angles)
    return prototypes


def geometry_distance(first, second, geometry="poincare", eps=1e-5):
    """Distance with broadcasting over all leading dimensions."""
    _validate_geometry(geometry)
    if first.size(-1) != second.size(-1):
        raise ValueError("geometry vectors must have the same final dimension")
    if geometry == "euclidean":
        distance = (first - second).norm(p=2, dim=-1)
    elif geometry == "spherical":
        first_unit = first / first.norm(
            p=2, dim=-1, keepdim=True).clamp_min(eps)
        second_unit = second / second.norm(
            p=2, dim=-1, keepdim=True).clamp_min(eps)
        cosine = (first_unit * second_unit).sum(dim=-1)
        distance = torch.acos(cosine.clamp(-1.0 + eps, 1.0 - eps))
    else:
        first_sq = (first * first).sum(dim=-1).clamp(max=1.0 - eps)
        second_sq = (second * second).sum(dim=-1).clamp(max=1.0 - eps)
        difference_sq = ((first - second) ** 2).sum(dim=-1)
        denominator = ((1.0 - first_sq)
                       * (1.0 - second_sq)).clamp_min(eps)
        argument = (1.0 + 2.0 * difference_sq / denominator).clamp_min(
            1.0 + eps)
        distance = torch.acosh(argument)
    if not torch.isfinite(distance).all():
        raise FloatingPointError(
            "nonfinite {} distance".format(geometry))
    return distance


def geometry_prototype_outputs(features, prototypes, geometry="poincare",
                               temperature=1.0, eps=1e-5):
    """Return prototype distances, logits, predictions, and typicality."""
    if features.ndim != 2 or prototypes.ndim != 2:
        raise ValueError("features and prototypes must be two-dimensional")
    if features.size(-1) != prototypes.size(-1):
        raise ValueError("features and prototypes must share their final dimension")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("wheel temperature must be finite and positive")
    distances = geometry_distance(
        features.unsqueeze(1), prototypes.unsqueeze(0), geometry, eps)
    nearest_distance, predictions = distances.min(dim=-1)
    logits = -distances / temperature
    typicality = torch.exp(-nearest_distance / temperature).clamp(0.0, 1.0)
    return distances, logits, predictions, typicality


def geometry_cpcc_loss(features, labels, geometry="poincare", eps=1e-8,
                       geometry_eps=1e-5, class_distance_matrix=None):
    """Return 1 minus the pairwise geometry/label-distance correlation."""
    if features.ndim != 2 or labels.ndim != 1:
        raise ValueError("CPCC expects [N,D] features and [N] labels")
    if features.size(0) != labels.numel():
        raise ValueError("CPCC feature and label counts differ")
    if features.size(0) < 2:
        return features.sum() * 0.0
    row, column = torch.triu_indices(
        features.size(0), features.size(0), offset=1, device=features.device)
    geometric = geometry_distance(
        features[row], features[column], geometry, geometry_eps)
    if class_distance_matrix is None:
        categorical = (labels[row] != labels[column]).type_as(geometric)
    else:
        if class_distance_matrix.ndim != 2 or (
                class_distance_matrix.size(0) != class_distance_matrix.size(1)):
            raise ValueError("class distance matrix must be square")
        if labels.numel() and (
                labels.min() < 0 or labels.max() >= class_distance_matrix.size(0)):
            raise ValueError("labels exceed the class distance matrix")
        categorical = class_distance_matrix[
            labels[row], labels[column]].type_as(geometric)
    geometric = geometric - geometric.mean()
    categorical = categorical - categorical.mean()
    denominator = (geometric.square().sum().sqrt()
                   * categorical.square().sum().sqrt())
    if denominator.detach().item() <= eps:
        return geometric.sum() * 0.0
    correlation = (geometric * categorical).sum() / denominator.clamp_min(eps)
    return 1.0 - correlation.clamp(-1.0, 1.0)
