"""Geometry-controlled emotion-wheel ablations for SDT."""

from .geometry import (GEOMETRIES, GeometryProjector, geometry_cpcc_loss,
                       geometry_distance, geometry_prototype_outputs,
                       wheel_prototypes)

__all__ = [
    "GEOMETRIES",
    "GeometryProjector",
    "geometry_cpcc_loss",
    "geometry_distance",
    "geometry_prototype_outputs",
    "wheel_prototypes",
]
