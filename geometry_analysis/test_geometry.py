import unittest

import torch

from geometry_analysis.geometry import (GEOMETRIES, GeometryProjector,
                                        geometry_cpcc_loss,
                                        geometry_distance,
                                        geometry_prototype_outputs,
                                        wheel_prototypes)


class GeometryTest(unittest.TestCase):
    def test_projector_constraints_and_gradients(self):
        source = torch.randn(7, 5)
        for geometry in GEOMETRIES:
            projector = GeometryProjector(5, 4, geometry)
            projected = projector(source)
            self.assertEqual(projected.shape, (7, 4))
            self.assertTrue(torch.isfinite(projected).all())
            if geometry == "spherical":
                torch.testing.assert_close(
                    projected.norm(dim=-1), torch.ones(7),
                    atol=1e-5, rtol=1e-5)
            if geometry == "poincare":
                self.assertTrue((projected.norm(dim=-1) < 1).all())
            projected.square().mean().backward()
            self.assertIsNotNone(projector.linear.weight.grad)

    def test_each_geometry_recovers_its_own_prototypes(self):
        angles = torch.arange(6) * (2 * torch.pi / 6)
        for geometry in GEOMETRIES:
            prototypes = wheel_prototypes(
                angles, 16, geometry, radius=0.75)
            _, logits, predictions, _ = geometry_prototype_outputs(
                prototypes, prototypes, geometry, temperature=1.0)
            torch.testing.assert_close(predictions, torch.arange(6))
            torch.testing.assert_close(
                logits.argmax(dim=-1), torch.arange(6))
            if geometry == "spherical":
                torch.testing.assert_close(
                    prototypes.norm(dim=-1), torch.ones(6))

    def test_cpcc_is_finite_and_backpropagates(self):
        labels = torch.tensor([0, 1, 2, 3, 0, 2])
        class_distance = torch.rand(4, 4)
        class_distance = (class_distance + class_distance.t()) / 2
        class_distance.fill_diagonal_(0)
        for geometry in GEOMETRIES:
            projector = GeometryProjector(5, 4, geometry)
            features = projector(torch.randn(6, 5))
            loss = geometry_cpcc_loss(
                features, labels, geometry,
                class_distance_matrix=class_distance)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertIsNotNone(projector.linear.weight.grad)

    def test_distances_are_symmetric(self):
        for geometry in GEOMETRIES:
            projector = GeometryProjector(3, 3, geometry)
            points = projector(torch.randn(4, 3))
            matrix = geometry_distance(
                points[:, None], points[None, :], geometry)
            torch.testing.assert_close(matrix, matrix.t())


if __name__ == "__main__":
    unittest.main()
