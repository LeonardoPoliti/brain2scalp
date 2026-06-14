"""
tests/test_projection.py
=========================
Unit tests for core/projection.py using a synthetic spherical scalp surface.

The ground truth is a sphere of radius R centred at the origin.  For any
cortical target inside the sphere the expected scalp entry point is the
sphere surface along the outward ray from the target.
"""

from __future__ import annotations

import numpy as np
import pytest

from brain2scalp.core.projection import find_scalp_entry

#---------------------------------------------------------------------------

def _make_sphere_surface_and_mask(
    radius: float = 80.0,
    n_theta: int = 40,
    n_phi: int = 40,
    voxel_size: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (scalp_surface_mni, filled_mask, affine) for a sphere.

    scalp_surface_mni: (N, 3) array of surface point coordinates in mm.
    filled_mask: binary 3-D array with a filled sphere.
    affine: 1 mm isotropic identity-like affine.
    """
    # Build a point cloud on the sphere surface
    theta = np.linspace(0, np.pi, n_theta)
    phi   = np.linspace(0, 2 * np.pi, n_phi)
    TH, PH = np.meshgrid(theta, phi)
    x = radius * np.sin(TH) * np.cos(PH)
    y = radius * np.sin(TH) * np.sin(PH)
    z = radius * np.cos(TH)
    surface = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)

    # Build a small filled mask
    side = int(2 * radius / voxel_size) + 4
    mask = np.zeros((side, side, side), dtype=np.uint8)
    centre = side // 2
    ii, jj, kk = np.ogrid[:side, :side, :side]
    mask[(ii - centre) ** 2 + (jj - centre) ** 2 + (kk - centre) ** 2 <= (radius / voxel_size) ** 2] = 1

    affine = np.eye(4)
    affine[:3, 3] = -centre * voxel_size

    return surface, mask, affine


class TestRayCastProjection:
    @classmethod
    def setup_class(cls):
        cls.surface, cls.mask, cls.affine = _make_sphere_surface_and_mask(
            radius=80.0, n_theta=60, n_phi=60
        )

    def test_ray_cast_used_for_interior_target(self):
        """A target well inside the sphere should use the ray-cast method."""
        target = [0.0, 0.0, 40.0]  # clearly inside
        entry, dist, method = find_scalp_entry(
            target, self.surface, self.affine
        )
        assert method == "ray_cast", f"Expected ray_cast, got {method}"

    def test_entry_on_surface(self):
        """Entry point should be close to the sphere surface (within ~2 voxels)."""
        target = [0.0, 0.0, 40.0]
        entry, _, _ = find_scalp_entry(
            target, self.surface, self.affine
        )
        dist_from_centre = float(np.linalg.norm(entry))
        assert abs(dist_from_centre - 80.0) < 4.0, (
            f"Entry point at radius {dist_from_centre:.1f} mm, expected ~80 mm"
        )

    def test_correct_hemisphere(self):
        """Positive-z target should map to positive-z scalp hemisphere."""
        target = [0.0, 0.0, 40.0]
        entry, _, _ = find_scalp_entry(
            target, self.surface, self.affine
        )
        assert entry[2] > 0, (
            f"Expected positive-z scalp entry for positive-z target, got z={entry[2]:.1f}"
        )

    def test_negative_z_target_maps_to_negative_z_scalp(self):
        target = [0.0, 0.0, -40.0]
        entry, _, _ = find_scalp_entry(
            target, self.surface, self.affine
        )
        assert entry[2] < 0, (
            f"Expected negative-z scalp entry for negative-z target, got z={entry[2]:.1f}"
        )

    def test_left_target_maps_to_left_scalp(self):
        """Negative-x (left hemisphere) target → negative-x scalp entry."""
        target = [-40.0, 0.0, 0.0]
        entry, _, _ = find_scalp_entry(
            target, self.surface, self.affine
        )
        assert entry[0] < 0, (
            f"Expected negative-x scalp entry for left-hemisphere target, got x={entry[0]:.1f}"
        )

    def test_return_types(self):
        entry, dist, method = find_scalp_entry([0.0, 0.0, 40.0], self.surface, self.affine)
        assert len(entry) == 3
        assert isinstance(float(dist), float)
        assert isinstance(method, str)

    def test_entry_always_finite_for_exterior_target(self):
        """A target outside the scalp surface should still yield valid coords."""
        target = [0.0, 0.0, 200.0]  # well beyond sphere radius 80
        entry, dist, _ = find_scalp_entry(target, self.surface, self.affine)
        assert np.all(np.isfinite(entry)), f"Non-finite entry point: {entry}"
        assert dist >= 0

    def test_distance_sensible(self):
        """Distance should be approximately R - |target| for a radial target."""
        target_r = 40.0  # 40 mm from centre
        target = [0.0, 0.0, target_r]
        _, dist, _ = find_scalp_entry(
            target, self.surface, self.affine
        )
        expected_dist = 80.0 - target_r  # sphere radius minus target radius
        assert abs(dist - expected_dist) < 5.0, (
            f"Expected ~{expected_dist:.1f} mm, got {dist:.1f} mm"
        )
