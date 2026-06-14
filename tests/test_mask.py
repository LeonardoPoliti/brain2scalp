"""
tests/test_mask.py
==================
Unit tests for core/mask.py using a synthetic spherical phantom.
"""

from __future__ import annotations

import numpy as np
import pytest

from brain2scalp.core.mask import (
    build_head_mask,
    erode_mask,
    extract_surface,
    fill_holes,
)

#------------------------------------------------------------------------------

def _make_sphere_volume(
    shape: tuple[int, int, int] = (60, 60, 60),
    radius: int = 20,
    center: tuple[int, int, int] | None = None,
    background: float = 2.0,
    signal: float = 100.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a synthetic 3-D volume with a solid sphere and a 1 mm³ affine.

    Returns (data, affine).
    """
    if center is None:
        center = (shape[0] // 2, shape[1] // 2, shape[2] // 2)

    data = np.full(shape, background, dtype=float)
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    dist = np.sqrt(
        (xx - center[2]) ** 2
        + (yy - center[1]) ** 2
        + (zz - center[0]) ** 2
    )
    data[dist <= radius] = signal
    affine = np.eye(4)  # 1 mm isotropic
    return data, affine


class TestBuildHeadMask:
    def test_returns_binary_array(self):
        data, _ = _make_sphere_volume()
        mask, _ = build_head_mask(data)
        unique = set(np.unique(mask).tolist())
        assert unique.issubset({0, 1}), f"Mask contains non-binary values: {unique}"

    def test_sphere_detected(self):
        data, _ = _make_sphere_volume()
        mask, _ = build_head_mask(data)
        assert mask.sum() > 0
        assert mask.max() == 1

    def test_explicit_threshold(self):
        data, _ = _make_sphere_volume(signal=200.0)
        mask, used = build_head_mask(data, threshold=50.0)
        assert used == 50.0
        assert mask.sum() > 0

    def test_background_excluded(self):
        data, _ = _make_sphere_volume(background=2.0, signal=100.0)
        mask, _ = build_head_mask(data)
        # Background voxels at the image corners should be 0
        assert mask[0, 0, 0] == 0
        assert mask[-1, -1, -1] == 0

    def test_auto_threshold_scales_with_intensity(self):
        """Doubling signal intensities should not break auto-threshold."""
        data_lo, _ = _make_sphere_volume(background=2.0, signal=100.0)
        data_hi, _ = _make_sphere_volume(background=20.0, signal=1000.0)
        mask_lo, _ = build_head_mask(data_lo)
        mask_hi, _ = build_head_mask(data_hi)
        # Both should detect a head
        assert mask_lo.sum() > 0
        assert mask_hi.sum() > 0

    def test_all_zeros_raises(self):
        data = np.zeros((30, 30, 30))
        with pytest.raises(ValueError):
            build_head_mask(data, threshold=1.0)


class TestFillHoles:
    def test_hollow_sphere_filled(self):
        """A hollow sphere should be fully filled after fill_holes."""
        shape = (60, 60, 60)
        r_outer, r_inner = 20, 15
        center = (30, 30, 30)
        data = np.zeros(shape)
        zz, yy, xx = np.ogrid[:60, :60, :60]
        dist = np.sqrt((xx - 30) ** 2 + (yy - 30) ** 2 + (zz - 30) ** 2)
        # Shell: between inner and outer radius
        data[(dist >= r_inner) & (dist <= r_outer)] = 1.0

        mask, _ = build_head_mask(data, threshold=0.5)
        filled = fill_holes(mask)

        # Interior voxels (dist < r_inner) should now be filled
        interior = dist < r_inner
        interior_filled = filled[interior].mean()
        assert interior_filled > 0.90, (
            f"Only {interior_filled*100:.1f}% of interior voxels were filled"
        )

    def test_fill_does_not_shrink(self):
        data, _ = _make_sphere_volume()
        mask, _ = build_head_mask(data)
        filled = fill_holes(mask)
        assert filled.sum() >= mask.sum()


class TestErodeMask:
    def test_eroded_is_subset_of_filled(self):
        data, affine = _make_sphere_volume()
        mask, _ = build_head_mask(data)
        filled = fill_holes(mask)
        eroded = erode_mask(filled, affine, surface_thickness_mm=1.5)
        # No voxel can be in eroded but not in filled
        assert np.all(eroded[filled == 0] == 0), \
            "Eroded mask has voxels outside the filled mask"

    def test_erode_shrinks_mask(self):
        data, affine = _make_sphere_volume()
        mask, _ = build_head_mask(data)
        filled = fill_holes(mask)
        eroded = erode_mask(filled, affine, surface_thickness_mm=1.0)
        assert eroded.sum() < filled.sum()

    def test_erode_thickness_scales_with_voxel_size(self):
        """Larger voxels → fewer iterations → relatively thicker surface remaining."""
        data, _ = _make_sphere_volume()
        mask, _ = build_head_mask(data)
        filled = fill_holes(mask)

        affine_1mm = np.eye(4)
        affine_2mm = np.diag([2.0, 2.0, 2.0, 1.0])

        eroded_1mm = erode_mask(filled, affine_1mm, surface_thickness_mm=2.0)
        eroded_2mm = erode_mask(filled, affine_2mm, surface_thickness_mm=2.0)

        # With 1 mm voxels: 2 mm → 2 erosion iterations (more erosion)
        # With 2 mm voxels: 2 mm → 1 erosion iteration (less erosion)
        assert eroded_2mm.sum() > eroded_1mm.sum()


class TestExtractSurface:
    def test_surface_is_shell(self):
        data, affine = _make_sphere_volume()
        mask, _ = build_head_mask(data)
        filled = fill_holes(mask)
        eroded = erode_mask(filled, affine)
        surface = extract_surface(filled, eroded)

        # Surface = filled AND NOT eroded
        expected = (filled & ~eroded).astype(np.uint8)
        np.testing.assert_array_equal(surface, expected)

    def test_surface_plus_interior_equals_filled(self):
        data, affine = _make_sphere_volume()
        mask, _ = build_head_mask(data)
        filled = fill_holes(mask)
        eroded = erode_mask(filled, affine)
        surface = extract_surface(filled, eroded)

        # surface | eroded should equal filled
        reconstructed = (surface | eroded).astype(np.uint8)
        np.testing.assert_array_equal(reconstructed, filled)
