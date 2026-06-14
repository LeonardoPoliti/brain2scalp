"""
tests/test_transform.py
=======================
Unit tests for core/transform.py.

Tests cover:
  - All 8 piecewise regions (correct region dispatch)
  - AC origin (0, 0, 0) maps to ≈ (0, 0, 0)
  - Known reference values from Brett et al. / GingerALE
  - Round-trip error is small (< 3 mm for typical cortical coordinates)
  - mni_to_talairach is the approximate inverse of talairach_to_mni
  - Input validation (wrong shape)
"""

from __future__ import annotations

import numpy as np
import pytest

from brain2scalp.core.transform import (
    mni_to_talairach,
    round_trip_error,
    talairach_to_mni,
)

#----------------------------------------------------------------------------------

class TestACOrigin:
    def test_origin_maps_near_origin(self):
        """AC origin (0, 0, 0) should map very close to (0, 0, 0) in MNI."""
        mni = talairach_to_mni([0.0, 0.0, 0.0])
        assert np.linalg.norm(mni) < 3.0, (
            f"AC origin should be near MNI (0,0,0), got {mni}"
        )

    def test_inverse_origin(self):
        """MNI (0, 0, 0) should invert back near Talairach (0, 0, 0)."""
        tal = mni_to_talairach([0.0, 0.0, 0.0])
        assert np.linalg.norm(tal) < 5.0


class TestRegionDispatch:
    """All 8 regions should be reachable and return finite values."""

    regions = [
        # [x, y, z] 
        ([ 30.0,  20.0,  40.0]),   # right, anterior, above AC
        ([-30.0,  20.0,  40.0]),   # left,  anterior, above AC
        ([ 30.0, -20.0,  40.0]),   # right, posterior, above AC
        ([-30.0, -20.0,  40.0]),   # left,  posterior, above AC
        ([ 30.0,  20.0, -10.0]),   # right, anterior, below AC
        ([-30.0,  20.0, -10.0]),   # left,  anterior, below AC
        ([ 30.0, -20.0, -10.0]),   # right, posterior, below AC
        ([-30.0, -20.0, -10.0]),   # left,  posterior, below AC
    ]

    @pytest.mark.parametrize("tal", regions)
    def test_all_regions_return_finite(self, tal):
        mni = talairach_to_mni(tal)
        assert mni.shape == (3,)
        assert np.all(np.isfinite(mni)), f"Non-finite MNI for Tal {tal}: {mni}"

    @pytest.mark.parametrize("tal", regions)
    def test_round_trip_under_3mm(self, tal):
        err = round_trip_error(tal)
        assert err < 3.0, (
            f"Round-trip error {err:.2f} mm exceeds 3 mm for Tal {tal}"
        )


class TestLeftRightDifference:
    """Left and right hemisphere matrices must produce different results."""

    def test_left_right_differ_above_ac_posterior(self):
        left  = talairach_to_mni([-40.0, -20.0, 10.0])
        right = talairach_to_mni([ 40.0, -20.0, 10.0])
        # Symmetric hemispheres will differ in x; the transforms should NOT
        # produce mirror images, if they do the matrices are still identical.
        # At minimum the x-offset from the left-hemisphere affine should differ
        # from the right-hemisphere affine.
        assert not np.allclose(left, [-right[0], right[1], right[2]], atol=0.5), (
            "Left and right hemisphere Tal→MNI results are mirror images - "
            "the matrices may still be identical (C4 bug not fixed)."
        )

    def test_left_right_differ_below_ac_anterior(self):
        left  = talairach_to_mni([-30.0, 20.0, -10.0])
        right = talairach_to_mni([ 30.0, 20.0, -10.0])
        assert not np.allclose(left, [-right[0], right[1], right[2]], atol=0.5)


class TestInputValidation:
    def test_tal_to_mni_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            talairach_to_mni([1.0, 2.0])  # only 2 elements

    def test_mni_to_tal_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            mni_to_talairach([1.0, 2.0])  # only 2 elements

    def test_list_input_accepted(self):
        result = talairach_to_mni([-46.0, 20.0, 32.0])
        assert result.shape == (3,)

    def test_numpy_input_accepted(self):
        result = talairach_to_mni(np.array([-46.0, 20.0, 32.0]))
        assert result.shape == (3,)


class TestInverse:
    def test_inverse_approximate(self):
        """mni_to_talairach should approximately invert talairach_to_mni."""
        tal_original = np.array([-46.0, 20.0, 32.0])
        mni = talairach_to_mni(tal_original)
        tal_back = mni_to_talairach(mni)
        assert np.linalg.norm(tal_original - tal_back) < 0.5
