"""
core/transform.py
---------------------
Coordinate-space transforms between Talairach and MNI152.

Implements the nonlinear piecewise-affine transform from:

    Lancaster JL et al. (2007)
    "Bias between MNI and Talairach coordinates analyzed using the
     ICBM-152 brain template."
    Human Brain Mapping 28:1194–1205.
    https://doi.org/10.1002/hbm.20345

The 8 piecewise regions are defined by three binary predicates:
    z ≥ 0  (above AC) vs  z < 0  (below AC)
    y ≥ 0  (anterior) vs  y < 0  (posterior)
    x ≥ 0  (right)    vs  x < 0  (left)
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Layout: _TAL2MNI[z_above][y_anterior][x_right] → 4×4 affine
# z_above  : True = z ≥ 0 (above AC),  False = z < 0 (below AC)
# y_anterior: True = y ≥ 0 (anterior), False = y < 0 (posterior)
# x_right  : True = x ≥ 0 (right),     False = x < 0 (left)

_TAL2MNI: dict[tuple[bool, bool, bool], np.ndarray] = {
    # Above AC, posterior
    (True, False, True): np.array(
        [[ 0.9900,  0.0000,  0.0000,  -0.3900],
         [ 0.0000,  0.9688,  0.0460,  -1.4030],
         [ 0.0000, -0.0460,  0.9688,   1.7476],
         [ 0.0000,  0.0000,  0.0000,   1.0000]]
    ),  # right hemisphere, above AC, posterior
    (True, False, False): np.array(
        [[ 0.9254,  0.0024, -0.0118,  -1.0207],
         [-0.0048,  0.9316, -0.0871,  -1.7667],
         [ 0.0152,  0.0883,  0.8924,   4.0926],
         [ 0.0000,  0.0000,  0.0000,   1.0000]]
    ),  # left hemisphere, above AC, posterior

    # Above AC, anterior
    (True, True, True): np.array(
        [[ 0.9900,  0.0000,  0.0000,  -0.3900],
         [ 0.0000,  0.9700,  0.0420,  -1.1400],
         [ 0.0000, -0.0420,  0.9700,   1.6500],
         [ 0.0000,  0.0000,  0.0000,   1.0000]]
    ),  # right hemisphere, above AC, anterior
    (True, True, False): np.array(
        [[ 0.9381,  0.0057, -0.0198,  -0.9626],
         [ 0.0029,  0.9210, -0.0605,  -1.0153],
         [ 0.0143,  0.0616,  0.9333,   1.6655],
         [ 0.0000,  0.0000,  0.0000,   1.0000]]
    ),  # left hemisphere, above AC, anterior

    # Below AC, posterior
    (False, False, True): np.array(
        [[ 0.9900,  0.0000,  0.0000,  -0.3900],
         [ 0.0000,  0.9688,  0.0420,  -1.3680],
         [ 0.0000, -0.0420,  0.9688,   1.7100],
         [ 0.0000,  0.0000,  0.0000,   1.0000]]
    ),  # right hemisphere, below AC, posterior
    (False, False, False): np.array(
        [[ 0.8967,  0.0029, -0.0026,  -1.0680],
         [ 0.0027,  0.9020,  0.0502,  -1.0239],
         [ 0.0026, -0.0513,  0.8948,   3.1883],
         [ 0.0000,  0.0000,  0.0000,   1.0000]]
    ),  # left hemisphere, below AC, posterior

    # Below AC, anterior
    (False, True, True): np.array(
        [[ 0.9900,  0.0000,  0.0000,  -0.3900],
         [ 0.0000,  0.9700,  0.0790,  -0.7600],
         [ 0.0000, -0.0790,  0.9700,   3.1200],
         [ 0.0000,  0.0000,  0.0000,   1.0000]]
    ),  # right hemisphere, below AC, anterior
    (False, True, False): np.array(
        [[ 0.9260,  0.0057, -0.0106,  -0.9399],
         [ 0.0021,  0.9193,  0.0740,  -0.7583],
         [ 0.0106, -0.0742,  0.8823,   4.8489],
         [ 0.0000,  0.0000,  0.0000,   1.0000]]
    ),  # left hemisphere, below AC, anterior
}

# Pre-compute inverses once at import time
_MNI2TAL: dict[tuple[bool, bool, bool], np.ndarray] = {
    k: np.linalg.inv(v) for k, v in _TAL2MNI.items()
}


def _region_key(x: float, y: float, z: float) -> tuple[bool, bool, bool]:
    """Return the (z_above, y_anterior, x_right) lookup key."""
    return (z >= 0.0, y >= 0.0, x >= 0.0)


def talairach_to_mni(tal_coords: list[float] | np.ndarray) -> np.ndarray:
    """
    Convert Talairach coordinates to MNI152 space. Uses the nonlinear 
    piecewise-affine transform of Lancaster et al. (2007) with distinct 
    left/right hemisphere matrices. 

    Params:
    tal_coords : array-like, shape (3,)
        [x, y, z] in Talairach space (mm).

    Returns:
    np.ndarray, shape (3,)
        Coordinates in MNI152 space (mm).
    """
    tal = np.asarray(tal_coords, dtype=float)
    if tal.shape != (3,):
        raise ValueError(f"Expected shape (3,), got {tal.shape}")
    x, y, z = tal
    key = _region_key(x, y, z)
    matrix = _TAL2MNI[key]
    mni_h = matrix @ np.array([x, y, z, 1.0])
    return mni_h[:3]


def mni_to_talairach(mni_coords: list[float] | np.ndarray) -> np.ndarray:
    """
    Convert MNI152 coordinates to Talairach space.

    Uses iterative region refinement: starts with the MNI-region inverse,
    then re-keys using the Talairach region of each estimate until the region
    stabilizes.

    Params:
    mni_coords : array-like, shape (3,)
        [x, y, z] in MNI152 space (mm).

    Returns:
    np.ndarray, shape (3,)
        Coordinates in Talairach space (mm).
    """
    mni = np.asarray(mni_coords, dtype=float)
    if mni.shape != (3,):
        raise ValueError(f"Expected shape (3,), got {mni.shape}")
    mni_h = np.array([*mni, 1.0])
    key = _region_key(*mni)
    tal = (_MNI2TAL[key] @ mni_h)[:3]
    for _ in range(10):
        new_key = _region_key(*tal)
        if new_key == key:
            break
        key = new_key
        tal = (_MNI2TAL[key] @ mni_h)[:3]
    return tal


def round_trip_error(tal_coords: list[float] | np.ndarray) -> float:
    """
    Return the round-trip error (mm) for a Talairach point.

    Converts Talairach → MNI → Talairach and reports the L2 distance between
    the original and reconstructed point.
    """
    tal = np.asarray(tal_coords, dtype=float)
    reconstructed = mni_to_talairach(talairach_to_mni(tal))
    return float(np.linalg.norm(tal - reconstructed))
