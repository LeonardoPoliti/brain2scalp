"""
core/atlas.py
-----------------
NIfTI scan loading, orientation normalization, and validation.

"""

from __future__ import annotations

import warnings
from pathlib import Path

import nibabel as nib
import numpy as np

from brain2scalp import _ISSUES_URL

#------------------------------------------------------------------------------

# Minimum ratio of lateral-border signal to global max required for a full-head template.
# 4 % of global_max is a conservative lower bound derived from tissue contrast:
#   T1 full-head : scalp fat 50–90 % of global_max
#   T2 full-head : scalp/bone 5–15 % of global_max 
#   Brain-extracted (any modality): border ≈ 0–1 %  
_LATERAL_SIGNAL_RATIO = 0.04

# Fraction of each axis dimension used as the lateral border slab for Check 3.
# ≥3 voxels at 1 mm resolution. Robust to the 1–3 voxel zero-padding in NIfTI files.
_BORDER_FRAC = 0.03

# Minimum number of distinct integer-rounded intensity levels expected in a continuous MRI volume.
# Label atlases and binary masks have far fewer unique values (typically <10 or <100).
# A continuous T1/T2 has thousands even at low resolution.
_MIN_UNIQUE_INTENSITIES = 100

# Minimum ratio of voxel signal to global max required in the inferior slab (Check 2).
# Brain-extracted volumes have no tissue below ~-50 mm MNI (≈ 0 signal).
# Any real head template has skull/scalp there at well above 2 % of global max.
_INFERIOR_SIGNAL_RATIO = 0.02

class AtlasError(ValueError):
    """Raised when the NIfTI file fails validation."""


def load_atlas(
    path: str | Path,
    *,
    require_scalp: bool = True,
    fov_z_min_mm: float = 50.0,
    warn_native_space: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a NIfTI HEAD template, reorient to RAS canonical, validate.

    Params:
    path : str or Path
        Path to a .nii or .nii.gz file.
    require_scalp : bool
        If True, raise AtlasError when the volume does not look like a
        full-head template.
    fov_z_min_mm : float
        MNI z threshold (mm) below which real signal must exist.

    Returns:
    data : np.ndarray
        Voxel intensity data (float64), shape matches the canonical header.
    affine : np.ndarray
        4×4 voxel-to-mm affine for the returned data (canonical orientation).

    Raises:
    FileNotFoundError
        If the path does not exist.
    AtlasError
        If the volume fails validation (no scalp signal, wrong FOV, …).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"NIfTI file not found: {path}")

    img = nib.load(str(path))

    # qform / sform sanity check
    qform_code = int(img.header.get("qform_code", 0))  # type: ignore[arg-type]
    sform_code = int(img.header.get("sform_code", 0))  # type: ignore[arg-type]
    best_code = sform_code if sform_code > 0 else qform_code
    if best_code == 0:
        warnings.warn(
            f"{path.name}: both qform_code and sform_code are 0. "
            "The affine may be the identity matrix.  Results may be wrong.",
            stacklevel=2,
        )
    elif best_code == 1 and warn_native_space:
        # code 1 = scanner-native space (not aligned to any standard space)
        warnings.warn(
            f"{path.name}: xform_code = 1 (scanner-native space). "
            "Brain target coordinates are assumed to be in MNI152 mm space, "
            "but this volume's affine maps to native scanner coordinates.  "
            "Use --space native if your target is also in native scanner space, "
            "or register the scan to MNI152 space first.",
            stacklevel=2,
        )

    # reorient to RAS canonical
    # Ensures axis 2 is always the inferior–superior direction.
    canonical = nib.as_closest_canonical(img)
    data = canonical.get_fdata(dtype=np.float64)
    affine = canonical.affine

    # Collapse 4-D volumes to 3-D by taking the first volume.  
    if data.ndim == 4:
        warnings.warn(
            f"{path.name}: 4-D NIfTI detected ({data.shape}). "
            "Using first volume (index 0).  For best results supply a "
            "single 3-D image.",
            stacklevel=2,
        )
        data = data[..., 0]
    elif data.ndim != 3:
        raise AtlasError(
            f"Expected a 3-D NIfTI volume but got shape {data.shape}.  "
            "Please supply a single full-head template.  "
            f"If the problem persists, open an issue at {_ISSUES_URL}"
        )

    if require_scalp:
        _validate_head_template(data, affine, fov_z_min_mm)

    return data, affine


def _validate_head_template(
    data: np.ndarray,
    affine: np.ndarray,
    fov_z_min_mm: float,
) -> None:
    """
    Raise AtlasError when the volume does not look like a full-head template.

    Three checks:
    1. Volume is non-empty and has a continuous intensity distribution
       (not a binary mask or integer label atlas).
    2. Real MRI signal exists below -fov_z_min_mm in MNI space. 
    3. A proportional border slab (not just the outermost face) carries scalp
       signal, catching brain-extracted volumes regardless of zero-padding.
    """
    nx, ny, nz = data.shape

    # check 1a: non-empty
    global_max = float(data.max())
    if global_max == 0.0:
        raise AtlasError(
            "Atlas is empty (all voxels are zero).  "
            "Please supply an MNI HEAD template (e.g. "
            "mni_icbm152_t1_tal_nlin_sym_09c.nii).  "
            f"If the problem persists, open an issue at {_ISSUES_URL}"
        )

    # check 1b: continuous image, not a binary mask or label atlas
    n_unique = int(np.unique(data.round()).size)
    if n_unique < _MIN_UNIQUE_INTENSITIES:
        raise AtlasError(
            f"Atlas has only {n_unique} distinct intensity levels.  "
            "This looks like a binary mask or label atlas, not a head template.  "
            "Please supply an MNI HEAD template (e.g. "
            "mni_icbm152_t1_tal_nlin_sym_09c.nii).  "
            f"If the problem persists, open an issue at {_ISSUES_URL}"
        )

    # check 2: real signal must exist below -fov_z_min_mm in MNI space
    # Compute the MNI z-coordinate for the centre voxel of each axial slice
    # and then check peak signal in the inferior slices.
    ci, cj = nx // 2, ny // 2
    z_per_k = (
        affine[2, 0] * ci
        + affine[2, 1] * cj
        + affine[2, 2] * np.arange(nz)
        + affine[2, 3]
    )
    inferior_k = np.where(z_per_k < -fov_z_min_mm)[0]
    if inferior_k.size == 0 or float(data[:, :, inferior_k].max()) < global_max * _INFERIOR_SIGNAL_RATIO:
        raise AtlasError(
            f"No signal found below -{fov_z_min_mm:.0f} mm MNI.  "
            "Please supply a full-head template with intact scalp and skull.  "
            f"If the problem persists, open an issue at {_ISSUES_URL}"
        )

    # check 3: lateral border signal
    bx = max(3, round(nx * _BORDER_FRAC))
    by = max(3, round(ny * _BORDER_FRAC))
    lateral_max = max(
        float(data[:bx,  :, :].max()),   # left  slab
        float(data[-bx:, :, :].max()),   # right slab
        float(data[:, :by,  :].max()),   # posterior slab
        float(data[:, -by:, :].max()),   # anterior slab
    )
    if lateral_max < global_max * _LATERAL_SIGNAL_RATIO:
        raise AtlasError(
            f"No signal at the volume's lateral borders "
            f"(lateral_max = {lateral_max:.1f}, global_max = {global_max:.1f}, "
            f"ratio = {lateral_max / global_max:.1%}).  "
            "This looks like a brain-extracted volume.  "
            "Please supply a full-head template (T1 or T2) with intact scalp "
            "and skull.  If you are certain the volume is full-head, pass "
            "--threshold explicitly to skip auto-detection.  "
            f"If the problem persists, open an issue at {_ISSUES_URL}"
        )


def voxel_size_mm(affine: np.ndarray) -> tuple[float, float, float]:
    """Return (dx, dy, dz) voxel dimensions in mm from an affine matrix."""
    zooms = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    return float(zooms[0]), float(zooms[1]), float(zooms[2])


def voxel_to_mni(voxel_coords: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """
    Convert an (N, 3) array of voxel indices to MNI mm coordinates.

    Params:
    voxel_coords : np.ndarray, shape (N, 3)
    affine : np.ndarray, shape (4, 4)

    Returns:
    np.ndarray, shape (N, 3)
    """
    vox_h = np.hstack([voxel_coords, np.ones((len(voxel_coords), 1))])
    return (affine @ vox_h.T).T[:, :3]


def mni_to_voxel_continuous(
    mni_coords: np.ndarray, affine: np.ndarray
) -> np.ndarray:
    """
    Convert MNI mm coordinates to continuous (non-rounded) voxel indices.

    Params:
    mni_coords : np.ndarray, shape (N, 3) or (3,)
    affine : np.ndarray, shape (4, 4)

    Returns:
    np.ndarray, shape (N, 3), float64
    """
    coords = np.atleast_2d(mni_coords).astype(float)
    inv = np.linalg.inv(affine)
    mni_h = np.hstack([coords, np.ones((len(coords), 1))])
    return (inv @ mni_h.T).T[:, :3]


def mni_to_voxel(mni_coords: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """
    Convert MNI mm coordinates to integer voxel indices (nearest voxel).
    """
    return np.round(mni_to_voxel_continuous(mni_coords, affine)).astype(int)
