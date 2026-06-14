"""
core/mask.py
------------------
Head-mask construction: threshold → largest component → fill holes →
erode → extract surface shell.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, label as nd_label

from brain2scalp import _ISSUES_URL
from skimage.filters import threshold_otsu
from skimage.morphology import convex_hull_image

#------------------------------------------------------------------------------

_MIN_SLICE_VOXELS = 10  # minimum voxels in a slice to run convex hull

def build_head_mask(
    data: np.ndarray,
    threshold: float | None = None,
) -> tuple[np.ndarray, float]:
    """
    Build a binary head mask from raw NIfTI data.

    Params:
    data : np.ndarray
        Voxel intensity data (canonical orientation, i.e. after atlas.load_atlas).
    threshold : float or None
        Explicit intensity threshold.  If None, it is derived automatically
        from the image border voxels.

    Returns:
    head_mask : np.ndarray, dtype uint8
        1 inside the head, 0 outside.  Only the largest connected component
        is retained to remove noise islands.
    threshold_used : float
        The threshold value that was applied.

    Notes Auto-threshold strategy:
    Otsu's method is used to find the optimal threshold between background
    and tissue.
    """
    if threshold is None:

        threshold = float(threshold_otsu(data))

    raw_mask = (data > threshold).astype(np.uint8)

    if raw_mask.sum() == 0:
        raise ValueError(
            f"build_head_mask: no voxels found above threshold {threshold:.4g}. "
            "Pass an explicit --threshold or make sure to pass a full-head scan.  "
            f"If the problem persists, open an issue at {_ISSUES_URL}"
        )

    # Keep only the largest connected component
    labeled, _ = nd_label(raw_mask)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0  # ignore background label
    head_mask = (labeled == int(sizes.argmax())).astype(np.uint8)
    return head_mask, float(threshold)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """
    Fill internal holes using per-slice convex hull (axial direction, axis 2).

    Params:
    mask : np.ndarray, dtype uint8, shape (nx, ny, nz)
        Binary head mask from build_head_mask.

    Returns:
    filled : np.ndarray, dtype uint8, same shape

    Notes:
    Convex hull per slice fills all concavities, regardless of background connectivity.  
    It requires the data to be in canonical orientation (axis 2 = inferior–superior), which
    atlas.load_atlas() guarantees via nib.as_closest_canonical().
    """
    filled = np.zeros_like(mask)
    for zi in range(mask.shape[2]):
        sl = mask[:, :, zi]
        if sl.sum() > _MIN_SLICE_VOXELS:
            filled[:, :, zi] = convex_hull_image(sl).astype(np.uint8)
        else:
            filled[:, :, zi] = sl
    return filled


def erode_mask(
    filled: np.ndarray,
    affine: np.ndarray,
    surface_thickness_mm: float = 1.5,
) -> np.ndarray:
    """
    Erode the filled mask by a fixed physical thickness.

    Params:
    filled : np.ndarray, dtype uint8
        Head mask from fill_holes.
    affine : np.ndarray, shape (4, 4)
        Voxel-to-mm affine.  Used to derive voxel size.
    surface_thickness_mm : float
        Desired surface-shell thickness in mm (default 1.5 mm).
        Converted to an integer number of erosion iterations based on the
        mean voxel size, with a minimum of 1.

    Returns:
    eroded : np.ndarray, dtype uint8
    """
    voxel_size = float(np.mean(np.sqrt((affine[:3, :3] ** 2).sum(axis=0))))
    iterations = max(1, round(surface_thickness_mm / voxel_size))
    return binary_erosion(filled, iterations=iterations).astype(np.uint8)


def extract_surface(filled: np.ndarray, eroded: np.ndarray) -> np.ndarray:
    """
    Extract the surface shell as filled XOR eroded.

    Params:
    filled : np.ndarray, dtype uint8
    eroded : np.ndarray, dtype uint8

    Returns:
    surface : np.ndarray, dtype uint8
        1 on the surface shell, 0 elsewhere.
    """
    return (filled & ~eroded).astype(np.uint8)
