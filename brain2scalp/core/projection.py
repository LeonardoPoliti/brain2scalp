"""
core/projection.py
---------------------------
Find the scalp entry point for a brain target coordinate.

Algo:
1.  Local-normal ray-cast:
    a.  Query the k-nearest scalp surface voxels to the brain target and
        take their centroid.  This estimates the scalp surface patch directly
        above the target.
    b.  Direction = centroid_of_k_nearest − brain_target  (outward).
    c.  Pre-compute all ray sample points as an (N, 3) array and batch-query
        the KD-tree in a single call.
    d.  Walk the first contiguous group of hits (samples within hit_tolerance_mm
        of a surface voxel) and return the last step of that group (the point
        where the ray exits the shell).
        Tolerance = voxel_size × √3/2 (half the space diagonal).

2.  Directed nearest-neighbour (fallback):
    Used when the ray misses the surface.  Restricts the search to the
    forward half-space of the ray direction so the result is never laterally
    displaced to the wrong side of the head.  Falls back to unconstrained
    nearest-neighbour only when no forward-facing voxels exist.
    After finding the nearest (inner-edge) surface voxel, _outer_edge_walk
    continues from that voxel along the ray direction to reach the outer skin.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

#------------------------------------------------------------------------------

_NN_CANDIDATE_POOL = 500  # candidates queried for directed nearest-neighbour fallback


def build_scalp_tree(scalp_surface_mni: np.ndarray) -> cKDTree:
    """Build a KD-tree from the scalp surface point cloud.
    """
    return cKDTree(scalp_surface_mni)


def find_scalp_entry(
    brain_target_mni: list[float] | np.ndarray,
    scalp_surface_mni: np.ndarray,
    affine: np.ndarray,
    ray_step_mm: float = 0.5,
    ray_max_mm: float = 100.0,
    k_local: int = 100,
    tree: cKDTree | None = None,
) -> tuple[np.ndarray, float, str]:
    """
    Find the scalp entry point for a brain target.

    Params:
    brain_target_mni : array-like, shape (3,)
        Brain target in MNI mm.
    scalp_surface_mni : np.ndarray, shape (N, 3)
        All scalp surface voxel centres in MNI mm.
    affine : np.ndarray, shape (4, 4)
        Voxel-to-mm affine (used to derive voxel size for hit tolerance).
    ray_step_mm : float
        Ray sampling step size (mm).  Must be ≤ hit_tolerance to avoid
        skipping surface voxels.  Default 0.5 mm.
    ray_max_mm : float
        Maximum ray length (mm).  100 mm covers any brain-to-scalp distance
        in MNI space.  Default 100 mm.
    k_local : int
        Number of nearest scalp voxels used to estimate the local ray
        direction (default 100).
    tree : cKDTree or None
        Pre-built KD-tree of scalp_surface_mni.  Built internally when None.
        Pass a pre-built tree when calling for multiple targets to avoid
        rebuilding it on every call.

    Returns:
    entry_mni : np.ndarray, shape (3,)
        Scalp entry point in MNI mm (outer/skin surface).
    distance_mm : float
        Distance from brain target to entry point (mm).
    method : str
        'ray_cast' or 'nearest_neighbor'.
    """
    target = np.asarray(brain_target_mni, dtype=float)
    if tree is None:
        tree = cKDTree(scalp_surface_mni)

    # Local ray direction
    k = min(k_local, len(scalp_surface_mni))
    _, near_idxs = tree.query(target, k=k)
    near_idxs = np.atleast_1d(near_idxs)
    local_centroid = scalp_surface_mni[near_idxs].mean(axis=0)

    direction = local_centroid - target
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        # Target sits exactly at the local scalp centroid.
        # Return nearest voxel without outer-edge walk.
        return _nearest_neighbor_directed(target, scalp_surface_mni, tree, unit=None)
    unit = direction / norm

    # hit_tolerance
    voxel_size = float(np.mean(np.sqrt((affine[:3, :3] ** 2).sum(axis=0))))
    hit_tolerance_mm = voxel_size * (3 ** 0.5 / 2)

    # Vectorised batch ray cast
    n_steps = int(ray_max_mm / ray_step_mm)
    steps = np.arange(n_steps, dtype=float)
    samples = target + unit * steps[:, np.newaxis] * ray_step_mm  # (N, 3)

    dists, idxs = tree.query(samples)
    hits = np.where(dists <= hit_tolerance_mm)[0]

    if len(hits):
        # hits[0] = ray enters the shell (inner edge).
        # Walk forward to the last step of the same crossing (outer/skin edge).
        # Gap tolerance = 1 step (0.5 mm)
        outer = hits[0]
        for h in hits[1:]:
            if h - outer <= 1:
                outer = h
            else:
                break
        entry_mni = scalp_surface_mni[idxs[outer]]
        distance_mm = float(np.linalg.norm(target - entry_mni))
        return entry_mni, distance_mm, "ray_cast"

    # Directed nearest-neighbour fallback
    # Find the nearest forward-facing surface voxel (likely inner edge), then
    # walk outward along the ray direction to reach the outer skin surface.
    inner_mni, _, method = _nearest_neighbor_directed(
        target, scalp_surface_mni, tree, unit=unit
    )
    entry_mni = _outer_edge_walk(
        inner_mni, unit, tree, scalp_surface_mni, hit_tolerance_mm, ray_step_mm
    )
    distance_mm = float(np.linalg.norm(target - entry_mni))
    return entry_mni, distance_mm, method


#------------------------------------------------------------------------------

def _nearest_neighbor_directed(
    target: np.ndarray,
    scalp_surface_mni: np.ndarray,
    tree: cKDTree,
    unit: np.ndarray | None,
) -> tuple[np.ndarray, float, str]:
    """
    Nearest scalp voxel in the forward half-space of *unit*.
    """
    if unit is not None:
        k = min(_NN_CANDIDATE_POOL, len(scalp_surface_mni))
        _, idxs = tree.query(target, k=k)
        candidates = scalp_surface_mni[idxs]           # (k, 3)
        forward_mask = (candidates - target) @ unit > 0
        if forward_mask.any():
            forward_idxs = idxs[forward_mask]
            dists = np.linalg.norm(scalp_surface_mni[forward_idxs] - target, axis=1)
            idx = forward_idxs[np.argmin(dists)]
            entry_mni = scalp_surface_mni[idx]
            distance_mm = float(np.linalg.norm(target - entry_mni))
            return entry_mni, distance_mm, "nearest_neighbor"

    # Unconstrained 
    _, idx = tree.query(target)
    entry_mni = scalp_surface_mni[idx]
    distance_mm = float(np.linalg.norm(target - entry_mni))
    return entry_mni, distance_mm, "nearest_neighbor"


def _outer_edge_walk(
    start: np.ndarray,
    unit: np.ndarray,
    tree: cKDTree,
    scalp_surface_mni: np.ndarray,
    hit_tolerance_mm: float,
    ray_step_mm: float,
    ray_max_mm: float = 20.0,
) -> np.ndarray:
    """
    Walk outward from *start* along *unit* and return the last surface voxel
    before the ray exits the shell.
    """
    n_steps = int(ray_max_mm / ray_step_mm)
    steps = np.arange(n_steps, dtype=float)
    samples = start + unit * steps[:, np.newaxis] * ray_step_mm

    dists, idxs = tree.query(samples)
    hits = np.where(dists <= hit_tolerance_mm)[0]

    if not len(hits):
        return start  # already at or beyond the outer edge

    outer = hits[0]
    for h in hits[1:]:
        if h - outer <= 1:
            outer = h
        else:
            break
    return scalp_surface_mni[idxs[outer]]
