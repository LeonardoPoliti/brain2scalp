"""
core/pipeline.py
-----------------
Pipeline orchestrator.

Public entry points (light = coords only, full = coords + volumetric arrays):

run(nii_path, brain_target, ...)       single target, light → ScalpResult
run_multi(nii_path, targets, ...)      multi target,  light → list[ScalpResult]
run_full(nii_path, brain_target, ...)  single target, full  → PipelineState
run_full_multi(nii_path, targets, ...) multi target,  full  → list[PipelineState]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .atlas import (
    load_atlas,
    mni_to_voxel_continuous,
    voxel_size_mm,
    voxel_to_mni,
)
from .mask import build_head_mask, erode_mask, extract_surface, fill_holes
from .models import PipelineState, ScalpResult
from .projection import build_scalp_tree, find_scalp_entry
from .transform import talairach_to_mni

#------------------------------------------------------------------------------

def run_full_multi(
    nii_path: str | Path,
    targets: list[list[float]],
    *,
    threshold: float | None = None,
    is_talairach: bool = False,
    is_native: bool = False,
    surface_thickness_mm: float = 1.5,
    k_local: int = 100,
    verbose: bool = False,
) -> list[PipelineState]:
    """
    Run the brain-to-scalp pipeline for one or more targets.

    Params:
    nii_path : str or Path
    targets  : list of [x, y, z] triplets (MNI or Talairach mm)
    Returns:
    list[PipelineState]  one per target; all share the same volumetric arrays ref.
    """
    nii_path = Path(nii_path)
    _log = _make_logger(verbose)
    n = len(targets)

    _log(f"\n{'='*90}")
    _log(f"{' '*40}brain2scalp")
    _log(f"{'='*90}\n")
    _log(f"  NIfTI file : {nii_path}")
    if n > 1:
        _log(f"  Targets    : {n}")
    _log("")

    # 1. Load atlas (once)
    _log("[1/5] Loading NIfTI and validating …")
    data, affine = load_atlas(nii_path, warn_native_space=not is_native)
    dx, dy, dz = voxel_size_mm(affine)
    _log(f"      Shape: {data.shape}  |  voxel size: ({dx:.2f}, {dy:.2f}, {dz:.2f}) mm")

    # 2. Binary head mask (once)
    _log("[2/5] Building binary head mask …")
    mask, used_thresh = build_head_mask(data, threshold=threshold)
    _log(f"      Threshold: {used_thresh:.2f}  |  head voxels: {int(mask.sum()):,}")

    # 3. Fill holes (once)
    _log("[3/5] Filling holes (convex hull per axial slice) …")
    filled = fill_holes(mask)
    _log(f"      Filled voxels: {int(filled.sum()):,}  (+{int(filled.sum()-mask.sum()):,} holes sealed)")

    # 4. Erode (once)
    _log(f"[4/5] Eroding by {surface_thickness_mm} mm …")
    eroded = erode_mask(filled, affine, surface_thickness_mm=surface_thickness_mm)

    # 5. Surface extraction (once)
    _log("[5/5] Extracting surface …")
    surface = extract_surface(filled, eroded)
    surface_voxels = np.argwhere(surface)
    scalp_mni = voxel_to_mni(surface_voxels, affine)
    scalp_tree = build_scalp_tree(scalp_mni)
    _log("")

    states: list[PipelineState] = []

    for t_idx, brain_target in enumerate(targets):
        # Coordinate conversion
        tal_target: list[float] | None = None
        target_mni: list[float]

        if is_native:
            target_mni = list(brain_target)
            coordinate_space = "native"
        elif is_talairach:
            tal_target = list(brain_target)
            mni_arr = talairach_to_mni(tal_target)
            target_mni = mni_arr.tolist()
            coordinate_space = "talairach"
        else:
            target_mni = list(brain_target)
            coordinate_space = "mni"

        # Warn if outside FOV
        vox = mni_to_voxel_continuous(np.array([target_mni]), affine)[0]
        nx, ny, nz = data.shape
        if not (0 <= vox[0] < nx and 0 <= vox[1] < ny and 0 <= vox[2] < nz):
            print(
                f"  Warning: target {np.round(target_mni, 1)} mm is outside "
                f"the FOV ({nx}×{ny}×{nz}) - results may be unreliable.\n",
                file=sys.stderr,
            )

        if is_talairach:
            assert tal_target is not None
            _log(f"  [{t_idx}] Tal {np.round(tal_target, 2)}  →  MNI {np.round(target_mni, 2)}")
        else:
            _log(f"  [{t_idx}] MNI {np.round(target_mni, 2)}")

        # Find scalp entry points
        entry_mni, dist, method = find_scalp_entry(
            target_mni,
            scalp_mni,
            affine,
            k_local=k_local,
            tree=scalp_tree,
        )

        _log(f"       entry : {np.round(entry_mni, 2)}")
        _log(f"       dist  : {dist:.2f} mm  |  method: {method}")
        _log("")

        states.append(PipelineState(
            brain_target_mni=target_mni,
            coordinate_space=coordinate_space,
            is_talairach=is_talairach,
            tal_target=tal_target,
            scalp_entry_mni=entry_mni.tolist(),
            distance_mm=round(dist, 4),
            projection_method=method,
            nii_path=str(nii_path),
            atlas_shape=tuple(data.shape),  # type: ignore[arg-type]
            voxel_size_mm=(round(dx, 4), round(dy, 4), round(dz, 4)),
            threshold_used=round(used_thresh, 4),
            surface_thickness_mm=round(surface_thickness_mm, 4),
            data=data,
            affine=affine,
            mask=mask,
            filled=filled,
            eroded=eroded,
            surface=surface,
            scalp_surface_mni=scalp_mni,
        ))

    # Summary 
    if not states:
        return states

    def _fmt(c) -> str:
        return f"[{float(c[0]):7.1f} {float(c[1]):7.1f} {float(c[2]):7.1f}]"

    rows = []
    for i, s in enumerate(states):
        rows.append((
            f"Target {i}",
            _fmt(s.tal_target) if s.is_talairach and s.tal_target is not None else None,
            _fmt(s.brain_target_mni),
            _fmt(s.scalp_entry_mni) if s.scalp_entry_mni is not None else "None",
            f"{s.distance_mm:.2f} mm",
        ))

    w_lbl = max(len(r[0]) for r in rows)

    content_lines = []
    for lbl, tal, t, e, d in rows:
        if tal is not None:
            content_lines.append(f"  {lbl:<{w_lbl}}  Tal {tal}  →  MNI {t}  →  entry {e}  |  {d}")
        else:
            content_lines.append(f"  {lbl:<{w_lbl}}  MNI {t}  →  entry {e}  |  {d}")
    sep_w = max(56, max(len(line) for line in content_lines))
    header = "brain2scalp - Results Summary"

    _log(f"\n{'='*sep_w}")
    _log(f"{header:^{sep_w}}")
    _log(f"{'='*sep_w}")
    for line in content_lines:
        _log(line)
    _log(f"{'='*sep_w}\n")

    return states


def run_full(
    nii_path: str | Path,
    brain_target: list[float],
    *,
    threshold: float | None = None,
    is_talairach: bool = False,
    is_native: bool = False,
    surface_thickness_mm: float = 1.5,
    k_local: int = 100,
    verbose: bool = False,
) -> PipelineState:
    """Single-target wrapper around run_full_multi."""
    return run_full_multi(
        nii_path,
        [brain_target],
        threshold=threshold,
        is_talairach=is_talairach,
        is_native=is_native,
        surface_thickness_mm=surface_thickness_mm,
        k_local=k_local,
        verbose=verbose,
    )[0]


def run(
    nii_path: str | Path,
    brain_target: list[float],
    *,
    threshold: float | None = None,
    is_talairach: bool = False,
    is_native: bool = False,
    surface_thickness_mm: float = 1.5,
    k_local: int = 100,
    verbose: bool = False,
) -> ScalpResult:
    """
    Run the pipeline for a single target; return a lightweight ScalpResult.

    Same params as run_full().
    """
    return run_full(
        nii_path,
        brain_target,
        threshold=threshold,
        is_talairach=is_talairach,
        is_native=is_native,
        surface_thickness_mm=surface_thickness_mm,
        k_local=k_local,
        verbose=verbose,
    ).to_result()


def run_multi(
    nii_path: str | Path,
    targets: list[list[float]],
    *,
    threshold: float | None = None,
    is_talairach: bool = False,
    is_native: bool = False,
    surface_thickness_mm: float = 1.5,
    k_local: int = 100,
    verbose: bool = False,
) -> list[ScalpResult]:
    """Run the pipeline for multiple targets; return lightweight ScalpResults."""
    return [
        s.to_result()
        for s in run_full_multi(
            nii_path,
            targets,
            threshold=threshold,
            is_talairach=is_talairach,
            is_native=is_native,
            surface_thickness_mm=surface_thickness_mm,
            k_local=k_local,
            verbose=verbose,
        )
    ]


def _make_logger(verbose: bool):
    if verbose:
        return print
    return lambda *_: None