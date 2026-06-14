"""
core/models.py
-----------------
Data models for the brain2scalp pipeline.

Two classes exist:

ScalpResult
    Lightweight, JSON-serialisable result.  Contains only the coordinate
    outputs and metadata that any consumer needs. No volumetric arrays.

PipelineState
    Extends ScalpResult with the heavy NumPy arrays produced during
    processing (raw data, masks, surface voxels …).  Constructed only when
    visualisation is requested.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields as dataclass_fields

import numpy as np

#------------------------------------------------------------------------------

@dataclass
class ScalpResult:
    """Serialisable output of one brain-to-scalp projection."""

    # INPUTS
    # Brain target in MNI152 space (mm). Already converted if Talairach input was given.
    brain_target_mni: list[float]

    # Coordinate space of the input target: 'mni', 'talairach', or 'native'.
    coordinate_space: str = "mni"

    # True when the original coordinates were in Talairach space.
    is_talairach: bool = False

    # Original Talairach coordinates (mm), only set when is_talairach=True
    tal_target: list[float] | None = None

    # OUTPUTS
    # Scalp entry point in MNI152 space (mm)
    scalp_entry_mni: list[float] | None = None

    # Euclidean distance from brain target to scalp entry point (mm)
    distance_mm: float | None = None

    # 'ray_cast' if the ray-cast succeeded; 'nearest_neighbor' if fallback
    # was used (target outside mask or ray missed the surface)
    projection_method: str = "ray_cast"

    # atlas metadata
    nii_path: str = ""
    atlas_shape: tuple[int, int, int] | None = None
    voxel_size_mm: tuple[float, float, float] | None = None
    threshold_used: float | None = None
    surface_thickness_mm: float | None = None

    # convenience
    def to_dict(self) -> dict:
        """Return a plain Python dict (all values JSON-serialisable).

        When called on a PipelineState subclass, numpy array fields are omitted.
        """
        d = asdict(self)
        result = {}
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                continue
            elif isinstance(v, np.generic):
                result[k] = v.item()
            elif isinstance(v, list):
                result[k] = [x.item() if isinstance(x, np.generic) else x for x in v]
            else:
                result[k] = v
        return result

    @property
    def scalp_entry_np(self) -> np.ndarray | None:
        """Scalp entry as a (3,) float64 array, or None."""
        return np.array(self.scalp_entry_mni) if self.scalp_entry_mni is not None else None

    @property
    def brain_target_np(self) -> np.ndarray:
        """Brain target as a (3,) float64 array."""
        return np.array(self.brain_target_mni)


@dataclass
class PipelineState(ScalpResult):
    """ScalpResult extended with the intermediate arrays needed for visualisation."""

    # raw NIfTI data (float64, as returned by atlas.load_atlas)
    data: np.ndarray | None = field(default=None, repr=False)
    affine: np.ndarray | None = field(default=None, repr=False)

    # PROCESSING STAGES
    # Binary head mask (uint8).  1 inside the head, 0 outside.
    mask: np.ndarray | None = field(default=None, repr=False)
    # Hole-filled head mask (uint8).
    filled: np.ndarray | None = field(default=None, repr=False)
    # Eroded mask (uint8).  One surface-thickness inward.
    eroded: np.ndarray | None = field(default=None, repr=False)
    # Surface shell: filled XOR eroded (uint8).
    surface: np.ndarray | None = field(default=None, repr=False)

    # surface in MNI mm
    scalp_surface_mni: np.ndarray | None = field(default=None, repr=False)
    """(N, 3) float64 array of all scalp voxel centres in MNI space."""

    def to_result(self) -> ScalpResult:
        """Return a lightweight ScalpResult (drops all arrays)."""
        return ScalpResult(**{f.name: getattr(self, f.name) for f in dataclass_fields(ScalpResult)})
