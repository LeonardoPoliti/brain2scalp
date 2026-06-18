"""
tests/test_pipeline.py
======================
Integration tests for core/pipeline.py using mocked IO and compute deps.

All heavy NIfTI IO and expensive compute steps are patched so the tests run
without a real NIfTI file.

  - run() with a list of targets returns the correct number of PipelineState objects (full=True)
  - NIfTI loading and mask building happen exactly once for N targets
  - find_scalp_entry is called once per target
  - All states in a multi-target run share the same volumetric array objects
  - run() with a single target and full=True returns a single PipelineState
  - run() with a single target returns a ScalpResult (not a PipelineState)
"""

from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from brain2scalp.core.models import PipelineState, ScalpResult
from brain2scalp.core.pipeline import run

# ---------------------------------------------------------------------------
# Shared fake data

_SHAPE = (10, 10, 10)
_AFFINE = np.eye(4)
_FAKE_DATA = np.zeros(_SHAPE)
_FAKE_MASK = np.zeros(_SHAPE, dtype=np.uint8)

_FAKE_SURFACE = np.zeros(_SHAPE, dtype=np.uint8)
_FAKE_SURFACE[5, 5, 9] = 1  # one surface voxel so argwhere returns something

_FAKE_SCALP_MNI = np.array([[0.0, 0.0, 80.0], [0.0, 1.0, 80.0]])
_FAKE_ENTRY = np.array([0.0, 0.0, 80.0])
_FAKE_DIST = 40.0
_FAKE_METHOD = "ray_cast"

_TARGET_A = [5.0, 5.0, 3.0]
_TARGET_B = [5.0, 5.0, 6.0]


def _mocked_pipeline():
    """ExitStack context manager that patches all pipeline IO/compute deps."""
    stack = ExitStack()
    mocks: dict[str, MagicMock] = {}

    entries = {
        "load_atlas":              patch("brain2scalp.core.pipeline.load_atlas",
                                         return_value=(_FAKE_DATA, _AFFINE)),
        "voxel_size_mm":           patch("brain2scalp.core.pipeline.voxel_size_mm",
                                         return_value=(1.0, 1.0, 1.0)),
        "build_head_mask":         patch("brain2scalp.core.pipeline.build_head_mask",
                                         return_value=(_FAKE_MASK, 50.0)),
        "fill_holes":              patch("brain2scalp.core.pipeline.fill_holes",
                                         return_value=_FAKE_MASK),
        "erode_mask":              patch("brain2scalp.core.pipeline.erode_mask",
                                         return_value=_FAKE_MASK),
        "extract_surface":         patch("brain2scalp.core.pipeline.extract_surface",
                                         return_value=_FAKE_SURFACE),
        "voxel_to_mni":            patch("brain2scalp.core.pipeline.voxel_to_mni",
                                         return_value=_FAKE_SCALP_MNI),
        "mni_to_voxel_continuous": patch("brain2scalp.core.pipeline.mni_to_voxel_continuous",
                                         return_value=np.array([[5.0, 5.0, 5.0]])),
        "find_scalp_entry":        patch("brain2scalp.core.pipeline.find_scalp_entry",
                                         return_value=(_FAKE_ENTRY, _FAKE_DIST, _FAKE_METHOD)),
    }

    for name, patcher in entries.items():
        mocks[name] = stack.enter_context(patcher)

    return stack, mocks


# ---------------------------------------------------------------------------
# multi-target, full=True  

class TestRunMultiFull:
    def test_single_target_returns_one_state(self):
        with _mocked_pipeline()[0]:
            states = run("fake.nii", [_TARGET_A], full=True)
        assert len(states) == 1
        assert isinstance(states[0], PipelineState)

    def test_two_targets_return_two_states(self):
        with _mocked_pipeline()[0]:
            states = run("fake.nii", [_TARGET_A, _TARGET_B], full=True)
        assert len(states) == 2

    def test_brain_target_stored_per_state(self):
        with _mocked_pipeline()[0]:
            states = run("fake.nii", [_TARGET_A, _TARGET_B], full=True)
        assert states[0].brain_target_mni == _TARGET_A
        assert states[1].brain_target_mni == _TARGET_B

    def test_scalp_entry_from_find_scalp_entry(self):
        with _mocked_pipeline()[0]:
            states = run("fake.nii", [_TARGET_A], full=True)
        assert states[0].scalp_entry_mni == pytest.approx(_FAKE_ENTRY.tolist())
        assert states[0].distance_mm == _FAKE_DIST
        assert states[0].projection_method == _FAKE_METHOD

    def test_states_share_same_data_array(self):
        with _mocked_pipeline()[0]:
            states = run("fake.nii", [_TARGET_A, _TARGET_B], full=True)
        assert states[0].data is states[1].data

    def test_states_share_same_mask(self):
        with _mocked_pipeline()[0]:
            states = run("fake.nii", [_TARGET_A, _TARGET_B], full=True)
        assert states[0].mask is states[1].mask

    def test_states_share_same_scalp_surface(self):
        with _mocked_pipeline()[0]:
            states = run("fake.nii", [_TARGET_A, _TARGET_B], full=True)
        assert states[0].scalp_surface_mni is states[1].scalp_surface_mni

    def test_nifti_loaded_exactly_once_for_two_targets(self):
        stack, mocks = _mocked_pipeline()
        with stack:
            run("fake.nii", [_TARGET_A, _TARGET_B], full=True)
        mocks["load_atlas"].assert_called_once()

    def test_head_mask_built_exactly_once_for_two_targets(self):
        stack, mocks = _mocked_pipeline()
        with stack:
            run("fake.nii", [_TARGET_A, _TARGET_B], full=True)
        mocks["build_head_mask"].assert_called_once()

    def test_find_scalp_entry_called_once_per_target(self):
        stack, mocks = _mocked_pipeline()
        with stack:
            run("fake.nii", [_TARGET_A, _TARGET_B], full=True)
        assert mocks["find_scalp_entry"].call_count == 2

    def test_talairach_flag_propagates(self):
        with _mocked_pipeline()[0]:
            states = run("fake.nii", [_TARGET_A], full=True, is_talairach=True)
        assert states[0].is_talairach is True
        assert states[0].coordinate_space == "talairach"

    def test_native_space_propagates(self):
        with _mocked_pipeline()[0]:
            states = run("fake.nii", [_TARGET_A], full=True, is_native=True)
        assert states[0].coordinate_space == "native"


# ---------------------------------------------------------------------------
# single target, full=True

class TestRunSingleFull:
    def test_returns_pipeline_state(self):
        with _mocked_pipeline()[0]:
            state = run("fake.nii", _TARGET_A, full=True)
        assert isinstance(state, PipelineState)

    def test_brain_target_matches_input(self):
        with _mocked_pipeline()[0]:
            state = run("fake.nii", _TARGET_A, full=True)
        assert state.brain_target_mni == _TARGET_A

    def test_volumetric_arrays_present(self):
        with _mocked_pipeline()[0]:
            state = run("fake.nii", _TARGET_A, full=True)
        assert state.data is not None
        assert state.affine is not None
        assert state.mask is not None
        assert state.scalp_surface_mni is not None


# ---------------------------------------------------------------------------
# single target, full=False 

class TestRunSingle:
    def test_returns_scalp_result(self):
        with _mocked_pipeline()[0]:
            result = run("fake.nii", _TARGET_A)
        assert isinstance(result, ScalpResult)

    def test_not_pipeline_state(self):
        with _mocked_pipeline()[0]:
            result = run("fake.nii", _TARGET_A)
        assert type(result) is ScalpResult

    def test_brain_target_matches_input(self):
        with _mocked_pipeline()[0]:
            result = run("fake.nii", _TARGET_A)
        assert result.brain_target_mni == _TARGET_A

    def test_distance_and_method_present(self):
        with _mocked_pipeline()[0]:
            result = run("fake.nii", _TARGET_A)
        assert result.distance_mm == _FAKE_DIST
        assert result.projection_method == _FAKE_METHOD


# ---------------------------------------------------------------------------
# dispatch: single [x,y,z] vs list [[x,y,z]]

class TestDispatch:
    def test_single_coords_returns_scalar(self):
        with _mocked_pipeline()[0]:
            result = run("fake.nii", _TARGET_A)
        assert not isinstance(result, list)

    def test_list_of_one_returns_list(self):
        with _mocked_pipeline()[0]:
            result = run("fake.nii", [_TARGET_A])
        assert isinstance(result, list)
        assert len(result) == 1

    def test_single_full_returns_scalar(self):
        with _mocked_pipeline()[0]:
            result = run("fake.nii", _TARGET_A, full=True)
        assert not isinstance(result, list)

    def test_list_of_one_full_returns_list(self):
        with _mocked_pipeline()[0]:
            result = run("fake.nii", [_TARGET_A], full=True)
        assert isinstance(result, list)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# multi-target, full=False

class TestRunMulti:
    def test_returns_list_of_scalp_results(self):
        with _mocked_pipeline()[0]:
            results = run("fake.nii", [_TARGET_A, _TARGET_B])
        assert len(results) == 2
        assert all(type(r) is ScalpResult for r in results)

    def test_no_volumetric_arrays(self):
        with _mocked_pipeline()[0]:
            results = run("fake.nii", [_TARGET_A, _TARGET_B])
        assert not isinstance(results[0], PipelineState)