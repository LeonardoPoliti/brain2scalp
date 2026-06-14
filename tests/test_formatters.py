"""
tests/test_formatters.py
========================
Unit tests for cli/formatters.py.

Each formatter accepts list[ScalpResult] and returns a string.
Single-target and multi-target paths are distinct in all three formats.
"""

from __future__ import annotations

import csv
import io
import json

import pytest

from brain2scalp.cli.formatters import format_csv, format_json, format_text
from brain2scalp.core.models import ScalpResult

#---------------------------------------------------------------------------

def _r(
    brain_target: list[float] | None = None,
    scalp_entry: list[float] | None = None,
    distance_mm: float = 18.34,
    projection_method: str = "ray_cast",
    is_talairach: bool = False,
    tal_target: list[float] | None = None,
) -> ScalpResult:
    return ScalpResult(
        brain_target_mni=brain_target or [-46.0, 20.0, 32.0],
        scalp_entry_mni=scalp_entry or [-64.2, 22.1, 35.8],
        distance_mm=distance_mm,
        projection_method=projection_method,
        is_talairach=is_talairach,
        tal_target=tal_target,
        nii_path="/fake/atlas.nii",
        atlas_shape=(193, 229, 193),
        voxel_size_mm=(1.0, 1.0, 1.0),
        threshold_used=10.2,
    )


# ---------------------------------------------------------------------------
# format_text

class TestFormatText:
    def test_no_brain2scalp_result_banner(self):
        out = format_text([_r()])
        assert "brain2scalp result" not in out

    def test_single_contains_target_coords(self):
        out = format_text([_r(brain_target=[10.0, 20.0, 30.0])])
        assert "10" in out
        assert "20" in out
        assert "30" in out

    def test_single_contains_entry_coords(self):
        out = format_text([_r(scalp_entry=[11.0, 21.0, 31.0])])
        assert "11" in out

    def test_single_contains_distance(self):
        out = format_text([_r(distance_mm=25.5)])
        assert "25.5" in out

    def test_single_has_target_column(self):
        out = format_text([_r()])
        assert "target" in out
        assert "0" in out

    def test_single_no_bracket_label(self):
        out = format_text([_r()])
        assert "[Target" not in out

    def test_multi_has_both_indices(self):
        out = format_text([_r(), _r()])
        assert "0" in out
        assert "1" in out

    def test_multi_contains_both_targets(self):
        r0 = _r(brain_target=[10.0, 20.0, 30.0])
        r1 = _r(brain_target=[-10.0, -20.0, -30.0])
        out = format_text([r0, r1])
        assert "10" in out
        assert "-10" in out



# ---------------------------------------------------------------------------
# format_json

class TestFormatJson:
    def test_single_is_dict(self):
        obj = json.loads(format_json([_r()]))
        assert isinstance(obj, dict)

    def test_single_has_required_keys(self):
        obj = json.loads(format_json([_r()]))
        assert "mni" in obj
        assert "entry" in obj
        assert "distance_mm" in obj

    def test_multi_is_list(self):
        obj = json.loads(format_json([_r(), _r()]))
        assert isinstance(obj, list)
        assert len(obj) == 2

    def test_single_values_correct(self):
        r = _r(brain_target=[10.0, 20.0, 30.0], distance_mm=15.0)
        obj = json.loads(format_json([r]))
        assert obj["mni"] == {"x": 10.0, "y": 20.0, "z": 30.0}
        assert abs(obj["distance_mm"] - 15.0) < 0.01

    def test_multi_order_preserved(self):
        r0 = _r(brain_target=[1.0, 2.0, 3.0])
        r1 = _r(brain_target=[4.0, 5.0, 6.0])
        objs = json.loads(format_json([r0, r1]))
        assert objs[0]["mni"] == {"x": 1.0, "y": 2.0, "z": 3.0}
        assert objs[1]["mni"] == {"x": 4.0, "y": 5.0, "z": 6.0}

    def test_output_is_valid_json(self):
        out = format_json([_r(), _r()])
        json.loads(out)  # raises if invalid

    def test_three_targets_list_length(self):
        obj = json.loads(format_json([_r(), _r(), _r()]))
        assert len(obj) == 3

    def test_null_entry_when_scalp_entry_none(self):
        r = ScalpResult(brain_target_mni=[0.0, 0.0, 0.0], scalp_entry_mni=None, distance_mm=None)
        obj = json.loads(format_json([r]))
        assert obj["entry"] is None
        assert obj["distance_mm"] is None


# ---------------------------------------------------------------------------
# format_csv

def _parse_csv(text: str) -> list[dict]:
    return list(csv.DictReader(io.StringIO(text)))


class TestFormatCsv:
    def test_single_one_row(self):
        rows = _parse_csv(format_csv([_r()]))
        assert len(rows) == 1

    def test_has_target_column(self):
        rows = _parse_csv(format_csv([_r()]))
        assert "target" in rows[0]

    def test_single_brain_coords(self):
        r = _r(brain_target=[10.0, 20.0, 30.0])
        rows = _parse_csv(format_csv([r]))
        assert float(rows[0]["mni_x"]) == pytest.approx(10.0)
        assert float(rows[0]["mni_y"]) == pytest.approx(20.0)
        assert float(rows[0]["mni_z"]) == pytest.approx(30.0)

    def test_single_entry_coords(self):
        r = _r(scalp_entry=[11.0, 21.0, 31.0])
        rows = _parse_csv(format_csv([r]))
        assert float(rows[0]["entry_x"]) == pytest.approx(11.0)
        assert float(rows[0]["entry_y"]) == pytest.approx(21.0)
        assert float(rows[0]["entry_z"]) == pytest.approx(31.0)

    def test_single_distance(self):
        rows = _parse_csv(format_csv([_r(distance_mm=5.123)]))
        assert float(rows[0]["dist_mm"]) == pytest.approx(5.123, abs=0.001)

    def test_multi_target_values(self):
        rows = _parse_csv(format_csv([_r(), _r()]))
        assert rows[0]["target"] == "0"
        assert rows[1]["target"] == "1"

    def test_multi_row_count(self):
        rows = _parse_csv(format_csv([_r(), _r(), _r()]))
        assert len(rows) == 3

    def test_multi_coords_per_row(self):
        r0 = _r(brain_target=[1.0, 2.0, 3.0])
        r1 = _r(brain_target=[4.0, 5.0, 6.0])
        rows = _parse_csv(format_csv([r0, r1]))
        assert float(rows[0]["mni_x"]) == pytest.approx(1.0)
        assert float(rows[1]["mni_x"]) == pytest.approx(4.0)