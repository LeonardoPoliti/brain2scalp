"""
tests/test_cli_helpers.py
=========================
Unit tests for the CLI parsing helpers in cli/main.py.

Covers:
  _parse_target_str  - comma-separated triplet parsing
  _parse_target_file - CSV file loading with header detection
  _merge_targets     - deduplication and ordering
  _resolve_indices   - viz flag → index list translation
"""

from __future__ import annotations

import argparse

import pytest

from brain2scalp.cli.main import (
    _merge_targets,
    _parse_target_file,
    _parse_target_str,
    _resolve_indices,
)


# ---------------------------------------------------------------------------
# _parse_target_str

class TestParseTargetStr:
    def test_single_triplet(self):
        assert _parse_target_str("10 20 30") == [[10.0, 20.0, 30.0]]

    def test_multiple_triplets(self):
        result = _parse_target_str("10 20 30, -10 5 45")
        assert result == [[10.0, 20.0, 30.0], [-10.0, 5.0, 45.0]]

    def test_three_triplets(self):
        result = _parse_target_str("1 2 3, 4 5 6, 7 8 9")
        assert len(result) == 3
        assert result[2] == [7.0, 8.0, 9.0]

    def test_floats(self):
        assert _parse_target_str("10.5 -20.1 30.7") == [[10.5, -20.1, 30.7]]

    def test_extra_whitespace_around_commas(self):
        result = _parse_target_str("  10 20 30  ,  -10 5 45  ")
        assert result == [[10.0, 20.0, 30.0], [-10.0, 5.0, 45.0]]

    def test_trailing_comma_ignored(self):
        assert _parse_target_str("10 20 30,") == [[10.0, 20.0, 30.0]]

    def test_too_few_values_raises(self):
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_target_str("10 20")

    def test_too_many_values_per_triplet_raises(self):
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_target_str("10 20 30 40")

    def test_non_numeric_raises(self):
        with pytest.raises((argparse.ArgumentTypeError, ValueError)):
            _parse_target_str("a b c")

    def test_empty_string_raises(self):
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_target_str("")

    def test_only_commas_raises(self):
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_target_str(",,,")


# ---------------------------------------------------------------------------
# _parse_target_file

class TestParseTargetFile:
    def test_no_header(self, tmp_path):
        f = tmp_path / "targets.csv"
        f.write_text("10,20,30\n-10,5,45\n")
        assert _parse_target_file(str(f)) == [[10.0, 20.0, 30.0], [-10.0, 5.0, 45.0]]

    def test_header_row_skipped(self, tmp_path):
        f = tmp_path / "targets.csv"
        f.write_text("x,y,z\n10,20,30\n-10,5,45\n")
        assert _parse_target_file(str(f)) == [[10.0, 20.0, 30.0], [-10.0, 5.0, 45.0]]

    def test_blank_lines_ignored(self, tmp_path):
        f = tmp_path / "targets.csv"
        f.write_text("10,20,30\n\n-10,5,45\n")
        assert _parse_target_file(str(f)) == [[10.0, 20.0, 30.0], [-10.0, 5.0, 45.0]]

    def test_extra_columns_ignored(self, tmp_path):
        f = tmp_path / "targets.csv"
        f.write_text("10,20,30,label\n-10,5,45,other\n")
        assert _parse_target_file(str(f)) == [[10.0, 20.0, 30.0], [-10.0, 5.0, 45.0]]

    def test_floats(self, tmp_path):
        f = tmp_path / "targets.csv"
        f.write_text("10.5,-20.1,30.7\n")
        result = _parse_target_file(str(f))
        assert result[0] == pytest.approx([10.5, -20.1, 30.7])

    def test_bad_data_mid_file_raises(self, tmp_path):
        f = tmp_path / "targets.csv"
        f.write_text("10,20,30\nbad,data,here\n")
        with pytest.raises(ValueError):
            _parse_target_file(str(f))

    def test_empty_file_raises(self, tmp_path):
        f = tmp_path / "empty.csv"
        f.write_text("")
        with pytest.raises(ValueError):
            _parse_target_file(str(f))

    def test_header_only_raises(self, tmp_path):
        f = tmp_path / "hdr.csv"
        f.write_text("x,y,z\n")
        with pytest.raises(ValueError):
            _parse_target_file(str(f))

    def test_too_few_columns_raises(self, tmp_path):
        f = tmp_path / "targets.csv"
        f.write_text("10,20\n")
        with pytest.raises(ValueError):
            _parse_target_file(str(f))


# ---------------------------------------------------------------------------
# _merge_targets

class TestMergeTargets:
    def test_file_targets_come_first(self):
        file_t = [[1.0, 2.0, 3.0]]
        cli_t  = [[4.0, 5.0, 6.0]]
        assert _merge_targets(file_t, cli_t) == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

    def test_exact_duplicate_dropped(self):
        t = [1.0, 2.0, 3.0]
        result = _merge_targets([t], [t])
        assert len(result) == 1
        assert result[0] == t

    def test_first_occurrence_wins(self):
        # file target appears first; same coords from CLI should be dropped
        file_t = [[1.0, 2.0, 3.0]]
        cli_t  = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        result = _merge_targets(file_t, cli_t)
        assert result == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

    def test_cli_only(self):
        cli_t = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        assert _merge_targets([], cli_t) == cli_t

    def test_file_only(self):
        file_t = [[1.0, 2.0, 3.0]]
        assert _merge_targets(file_t, []) == file_t

    def test_both_empty(self):
        assert _merge_targets([], []) == []

    def test_multiple_duplicates_all_dropped(self):
        t = [5.0, 5.0, 5.0]
        result = _merge_targets([t, t], [t])
        assert result == [[5.0, 5.0, 5.0]]

    def test_near_duplicates_not_dropped(self):
        t1 = [1.0, 2.0, 3.0]
        t2 = [1.0, 2.0, 3.1]
        result = _merge_targets([t1], [t2])
        assert len(result) == 2


# ---------------------------------------------------------------------------
# _resolve_indices

class TestResolveIndices:
    def test_none_returns_empty(self):
        assert _resolve_indices(None, 5) == []

    def test_empty_list_returns_all(self):
        assert _resolve_indices([], 3) == [0, 1, 2]

    def test_empty_list_single_target(self):
        assert _resolve_indices([], 1) == [0]

    def test_specific_indices(self):
        assert _resolve_indices([0, 2], 3) == [0, 2]

    def test_single_index(self):
        assert _resolve_indices([1], 4) == [1]

    def test_n_zero_empty_list_returns_empty(self):
        assert _resolve_indices([], 0) == []