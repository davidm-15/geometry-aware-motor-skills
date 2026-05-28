"""
Unit tests for the path parser module.

Covers:
  - load_path()                  – file I/O, schema, error handling
  - load_robotwin_trajectory()   – file I/O, schema, error handling
  - convert_path_2_trajectory()  – column count, dtype, quaternion norms
  - split_path_2_segments()      – output length, boolean dtype, multi-stroke

All tests are self-contained; no network or GPU access required.
"""

import sys
import os
import pytest
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Make the reference project importable (SkillTrace2-carrot_on_stick acts as
# the package root until the new src/ layout is established).
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
_REF_PKG = _REPO_ROOT / "SkillTrace2-carrot_on_stick"
if str(_REF_PKG) not in sys.path:
    sys.path.insert(0, str(_REF_PKG))

from data_generation.load_data import (
    load_path,
    load_robotwin_trajectory,
    convert_path_2_trajectory,
    split_path_2_segments,
)


# ===========================================================================
# load_path()
# ===========================================================================

class TestLoadPath:
    """Tests for load_path() – reads semicolon-delimited MaskPlanner .txt files."""

    def test_returns_numpy_array(self, tmp_raw_path_file):
        result = load_path(tmp_raw_path_file)
        assert isinstance(result, np.ndarray)

    def test_shape(self, tmp_raw_path_file):
        """Loaded array must have 7 columns: X Y Z A B C strokeId."""
        result = load_path(tmp_raw_path_file)
        assert result.ndim == 2
        assert result.shape[1] == 7, f"Expected 7 columns, got {result.shape[1]}"

    def test_row_count(self, tmp_raw_path_file):
        """All data rows (after skipping the header) should be loaded."""
        result = load_path(tmp_raw_path_file)
        assert result.shape[0] == 20

    def test_dtype_float(self, tmp_raw_path_file):
        result = load_path(tmp_raw_path_file)
        assert np.issubdtype(result.dtype, np.floating)

    def test_stroke_ids_preserved(self, tmp_raw_path_file):
        """The strokeId column (last) must contain exactly 0 and 1."""
        result = load_path(tmp_raw_path_file)
        unique_ids = np.unique(result[:, -1])
        assert set(unique_ids) == {0.0, 1.0}

    def test_accepts_path_object(self, tmp_raw_path_file):
        """load_path should accept both str and Path objects."""
        result_str = load_path(str(tmp_raw_path_file))
        result_path = load_path(tmp_raw_path_file)
        np.testing.assert_array_equal(result_str, result_path)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_path(tmp_path / "does_not_exist.txt")

    def test_coordinates_match_expected(self, tmp_raw_path_file):
        """Spot-check: first row X=0, Y=-750."""
        result = load_path(tmp_raw_path_file)
        assert result[0, 0] == pytest.approx(0.0)
        assert result[0, 1] == pytest.approx(-750.0)


# ===========================================================================
# load_robotwin_trajectory()
# ===========================================================================

class TestLoadRoboTwinTrajectory:
    """Tests for load_robotwin_trajectory() – reads comma-delimited RoboTwin .csv files."""

    def test_returns_numpy_array(self, tmp_robotwin_csv):
        result = load_robotwin_trajectory(tmp_robotwin_csv)
        assert isinstance(result, np.ndarray)

    def test_shape(self, tmp_robotwin_csv):
        """Must have 10 columns: time x y z qx qy qz qw velocity ID."""
        result = load_robotwin_trajectory(tmp_robotwin_csv)
        assert result.ndim == 2
        assert result.shape[1] == 10, f"Expected 10 columns, got {result.shape[1]}"

    def test_row_count(self, tmp_robotwin_csv):
        result = load_robotwin_trajectory(tmp_robotwin_csv)
        assert result.shape[0] == 5

    def test_dtype_float(self, tmp_robotwin_csv):
        result = load_robotwin_trajectory(tmp_robotwin_csv)
        assert np.issubdtype(result.dtype, np.floating)

    def test_time_column_monotonically_increasing(self, tmp_robotwin_csv):
        result = load_robotwin_trajectory(tmp_robotwin_csv)
        time_col = result[:, 0]
        assert np.all(np.diff(time_col) > 0), "Time column must be strictly ascending"

    def test_accepts_path_object(self, tmp_robotwin_csv):
        result_str = load_robotwin_trajectory(str(tmp_robotwin_csv))
        result_path = load_robotwin_trajectory(tmp_robotwin_csv)
        np.testing.assert_array_equal(result_str, result_path)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_robotwin_trajectory(tmp_path / "ghost.csv")


# ===========================================================================
# convert_path_2_trajectory()
# ===========================================================================

class TestConvertPath2Trajectory:
    """Tests for convert_path_2_trajectory() – MaskPlanner → RoboTwin format."""

    def test_returns_numpy_array(self, raw_path):
        result = convert_path_2_trajectory(raw_path)
        assert isinstance(result, np.ndarray)

    def test_output_has_10_columns(self, raw_path):
        """Output must have exactly 10 columns: time x y z qx qy qz qw velocity ID."""
        result = convert_path_2_trajectory(raw_path)
        assert result.shape[1] == 10, (
            f"Expected 10 output columns (time,x,y,z,qx,qy,qz,qw,velocity,ID), got {result.shape[1]}"
        )

    def test_output_row_count_matches_input(self, raw_path):
        result = convert_path_2_trajectory(raw_path)
        assert result.shape[0] == raw_path.shape[0]

    def test_quaternions_are_unit_vectors(self, raw_path):
        """Columns 4-7 (qx, qy, qz, qw) must each have norm ≈ 1."""
        result = convert_path_2_trajectory(raw_path)
        quats = result[:, 4:8]
        norms = np.linalg.norm(quats, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6,
                                   err_msg="All quaternions must be unit quaternions")

    def test_xyz_coordinates_preserved(self, raw_path):
        """X, Y, Z from input must be copied to columns 1-3 of the output."""
        result = convert_path_2_trajectory(raw_path)
        np.testing.assert_allclose(result[:, 1:4], raw_path[:, 0:3], atol=1e-6)

    def test_stroke_ids_preserved(self, raw_path):
        """The last column (ID) must match the original strokeId column."""
        result = convert_path_2_trajectory(raw_path)
        np.testing.assert_array_equal(result[:, -1], raw_path[:, -1])

    def test_time_column_starts_at_zero(self, raw_path):
        result = convert_path_2_trajectory(raw_path)
        assert result[0, 0] == pytest.approx(0.0)

    def test_velocity_column_is_non_negative(self, raw_path):
        """Velocity (column 8) must be ≥ 0 everywhere."""
        result = convert_path_2_trajectory(raw_path)
        assert np.all(result[:, 8] >= 0), "Velocity must be non-negative"

    def test_custom_base_velocity_applied(self, raw_path):
        """
        Passing a higher v_base should produce different (larger) velocities
        than the default – the rule is applied somewhere in the trajectory.
        """
        default = convert_path_2_trajectory(raw_path, v_base=0.5)
        fast = convert_path_2_trajectory(raw_path, v_base=2.0)
        # At least *some* velocity values should differ
        assert not np.allclose(default[:, 8], fast[:, 8]), (
            "v_base parameter must affect the velocity column"
        )


# ===========================================================================
# split_path_2_segments()
# ===========================================================================

class TestSplitPath2Segments:
    """Tests for split_path_2_segments() – labels each point as straight or curved."""

    def test_returns_array(self, raw_path):
        result = split_path_2_segments(raw_path)
        assert isinstance(result, (np.ndarray, list))

    def test_output_length_matches_input(self, raw_path):
        result = split_path_2_segments(raw_path)
        assert len(result) == len(raw_path)

    def test_straight_line_labeled_straight(self):
        """A perfectly collinear set of points must all be labeled True (straight)."""
        # Build a single-ID straight path
        n = 20
        z = np.linspace(-123.0, -103.0, n)
        data = np.zeros((n, 7))
        data[:, 0] = 0.0    # X constant
        data[:, 1] = -750.0 # Y constant
        data[:, 2] = z      # Z varies linearly → straight line
        data[:, -1] = 0.0   # strokeId = 0
        result = np.array(split_path_2_segments(data))
        # The sliding-window algorithm pads ends; central points must be True
        n_pad = 7 // 2
        central = result[n_pad:-n_pad]
        assert np.all(central), "All central points of a perfectly straight line must be labeled True"

    def test_multi_stroke_processed(self, raw_path):
        """With 2 stroke IDs the returned array still has one entry per row."""
        result = split_path_2_segments(raw_path)
        assert len(result) == len(raw_path)

    def test_output_values_are_boolean(self, raw_path):
        """Each element of the result must be truthy or falsy (bool or int 0/1)."""
        result = split_path_2_segments(raw_path)
        for val in result:
            assert val in (True, False, 0, 1), f"Unexpected value {val!r}"
