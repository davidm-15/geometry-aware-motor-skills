"""
Shared pytest fixtures for the SkillTrace test suite.

All fixtures provide in-memory data that mirrors the data contracts
defined in SPEC.md so the tests remain fast and self-contained.
"""

import io
import textwrap
import numpy as np
import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# Raw "MaskPlanner" path data  (X; Y; Z; A; B; C; strokeId)
# ---------------------------------------------------------------------------

# Two strokes: IDs 0 and 1
RAW_PATH_ROWS = np.array([
    # stroke 0 – straight line along Z
    [0.0, -750.0, -123.0, -90.0, -70.0, 0.0, 0.0],
    [0.0, -750.0, -122.0, -90.0, -70.0, 0.0, 0.0],
    [0.0, -750.0, -121.0, -90.0, -70.0, 0.0, 0.0],
    [0.0, -750.0, -120.0, -90.0, -70.0, 0.0, 0.0],
    [0.0, -750.0, -119.0, -90.0, -70.0, 0.0, 0.0],
    [0.0, -750.0, -118.0, -90.0, -70.0, 0.0, 0.0],
    [0.0, -750.0, -117.0, -90.0, -70.0, 0.0, 0.0],
    [0.0, -750.0, -116.0, -90.0, -70.0, 0.0, 0.0],
    [0.0, -750.0, -115.0, -90.0, -70.0, 0.0, 0.0],
    [0.0, -750.0, -114.0, -90.0, -70.0, 0.0, 0.0],
    # stroke 1 – same shape, second ID
    [1.0, -750.0, -123.0, -90.0, -70.0, 0.0, 1.0],
    [1.0, -750.0, -122.0, -90.0, -70.0, 0.0, 1.0],
    [1.0, -750.0, -121.0, -90.0, -70.0, 0.0, 1.0],
    [1.0, -750.0, -120.0, -90.0, -70.0, 0.0, 1.0],
    [1.0, -750.0, -119.0, -90.0, -70.0, 0.0, 1.0],
    [1.0, -750.0, -118.0, -90.0, -70.0, 0.0, 1.0],
    [1.0, -750.0, -117.0, -90.0, -70.0, 0.0, 1.0],
    [1.0, -750.0, -116.0, -90.0, -70.0, 0.0, 1.0],
    [1.0, -750.0, -115.0, -90.0, -70.0, 0.0, 1.0],
    [1.0, -750.0, -114.0, -90.0, -70.0, 0.0, 1.0],
], dtype=float)


@pytest.fixture()
def raw_path():
    """Returns a (20, 7) float numpy array matching the MaskPlanner format."""
    return RAW_PATH_ROWS.copy()


# ---------------------------------------------------------------------------
# A minimal .txt file on disk for load_path() tests
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_raw_path_file(tmp_path):
    """Writes RAW_PATH_ROWS to a semicolon-delimited .txt with a header row."""
    p = tmp_path / "trajectory.txt"
    header = "X;Y;Z;A;B;C;strokeId"
    lines = [header] + [";".join(f"{v:.6f}" for v in row) for row in RAW_PATH_ROWS]
    p.write_text("\n".join(lines))
    return p


# ---------------------------------------------------------------------------
# A minimal RoboTwin .csv file for load_robotwin_trajectory() tests
# ---------------------------------------------------------------------------

ROBOTWIN_COLS = "time(s),x,y,z,qx,qy,qz,qw,velocity,ID"

ROBOTWIN_ROWS = np.array([
    [0.00, 0.0, -750.0, -123.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.0],
    [0.01, 0.0, -750.0, -122.5, 0.0, 0.0, 0.0, 1.0, 0.5, 0.0],
    [0.02, 0.0, -750.0, -122.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.0],
    [0.03, 0.0, -750.0, -121.5, 0.0, 0.0, 0.0, 1.0, 0.5, 0.0],
    [0.04, 0.0, -750.0, -121.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.0],
], dtype=float)


@pytest.fixture()
def tmp_robotwin_csv(tmp_path):
    """Writes ROBOTWIN_ROWS to a comma-delimited .csv with a header row."""
    p = tmp_path / "trajectory.csv"
    lines = [ROBOTWIN_COLS] + [",".join(f"{v:.6f}" for v in row) for row in ROBOTWIN_ROWS]
    p.write_text("\n".join(lines))
    return p


# ---------------------------------------------------------------------------
# Pre-built RoboTwin numpy array for assertions without touching the disk
# ---------------------------------------------------------------------------

@pytest.fixture()
def robotwin_data():
    """Returns a (5, 10) float numpy array in RoboTwin format."""
    return ROBOTWIN_ROWS.copy()
