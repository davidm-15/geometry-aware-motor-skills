"""
Unit tests for the data generator module.

Covers:
  - VirtualEndEffector   – initialisation, step(), quaternion stays unit norm
  - PathFollower         – get_lookahead_point() returns valid path point
  - compute_pd_force()   – correctness of PD controller formula
  - quat_mult()          – identity, norm preservation, associativity
  - get_angular_distances() – output length, identity quaternion gives 0 distance

All tests are purely in-memory; no disk, GPU, or network access needed.
"""

import sys
import pytest
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup – same strategy as test_path_parser.py
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
_REF_PKG = _REPO_ROOT / "SkillTrace2-carrot_on_stick"
if str(_REF_PKG) not in sys.path:
    sys.path.insert(0, str(_REF_PKG))

from data_generation.pure_pursuit import (
    VirtualEndEffector,
    PathFollower,
    compute_pd_force,
)
from utils.quaternions import quat_mult, get_angular_distances


# ===========================================================================
# Helpers shared within this module
# ===========================================================================

def _identity_quat():
    """[w, x, y, z] = [1, 0, 0, 0]"""
    return np.array([1.0, 0.0, 0.0, 0.0])


def _make_straight_path(n=20):
    """
    Returns an (n, 7) MaskPlanner-like array representing a straight line
    along the Z-axis.  Only columns 0-2 (XYZ) and -1 (strokeId) are used
    by PathFollower.
    """
    data = np.zeros((n, 7))
    data[:, 2] = np.linspace(0.0, 1.0, n)  # Z from 0 to 1
    data[:, -1] = 0.0
    return data


# ===========================================================================
# VirtualEndEffector
# ===========================================================================

class TestVirtualEndEffector:
    """Tests for VirtualEndEffector initialisation and physics step."""

    # --- Initialisation -------------------------------------------------------

    def test_initial_position_stored(self):
        pos = [1.0, 2.0, 3.0]
        eff = VirtualEndEffector(pos, _identity_quat())
        np.testing.assert_array_equal(eff.pos, pos)

    def test_initial_quaternion_stored(self):
        q = _identity_quat()
        eff = VirtualEndEffector([0, 0, 0], q)
        np.testing.assert_array_equal(eff.quat, q)

    def test_initial_velocity_is_zero(self):
        eff = VirtualEndEffector([0, 0, 0], _identity_quat())
        np.testing.assert_array_equal(eff.vel, [0.0, 0.0, 0.0])

    def test_initial_angular_velocity_is_zero(self):
        eff = VirtualEndEffector([0, 0, 0], _identity_quat())
        np.testing.assert_array_equal(eff.omega, [0.0, 0.0, 0.0])

    def test_custom_mass_stored(self):
        eff = VirtualEndEffector([0, 0, 0], _identity_quat(), mass=3.5)
        assert eff.mass == pytest.approx(3.5)

    def test_inertia_inverse_computed(self):
        I = 2.0 * np.eye(3)
        eff = VirtualEndEffector([0, 0, 0], _identity_quat(), inertia=I)
        expected_inv = 0.5 * np.eye(3)
        np.testing.assert_allclose(eff.inertia_inv, expected_inv, atol=1e-10)

    # --- step() ---------------------------------------------------------------

    def test_step_returns_pos_and_quat(self):
        eff = VirtualEndEffector([0, 0, 0], _identity_quat())
        force = np.array([1.0, 0.0, 0.0])
        torque = np.zeros(3)
        pos, quat = eff.step(force, torque, cv=0.1, cw=0.1, t=0.0, dt=0.01)
        assert pos.shape == (3,)
        assert quat.shape == (4,)

    def test_step_quaternion_stays_unit_norm(self):
        """After one step the quaternion must still be a unit quaternion."""
        eff = VirtualEndEffector([0, 0, 0], _identity_quat())
        force = np.array([10.0, 5.0, 2.0])
        torque = np.array([0.1, 0.2, 0.05])
        eff.step(force, torque, cv=0.5, cw=0.5, t=0.0, dt=0.01)
        norm = np.linalg.norm(eff.quat)
        assert norm == pytest.approx(1.0, abs=1e-6)

    def test_step_quaternion_stays_unit_norm_many_steps(self):
        """Unit-norm invariant must hold over many integration steps."""
        eff = VirtualEndEffector([0, 0, 0], _identity_quat())
        force = np.array([5.0, 0.0, 0.0])
        torque = np.array([0.0, 0.1, 0.0])
        for i in range(100):
            eff.step(force, torque, cv=0.5, cw=0.5, t=i * 0.01, dt=0.01)
        assert np.linalg.norm(eff.quat) == pytest.approx(1.0, abs=1e-5)

    def test_zero_force_effector_decelerates(self):
        """
        With zero applied force and non-zero damping the effector's speed
        should decrease (or stay zero if already at rest).
        """
        eff = VirtualEndEffector([0, 0, 0], _identity_quat())
        eff.vel = np.array([1.0, 0.0, 0.0])  # give it an initial velocity
        force = np.zeros(3)
        torque = np.zeros(3)
        eff.step(force, torque, cv=2.0, cw=0.0, t=0.0, dt=0.01)
        speed_after = np.linalg.norm(eff.vel)
        assert speed_after < 1.0, "Damping must reduce speed when no force is applied"

    def test_step_moves_effector_in_force_direction(self):
        """A force along X should eventually move the position in +X."""
        eff = VirtualEndEffector([0, 0, 0], _identity_quat(), mass=1.0)
        force = np.array([100.0, 0.0, 0.0])
        torque = np.zeros(3)
        for i in range(50):
            eff.step(force, torque, cv=0.0, cw=0.0, t=i * 0.01, dt=0.01)
        assert eff.pos[0] > 0, "Position must move in the force direction"


# ===========================================================================
# PathFollower
# ===========================================================================

class TestPathFollower:
    """Tests for PathFollower.get_lookahead_point()."""

    def test_lookahead_returns_tuple_of_two(self):
        path = _make_straight_path()
        follower = PathFollower(path, lookahead_dist=0.1)
        result = follower.get_lookahead_point(np.array([0.0, 0.0, 0.0]))
        assert len(result) == 2

    def test_lookahead_point_is_3d_vector(self):
        path = _make_straight_path()
        follower = PathFollower(path, lookahead_dist=0.1)
        point, _ = follower.get_lookahead_point(np.array([0.0, 0.0, 0.0]))
        assert point.shape == (3,)

    def test_target_index_is_int_like(self):
        path = _make_straight_path()
        follower = PathFollower(path, lookahead_dist=0.1)
        _, idx = follower.get_lookahead_point(np.array([0.0, 0.0, 0.0]))
        assert isinstance(idx, (int, np.integer))

    def test_target_index_within_bounds(self):
        path = _make_straight_path(n=20)
        follower = PathFollower(path, lookahead_dist=0.05)
        _, idx = follower.get_lookahead_point(np.array([0.0, 0.0, 0.0]))
        assert 0 <= idx < len(path)

    def test_lookahead_point_in_path(self):
        """The returned point must be exactly one of the path points."""
        path = _make_straight_path(n=20)
        follower = PathFollower(path, lookahead_dist=0.1)
        point, _ = follower.get_lookahead_point(np.array([0.0, 0.0, 0.0]))
        # Check that the returned point is a row of self.path (XYZ only)
        path_xyz = path[:, 0:3]
        dists = np.linalg.norm(path_xyz - point, axis=1)
        assert np.min(dists) == pytest.approx(0.0, abs=1e-10)

    def test_current_index_advances(self):
        """After calling get_lookahead_point(), current_idx must not regress."""
        path = _make_straight_path(n=20)
        follower = PathFollower(path, lookahead_dist=0.2)
        idx_before = follower.current_idx
        follower.get_lookahead_point(np.array([0.0, 0.0, 0.5]))
        assert follower.current_idx >= idx_before


# ===========================================================================
# compute_pd_force()
# ===========================================================================

class TestComputePdForce:
    """Tests for the PD-controller force computation."""

    def test_returns_3d_vector(self):
        force = compute_pd_force(
            robot_pos=np.zeros(3),
            robot_vel=np.zeros(3),
            target_pos=np.ones(3),
            kp=10.0, kd=1.0,
        )
        assert force.shape == (3,)

    def test_zero_error_zero_velocity_gives_zero_force(self):
        """When the robot is at the target with zero velocity, force must be 0."""
        force = compute_pd_force(
            robot_pos=np.array([1.0, 2.0, 3.0]),
            robot_vel=np.zeros(3),
            target_pos=np.array([1.0, 2.0, 3.0]),
            kp=50.0, kd=10.0,
        )
        np.testing.assert_allclose(force, np.zeros(3), atol=1e-12)

    def test_proportional_term_direction(self):
        """Force must point toward the target when velocity is zero."""
        rob = np.array([0.0, 0.0, 0.0])
        tgt = np.array([1.0, 0.0, 0.0])
        force = compute_pd_force(rob, np.zeros(3), tgt, kp=10.0, kd=0.0)
        assert force[0] > 0, "Force must have a positive X component toward target"
        assert force[1] == pytest.approx(0.0)
        assert force[2] == pytest.approx(0.0)

    def test_proportional_gain_scales_force(self):
        """Doubling kp must double the spring force (with zero velocity)."""
        rob = np.zeros(3)
        tgt = np.array([1.0, 0.0, 0.0])
        f1 = compute_pd_force(rob, np.zeros(3), tgt, kp=10.0, kd=0.0)
        f2 = compute_pd_force(rob, np.zeros(3), tgt, kp=20.0, kd=0.0)
        np.testing.assert_allclose(f2, 2.0 * f1, atol=1e-12)

    def test_damping_term_opposes_velocity(self):
        """The velocity term must subtract from the total force (opposed motion)."""
        rob = np.zeros(3)
        tgt = np.array([1.0, 0.0, 0.0])
        vel = np.array([1.0, 0.0, 0.0])
        f_no_damp = compute_pd_force(rob, np.zeros(3), tgt, kp=10.0, kd=0.0)
        f_with_damp = compute_pd_force(rob, vel, tgt, kp=10.0, kd=5.0)
        assert f_with_damp[0] < f_no_damp[0], "Damping must reduce force along velocity direction"

    def test_formula_correctness(self):
        """
        Manual verification:  force = kp*(target-robot) - kd*vel
        """
        rob = np.array([1.0, 2.0, 3.0])
        tgt = np.array([4.0, 5.0, 6.0])
        vel = np.array([0.1, 0.2, 0.3])
        kp, kd = 10.0, 2.0
        expected = kp * (tgt - rob) - kd * vel
        result = compute_pd_force(rob, vel, tgt, kp=kp, kd=kd)
        np.testing.assert_allclose(result, expected, atol=1e-12)


# ===========================================================================
# quat_mult()
# ===========================================================================

class TestQuatMult:
    """Tests for quaternion multiplication (convention [w, x, y, z])."""

    def test_identity_times_identity(self):
        q_id = _identity_quat()
        result = quat_mult(q_id, q_id)
        np.testing.assert_allclose(result, q_id, atol=1e-12)

    def test_identity_left_neutral(self):
        """q_id ⊗ q must equal q for any unit quaternion q."""
        q = np.array([0.0, 1.0, 0.0, 0.0])  # 180° rotation around X
        result = quat_mult(_identity_quat(), q)
        np.testing.assert_allclose(result, q, atol=1e-12)

    def test_identity_right_neutral(self):
        q = np.array([0.0, 0.0, 1.0, 0.0])  # 180° rotation around Y
        result = quat_mult(q, _identity_quat())
        np.testing.assert_allclose(result, q, atol=1e-12)

    def test_norm_preservation(self):
        """Product of two unit quaternions must also be a unit quaternion."""
        q1 = np.array([0.7071068, 0.7071068, 0.0, 0.0])  # 90° around X
        q2 = np.array([0.7071068, 0.0, 0.7071068, 0.0])  # 90° around Y
        result = quat_mult(q1, q2)
        assert np.linalg.norm(result) == pytest.approx(1.0, abs=1e-6)

    def test_returns_4_element_array(self):
        result = quat_mult(_identity_quat(), _identity_quat())
        assert result.shape == (4,)

    def test_180_around_x_twice_is_identity(self):
        """Rotating 180° around X twice should return to identity (or its negation)."""
        q = np.array([0.0, 1.0, 0.0, 0.0])
        result = quat_mult(q, q)
        # Could be +identity or -identity (both represent the same rotation)
        np.testing.assert_allclose(np.abs(result), np.array([1.0, 0.0, 0.0, 0.0]), atol=1e-6)


# ===========================================================================
# get_angular_distances()
# ===========================================================================

class TestGetAngularDistances:
    """Tests for angular distance computation between consecutive quaternions."""

    def test_returns_numpy_array(self):
        qs = np.tile(_identity_quat(), (5, 1))
        result = get_angular_distances(qs)
        assert isinstance(result, np.ndarray)

    def test_output_length_is_n_minus_1(self):
        """For n quaternions the output must have n-1 distances."""
        n = 8
        qs = np.tile(_identity_quat(), (n, 1))
        result = get_angular_distances(qs)
        assert len(result) == n - 1

    def test_identical_quaternions_give_zero_distance(self):
        """Distance between the same orientation must be 0."""
        qs = np.tile(_identity_quat(), (6, 1))
        result = get_angular_distances(qs)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_distances_are_non_negative(self):
        """Angular distances must always be ≥ 0."""
        rng = np.random.default_rng(42)
        raw = rng.standard_normal((10, 4))
        qs = raw / np.linalg.norm(raw, axis=1, keepdims=True)
        result = get_angular_distances(qs)
        assert np.all(result >= 0)

    def test_90_degree_rotation(self):
        """
        Distance between q_id and a 90° rotation must be ≈ π/2.
        """
        q_id = _identity_quat()
        q_90x = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0])
        qs = np.stack([q_id, q_90x])
        result = get_angular_distances(qs)
        assert result[0] == pytest.approx(np.pi / 2, abs=1e-6)

    def test_two_element_input(self):
        """Minimum valid input is 2 quaternions → 1 distance."""
        q1 = _identity_quat()
        q2 = np.array([0.0, 1.0, 0.0, 0.0])
        result = get_angular_distances(np.stack([q1, q2]))
        assert len(result) == 1
        assert result[0] >= 0
