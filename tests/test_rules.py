"""
Tests for data_generation/rules.py — GeometryProximityRule behaviour.

All tests exercise the public interface only (no implementation details).
"""

import numpy as np
import pytest
from data_generation.rules import Rule, GeometryProximityRule, VelocityScalingRule, apply_rules_max_wins
from data_generation.load_data import compute_proximity_data, convert_path_2_trajectory
from data_generation.generate_dataset import sample_geometric_proximity_rule


POS  = np.array([1.0, 2.0, 3.0])
QUAT = np.array([0.0, 0.0, 0.0, 1.0])
F    = np.array([10.0, 0.0, 0.0])
TAU  = np.array([0.0, 5.0, 0.0])


# ---------------------------------------------------------------------------
# Base class default
# ---------------------------------------------------------------------------

class TestRuleBaseDefaults:
    def test_base_modify_with_proximity_passthrough(self):
        """Default base-class hook must return all four values unchanged."""

        class _Stub(Rule):
            @property
            def name(self): return "stub"

        stub = _Stub()
        out_pos, out_quat, out_f, out_tau = stub.modify_with_proximity(
            POS, QUAT, F, TAU, {}
        )
        np.testing.assert_array_equal(out_pos, POS)
        np.testing.assert_array_equal(out_quat, QUAT)
        np.testing.assert_array_equal(out_f, F)
        np.testing.assert_array_equal(out_tau, TAU)


# ---------------------------------------------------------------------------
# GeometryProximityRule — signature and passthrough
# ---------------------------------------------------------------------------

class TestGeometryProximityRuleInterface:
    def test_returns_four_values(self):
        rule = GeometryProximityRule(node_type='edge', dist_threshold=0.05, f_scale=0.5)
        result = rule.modify_with_proximity(POS, QUAT, F, TAU, {})
        assert len(result) == 4

    def test_pos_always_passed_through(self):
        rule = GeometryProximityRule(node_type='edge', dist_threshold=0.05, f_scale=0.5)
        out_pos, _, _, _ = rule.modify_with_proximity(POS, QUAT, F, TAU, {})
        np.testing.assert_array_equal(out_pos, POS)

    def test_quat_always_passed_through(self):
        rule = GeometryProximityRule(node_type='edge', dist_threshold=0.05, f_scale=0.5)
        _, out_quat, _, _ = rule.modify_with_proximity(POS, QUAT, F, TAU, {})
        np.testing.assert_array_equal(out_quat, QUAT)


# ---------------------------------------------------------------------------
# GeometryProximityRule — threshold logic
# ---------------------------------------------------------------------------

class TestGeometryProximityRuleThreshold:
    def test_no_effect_when_proximity_data_empty(self):
        """Rule must not modify forces when proximity_data has no relevant key."""
        rule = GeometryProximityRule(node_type='edge', dist_threshold=0.05, f_scale=0.5)
        _, _, out_f, out_tau = rule.modify_with_proximity(POS, QUAT, F, TAU, {})
        np.testing.assert_array_equal(out_f, F)
        np.testing.assert_array_equal(out_tau, TAU)

    def test_no_effect_when_dist_above_threshold(self):
        rule = GeometryProximityRule(node_type='edge', dist_threshold=0.05, f_scale=0.5)
        _, _, out_f, out_tau = rule.modify_with_proximity(
            POS, QUAT, F, TAU, {"dist_to_edge": 0.10}
        )
        np.testing.assert_array_equal(out_f, F)
        np.testing.assert_array_equal(out_tau, TAU)

    def test_no_effect_when_dist_exactly_at_threshold(self):
        """Boundary: dist == threshold must NOT fire (strictly less than)."""
        rule = GeometryProximityRule(node_type='edge', dist_threshold=0.05, f_scale=0.5)
        _, _, out_f, out_tau = rule.modify_with_proximity(
            POS, QUAT, F, TAU, {"dist_to_edge": 0.05}
        )
        np.testing.assert_array_equal(out_f, F)
        np.testing.assert_array_equal(out_tau, TAU)

    def test_f_scale_applied_when_dist_below_threshold(self):
        rule = GeometryProximityRule(node_type='edge', dist_threshold=0.05, f_scale=0.5)
        _, _, out_f, _ = rule.modify_with_proximity(
            POS, QUAT, F, TAU, {"dist_to_edge": 0.01}
        )
        np.testing.assert_allclose(out_f, F * 0.5)

    def test_tau_scale_applied_when_dist_below_threshold(self):
        rule = GeometryProximityRule(node_type='edge', dist_threshold=0.05, f_scale=1.0, tau_scale=2.0)
        _, _, _, out_tau = rule.modify_with_proximity(
            POS, QUAT, F, TAU, {"dist_to_edge": 0.01}
        )
        np.testing.assert_allclose(out_tau, TAU * 2.0)

    def test_fires_when_dist_is_zero(self):
        """End-effector exactly on a node must trigger the rule."""
        rule = GeometryProximityRule(node_type='edge', dist_threshold=0.05, f_scale=0.7)
        _, _, out_f, _ = rule.modify_with_proximity(
            POS, QUAT, F, TAU, {"dist_to_edge": 0.0}
        )
        np.testing.assert_allclose(out_f, F * 0.7)

    def test_crossing_rule_ignores_edge_key(self):
        """A crossing-type rule must not fire when only dist_to_edge is present."""
        rule = GeometryProximityRule(node_type='crossing', dist_threshold=0.05, f_scale=0.5)
        _, _, out_f, out_tau = rule.modify_with_proximity(
            POS, QUAT, F, TAU, {"dist_to_edge": 0.001}
        )
        np.testing.assert_array_equal(out_f, F)
        np.testing.assert_array_equal(out_tau, TAU)

    def test_edge_rule_ignores_crossing_key(self):
        rule = GeometryProximityRule(node_type='edge', dist_threshold=0.05, f_scale=0.5)
        _, _, out_f, out_tau = rule.modify_with_proximity(
            POS, QUAT, F, TAU, {"dist_to_crossing": 0.001}
        )
        np.testing.assert_array_equal(out_f, F)
        np.testing.assert_array_equal(out_tau, TAU)


# ---------------------------------------------------------------------------
# apply_rules_max_wins
# ---------------------------------------------------------------------------

class TestApplyRulesMaxWins:
    def test_no_rules_returns_base_forces(self):
        _, _, out_f, out_tau = apply_rules_max_wins([], POS, QUAT, F, TAU, 'straight', {})
        np.testing.assert_array_equal(out_f, F)
        np.testing.assert_array_equal(out_tau, TAU)

    def test_single_scaling_rule_applies(self):
        rule = VelocityScalingRule('any', f_scale=1.5, tau_scale=1.5)
        _, _, out_f, _ = apply_rules_max_wins([rule], POS, QUAT, F, TAU, 'straight', {})
        np.testing.assert_allclose(out_f, F * 1.5)

    def test_two_scale_up_rules_do_not_compound(self):
        """1.5× and 1.3× rules must not yield 1.5×1.3×=1.95×; max wins."""
        rule_a = VelocityScalingRule('any', f_scale=1.5, tau_scale=1.5)
        rule_b = VelocityScalingRule('any', f_scale=1.3, tau_scale=1.3)
        _, _, out_f, _ = apply_rules_max_wins([rule_a, rule_b], POS, QUAT, F, TAU, 'straight', {})
        np.testing.assert_allclose(out_f, F * 1.5)

    def test_larger_deviation_wins_over_smaller_deviation(self):
        """A 1.6× rule (dev 0.6) must beat a 1.2× rule (dev 0.2)."""
        rule_big   = VelocityScalingRule('any', f_scale=1.6, tau_scale=1.0)
        rule_small = VelocityScalingRule('any', f_scale=1.2, tau_scale=1.0)
        _, _, out_f, _ = apply_rules_max_wins([rule_big, rule_small], POS, QUAT, F, TAU, 'straight', {})
        np.testing.assert_allclose(out_f, F * 1.6)

    def test_decelerating_rule_fires_when_no_competition(self):
        """A 0.5× rule must override the baseline (deviation 0.5 > 0)."""
        rule_down = VelocityScalingRule('any', f_scale=0.5, tau_scale=1.0)
        _, _, out_f, _ = apply_rules_max_wins([rule_down], POS, QUAT, F, TAU, 'straight', {})
        np.testing.assert_allclose(out_f, F * 0.5)

    def test_stronger_decel_beats_weaker_decel(self):
        """A 0.4× rule (dev 0.6) must beat a 0.8× rule (dev 0.2)."""
        rule_strong = VelocityScalingRule('any', f_scale=0.4, tau_scale=1.0)
        rule_weak   = VelocityScalingRule('any', f_scale=0.8, tau_scale=1.0)
        _, _, out_f, _ = apply_rules_max_wins([rule_strong, rule_weak], POS, QUAT, F, TAU, 'straight', {})
        np.testing.assert_allclose(out_f, F * 0.4)

    def test_f_and_tau_selected_independently(self):
        """Rule A wins on f; rule B wins on tau — they should not force each other."""
        # rule_a: big f, small tau
        rule_a = VelocityScalingRule('any', f_scale=2.0, tau_scale=0.5)
        # rule_b: small f, big tau
        rule_b = VelocityScalingRule('any', f_scale=0.5, tau_scale=2.0)
        _, _, out_f, out_tau = apply_rules_max_wins([rule_a, rule_b], POS, QUAT, F, TAU, 'straight', {})
        np.testing.assert_allclose(out_f, F * 2.0)
        np.testing.assert_allclose(out_tau, TAU * 2.0)

    def test_proximity_rule_participates_in_same_pool(self):
        """GeometryProximityRule (0.6×) must not override VelocityScalingRule (1.4×)."""
        vel_rule  = VelocityScalingRule('any', f_scale=1.4, tau_scale=1.0)
        prox_rule = GeometryProximityRule('edge', dist_threshold=0.05, f_scale=0.6)
        proximity = {"dist_to_edge": 0.01}  # within threshold
        _, _, out_f, _ = apply_rules_max_wins(
            [vel_rule, prox_rule], POS, QUAT, F, TAU, 'straight', proximity
        )
        np.testing.assert_allclose(out_f, F * 1.4)

    def test_proximity_rule_wins_when_larger(self):
        """GeometryProximityRule (1.5×) must beat VelocityScalingRule (1.2×)."""
        vel_rule  = VelocityScalingRule('any', f_scale=1.2, tau_scale=1.0)
        prox_rule = GeometryProximityRule('edge', dist_threshold=0.05, f_scale=1.5)
        proximity = {"dist_to_edge": 0.01}
        _, _, out_f, _ = apply_rules_max_wins(
            [vel_rule, prox_rule], POS, QUAT, F, TAU, 'straight', proximity
        )
        np.testing.assert_allclose(out_f, F * 1.5)

    def test_pos_and_quat_passed_through(self):
        rule = VelocityScalingRule('any', f_scale=1.0)
        out_pos, out_quat, _, _ = apply_rules_max_wins([rule], POS, QUAT, F, TAU, 'straight', {})
        np.testing.assert_array_equal(out_pos, POS)
        np.testing.assert_array_equal(out_quat, QUAT)


# ---------------------------------------------------------------------------
# compute_proximity_data
# ---------------------------------------------------------------------------

class TestComputeProximityData:
    def _make_nodes(self, positions, node_type, other_type):
        """Build locations list from a list of positions."""
        return [{"position": np.array(p), "type": node_type} for p in positions]

    def test_empty_locations_gives_empty_dict(self):
        edge_nodes    = np.empty((0, 3))
        crossing_nodes = np.empty((0, 3))
        result = compute_proximity_data(np.zeros(3), edge_nodes, crossing_nodes)
        assert result == {}

    def test_no_crossing_nodes_key_absent(self):
        edge_nodes    = np.array([[1.0, 0.0, 0.0]])
        crossing_nodes = np.empty((0, 3))
        result = compute_proximity_data(np.zeros(3), edge_nodes, crossing_nodes)
        assert "dist_to_crossing" not in result

    def test_no_edge_nodes_key_absent(self):
        edge_nodes    = np.empty((0, 3))
        crossing_nodes = np.array([[1.0, 0.0, 0.0]])
        result = compute_proximity_data(np.zeros(3), edge_nodes, crossing_nodes)
        assert "dist_to_edge" not in result

    def test_single_edge_node_exact_distance(self):
        node = np.array([3.0, 4.0, 0.0])
        pos  = np.array([0.0, 0.0, 0.0])
        edge_nodes    = node.reshape(1, 3)
        crossing_nodes = np.empty((0, 3))
        result = compute_proximity_data(pos, edge_nodes, crossing_nodes)
        assert result["dist_to_edge"] == pytest.approx(5.0)

    def test_returns_minimum_distance_among_multiple_nodes(self):
        edge_nodes = np.array([
            [10.0, 0.0, 0.0],
            [3.0,  4.0, 0.0],  # dist = 5
            [1.0,  0.0, 0.0],  # dist = 1  ← minimum
        ])
        crossing_nodes = np.empty((0, 3))
        result = compute_proximity_data(np.zeros(3), edge_nodes, crossing_nodes)
        assert result["dist_to_edge"] == pytest.approx(1.0)

    def test_edge_and_crossing_distances_independent(self):
        edge_nodes    = np.array([[2.0, 0.0, 0.0]])   # dist = 2
        crossing_nodes = np.array([[0.0, 5.0, 0.0]])  # dist = 5
        result = compute_proximity_data(np.zeros(3), edge_nodes, crossing_nodes)
        assert result["dist_to_edge"]    == pytest.approx(2.0)
        assert result["dist_to_crossing"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Integration: proximity rule affects velocity in convert_path_2_trajectory
# ---------------------------------------------------------------------------

def _make_straight_path(n=30, length=1.0):
    """Single-stroke path along Z."""
    data = np.zeros((n, 7))
    data[:, 2] = np.linspace(0.0, length, n)
    data[:, 3] = -90.0
    data[:, 4] = -70.0
    return data


class TestProximityRuleIntegration:
    def test_proximity_rule_slows_robot_near_node(self):
        """
        A strongly decelerating proximity rule (f_scale=0.1) covering the entire
        path must cause the robot to take longer to traverse the arc — the final
        time stamp in column 0 must be larger than the no-rule baseline.
        """
        path = _make_straight_path(n=30, length=1.0)

        # Threshold=2.0 covers the whole 1m path from any position on it
        locations = [{"position": path[0, 0:3].copy(), "type": "edge", "degree": 1}]
        rule = GeometryProximityRule(node_type='edge', dist_threshold=2.0, f_scale=0.1)

        baseline  = convert_path_2_trajectory(path, rules=[], locations=[])
        with_rule = convert_path_2_trajectory(path, rules=[rule], locations=locations)

        assert with_rule[-1, 0] > baseline[-1, 0], (
            "A decelerating proximity rule must increase traversal time"
        )

    def test_no_locations_baseline_unchanged(self):
        """With empty locations and a proximity rule, output must equal no-rule baseline."""
        path = _make_straight_path()
        rule = GeometryProximityRule(node_type='edge', dist_threshold=1.0, f_scale=0.5)

        baseline  = convert_path_2_trajectory(path, rules=[], locations=[])
        with_rule = convert_path_2_trajectory(path, rules=[rule], locations=[])

        np.testing.assert_allclose(baseline[:, 0], with_rule[:, 0], atol=1e-10)


# ---------------------------------------------------------------------------
# sample_geometric_proximity_rule
# ---------------------------------------------------------------------------

class TestSampleGeometricProximityRule:
    N_SAMPLES = 200  # enough to cover both branches

    def _collect(self):
        rules, metas = [], []
        for _ in range(self.N_SAMPLES):
            rule, meta = sample_geometric_proximity_rule(is_ood=False)
            if rule is not None:
                rules.append(rule)
                metas.append(meta)
        return rules, metas

    def test_returns_tuple_of_rule_and_dict(self):
        result = sample_geometric_proximity_rule(is_ood=False)
        assert isinstance(result, tuple) and len(result) == 2
        assert isinstance(result[1], dict)

    def test_rule_is_geometry_proximity_rule_or_none(self):
        for _ in range(50):
            rule, _ = sample_geometric_proximity_rule(is_ood=False)
            assert rule is None or isinstance(rule, GeometryProximityRule)

    def test_node_type_is_edge_or_crossing(self):
        rules, _ = self._collect()
        for rule in rules:
            assert rule.node_type in ('edge', 'crossing')

    def test_both_node_types_appear(self):
        rules, _ = self._collect()
        types = {r.node_type for r in rules}
        assert 'edge' in types
        assert 'crossing' in types

    def test_dist_threshold_in_range(self):
        rules, _ = self._collect()
        for rule in rules:
            assert 0.02 <= rule.dist_threshold <= 0.08

    def test_f_scale_avoids_neutral_zone(self):
        """f_scale must be in [0.5, 0.9] or [1.1, 1.5] — never [0.9, 1.1]."""
        rules, _ = self._collect()
        for rule in rules:
            in_low  = 0.5 <= rule.f_scale <= 0.9
            in_high = 1.1 <= rule.f_scale <= 1.5
            assert in_low or in_high, f"f_scale {rule.f_scale} outside allowed ranges"

    def test_both_f_scale_ranges_appear(self):
        rules, _ = self._collect()
        low  = any(rule.f_scale <= 0.9 for rule in rules)
        high = any(rule.f_scale >= 1.1 for rule in rules)
        assert low and high

    def test_tau_scale_is_one(self):
        rules, _ = self._collect()
        for rule in rules:
            assert rule.tau_scale == pytest.approx(1.0)

    def test_metadata_keys_present(self):
        _, meta = sample_geometric_proximity_rule(is_ood=False)
        assert 'geometric_proximity' in meta
        assert 'geometric_proximity_geom' in meta

    def test_is_ood_accepted_without_error(self):
        sample_geometric_proximity_rule(is_ood=True)
