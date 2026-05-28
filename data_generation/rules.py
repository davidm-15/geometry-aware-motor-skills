import random
import math
import numpy as np
from abc import ABC, abstractmethod
from scipy.spatial.transform import Rotation as R

class Rule(ABC):
    @property
    @abstractmethod
    def name(self):
        pass

    def modify_target(self, target_pos, target_quat, segment_type):
        """Override to modify the lookahead target before pure pursuit calculations."""
        return target_pos, target_quat

    def modify_control(self, f_ctrl, tau_ctrl, segment_type):
        """Override to modify the resulting control forces applied to the rigid body."""
        return f_ctrl, tau_ctrl

    def modify_with_proximity(self, pos, quat, f_ctrl, tau_ctrl, proximity_data):
        """
        New method to handle rules that depend on proximity to skeletal nodes.
        proximity_data: dict with 'dist_to_edge', 'dist_to_crossing', 'max_extent'
        """
        return pos, quat, f_ctrl, tau_ctrl


class VelocityScalingRule(Rule):
    """Scales the control thrust applied to the end-effector to increase or decrease speed."""
    def __init__(self, target_segment, f_scale=1.0, tau_scale=1.0):
        self.target_segment = target_segment
        self.f_scale = f_scale
        self.tau_scale = tau_scale
        
    @property
    def name(self):
        return f"VelocityScale_{self.target_segment}_f{self.f_scale:.2f}"
        
    def modify_control(self, f_ctrl, tau_ctrl, segment_type):
        if segment_type == self.target_segment or self.target_segment == 'any':
            return f_ctrl * self.f_scale, tau_ctrl * self.tau_scale
        return f_ctrl, tau_ctrl


class GeometryProximityRule(Rule):
    """
    Modifies control forces based on proximity to skeletal nodes (edges or crossings).
    dist_threshold is in metres (absolute).
    orientation_offset is stored but unused — reserved for future use.
    """
    def __init__(self, node_type, dist_threshold=0.1, tau_scale=1.0, f_scale=1.0, orientation_offset=None):
        self.node_type = node_type  # 'edge' (deg 2) or 'crossing' (deg 3+)
        self.dist_threshold = dist_threshold
        self.f_scale = f_scale
        self.tau_scale = tau_scale
        self.orientation_offset = orientation_offset  # Optional [rv_x, rv_y, rv_z]

    @property
    def name(self):
        return f"Proximity_{self.node_type}_t{self.dist_threshold}"

    def modify_with_proximity(self, pos, quat, f_ctrl, tau_ctrl, proximity_data):
        dist = proximity_data.get(f'dist_to_{self.node_type}', float('inf'))
        if dist < self.dist_threshold:
            return pos, quat, f_ctrl * self.f_scale, tau_ctrl * self.tau_scale
        return pos, quat, f_ctrl, tau_ctrl


class SpatialPositionRule(Rule):
    """Applies a continuous 3D offset to the target path (e.g. pulling the spray gun further away)."""
    def __init__(self, target_segment, offset):
        self.target_segment = target_segment
        self.offset = np.array(offset, dtype=float)
        
    @property
    def name(self):
        return f"PosOffset_{self.target_segment}_{self.offset}"
        
    def modify_target(self, target_pos, target_quat, segment_type):
        if segment_type == self.target_segment or self.target_segment == 'any':
            return target_pos + self.offset, target_quat
        return target_pos, target_quat


class SpatialOrientationRule(Rule):
    """Applies a continuous rotational offset to the target pose."""
    def __init__(self, target_segment, rotvec):
        self.target_segment = target_segment
        self.rot_offset = R.from_rotvec(rotvec, degrees=False)
        
    @property
    def name(self):
        return f"OriOffset_{self.target_segment}"
        
    def modify_target(self, target_pos, target_quat, segment_type):
        if segment_type == self.target_segment or self.target_segment == 'any':
            r_target = R.from_quat(target_quat)
            r_modified = self.rot_offset * r_target
            return target_pos, r_modified.as_quat()

        return target_pos, target_quat


def apply_rules_max_wins(rules, pos, quat, f_ctrl, tau_ctrl, segment_type, proximity_data):
    """
    Apply all rules using a max-deviation-wins policy: every rule sees the base
    forces, proposes its own scaled result, and the candidate with the largest
    deviation from the base force wins for f and tau independently.
    Rules never compound, and decelerating rules can win just as accelerating
    ones can.

    Returns (pos, quat, f_ctrl, tau_ctrl).
    """
    best_f       = f_ctrl.copy()
    best_tau     = tau_ctrl.copy()
    best_f_dev   = 0.0
    best_tau_dev = 0.0

    for rule in rules:
        cf, ct = rule.modify_control(f_ctrl, tau_ctrl, segment_type)
        f_dev = np.linalg.norm(cf - f_ctrl)
        if f_dev > best_f_dev:
            best_f, best_f_dev = cf, f_dev
        tau_dev = np.linalg.norm(ct - tau_ctrl)
        if tau_dev > best_tau_dev:
            best_tau, best_tau_dev = ct, tau_dev

    for rule in rules:
        _, _, cf, ct = rule.modify_with_proximity(pos, quat, f_ctrl, tau_ctrl, proximity_data)
        f_dev = np.linalg.norm(cf - f_ctrl)
        if f_dev > best_f_dev:
            best_f, best_f_dev = cf, f_dev
        tau_dev = np.linalg.norm(ct - tau_ctrl)
        if tau_dev > best_tau_dev:
            best_tau, best_tau_dev = ct, tau_dev

    return pos, quat, best_f, best_tau


class HumanNoiseRule(Rule):
    """
    Simulates smooth human motor noise (tremor/drift) by applying pseudo-Perlin 
    noise (summed sine waves) to the lookahead target.
    """
    def __init__(self, target_segment='any', amplitude=0.005, frequency=10.0):
        self.target_segment = target_segment
        self.amplitude = amplitude # Max distance of the wobble (e.g., 0.005 meters)
        self.frequency = frequency # How fast the wobble oscillates
        
        # Random offsets so every simulation has a unique noise pattern
        self.seed_x = random.uniform(0, 1000)
        self.seed_y = random.uniform(0, 1000)
        self.seed_z = random.uniform(0, 1000)
        
        # Internal counter to track "time" since modify_target doesn't receive simulator.time
        self.step_count = 0 
        self.dt = 0.01 # Approximation of time passing

    @property
    def name(self):
        return f"Tremor_{self.target_segment}_a{self.amplitude}_f{self.frequency}"

    def _fractal_noise(self, t, seed):
        """Generates smooth, organic noise between -1 and 1 using summed sines."""
        noise = (
            math.sin(t * self.frequency + seed) * 0.5 + 
            math.sin(t * self.frequency * 2.1 + seed * 1.5) * 0.25 + 
            math.sin(t * self.frequency * 4.3 + seed * 2.0) * 0.125
        )
        return noise / 0.875 # Normalize back to roughly [-1, 1]

    def modify_target(self, target_pos, target_quat, segment_type):
        if segment_type == self.target_segment or self.target_segment == 'any':
            # Advance internal time
            t = self.step_count * self.dt
            self.step_count += 1
            
            # Generate 3D noise vector
            noise_x = self._fractal_noise(t, self.seed_x) * self.amplitude
            noise_y = self._fractal_noise(t, self.seed_y) * self.amplitude
            noise_z = self._fractal_noise(t, self.seed_z) * self.amplitude
            
            noise_vector = np.array([noise_x, noise_y, noise_z])
            
            # Apply the continuous noise offset to the target position
            return target_pos + noise_vector, target_quat
            
        return target_pos, target_quat