import numpy as np
from scipy.spatial.transform import Rotation as R

class PurePursuitController:
    def __init__(self, lookahead_distance=5.0, kp_linear=10.0, kp_angular=10.0):
        self.lookahead_distance = lookahead_distance
        self.kp_linear = kp_linear
        self.kp_angular = kp_angular

    def get_lookahead_point(self, current_pos, path_positions, path_quats, start_idx=0):
        n = len(path_positions)
        # Advance start_idx to the nearest path point — handles overshoot at corners
        while start_idx + 1 < n:
            if np.linalg.norm(path_positions[start_idx + 1] - current_pos) < \
               np.linalg.norm(path_positions[start_idx] - current_pos):
                start_idx += 1
            else:
                break
        # Scan forward for the first point at lookahead_distance
        for i in range(start_idx, n):
            if np.linalg.norm(path_positions[i] - current_pos) >= self.lookahead_distance:
                return path_positions[i], path_quats[i], i
        return path_positions[-1], path_quats[-1], n - 1

    def compute_control(self, current_pos, current_quat, target_pos, target_quat):
        # Linear tracking
        pos_error = target_pos - current_pos
        f_ctrl = self.kp_linear * pos_error
        
        # Angular tracking
        r_current = R.from_quat(current_quat)
        # Handle zero-norms which can happen accidentally 
        norm_tq = np.linalg.norm(target_quat)
        if norm_tq > 0:
            target_quat_norm = target_quat / norm_tq
        else:
            target_quat_norm = target_quat
        
        r_target = R.from_quat(target_quat_norm)
        r_error = r_target * r_current.inv()
        rot_vec = r_error.as_rotvec()
        tau_ctrl = self.kp_angular * rot_vec
        
        return f_ctrl, tau_ctrl
