import numpy as np
from scipy.spatial.transform import Rotation as R

def euler_to_quaternion(euler_angles_deg):
    """
    Converts Euler angles (A, B, C) in degrees to quaternions (qx, qy, qz, qw).
    Expected order: XYZ extrinsic/intrinsic matching the reference project.
    """
    r = R.from_euler('xyz', euler_angles_deg, degrees=True)
    return r.as_quat()

def get_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))
