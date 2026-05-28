import numpy as np


def get_angular_distances(quaternions: np.ndarray) -> np.ndarray:
    """
    Input is numpy array of quarternoins, output is an array with distances between them
    
    """

    dists = []
    for q1, q2 in zip(quaternions[:-1], quaternions[1:]):
        dot = np.dot(q1, q2)

        if dot < 0:
            dot = -dot

        dot = np.clip(dot, 0.0, 1.0)

        theta = 2*np.arccos(dot)
        dists.append(theta)

    return np.array(dists)



def quat_mult(q1: np.ndarray, q2: np.ndarray):
    """Multiplies two quaternions [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])



if __name__ == "__main__":
    q1 = np.array((0, 0, 0, 1))
    q2 = np.array((0, 0, 0, 1))


    print(quat_mult(q1, q2))