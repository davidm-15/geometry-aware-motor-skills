import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

def plot_trajectory(trajectory: np.ndarray) -> None:
    """
    Plots the trajectory in 3D space (x, y, z)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2])
    set_axes_equal(ax)
    plt.show()


def plot_velocity(velocity: np.ndarray) -> None:
    """
    Plots the velocity of the trajectory over time
    """
    plt.plot(velocity)
    plt.xlabel("Time step")
    plt.ylabel("velocity")
    plt.title("Trajectory Velocity over Time")
    plt.show()

def plot_velocity_label(velocity_label: np.ndarray) -> None:
    """
    Plots the velocity of the trajectory over time
    """
    plt.plot(velocity_label[:, 0])
    neighb_labels = velocity_label[1:, 1]-velocity_label[:-1, 1]
    switch_points = np.where(neighb_labels != 0)[0]
    for switch_point in switch_points:
        plt.axvline(x=switch_point, color='r', linestyle='--', alpha=0.7)
    plt.xlabel("Time step")
    plt.ylabel("velocity")
    plt.title("Trajectory Velocity over Time")
    plt.show()


def plot_jerk_acceleration_speed(data: np.ndarray) -> None:
    """
    Plots the jerk, acceleration, and speed of the trajectory over time
    """
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(data[:, -3])
    plt.title("Jerk over Time")
    plt.xlabel("Time step")
    plt.ylabel("Jerk")
    
    plt.subplot(3, 1, 2)
    plt.plot(data[:, -2])
    plt.title("Acceleration over Time")
    plt.xlabel("Time step")
    plt.ylabel("Acceleration")
    
    plt.subplot(3, 1, 3)
    plt.plot(data[:, -1])
    plt.title("Speed over Time")
    plt.xlabel("Time step")
    plt.ylabel("Speed")
    
    plt.tight_layout()
    plt.show()


def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])




def _plot_trajectory_base(ax, data: np.ndarray, label: str = None):
    """Internal helper to plot the 3D line and orientation triads."""
    positions = data[:, :3]
    quaternions = data[:, 3:7]

    # Standard axes
    axes = {'red': [1, 0, 0], 'green': [0, 1, 0], 'blue': [0, 0, 1]}
    
    # Plot orientation arrows (Quivers)
    for color, axis in axes.items():
        directions = np.array([Rotation.from_quat(q).apply(axis) for q in quaternions])
        ax.quiver(
            positions[:, 0], positions[:, 1], positions[:, 2],
            directions[:, 0], directions[:, 1], directions[:, 2],
            length=0.05, normalize=False, alpha=0.6, color=color, arrow_length_ratio=0.3
        )

    # Plot the actual path line
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label=label)


def plot_recorded_trajectory(recorded_trajectory: np.ndarray) -> None:
    """
    Plots the recorded trajectory in 3D space with orientation triads.
    The input trajectory should be in the format (x, y, z, qx, qy, qz, qw)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    _plot_trajectory_base(ax, recorded_trajectory)
    
    plt.title("Recorded Trajectory")
    set_axes_equal(ax)
    plt.show()


def plot_recorded_trajectory_with_labels(recorded_trajectory: np.ndarray) -> None:
    """
    Plots the recorded trajectory in 3D space with orientation triads and labels.
    The input trajectory should be in the format (x, y, z, qx, qy, qz, qw, label)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    

    labels = np.unique(recorded_trajectory[:, -1])
    for label in labels:
        traj_part = recorded_trajectory[recorded_trajectory[:, -1] == label]
        _plot_trajectory_base(ax, traj_part, label=f"Label {int(label)}")    
    
    plt.title("Recorded Trajectory with Labels")
    set_axes_equal(ax)
    plt.legend()
    plt.show()


def plot_continuous_p2p(continuous: np.ndarray, p2p: np.ndarray) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Use the helper for the main line and triads
    _plot_trajectory_base(ax, continuous)
    
    # Add the unique P2P overlay
    ax.scatter(p2p[:, 0], p2p[:, 1], p2p[:, 2], color='black', s=50, marker='o', label='P2P Points')
    
    plt.title("Continuous Trajectory with P2P Points")
    set_axes_equal(ax)
    plt.show()


if __name__ == "__main__":
    from data_generation.load_data import load_robotwin_trajectory
    trajectory = load_robotwin_trajectory("datasets/RoboTeach/CSV_15_31_41.csv")
    plot_recorded_trajectory(trajectory[:, 1:8])

    accleration = np.diff(trajectory[:, 8], prepend=0)
    jerk = np.diff(accleration, prepend=0)
    
    plot_jerk_acceleration_speed(np.hstack((jerk.reshape(-1, 1), accleration.reshape(-1, 1), trajectory[:, 8].reshape(-1, 1))))
