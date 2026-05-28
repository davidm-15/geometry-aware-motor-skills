import json
import numpy as np
from pathlib import Path
from typing import Union
from scipy.spatial.transform import Rotation as R
from more_itertools import sliding_window
from data_generation.simulation import VirtualEndEffector
from data_generation.rules import VelocityScalingRule, SpatialPositionRule, SpatialOrientationRule, apply_rules_max_wins
from data_generation.skeleton_converter import WindowSkeletonConverter
import trimesh
import os

def load_path(file_path: Union[str, Path]) -> np.ndarray:
    """
    Loads data from MaskPlanner format csv file (X, Y, Z, A, B, C, index)
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Could not find file: {file_path}")

    data = np.loadtxt(file_path, delimiter=";", skiprows=1)

    return data


def compute_proximity_data(pos: np.ndarray, edge_nodes: np.ndarray, crossing_nodes: np.ndarray) -> dict:
    """
    Return a dict with the minimum distances from pos to each node type.
    Keys are only present when the corresponding node array is non-empty.
    """
    result = {}
    if len(edge_nodes):
        result["dist_to_edge"] = float(np.min(np.linalg.norm(edge_nodes - pos, axis=1)))
    if len(crossing_nodes):
        result["dist_to_crossing"] = float(np.min(np.linalg.norm(crossing_nodes - pos, axis=1)))
    return result


def convert_path_2_trajectory(
    data: np.ndarray,
    mass: float = 0.5,
    cv: float = 1.0,
    comega: float = 4.0,
    kp_linear: float = 50.0,
    kp_angular: float = 10.0,
    lookahead_distance: float = 0.05,
    inertia_scale: float = 2.0,
    dt = 0.01,
    rules: list = None,
    segments: list = None,
    locations: list = None,
) -> np.ndarray:
    """
    Converts the MaskPlanner (X,Y,Z,A,B,C,ID) format to RoboTwin format
    (time(s),x,y,z,qx,qy,qz,qw,velocity,ID)

    Parameters for Physical Dynamics:
    - mass: Mass of the virtual end-effector (Linear inertia)
    - cv: Translational damping coefficient (Prevents overshoot)
    - comega: Rotational damping coefficient (Reduces rotational oscillation)
    - kp_linear: P-gain for position tracking stiffness
    - kp_angular: P-gain for orientation tracking stiffness
    - lookahead_distance: Target distance for the Pure Pursuit controller
    - inertia_scale: Multiplier for the Identity Inertia Matrix (Rotational mass)
    """

    I_matrix = np.eye(3) * inertia_scale


    if data.size == 0:
        return np.empty((0, 10), dtype=float)
    

    r = R.from_euler('xyz', data[:, 3:6], degrees=True) 
    quats = r.as_quat()
    ids = data[:, -1]
    stroke_starts = np.flatnonzero(np.diff(ids) != 0) + 1
    stroke_bounds = np.concatenate(([0], stroke_starts, [len(data)]))

    all_velocities = []
    all_time_steps = []

    all_out_positions = []
    all_out_quats = []
    all_ids = []

    eps = 1e-9

    if segments is None:
        segments = split_path_2_segments(data, curvature_threshold=0.001, slice_size=15)

    locs = locations or []
    edge_nodes     = np.array([l["position"] for l in locs if l["type"] == "edge"],    dtype=float).reshape(-1, 3)
    crossing_nodes = np.array([l["position"] for l in locs if l["type"] == "crossing"], dtype=float).reshape(-1, 3)

    for stroke_idx, (start, end) in enumerate(zip(stroke_bounds[:-1], stroke_bounds[1:])):

        positions = data[start:end, 0:3]
        stroke_quats = quats[start:end]
        n_points = end - start

        # If segments are provided, they should match the total data length
        stroke_segments = segments[start:end] if segments is not None else ['straight'] * n_points


        if n_points == 1:
            continue

        simulator = VirtualEndEffector(
            m=mass, 
            I=I_matrix, 
            cv=cv, 
            comega=comega, 
            dt=dt, 
            lookahead_distance=lookahead_distance,
            kp_linear=kp_linear,
            kp_angular=kp_angular,
        )

        simulator.reset(positions[0], stroke_quats[0])

        current_idx = 1
        current_speed = 0.0
        max_steps = n_points * 50

        velocities = [np.linalg.norm(simulator.v.copy())]
        out_positions = [simulator.p.copy()]
        out_quats = [simulator.q.copy()]
        ids = [stroke_idx]

        for step_idx in range(max_steps):
            # 1. Get Lookahead Point
            target_pos, target_quat, lp_idx = simulator.controller.get_lookahead_point(
                simulator.p, positions, stroke_quats, start_idx=current_idx
            )
            segment_type = stroke_segments[current_idx]
            
            current_idx = lp_idx
            
            # Apply Rule-based Target Modifications
            if rules:
                for rule in rules:
                    target_pos, target_quat = rule.modify_target(target_pos, target_quat, segment_type)

            # Compute Base Control
            f_ctrl, tau_ctrl = simulator.controller.compute_control(
                simulator.p,
                simulator.q,
                target_pos,
                target_quat,
            )

            velocities.append(np.linalg.norm(simulator.v.copy()))
            out_positions.append(simulator.p.copy())
            out_quats.append(simulator.q.copy())
            ids.append(stroke_idx)

            if rules:
                proximity_data = compute_proximity_data(simulator.p, edge_nodes, crossing_nodes)
                _, _, f_ctrl, tau_ctrl = apply_rules_max_wins(
                    rules, simulator.p, simulator.q, f_ctrl, tau_ctrl, segment_type, proximity_data
                )

            simulator.step(f_ctrl, tau_ctrl, f_noise_std=0.0, tau_noise_std=0.0)

            

            acceleration = velocities[-1] - velocities[-2] if len(velocities) > 1 else np.zeros(3)
            full_vel = np.linalg.norm(simulator.v) 
            
            

            if (current_idx >= n_points - 1) and (acceleration < 0.1) and (full_vel < 0.1):
                break


        

        # Subsample simulated trajectory to n_points equidistant along arc length,
        # matching the original path point count so sequence lengths stay manageable.
        out_pos_arr = np.array(out_positions)
        out_quat_arr = np.array(out_quats)
        vel_arr = np.array(velocities, dtype=float)
        time_arr = np.arange(len(velocities), dtype=float) * dt

        seg_diffs = np.diff(out_pos_arr, axis=0)
        cum_lengths = np.concatenate([[0.0], np.cumsum(np.linalg.norm(seg_diffs, axis=1))])
        total_length = cum_lengths[-1]

        if total_length > eps and n_points > 1:
            targets = np.linspace(0.0, total_length, n_points)
            pos_r = np.column_stack([np.interp(targets, cum_lengths, out_pos_arr[:, i]) for i in range(3)])
            quat_r = np.column_stack([np.interp(targets, cum_lengths, out_quat_arr[:, i]) for i in range(4)])
            quat_r /= np.maximum(np.linalg.norm(quat_r, axis=1, keepdims=True), eps)
            vel_r = np.interp(targets, cum_lengths, vel_arr)
            time_r = np.interp(targets, cum_lengths, time_arr)
            ids_r = np.full(n_points, stroke_idx)
        else:
            n = min(n_points, len(out_pos_arr))
            pos_r, quat_r = out_pos_arr[:n], out_quat_arr[:n]
            vel_r, time_r = vel_arr[:n], time_arr[:n]
            ids_r = np.full(n, stroke_idx)

        all_velocities.append(vel_r)
        all_time_steps.append(time_r)
        all_out_positions.append(pos_r)
        all_out_quats.append(quat_r)
        all_ids.append(ids_r)
    

    all_velocities = np.concatenate(all_velocities, axis=0)
    all_time_steps = np.concatenate(all_time_steps, axis=0)
    out_positions = np.concatenate(all_out_positions, axis=0)
    out_quats = np.concatenate(all_out_quats, axis=0)
    all_ids = np.concatenate(all_ids, axis=0)

    all_velocities = all_velocities.reshape(-1, 1)
    all_time_steps = all_time_steps.reshape(-1, 1)

    # Positions and velocities remain in simulation units (mm).

    data = np.hstack((all_time_steps, out_positions, out_quats, all_velocities, all_ids.reshape(-1, 1)))

    return data
    

def load_robotwin_trajectory(file_path: Union[str, Path]) -> np.ndarray:
    """
    Loads data from RoboTwin format csv file (time(s),x,y,z,qx,qy,qz,qw,velocity,ID)
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Could not find file: {file_path}")

    data = np.loadtxt(file_path, delimiter=",", skiprows=1)

    return data


def split_path_2_segments(path: np.ndarray, curvature_threshold: float = 0.001, slice_size: int = 15) -> list:
    """
    Splits the given path to segments by their curvature.
    The structure must be at leasr (x, y, z, ..., ID) where ID is used to separate strokes,
    x, y, z has to be in the first 3 columns, ID has to be in the last column. 
    The output is a list of segment types (e.g. 'straight' or 'corner') for each point in the path.
    """
    IDs = np.unique(path[:, -1])
    all_straight = []


    for ID in IDs:
        points = path[path[:, -1] == ID][:, 0:3]
        straight = []

        for window in sliding_window(points, slice_size):
            pts = np.array(window)
            centroid = pts.mean(axis=0)
            centered_pts = pts - centroid

            _, _, vh = np.linalg.svd(centered_pts)
            direction = vh[0]

            dist_sq = np.sum(centered_pts**2) - np.sum(np.dot(centered_pts, direction)**2)

            straight.append(dist_sq < curvature_threshold)

        pad_width = slice_size // 2
        straight_padded = [straight[0]] * pad_width + straight + [straight[-1]] * (len(points) - len(straight) - pad_width)

        # Majority-vote smoothing: a point is straight only if most windows covering it agree.
        straight_arr = np.array(straight_padded, dtype=float)
        kernel = np.ones(slice_size) / slice_size
        smoothed = np.convolve(straight_arr, kernel, mode='same')
        straight_padded = list(smoothed >= 0.5)

        all_straight.append(straight_padded)
    

    all_straight = np.concatenate(all_straight)
    all_straight = ['straight' if x else 'corner' for x in all_straight]

    return all_straight



def split_mesh_2_locations_from_file(cad_model_path: str) -> list:
    """
    Loads a CAD model from a file path and splits it into locations by their geometry.
    
    Args:
        cad_model_path (str): Path to the .obj CAD model file.
        
    Returns:
        list: A list of dictionaries containing the position, type, and degree of each location.
    """
    if not os.path.exists(cad_model_path):
        print(f"Error: File {cad_model_path} does not exist.")
        return []
        
    try:
        # Load the mesh from the file path
        mesh = trimesh.load(cad_model_path, force='mesh')
        
        # Use the existing function to process the mesh
        return split_mesh_2_locations(mesh)
        
    except Exception as e:
        print(f"Error loading mesh from {cad_model_path}: {e}")
        return []


def split_mesh_2_locations(mesh) -> list:
    """
    Splits the given trimesh object to locations by their geometry.
    Output is a list of locations and the type of location, either 'edge' or 'crossing'.
    
    Args:
        mesh (trimesh.Trimesh): The loaded 3D mesh object.
        
    Returns:
        list: A list of dictionaries containing the position, type, and degree of each location.
    """
    converter = WindowSkeletonConverter()
    
    try:
        # Pass the mesh directly to the updated method
        skeleton = converter.extract_skeleton(mesh)
    except Exception as e:
        print(f"Error extracting skeleton: {e}")
        return []

    nodes = skeleton["nodes"]
    degrees = skeleton["degrees"]
    
    locations = []
    
    for i, node in enumerate(nodes):
        degree = degrees[i]
        
        # A node with 3 or more connected edges is an intersection/crossing.
        # Nodes with 1 or 2 connections are endpoints or segments along an 'edge'.
        location_type = "crossing" if degree >= 3 else "edge"
        
        locations.append({
            "position": node,
            "type": location_type,
            "degree": degree
        })
        
    return locations




if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    file_path = "datasets/L_shape/7_L_shape/trajectory.txt"
    path = load_path(file_path)
    path_segmented = split_path_2_segments(path, curvature_threshold=0.001, slice_size=100)
    print(np.unique(path_segmented, return_counts=True))

    speed_rule = VelocityScalingRule("corner", f_scale=2, tau_scale=2)
    orientation_rule = SpatialOrientationRule("straight", rotvec=[0.3, 0.0, 0.0])
    human_noise_rule = HumanNoiseRule('any', amplitude=5, frequency=10.0)
    

    rules = [speed_rule, orientation_rule, human_noise_rule]
    trajectory = convert_path_2_trajectory(path, v_base=20.0, a_base=30.0, segments=path_segmented, rules=rules)
    output_dir = Path("datasets/test_one_by_one")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "trajectory.csv"
    obj_files = list(Path(file_path).parent.glob("*.obj"))
    if not obj_files:
        raise FileNotFoundError(f"No .obj file found in {Path(file_path).parent}")

    shutil.copy2(obj_files[0], output_dir / "7_L_shape")

    window_id = Path(file_path).parent.name
    header = [
        "time(s)",
        "x",
        "y",
        "z",
        "qx",
        "qy",
        "qz",
        "qw",
        "velocity",
        "stroke_id",
        "segment_type",
        "window_id",
        "rule_vel_scale",
        "rule_pos_y",
        "rule_ori_x",
        "rule_vel_scale_geom",
        "rule_pos_y_geom",
        "rule_ori_x_geom",
    ]

    rule_vel_scale = 2.0
    rule_pos_y = 0.0
    rule_ori_x = 0.1
    rule_vel_scale_geom = "corner"
    rule_pos_y_geom = "none"
    rule_ori_x_geom = "straight"

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for idx, row in enumerate(trajectory):
            writer.writerow([
                row[0],
                row[1],
                row[2],
                row[3],
                row[4],
                row[5],
                row[6],
                row[7],
                row[8],
                int(row[9]),
                path_segmented[idx],
                window_id,
                rule_vel_scale,
                rule_pos_y,
                rule_ori_x,
                rule_vel_scale_geom,
                rule_pos_y_geom,
                rule_ori_x_geom,
            ])

    print(f"Saved trajectory to {output_file}")