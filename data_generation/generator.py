import numpy as np
from pathlib import Path
import pandas as pd
import json
import random

from data_generation.path_parser import parse_trajectory_file
from data_generation.simulation import VirtualEndEffector
from data_generation.rules import VelocityScalingRule, SpatialPositionRule, SpatialOrientationRule

# python -m data_generation.generator

def simulate_stroke(window_id, stroke_subpath, is_ood, dt=0.01):
    """
    Simulates a single stroke over a subpath using either ID or OOD rule ranges.
    Returns a list of data rows for the trajectory.
    """
    stroke_id = stroke_subpath['stroke_id']
    positions = stroke_subpath['positions']
    quats = stroke_subpath['quaternions']
    segments = stroke_subpath['segments']
    
    if len(positions) == 0:
        return []

    # Load skeleton for proximity rules
    skeleton_file = Path("datasets/windows-v2/0_skeletons") / f"{window_id}.json"
    skeleton_nodes = []
    edge_nodes = []
    crossing_nodes = []
    max_extent = 1000.0 # fallback
    
    if skeleton_file.exists():
        with open(skeleton_file, 'r') as f:
            skel = json.load(f)
            skeleton_nodes = np.array(skel['nodes'])
            degrees = skel.get('degrees', [])
            max_extent = skel.get('metadata', {}).get('max_extent', 1000.0)
            
            for i, deg in enumerate(degrees):
                if deg == 2:
                    edge_nodes.append(skeleton_nodes[i])
                elif deg >= 3:
                    crossing_nodes.append(skeleton_nodes[i])
    
    edge_nodes = np.array(edge_nodes) if edge_nodes else None
    crossing_nodes = np.array(crossing_nodes) if crossing_nodes else None

    from data_generation.rules import GeometryProximityRule
    
    # Define ranges based on In-Distribution (ID) vs Out-Of-Distribution (OOD)
    if not is_ood:
        vel_scale_range = (0.8, 1.2)
        pos_z_range = (-0.02, 0.02)
        ori_x_range = (-0.05, 0.05)
        # Proximity thresholds and scales
        prox_thresh_range = (0.05, 0.15)
        prox_f_scale_range = (0.5, 0.8) # Slow down near nodes
    else:
        # OOD settings
        vel_scale_range = random.choice([(0.4, 0.7), (1.3, 1.8)])
        pos_z_range = random.choice([(-0.06, -0.03), (0.03, 0.06)])
        ori_x_range = random.choice([(-0.15, -0.08), (0.08, 0.15)])
        # OOD proximity: extreme slow down or speed up
        prox_thresh_range = (0.2, 0.3)
        prox_f_scale_range = random.choice([(0.2, 0.4), (1.2, 1.5)])

    # Randomly select actual values from ranges
    f_scale_val = np.random.uniform(*vel_scale_range) if np.random.rand() > 0.5 else 1.0
    pos_z_val = np.random.uniform(*pos_z_range) if np.random.rand() > 0.5 else 0.0
    ori_x_val = np.random.uniform(*ori_x_range) if np.random.rand() > 0.5 else 0.0
    
    active_rules = []
    if f_scale_val != 1.0:
        active_rules.append(VelocityScalingRule('straight', f_scale=f_scale_val, tau_scale=f_scale_val))
    if pos_z_val != 0.0:
        active_rules.append(SpatialPositionRule('straight', offset=[0.0, 0.0, pos_z_val]))
    if ori_x_val != 0.0:
        active_rules.append(SpatialOrientationRule('corner', rotvec=[ori_x_val, 0, 0]))

    # Add Proximity Rules
    if np.random.rand() > 0.3:
        n_type = random.choice(['edge', 'crossing'])
        t_val = np.random.uniform(*prox_thresh_range)
        s_val = np.random.uniform(*prox_f_scale_range)
        active_rules.append(GeometryProximityRule(n_type, dist_threshold=t_val, f_scale=s_val))

    simulator = VirtualEndEffector(dt=dt)
    simulator.reset(positions[0], quats[0])
    
    step_count = 0
    max_steps = min(len(positions) * 10, 1500)
    stroke_data = []

    while step_count < max_steps:
        target_pos, target_quat, target_idx = simulator.controller.get_lookahead_point(
            simulator.p, positions, quats
        )
        segment_type = segments[target_idx]
        
        # 1. Standard modifications
        for rule in active_rules:
            target_pos, target_quat = rule.modify_target(target_pos, target_quat, segment_type)
        
        # 2. Control computation
        f_ctrl, tau_ctrl = simulator.controller.compute_control(
            simulator.p, simulator.q, target_pos, target_quat
        )
        
        # 3. Proximity calculations
        prox_data = {'max_extent': max_extent}
        if edge_nodes is not None:
            dist_e = np.min(np.linalg.norm(edge_nodes - simulator.p, axis=1))
            prox_data['dist_to_edge'] = dist_e
        if crossing_nodes is not None:
            dist_c = np.min(np.linalg.norm(crossing_nodes - simulator.p, axis=1))
            prox_data['dist_to_crossing'] = dist_c
            
        # 4. Proximity rule application
        for rule in active_rules:
            f_ctrl, tau_ctrl = rule.modify_control(f_ctrl, tau_ctrl, segment_type)
            # Use the new combined method for proximity-aware modification
            target_pos, target_quat, f_ctrl, tau_ctrl = rule.modify_with_proximity(
                target_pos, target_quat, f_ctrl, tau_ctrl, prox_data
            )
        
        t, p, q, v_norm, w_norm = simulator.step(f_ctrl, tau_ctrl)
        
        # Simulation runs in mm; convert positions and speed to metres for output.
        stroke_data.append([
            t, p[0]*1e-3, p[1]*1e-3, p[2]*1e-3, q[0], q[1], q[2], q[3], v_norm*1e-3, stroke_id, window_id,
            f_scale_val, pos_z_val, ori_x_val
        ])
        
        dist_to_end = np.linalg.norm(simulator.p - positions[-1])
        if dist_to_end < 1.0: 
            break
            
        step_count += 1

    return stroke_data

def generate_multi_window_dataset(window_list, output_csv, is_ood, num_samples):
    """
    Generates a dataset by randomly sampling paths from the provided window list.
    """
    all_data = []
    samples_generated = 0
    
    print(f"Generating {'OOD' if is_ood else 'ID'} dataset: target {num_samples} samples...")
    
    while samples_generated < num_samples:
        window_id = random.choice(window_list)
        traj_file = Path("datasets/windows-v2") / window_id / "trajectory.txt"
        
        if not traj_file.exists():
            continue
            
        try:
            subpaths = parse_trajectory_file(str(traj_file))
        except Exception:
            continue
            
        if not subpaths:
            continue
            
        # We consider a full 'window pass' (all its subpaths) as 1 sample
        # Or you can consider a single subpath as 1 sample. Let's do single subpath.
        sub = random.choice(subpaths)
        stroke_data = simulate_stroke(window_id, sub, is_ood=is_ood)
        
        if stroke_data:
            all_data.extend(stroke_data)
            samples_generated += 1
            
        if samples_generated % 50 == 0:
            print(f"Progress: {samples_generated}/{num_samples}")

    columns = [
        'time(s)', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'velocity', 'stroke_id', 'window_id',
        'rule_vel_scale', 'rule_pos_z', 'rule_ori_x'
    ]
    df = pd.DataFrame(all_data, columns=columns)
    
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[✓] Saved {len(df)} trajectory steps to {output_csv}")

def main():
    base_dir = Path("datasets/windows-v2")
    
    with open(base_dir / "train_split.json", 'r') as f:
        train_windows = json.load(f)
        
    with open(base_dir / "test_split.json", 'r') as f:
        test_windows = json.load(f)
        
    # Generate 800 ID samples for training
    generate_multi_window_dataset(
        window_list=train_windows, 
        output_csv="datasets/training/train.csv",
        is_ood=False, 
        num_samples=800
    )
    
    # Generate OOD samples for testing
    generate_multi_window_dataset(
        window_list=test_windows, 
        output_csv="datasets/training/test.csv",
        is_ood=True, 
        num_samples=200
    )

if __name__ == "__main__":
    main()
