"""
Generate train/val/test CSV datasets from a directory of shape samples.

Each sample directory must contain a ``trajectory.txt`` (MaskPlanner format).

Defaults: generates L-shape dataset from datasets/L_shape/ into datasets/L_shape/
(original behaviour of datasets/L_shape/split_and_generate.py).

Usage:
    # Default L-shape behaviour
    python -m data_generation.generate_dataset

    # Custom dataset
    python -m data_generation.generate_dataset \\
        --samples-dir datasets/MyShape \\
        --sample-name my_shape \\
        --output-dir datasets/MyShape \\
        --total-windows 60 \\
        --train-ratio 0.7 \\
        --val-ratio 0.1


    for I_shape:
    python -m data_generation.generate_dataset --samples-dir datasets/I_shape --sample-name I_shape --output-dir datasets/I_shape --total-windows 1000

    for L-shape
    python -m data_generation.generate_dataset --samples-dir datasets/L_shape --sample-name L_shape --output-dir datasets/L_shape --total-windows 1000

    for Windows dataset:
    python -m data_generation.generate_dataset --samples-dir datasets/windows-v2 --sample-name wr1fr_1 --output-dir datasets/windows-v2 --total-windows 1000


    python -m data_generation.generate_dataset --total-windows 1000 --train-ratio 1.0


    for window_cross dataset:
    python -m data_generation.generate_dataset --samples-dir datasets/window_cross --sample-name window_cross --output-dir datasets/window_cross --total-windows 50

"""

import json
import random
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

from data_generation.rules import HumanNoiseRule, VelocityScalingRule, SpatialPositionRule, SpatialOrientationRule, GeometryProximityRule
from data_generation.load_data import convert_path_2_trajectory, split_mesh_2_locations_from_file, load_path, split_path_2_segments


def _assign_segments_by_position(raw_data: np.ndarray, segments: list, traj_data: np.ndarray) -> list:
    """
    For each simulated trajectory point find the nearest original path point IN THE
    SAME STROKE and return its segment label.

    Direct index assignment (segments[i]) is wrong because arc-length resampling
    shifts simulated positions relative to original path indices by up to ~70 mm.
    """
    orig_positions  = raw_data[:, 0:3]
    orig_stroke_ids = raw_data[:, -1].astype(int)
    sim_positions   = traj_data[:, 1:4]
    sim_stroke_ids  = traj_data[:, 9].astype(int)
    seg_arr         = np.array(segments)

    result = ['straight'] * len(traj_data)
    for sid in np.unique(sim_stroke_ids):
        o_mask = orig_stroke_ids == sid
        s_mask = sim_stroke_ids  == sid
        if not o_mask.any():
            continue
        tree = cKDTree(orig_positions[o_mask])
        orig_segs = seg_arr[o_mask]
        _, nn_idx = tree.query(sim_positions[s_mask])
        for k, global_i in enumerate(np.where(s_mask)[0]):
            result[global_i] = orig_segs[nn_idx[k]]
    return result

# python -m data_generation.generate_dataset


# ── Rule samplers ─────────────────────────────────────────────────────────────;

def sample_velocity_scaling_rule(is_ood):
    # vel_scale_range = random.choice([(0.4, 0.7), (1.3, 1.8)]) if is_ood else random.choice([(0.7, 0.8)])
    vel_scale_range = random.choice([(0.1, 0.9), (1.1, 3.0)])
    f_scale_val = 1.0
    only_seg = False
    rule = None
    if random.random() > 0.33:
        f_scale_val = np.random.uniform(*vel_scale_range)
        if only_seg:
            seg = "straight"
        else:
            seg = random.choice(['straight', 'corner'])
        rule = VelocityScalingRule(seg, f_scale=f_scale_val, tau_scale=f_scale_val)
        geom = seg
    else:
        geom = "none"
    return rule, {"rule_vel_scale": f_scale_val, "rule_vel_scale_geom": geom}


def sample_spatial_position_rule(is_ood):
    pos_y_val = 0.0
    rule = None
    pos_scale_range = random.choice([(-15, -10), (5, 10)]) if is_ood else random.choice([(-10, -5), (10, 15)])
    if random.random() > 0.5:
        pos_y_val = np.random.uniform(*pos_scale_range)
        seg = random.choice(['straight', 'corner'])
        rule = SpatialPositionRule(seg, offset=[0, pos_y_val, 0])
        geom = seg
    else:
        geom = "none"
    return rule, {"rule_pos_y": pos_y_val, "rule_pos_y_geom": geom}


def sample_spatial_orientation_rule(is_ood):
    ori_x_val = 0.0
    rule = None
    ori_scale_range = random.choice([(-0.2, -0.1), (0.2, 0.3)]) if is_ood else random.choice([(-0.3, -0.2), (0.1, 0.2)])
    # ori_scale_range = random.choice([(-0.3, -0.2), (0.1, 0.2)]) if is_ood else random.choice([(-0.3, -0.2), (0.1, 0.2)])
    ori_scale_range = random.choice(((-0.5, -0.1), (0.1, 0.5)))
    if random.random() > 0.5:
        ori_x_val = np.random.uniform(*ori_scale_range)
        seg = random.choice(['straight', 'corner'])
        rule = SpatialOrientationRule(seg, rotvec=[ori_x_val, 0, 0])
        geom = seg
    else:
        geom = "none"
    return rule, {"rule_ori_x": ori_x_val, "rule_ori_x_geom": geom}

def sample_geometric_proximity_rule(is_ood):
    geom_prox_val = 1.0
    rule = None
    if random.random() > 0.5:
        node_type = random.choice(["edge", "crossing"])
        dist_threshold = np.random.uniform(20, 80)   # mm — simulation runs in mm
        f_scale_range = random.choice([(0.5, 0.9), (1.1, 1.5)])
        geom_prox_val = np.random.uniform(*f_scale_range)
        rule = GeometryProximityRule(node_type, dist_threshold, f_scale=geom_prox_val, tau_scale=1.0)
        geom = node_type
    else:
        geom = "none"
    return rule, {"geometric_proximity": geom_prox_val, "geometric_proximity_geom": geom}


def build_active_rules(is_ood, enabled_rules=None):
    if enabled_rules is None:
        enabled_rules = {"velocity_scaling": True, "spatial_position": False, "spatial_orientation": False, "geometric_proximity": True}

    active_rules = []
    metadata = {
        "rule_vel_scale": 1.0, "rule_pos_y": 0.0, "rule_ori_x": 0.0, "geometric_proximity": 0.0,
        "rule_vel_scale_geom": "none", "rule_pos_y_geom": "none", "rule_ori_x_geom": "none", "geometric_proximity_geom": "none"
    }

    if enabled_rules.get("velocity_scaling", False):
        rule, meta = sample_velocity_scaling_rule(is_ood)
        if rule: active_rules.append(rule)
        metadata.update(meta)

    if enabled_rules.get("spatial_position", False):
        rule, meta = sample_spatial_position_rule(is_ood)
        if rule: active_rules.append(rule)
        metadata.update(meta)

    if enabled_rules.get("spatial_orientation", False):
        rule, meta = sample_spatial_orientation_rule(is_ood)
        if rule: active_rules.append(rule)
        metadata.update(meta)
    
    if enabled_rules.get("geometric_proximity", False):
        rule, meta = sample_geometric_proximity_rule(is_ood)
        if rule: active_rules.append(rule)
        metadata.update(meta)

    return active_rules, metadata


# ── Per-window simulation ─────────────────────────────────────────────────────

def simulate_window(window_id, traj_file, is_ood, unique_id, locations, enabled_rules=None):
    np.random.seed(None)
    random.seed(None)

    raw_data = load_path(str(traj_file))
    if raw_data.size == 0:
        return []

    segments = split_path_2_segments(raw_data)
    active_rules, metadata = build_active_rules(is_ood, enabled_rules)

    with open("configs/sim_params.json", "r") as f:
        params = json.load(f)
        

    try:
        traj_data = convert_path_2_trajectory(
            raw_data,
            rules=active_rules, segments=segments, locations=locations, **params
        )
    except Exception as e:
        import traceback
        print(f"[!] Error simulating window {window_id}: {e}")
        traceback.print_exc()
        return []

    resampled_segments = _assign_segments_by_position(raw_data, segments, traj_data)

    n_steps = traj_data.shape[0]
    window_data = []
    for i in range(n_steps):
        row = list(traj_data[i, :9])
        row.append(int(traj_data[i, 9]))      # stroke_id
        row.append(unique_id)                  # demonstration_id
        row.append(resampled_segments[i])      # segment_type (position-based, not index-based)
        row.append(window_id)                  # window_id
        row.extend([
            metadata["rule_vel_scale"], metadata["rule_pos_y"], metadata["rule_ori_x"],
            metadata["geometric_proximity"],
            metadata["rule_vel_scale_geom"], metadata["rule_pos_y_geom"], metadata["rule_ori_x_geom"],
            metadata["geometric_proximity_geom"],
        ])
        window_data.append(row)

    return window_data


COLUMNS = [
    'time(s)', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'velocity',
    'stroke_id', 'demonstration_id', 'segment_type', 'window_id',
    'rule_vel_scale', 'rule_pos_y', 'rule_ori_x', 'geometric_proximity',
    'rule_vel_scale_geom', 'rule_pos_y_geom', 'rule_ori_x_geom', 'geometric_proximity_geom',
]


def _load_or_extract_skeleton(mesh_file: Path) -> list:
    """Load skeleton from .npz cache if it exists, otherwise extract and save it."""
    cache_path = mesh_file.with_suffix(".skeleton.npz")
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=False)
        positions = data["positions"]   # (N, 3)
        types     = data["types"]       # (N,) bytes → str
        degrees   = data["degrees"]     # (N,) int
        return [
            {"position": positions[i], "type": types[i].decode(), "degree": int(degrees[i])}
            for i in range(len(positions))
        ]

    locations = split_mesh_2_locations_from_file(str(mesh_file))
    if locations:
        np.savez(
            cache_path,
            positions = np.array([l["position"] for l in locations], dtype=np.float64),
            types     = np.array([l["type"].encode() for l in locations]),
            degrees   = np.array([l["degree"] for l in locations], dtype=np.int32),
        )
    return locations


def generate_split(window_list, output_csv, samples_dir, is_ood, num_samples, enabled_rules=None):
    """Simulate `num_samples` strokes from `window_list` and save to `output_csv`."""
    all_data = []
    tasks = []

    for i in range(num_samples):
        window_id = random.choice(window_list)
        traj_file = Path(samples_dir) / window_id / "trajectory.txt"
        mesh_file = Path(samples_dir) / window_id / f"{window_id}.obj"

        if not traj_file.exists():
            continue
        try:
            need_locations = enabled_rules is not None and enabled_rules.get("geometric_proximity", False)
            locations = _load_or_extract_skeleton(mesh_file) if need_locations else []
            tasks.append((window_id, traj_file, is_ood, i, locations, enabled_rules))
        except Exception as e:
            print(f"Error generating window {window_id}: {e}")
            continue

    label = 'OOD' if is_ood else 'ID'
    print(f"Generating {label} dataset: {len(tasks)} tasks on multiple CPU cores...")

    samples_generated = 0
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(simulate_window, *t) for t in tasks]
        for future in as_completed(futures):
            window_data = future.result()
            if window_data:
                all_data.extend(window_data)
                samples_generated += 1
                if samples_generated % 50 == 0:
                    print(f"  Progress: {samples_generated}/{len(tasks)}")

    df = pd.DataFrame(all_data, columns=COLUMNS)
    df = df.sort_values(['demonstration_id', 'stroke_id']).reset_index(drop=True)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[✓] Saved {len(df)} rows to {output_csv}")


# ── Main entry point ──────────────────────────────────────────────────────────

def main(
    samples_dir="datasets/L_shape",
    sample_name="L_shape",
    output_dir="datasets/L_shape",
    total_windows=100,
    train_ratio=0.80,
    val_ratio=0.10,
    enabled_rules=None,
):
    if enabled_rules is None:
        enabled_rules = {"velocity_scaling": True, "spatial_position": False, "spatial_orientation": False, "geometric_proximity": True}

    base_dir = Path(samples_dir)

    # Build window list from directories matching <i>_<sample_name>
    all_windows = [f"{i}_{sample_name}" for i in range(1, total_windows + 1)
                   if (base_dir / f"{i}_{sample_name}").exists()]

    if not all_windows:
        print(f"[!] No sample directories found in {samples_dir}. Run generate_samples first.")
        return

    random.shuffle(all_windows)

    n_train = int(len(all_windows) * train_ratio)
    n_val   = int(len(all_windows) * val_ratio)

    train_windows = all_windows[:n_train]
    val_windows   = all_windows[n_train:n_train + n_val]
    test_windows  = all_windows[n_train + n_val:]

    split_dir = Path(output_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    with open(split_dir / "train_split.json", "w") as f: json.dump(train_windows, f)
    with open(split_dir / "val_split.json",   "w") as f: json.dump(val_windows, f)
    with open(split_dir / "test_split.json",  "w") as f: json.dump(test_windows, f)

    print(f"Split: {len(train_windows)} train / {len(val_windows)} val / {len(test_windows)} test windows")

    generate_split(train_windows, f"{output_dir}/train.csv", samples_dir, is_ood=False,
                   num_samples=len(train_windows), enabled_rules=enabled_rules)
    generate_split(val_windows,   f"{output_dir}/val.csv",   samples_dir, is_ood=False,
                   num_samples=len(val_windows),   enabled_rules=enabled_rules)
    generate_split(test_windows,  f"{output_dir}/test.csv",  samples_dir, is_ood=True,
                   num_samples=len(test_windows),  enabled_rules=enabled_rules)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate train/val/test CSV datasets from a directory of shape samples."
    )
    parser.add_argument(
        "--samples-dir", default="datasets/L_shape",
        help="Root directory containing <i>_<sample-name>/ sub-folders (default: datasets/L_shape)"
    )
    parser.add_argument(
        "--sample-name", default="L_shape",
        help="Base name used to discover sample sub-folders (default: L_shape)"
    )
    parser.add_argument(
        "--output-dir", default="datasets/L_shape",
        help="Where to write train/val/test CSVs and split JSONs (default: datasets/L_shape)"
    )
    parser.add_argument(
        "--total-windows", type=int, default=100,
        help="Maximum number of sample windows to include (default: 100)"
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.80,
        help="Fraction of windows used for training (default: 0.80)"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.10,
        help="Fraction of windows used for validation (default: 0.10)"
    )
    parser.add_argument(
        "--enable-pos-rule", action="store_true",
        help="Enable spatial position rule during simulation"
    )
    parser.add_argument(
        "--enable-ori-rule", action="store_true",
        help="Enable spatial orientation rule during simulation"
    )
    parser.add_argument(
        "--disable-vel-rule", action="store_true",
        help="Disable velocity scaling rule during simulation"
    )
    parser.add_argument(
        "--enable-prox-rule", action="store_true",
        help="Enable geometric proximity rule during simulation"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    enabled_rules = {
        "velocity_scaling": not args.disable_vel_rule,
        "spatial_position": args.enable_pos_rule,
        "spatial_orientation": args.enable_ori_rule,
        "geometric_proximity": args.enable_prox_rule,
    }
    main(
        samples_dir=args.samples_dir,
        sample_name=args.sample_name,
        output_dir=args.output_dir,
        total_windows=args.total_windows,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        enabled_rules=enabled_rules,
    )
