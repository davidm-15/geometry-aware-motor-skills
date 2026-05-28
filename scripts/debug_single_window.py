"""
Generate a trajectory for a single window using the full generate_dataset pipeline,
then launch the RViz2 visualizer on it.

Usage:
    python -m scripts.debug_single_window
    python -m scripts.debug_single_window --window 52_wr1fr_1 --no-rules
    python -m scripts.debug_single_window --window 52_wr1fr_1 --playback_speed 50
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_generation.generate_dataset import COLUMNS, build_active_rules, _assign_segments_by_position
from data_generation.load_data import load_path, split_path_2_segments, convert_path_2_trajectory


def _scatter_seg(ax, xs, ys, mask_straight, xlabel, ylabel):
    c_s, c_c = (0.1, 0.85, 0.1), (0.9, 0.1, 0.1)
    if mask_straight.any():
        ax.scatter(xs[mask_straight],  ys[mask_straight],  c=[c_s], s=4, label='straight')
    if (~mask_straight).any():
        ax.scatter(xs[~mask_straight], ys[~mask_straight], c=[c_c], s=4, label='corner')
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.legend(markerscale=2, fontsize=7)


def _save_comparison_plot(raw_data: np.ndarray, segments: list, df: pd.DataFrame, out_dir: Path):
    """Side-by-side YZ projection: original path labels vs. simulated trajectory labels."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Segment labelling: original path  vs.  simulated trajectory (YZ)', fontsize=10)

    # ── Left: original path ───────────────────────────────────────────────────
    ox, oy, oz = raw_data[:, 0], raw_data[:, 1], raw_data[:, 2]
    orig_straight = np.array([s == 'straight' for s in segments])
    _scatter_seg(axes[0], oy, oz, orig_straight, 'Y (original path)', 'Z')
    axes[0].set_title('Original path (trajectory.txt)', fontsize=9)

    # ── Right: simulated CSV ──────────────────────────────────────────────────
    sx, sy, sz = df['x'].values, df['y'].values, df['z'].values
    sim_straight = (df['segment_type'] == 'straight').values
    _scatter_seg(axes[1], sy, sz, sim_straight, 'Y (simulated trajectory)', 'Z')
    axes[1].set_title('Simulated trajectory (CSV)', fontsize=9)

    fig.tight_layout()
    out = out_dir / 'comparison_segmentation.png'
    fig.savefig(out, dpi=150)
    print(f"Saved comparison plot → {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--window', default='52_wr1fr_1',
                        help='Window ID inside datasets/windows-v2/')
    parser.add_argument('--samples_dir', default='datasets/windows-v2')
    parser.add_argument('--no-rules', action='store_true',
                        help='Disable all rules so segment_type is the only thing to inspect')
    parser.add_argument('--playback_speed', type=float, default=50.0)
    parser.add_argument('--hold_time', type=float, default=0.5)
    args = parser.parse_args()

    window_id  = args.window
    samples_dir = Path(args.samples_dir)
    traj_file   = samples_dir / window_id / 'trajectory.txt'
    mesh_file   = samples_dir / window_id / f'{window_id}.obj'

    if not traj_file.exists():
        sys.exit(f"[!] trajectory.txt not found: {traj_file}")

    # ── Output directory ──────────────────────────────────────────────────────
    out_dir = Path('outputs') / f'debug_{window_id}'
    out_dir.mkdir(parents=True, exist_ok=True)

    # The visualizer looks for the mesh at:
    #   <csv_parent> / <window_id> / <window_id>.obj
    # So we symlink the original window directory under out_dir.
    link_target = out_dir / window_id
    if not link_target.exists():
        link_target.symlink_to(samples_dir.resolve() / window_id)
        print(f"Symlinked mesh directory → {link_target}")

    out_csv = out_dir / 'trajectory.csv'

    # ── Simulate ──────────────────────────────────────────────────────────────
    print(f"Simulating {window_id} ({'no rules' if args.no_rules else 'with rules'}) ...")

    enabled_rules = {
        'velocity_scaling':   not args.no_rules,
        'spatial_position':   False,
        'spatial_orientation': False,
        'geometric_proximity': False,
    }

    raw_data = load_path(str(traj_file))
    segments = split_path_2_segments(raw_data)
    active_rules, metadata = build_active_rules(is_ood=False, enabled_rules=enabled_rules)

    with open('configs/sim_params.json') as f:
        params = json.load(f)

    traj_data = convert_path_2_trajectory(
        raw_data, rules=active_rules, segments=segments, locations=[], **params
    )

    resampled_segments = _assign_segments_by_position(raw_data, segments, traj_data)

    n_steps = traj_data.shape[0]
    rows = []
    for i in range(n_steps):
        row = list(traj_data[i, :9])
        row.append(int(traj_data[i, 9]))   # stroke_id
        row.append(0)                       # demonstration_id
        row.append(resampled_segments[i])   # segment_type (position-based lookup)
        row.append(window_id)
        row.extend([
            metadata['rule_vel_scale'], metadata['rule_pos_y'], metadata['rule_ori_x'],
            metadata['geometric_proximity'],
            metadata['rule_vel_scale_geom'], metadata['rule_pos_y_geom'],
            metadata['rule_ori_x_geom'], metadata['geometric_proximity_geom'],
        ])
        rows.append(row)

    df = pd.DataFrame(rows, columns=COLUMNS)
    df.to_csv(out_csv, index=False)

    straight = (df['segment_type'] == 'straight').sum()
    corner   = (df['segment_type'] == 'corner').sum()
    print(f"Saved {len(df)} rows → {out_csv}")
    print(f"  segment_type: straight={straight}  corner={corner}")
    print(f"  strokes: {sorted(df['stroke_id'].unique())}")

    _save_comparison_plot(raw_data, segments, df, out_dir)

    # ── Launch visualizer ─────────────────────────────────────────────────────
    # Kill any leftover visualizer / RViz instances so topics don't conflict.
    subprocess.run(['pkill', '-f', 'visualize_trajectory'], capture_output=True)
    subprocess.run(['pkill', '-f', 'rviz2'], capture_output=True)

    # The visualizer needs ROS Python 3.12 (system), not the conda Python 3.10.
    # We set PYTHONPATH so the ros2 package is importable from the project root.
    ros_python = '/usr/bin/python3'
    cmd = [
        ros_python, '-m', 'ros2.visualize_trajectory',
        str(out_csv),
        '--playback_speed', str(args.playback_speed),
        '--hold_time',      str(args.hold_time),
    ]
    env = os.environ.copy()
    project_root = str(Path(__file__).resolve().parents[1])
    env['PYTHONPATH'] = project_root + ':' + env.get('PYTHONPATH', '')

    print(f"\nLaunching visualizer with {ros_python}:")
    print(f"  {' '.join(cmd)}\n")
    subprocess.run(cmd, env=env)


if __name__ == '__main__':
    main()
