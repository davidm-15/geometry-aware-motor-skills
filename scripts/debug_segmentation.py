"""
Debug straight/corner segmentation for a single window.

Usage:
    python -m scripts.debug_segmentation
    python -m scripts.debug_segmentation --window datasets/windows-v2/52_wr1fr_1 --threshold 0.001 --slice_size 15
    python -m scripts.debug_segmentation --show   # open interactive matplotlib window
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

from data_generation.load_data import load_path, split_path_2_segments


def plot_segmentation(path: np.ndarray, segments: list, title: str, ax3d, ax_xy, ax_xz, ax_yz):
    positions = path[:, 0:3]
    stroke_ids = path[:, -1]

    straight_mask = np.array([s == 'straight' for s in segments])
    corner_mask = ~straight_mask

    color_straight = (0.1, 0.85, 0.1)
    color_corner = (0.9, 0.1, 0.1)

    def scatter_seg(ax, xs, ys, label_x, label_y):
        if straight_mask.any():
            ax.scatter(xs[straight_mask], ys[straight_mask], c=[color_straight],
                       s=6, label='straight', zorder=2)
        if corner_mask.any():
            ax.scatter(xs[corner_mask], ys[corner_mask], c=[color_corner],
                       s=6, label='corner', zorder=3)
        ax.set_xlabel(label_x)
        ax.set_ylabel(label_y)
        ax.legend(markerscale=2, fontsize=7)

    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]

    # 3D view
    if straight_mask.any():
        ax3d.scatter(x[straight_mask], y[straight_mask], z[straight_mask],
                     c=[color_straight], s=6, label='straight')
    if corner_mask.any():
        ax3d.scatter(x[corner_mask], y[corner_mask], z[corner_mask],
                     c=[color_corner], s=6, label='corner')
    ax3d.set_xlabel('X'); ax3d.set_ylabel('Y'); ax3d.set_zlabel('Z')
    ax3d.set_title(title, fontsize=9)
    ax3d.legend(markerscale=2, fontsize=7)

    # 2D projections
    scatter_seg(ax_xy, x, y, 'X', 'Y')
    ax_xy.set_title('XY projection', fontsize=8)

    scatter_seg(ax_xz, x, z, 'X', 'Z')
    ax_xz.set_title('XZ projection', fontsize=8)

    scatter_seg(ax_yz, y, z, 'Y', 'Z')
    ax_yz.set_title('YZ projection', fontsize=8)


def plot_dist_sq_profile(path: np.ndarray, threshold: float, slice_size: int):
    """Show the raw SVD residual curve so you can judge threshold placement."""
    from more_itertools import sliding_window

    IDs = np.unique(path[:, -1])
    fig, axes = plt.subplots(len(IDs), 1, figsize=(12, 3 * len(IDs)), squeeze=False)
    fig.suptitle(f'SVD residual dist_sq per window  (threshold={threshold}, slice={slice_size})', fontsize=10)

    for ax, ID in zip(axes[:, 0], IDs):
        points = path[path[:, -1] == ID][:, 0:3]
        errors = []
        for window in sliding_window(points, slice_size):
            pts = np.array(window)
            centroid = pts.mean(axis=0)
            centered = pts - centroid
            _, _, vh = np.linalg.svd(centered)
            direction = vh[0]
            dist_sq = np.sum(centered ** 2) - np.sum(np.dot(centered, direction) ** 2)
            errors.append(dist_sq)

        pad = slice_size // 2
        xs = np.arange(pad, pad + len(errors))
        ax.semilogy(xs, errors, lw=1, color='steelblue', label='dist_sq')
        ax.axhline(threshold, color='red', lw=1.2, linestyle='--', label=f'threshold={threshold}')
        ax.set_title(f'Stroke ID={ID}  ({len(points)} pts)', fontsize=8)
        ax.set_xlabel('point index'); ax.set_ylabel('dist_sq (log)')
        ax.legend(fontsize=7)

    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--window', default='datasets/windows-v2/52_wr1fr_1',
                        help='Path to window directory containing trajectory.txt')
    parser.add_argument('--threshold', type=float, default=0.001)
    parser.add_argument('--slice_size', type=int, default=15)
    parser.add_argument('--show', action='store_true', help='Open interactive matplotlib window')
    args = parser.parse_args()
    if args.show:
        matplotlib.use('TkAgg')

    window_dir = Path(args.window)
    traj_file = window_dir / 'trajectory.txt'

    print(f"Loading {traj_file}")
    path = load_path(str(traj_file))
    print(f"  {path.shape[0]} points, stroke IDs: {np.unique(path[:, -1])}")

    segments = split_path_2_segments(path, curvature_threshold=args.threshold, slice_size=args.slice_size)

    straight_count = segments.count('straight')
    corner_count = segments.count('corner')
    print(f"  straight: {straight_count}  corner: {corner_count}")

    # Main segmentation plot
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f'{window_dir.name}  threshold={args.threshold}  slice={args.slice_size}', fontsize=10)
    ax3d  = fig.add_subplot(2, 2, 1, projection='3d')
    ax_xy = fig.add_subplot(2, 2, 2)
    ax_xz = fig.add_subplot(2, 2, 3)
    ax_yz = fig.add_subplot(2, 2, 4)
    plot_segmentation(path, segments, window_dir.name, ax3d, ax_xy, ax_xz, ax_yz)
    fig.tight_layout()

    out_seg = window_dir / 'debug_segmentation.png'
    fig.savefig(out_seg, dpi=150)
    print(f"Saved {out_seg}")

    # Residual profile plot
    fig2 = plot_dist_sq_profile(path, args.threshold, args.slice_size)
    out_res = window_dir / 'debug_residuals.png'
    fig2.savefig(out_res, dpi=150)
    print(f"Saved {out_res}")

    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
