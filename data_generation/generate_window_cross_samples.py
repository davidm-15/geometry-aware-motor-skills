"""
Generate window-cross shape samples for proximity-rule training.

A window cross is a rectangle (W × H) with a "+" inside, like a window frame
divided into 4 panes by two cross-bars. It has a 9-node skeleton:
  4 corners (degree 2, type "edge")
  4 T-junctions (degree 3, type "crossing")
  1 center crossing (degree 4, type "crossing")

Each sample directory:
  <i>_window_cross/
    <i>_window_cross.obj          programmatic mesh (single solid body)
    <i>_window_cross.skeleton.npz 9-node skeleton (loaded by generate_dataset.py)
    trajectory.txt                random-walk path (MaskPlanner format)

Usage:
    # Generate 500 asymmetric (training) samples
    python -m data_generation.generate_window_cross_samples

    # Generate 500 symmetric (test) samples into a different dir
    python -m data_generation.generate_window_cross_samples \\
        --symmetric --output-dir datasets/window_cross_sym

    # Custom count
    python -m data_generation.generate_window_cross_samples --n-samples 100

Then generate the CSV dataset with:
    python -m data_generation.generate_dataset \\
        --samples-dir datasets/window_cross \\
        --sample-name window_cross \\
        --output-dir datasets/window_cross \\
        --total-windows 500 \\
        --enable-prox-rule
"""

import argparse
import random
from pathlib import Path

import numpy as np
import trimesh
from shapely.geometry import box as shapely_box

# ── Fixed geometry constants ───────────────────────────────────────────────────
BAR_W    = 20.0   # bar / frame member width (mm)
DEPTH    = 10.0   # mesh thickness along X axis (mm)
STANDOFF = 20.0   # X coordinate of trajectory (front face of mesh)
STEP_MM  = 2.0    # waypoint spacing along path (mm)
STROKE_ID = 7.0
ORIENT_A  = 90.0  # Euler X rotation (degrees) — same convention as L/I shapes

# ── Dimension ranges ───────────────────────────────────────────────────────────
W_RANGE          = (150.0, 500.0)  # outer width  (mm)
H_RANGE          = (150.0, 500.0)  # outer height (mm)
BAR_OFFSET_FRAC  = 0.30            # max bar offset as fraction of half-dimension
MIN_PANE         = 30.0            # minimum pane opening (mm)
HOP_RANGE        = (4, 12)         # random walk length (inclusive)

# ── Skeleton node indices ──────────────────────────────────────────────────────
# Corners (degree 2, type "edge")
BL, BR, TL, TR                = 0, 1, 2, 3
# T-junctions (degree 3, type "crossing")
T_BOT, T_TOP, T_LEFT, T_RIGHT = 4, 5, 6, 7
# Center crossing (degree 4, type "crossing")
CENTER = 8

NODE_DEGREES = [2, 2, 2, 2, 3, 3, 3, 3, 4]
NODE_TYPES   = ["edge", "edge", "edge", "edge",
                "crossing", "crossing", "crossing", "crossing",
                "crossing"]

# Undirected adjacency: each entry is the list of directly connected nodes.
# Bottom outer bar: BL — T_BOT — BR
# Top outer bar:    TL — T_TOP — TR
# Left outer bar:   BL — T_LEFT — TL
# Right outer bar:  BR — T_RIGHT — TR
# Vertical inner:   T_BOT — CENTER — T_TOP
# Horizontal inner: T_LEFT — CENTER — T_RIGHT
ADJACENCY = {
    BL:      [T_BOT,  T_LEFT ],
    BR:      [T_BOT,  T_RIGHT],
    TL:      [T_TOP,  T_LEFT ],
    TR:      [T_TOP,  T_RIGHT],
    T_BOT:   [BL,     BR,     CENTER],
    T_TOP:   [TL,     TR,     CENTER],
    T_LEFT:  [BL,     TL,     CENTER],
    T_RIGHT: [BR,     TR,     CENTER],
    CENTER:  [T_BOT,  T_TOP,  T_LEFT, T_RIGHT],
}


# ── Geometry helpers ───────────────────────────────────────────────────────────

def _sample_bar_pos(dim, symmetric):
    """Sample bar center along an axis of length `dim` (mm)."""
    center = dim / 2.0
    if symmetric:
        return center
    lo = BAR_W + MIN_PANE
    hi = dim - BAR_W - MIN_PANE
    offset = random.uniform(-BAR_OFFSET_FRAC, BAR_OFFSET_FRAC) * center
    return float(np.clip(center + offset, lo, hi))


def _skeleton_positions(W, H, bar_y, bar_z):
    """
    Return (9, 3) skeleton node positions in world coords.
    Trajectory plane: X = STANDOFF (constant), Y ∈ [0, W], Z ∈ [0, H].
    bar_y: Y position of the vertical cross-bar center line.
    bar_z: Z position of the horizontal cross-bar center line.
    """
    hw = BAR_W / 2.0
    return np.array([
        [STANDOFF, hw,       hw      ],  # 0  BL      corner
        [STANDOFF, W - hw,   hw      ],  # 1  BR      corner
        [STANDOFF, hw,       H - hw  ],  # 2  TL      corner
        [STANDOFF, W - hw,   H - hw  ],  # 3  TR      corner
        [STANDOFF, bar_y,    hw      ],  # 4  T_BOT   T-junction
        [STANDOFF, bar_y,    H - hw  ],  # 5  T_TOP   T-junction
        [STANDOFF, hw,       bar_z   ],  # 6  T_LEFT  T-junction
        [STANDOFF, W - hw,   bar_z   ],  # 7  T_RIGHT T-junction
        [STANDOFF, bar_y,    bar_z   ],  # 8  CENTER  crossing
    ], dtype=np.float64)


def _make_mesh(W, H, bar_y, bar_z):
    """
    Build the window-cross mesh as a single solid body.
    Cross-section: outer rectangle minus the 4 open panes, extruded along X.
    The mesh occupies X ∈ [STANDOFF-DEPTH, STANDOFF], Y ∈ [0, W], Z ∈ [0, H].
    """
    hw = BAR_W / 2.0

    # 2-D profile in the local (u=Y, v=Z) plane — shapely works in XY so we
    # map u→x, v→y and extrude along z, then rotate into world space.
    outer  = shapely_box(0,          0,         W,          H         )
    pane_bl = shapely_box(BAR_W,      BAR_W,     bar_y - hw, bar_z - hw)
    pane_br = shapely_box(bar_y + hw, BAR_W,     W - BAR_W,  bar_z - hw)
    pane_tl = shapely_box(BAR_W,      bar_z + hw, bar_y - hw, H - BAR_W )
    pane_tr = shapely_box(bar_y + hw, bar_z + hw, W - BAR_W,  H - BAR_W )
    profile = outer - pane_bl - pane_br - pane_tl - pane_tr

    # Extrude along local z (= world X direction) by DEPTH.
    mesh = trimesh.creation.extrude_polygon(profile, height=DEPTH)
    # mesh is now: local_x∈[0,W], local_y∈[0,H], local_z∈[0,DEPTH]

    # Rotate so that (local_x, local_y, local_z) → (world_Y, world_Z, world_X)
    # and translate so the front face lands at world X = STANDOFF.
    #   world_X = local_z + (STANDOFF - DEPTH)
    #   world_Y = local_x
    #   world_Z = local_y
    T = np.array([
        [0, 0, 1, STANDOFF - DEPTH],
        [1, 0, 0, 0               ],
        [0, 1, 0, 0               ],
        [0, 0, 0, 1               ],
    ], dtype=float)
    mesh.apply_transform(T)
    return mesh


# ── Path generation ────────────────────────────────────────────────────────────

def _random_walk(n_hops):
    """
    Random walk of n_hops steps on the skeleton graph.
    Never steps back to the immediately previous node.
    Returns list of node indices (length = n_hops + 1).
    """
    current = random.randint(0, 8)
    prev = None
    walk = [current]
    for _ in range(n_hops):
        choices = [n for n in ADJACENCY[current] if n != prev]
        nxt = random.choice(choices)
        prev, current = current, nxt
        walk.append(current)
    return walk


def _walk_to_trajectory(walk, positions):
    """
    Interpolate a node-index walk into (N, 7) MaskPlanner waypoints.
    Each row: [X, Y, Z, A, B, C, strokeId].
    Segment endpoints are not duplicated (each node appears once).
    """
    rows = []
    n_segs = len(walk) - 1
    for i in range(n_segs):
        p0 = positions[walk[i]]
        p1 = positions[walk[i + 1]]
        seg = p1 - p0
        length = np.linalg.norm(seg)
        if length < 1e-6:
            continue
        n_pts = max(2, int(np.ceil(length / STEP_MM)))
        # Include endpoint only on the last segment to avoid duplicates.
        include_end = (i == n_segs - 1)
        for t in np.linspace(0, 1, n_pts, endpoint=include_end):
            pt = p0 + t * seg
            rows.append([pt[0], pt[1], pt[2], ORIENT_A, 0.0, 0.0, STROKE_ID])
    return np.array(rows)


# ── Per-sample generation ──────────────────────────────────────────────────────

def generate_sample(output_dir, idx, symmetric=False):
    """Generate one sample: OBJ + skeleton NPZ + trajectory.txt."""
    W     = random.uniform(*W_RANGE)
    H     = random.uniform(*H_RANGE)
    bar_y = _sample_bar_pos(W, symmetric)
    bar_z = _sample_bar_pos(H, symmetric)

    name       = f"{idx}_window_cross"
    sample_dir = Path(output_dir) / name
    sample_dir.mkdir(parents=True, exist_ok=True)

    # ── Mesh ──────────────────────────────────────────────────────────────────
    mesh = _make_mesh(W, H, bar_y, bar_z)
    mesh.export(str(sample_dir / f"{name}.obj"))

    # ── Skeleton NPZ ──────────────────────────────────────────────────────────
    # Loaded by _load_or_extract_skeleton() in generate_dataset.py.
    # Fields: positions (N,3), types (N,) bytes, degrees (N,) int32.
    positions = _skeleton_positions(W, H, bar_y, bar_z)
    np.savez(
        sample_dir / f"{name}.skeleton.npz",
        positions = positions,
        types     = np.array([t.encode() for t in NODE_TYPES]),
        degrees   = np.array(NODE_DEGREES, dtype=np.int32),
    )

    # ── Trajectory ────────────────────────────────────────────────────────────
    n_hops = random.randint(*HOP_RANGE)
    walk   = _random_walk(n_hops)
    traj   = _walk_to_trajectory(walk, positions)

    with open(sample_dir / "trajectory.txt", "w") as f:
        f.write("X;Y;Z;A;B;C;strokeId\n\n")
        for row in traj:
            f.write(f"{row[0]};{row[1]};{row[2]};{row[3]};{row[4]};{row[5]};{row[6]}\n")


# ── Batch generation ───────────────────────────────────────────────────────────

def generate_samples(n=500, output_dir="datasets/window_cross", symmetric=False):
    """Generate n window-cross samples."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Generating {n} {'symmetric' if symmetric else 'asymmetric'} samples → {output_dir}/")
    for i in range(1, n + 1):
        generate_sample(output_dir, i, symmetric=symmetric)
        if i % 50 == 0:
            print(f"  {i}/{n}")
    print(f"[✓] Done. Run dataset generation with:")
    print(f"    python -m data_generation.generate_dataset \\")
    print(f"        --samples-dir {output_dir} --sample-name window_cross \\")
    print(f"        --output-dir {output_dir} --total-windows {n} --enable-prox-rule")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate window-cross shape samples."
    )
    parser.add_argument("--n-samples",   type=int, default=500)
    parser.add_argument("--output-dir",  default="datasets/window_cross")
    parser.add_argument("--symmetric",   action="store_true",
                        help="Keep crossbars centered (use for symmetric test set)")
    args = parser.parse_args()
    generate_samples(
        n=args.n_samples,
        output_dir=args.output_dir,
        symmetric=args.symmetric,
    )
