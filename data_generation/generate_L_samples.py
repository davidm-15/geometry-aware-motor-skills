"""
Generate geometric samples by randomly prolonging arms of an OBJ model
and updating the corresponding trajectory file.

Defaults: generates 100 L-shape samples from datasets/L_shape/00_source/
into datasets/L_shape/ (original behaviour).

Usage:
    # Default L-shape behaviour
    python -m data_generation.generate_L_samples

    # Custom
    python -m data_generation.generate_samples \\
        --source-dir datasets/MyShape/00_source \\
        --output-dir datasets/MyShape \\
        --sample-name my_shape \\
        --num-samples 50
"""

import os
import random
import argparse


def generate_samples(
    source_dir: str = "datasets/L_shape/00_source",
    output_dir: str = "datasets/L_shape",
    sample_name: str = "L_shape",
    num_samples: int = 1000,
    min_length: int = 100,
    max_length: int = 500,
    step: int = 2,
):
    """
    Generate `num_samples` geometric variants of an OBJ + trajectory pair.

    The source directory must contain:
      - ``<sample_name>.obj``   — template mesh
      - ``trajectory.txt``      — template path (semicolon-delimited, with header)

    Each generated sample gets its own sub-directory ``<output_dir>/<i>_<sample_name>/``.

    Arm prolongation rules (L-shape specific, vertex indices 1-based):
      - Vertices 1, 2, 7, 8  → Z coordinate extended (arm 1)
      - Vertices 4, 5, 10, 11 → Y coordinate extended (arm 2)
    """
    obj_path  = os.path.join(source_dir, f"{sample_name}.obj")
    traj_path = os.path.join(source_dir, "trajectory.txt")

    if not os.path.exists(obj_path) or not os.path.exists(traj_path):
        print(f"[!] Source files not found in {source_dir}")
        print(f"    Expected: {obj_path}")
        print(f"    Expected: {traj_path}")
        return

    with open(obj_path, 'r') as f:
        obj_lines = f.readlines()

    with open(traj_path, 'r') as f:
        traj_lines = f.readlines()

    header     = traj_lines[0]
    data_lines = [l.strip() for l in traj_lines[1:] if l.strip()]

    if not data_lines:
        print("[!] Trajectory file is empty.")
        return

    for i in range(1, num_samples + 1):
        sample_dir = os.path.join(output_dir, f"{i}_{sample_name}")
        os.makedirs(sample_dir, exist_ok=True)

        # Random independent arm lengths, rounded to `step` for alignment
        L_z = float(random.randint(min_length // step, max_length // step) * step)
        L_y = float(random.randint(min_length // step, max_length // step) * step)

        # ── OBJ: extend arm vertices ──────────────────────────────────────
        new_obj_lines = []
        v_count = 0
        for line in obj_lines:
            if line.startswith('v '):
                v_count += 1
                parts = line.split()
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])

                # Arm 1: vertices 1, 2, 7, 8 → extend along Z
                if v_count in [1, 2, 7, 8]:
                    z -= L_z
                # Arm 2: vertices 4, 5, 10, 11 → extend along Y
                elif v_count in [4, 5, 10, 11]:
                    y -= L_y

                new_obj_lines.append(f"v {x:.6f} {y:.6f} {z:.6f}\n")
            else:
                new_obj_lines.append(line)

        with open(os.path.join(sample_dir, f"{i}_{sample_name}.obj"), 'w') as f:
            f.writelines(new_obj_lines)

        # ── Trajectory: prepend Z-arm & append Y-arm extension ───────────
        first_point = data_lines[0].split(';')
        last_point  = data_lines[-1].split(';')

        new_trajectory_data = []

        # Prepend points along Z-arm
        z_orig = float(first_point[2])
        for offset in range(int(L_z), 0, -step):
            p    = list(first_point)
            p[2] = f"{z_orig - offset:.6f}"
            new_trajectory_data.append(";".join(p))

        # Original path points
        new_trajectory_data.extend(data_lines)

        # Append points along Y-arm
        y_orig = float(last_point[1])
        for offset in range(step, int(L_y) + step, step):
            p    = list(last_point)
            p[1] = f"{y_orig - offset:.6f}"
            new_trajectory_data.append(";".join(p))

        with open(os.path.join(sample_dir, "trajectory.txt"), 'w') as f:
            f.write(header + "\n")
            for line in new_trajectory_data:
                f.write(line + "\n")

    print(f"[✓] Generated {num_samples} samples in {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Generate geometric shape samples (OBJ + trajectory) by random arm prolongation."
    )
    parser.add_argument(
        "--source-dir", default="datasets/L_shape/00_source",
        help="Directory containing the template OBJ and trajectory.txt (default: datasets/L_shape/00_source)"
    )
    parser.add_argument(
        "--output-dir", default="datasets/L_shape",
        help="Root output directory; samples go into <output-dir>/<i>_<sample-name>/ (default: datasets/L_shape)"
    )
    parser.add_argument(
        "--sample-name", default="L_shape",
        help="Base name of the shape (used for OBJ filename and sub-directory naming, default: L_shape)"
    )
    parser.add_argument(
        "--num-samples", type=int, default=1000,
        help="How many samples to generate (default: 1000)"
    )
    parser.add_argument(
        "--min-length", type=int, default=100,
        help="Minimum arm extension in model units (default: 100)"
    )
    parser.add_argument(
        "--max-length", type=int, default=500,
        help="Maximum arm extension in model units (default: 500)"
    )
    args = parser.parse_args()

    generate_samples(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        sample_name=args.sample_name,
        num_samples=args.num_samples,
        min_length=args.min_length,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()
