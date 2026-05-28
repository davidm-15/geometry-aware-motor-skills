"""
Generate geometric samples by randomly prolonging both ends of an I-shape (log stick)
and updating the corresponding trajectory file.

Defaults: generates 100 I-shape samples from datasets/I_shape/00_source/
into datasets/I_shape/.

Usage:
    # Default I-shape behaviour
    python -m data_generation.generate_I_samples --num-samples 1000

    # Custom
    python -m data_generation.generate_I_samples \
        --source-dir datasets/MyIShape/00_source \
        --output-dir datasets/MyIShape \
        --sample-name my_i_shape \
        --num-samples 50
"""

import os
import random
import argparse


def generate_samples(
    source_dir: str = "datasets/I_shape/00_source",
    output_dir: str = "datasets/I_shape",
    sample_name: str = "I_shape",
    num_samples: int = 100,
    min_length: int = 80,
    max_length: int = 500,
    step: int = 2,
):
    """
    Generate `num_samples` geometric variants of an OBJ + trajectory pair (I-shape).

    The source directory must contain:
      - ``<sample_name>.obj``   — template mesh
      - ``trajectory.txt``      — template path (semicolon-delimited, with header)

    Each generated sample gets its own sub-directory ``<output_dir>/<i>_<sample_name>/``.
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

    header     = traj_lines[0].strip()
    data_lines = [l.strip() for l in traj_lines[1:] if l.strip()]

    if not data_lines:
        print("[!] Trajectory file is empty.")
        return

    for i in range(1, num_samples + 1):
        sample_dir = os.path.join(output_dir, f"{i}_{sample_name}")
        os.makedirs(sample_dir, exist_ok=True)

        # Random independent side lengths, rounded to `step` for alignment
        L_y_start = float(random.randint(min_length // step, max_length // step) * step)
        L_y_end   = float(random.randint(min_length // step, max_length // step) * step)

        # ── OBJ: extend stick vertices ────────────────────────────────────
        # Based on I_shape.obj:
        # Side with Y = -221.5 (start): vertices 2, 3, 6, 7 (1-based)
        # Side with Y = -32.5  (end):   vertices 1, 4, 5, 8 (1-based)
        new_obj_lines = []
        v_count = 0
        for line in obj_lines:
            if line.startswith('v '):
                v_count += 1
                parts = line.split()
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])

                # Start side: vertices 2, 3, 6, 7 → extend along -Y
                if v_count in [2, 3, 6, 7]:
                    y -= L_y_start
                # End side: vertices 1, 4, 5, 8 → extend along +Y
                elif v_count in [1, 4, 5, 8]:
                    y += L_y_end

                new_obj_lines.append(f"v {x:.6f} {y:.6f} {z:.6f}\n")
            else:
                new_obj_lines.append(line)

        with open(os.path.join(sample_dir, f"{i}_{sample_name}.obj"), 'w') as f:
            f.writelines(new_obj_lines)

        # ── Trajectory: prepend & append extensions ──────────────────────
        first_point = data_lines[0].split(';')
        last_point  = data_lines[-1].split(';')

        new_trajectory_data = []

        # Prepend points (Side 2 - start side, more negative Y)
        y_orig_start = float(first_point[1])
        for offset in range(int(L_y_start), 0, -step):
            p    = list(first_point)
            p[1] = f"{y_orig_start - offset:.6f}"
            new_trajectory_data.append(";".join(p))

        # Original path points
        new_trajectory_data.extend(data_lines)

        # Append points (Side 1 - end side, more positive Y)
        y_orig_end = float(last_point[1])
        for offset in range(step, int(L_y_end) + step, step):
            p    = list(last_point)
            p[1] = f"{y_orig_end + offset:.6f}"
            new_trajectory_data.append(";".join(p))

        with open(os.path.join(sample_dir, "trajectory.txt"), 'w') as f:
            f.write(header + "\n")
            for line in new_trajectory_data:
                f.write(line + "\n")

    print(f"[✓] Generated {num_samples} samples in {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Generate geometric shape samples (OBJ + trajectory) for I-shape by random prolongation."
    )
    parser.add_argument(
        "--source-dir", default="datasets/I_shape/00_source",
        help="Directory containing the template OBJ and trajectory.txt"
    )
    parser.add_argument(
        "--output-dir", default="datasets/I_shape",
        help="Root output directory; samples go into <output-dir>/<i>_<sample-name>/"
    )
    parser.add_argument(
        "--sample-name", default="I_shape",
        help="Base name of the shape (used for OBJ filename and sub-directory naming)"
    )
    parser.add_argument(
        "--num-samples", type=int, default=100,
        help="How many samples to generate (default: 100)"
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
