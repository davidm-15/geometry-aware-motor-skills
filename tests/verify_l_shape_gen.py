import numpy as np
import sys
import os
from pathlib import Path

# Add project root to sys.path
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from datasets.L_shape.split_and_generate import simulate_stroke
from data_generation.path_parser import parse_trajectory_file

def verify_l_shape_generation():
    traj_path = "datasets/L_shape/1_L_shape/trajectory.txt"
    if not os.path.exists(traj_path):
        print(f"Error: {traj_path} not found.")
        return

    subpaths = parse_trajectory_file(traj_path)
    sub = subpaths[0]
    
    print(f"Verifying simulate_stroke for {traj_path}...")
    
    # Try ID (is_ood=False)
    data_id = simulate_stroke("1_L_shape", sub, is_ood=False, unique_id=999)
    print(f"  ID Generation: {len(data_id)} steps.")
    
    # Try OOD (is_ood=True)
    data_ood = simulate_stroke("1_L_shape", sub, is_ood=True, unique_id=888)
    print(f"  OOD Generation: {len(data_ood)} steps.")

    if not data_id or not data_ood:
        print("  ❌ Simulation failed to return data.")
        return

    # Check column count
    expected_cols = 14
    if len(data_id[0]) != expected_cols:
        print(f"  ❌ Wrong column count! Expected {expected_cols}, got {len(data_id[0])}")
    else:
        print(f"  ✅ Column count matches: {expected_cols}")

    # Check for full duration (longer than 1500 steps if path is long)
    # The L-shape is ~891 units. v_base=0.5. Time = 1782s. dt=0.02.
    # Steps = 1782 / 0.02 = 89100 steps.
    if len(data_id) > 2000:
        print(f"  ✅ Successfully captured full duration: {len(data_id)} steps.")
    else:
        print(f"  ❌ Simulation still seems short: {len(data_id)} steps.")

    # Check for strictly increasing timestamps
    times = [row[0] for row in data_id]
    if all(t1 < t2 for t1, t2 in zip(times, times[1:])):
        print("  ✅ Timestamps are strictly increasing (no jumps).")
    else:
        print("  ❌ FOUND duplicate or non-increasing timestamps!")

if __name__ == "__main__":
    verify_l_shape_generation()
