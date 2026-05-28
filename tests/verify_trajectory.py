import numpy as np
import sys
import os
from pathlib import Path

# Add project root to sys.path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from data_generation.load_data import load_path_as_trajectory

def verify_trajectory(file_path, v_base=0.5):
    print(f"Verifying {file_path} with v_base={v_base}...")
    
    try:
        traj = load_path_as_trajectory(file_path, v_base=v_base)
    except Exception as e:
        print(f"FAILED to load/convert trajectory: {e}")
        return

    # Columns: time(0), x(1), y(2), z(3), qx(4), qy(5), qz(6), qw(7), velocity(8), ID(9)
    times = traj[:, 0]
    positions = traj[:, 1:4]
    
    # 1. Check for duplicate timestamps
    diff_times = np.diff(times)
    duplicates = np.where(diff_times <= 0)[0]
    if len(duplicates) > 0:
        print(f"  ❌ FOUND {len(duplicates)} duplicate or non-increasing timestamps!")
        for idx in duplicates[:5]:
            print(f"    Index {idx}: t={times[idx]}, next_t={times[idx+1]}")
    else:
        print("  ✅ All timestamps are strictly increasing.")

    # 2. Check total duration
    total_time = times[-1] - times[0]
    
    # Calculate path length
    path_diffs = np.diff(positions, axis=0)
    path_lengths = np.linalg.norm(path_diffs, axis=1)
    total_length = np.sum(path_lengths)
    
    expected_time = total_length / v_base
    print(f"  Path length: {total_length:.2f}")
    print(f"  Total time: {total_time:.2f}s (Expected approx {expected_time:.2f}s)")
    
    if total_time < expected_time * 0.5:
         print(f"  ❌ Simulation finished much too fast! ({total_time:.1f}s vs {expected_time:.1f}s)")
    elif total_time > expected_time * 2.0:
         print(f"  ⚠️ Simulation took much longer than expected. ({total_time:.1f}s vs {expected_time:.1f}s)")
    else:
         print(f"  ✅ Simulation duration is reasonable.")

    # 3. Check velocities
    avg_vel = np.mean(traj[:, 8])
    print(f"  Average velocity record: {avg_vel:.3f} (Target v_base={v_base})")
    
    # Check max velocity
    max_vel = np.max(traj[:, 8])
    print(f"  Max velocity record: {max_vel:.3f}")
    if max_vel > v_base * 1.5:
        print(f"  ❌ Max velocity exceeds v_base significantly!")

if __name__ == "__main__":
    test_file = "datasets/L_shape/1_L_shape/trajectory.txt"
    if not os.path.exists(test_file):
        print(f"Test file {test_file} not found. Please ensure it exists.")
    else:
        verify_trajectory(test_file, v_base=0.5)
