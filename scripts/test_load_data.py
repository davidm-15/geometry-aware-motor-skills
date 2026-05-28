from data_generation.load_data import convert_path_2_trajectory, load_path, split_path_2_segments
from data_generation.rules import VelocityScalingRule
from pathlib import Path
import pandas as pd
import numpy as np
import json


# python -m scripts.test_load_data





COLUMNS = [
    'time(s)', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'velocity',
    'stroke_id', 'window_id', 'segment_type'
]



if __name__ == "__main__":
    path_path = "datasets/windows-v2/1_wr1fr_1/trajectory.txt"


    with open("configs/sim_params.json", "r") as f:
        params = json.load(f)
        

    path = load_path(path_path)
    segments = split_path_2_segments(path)

    rule = VelocityScalingRule("straight", f_scale=4.0, tau_scale=1.0)

    traj = convert_path_2_trajectory(path, segments=segments, rules=[rule], **params)
    
    
    traj = np.column_stack((traj, np.zeros(traj.shape[0])))
    traj = np.column_stack((traj, segment_ids := np.zeros(traj.shape[0], dtype=int)))
    print(traj.shape)

    output_dir = Path("outputs/test_load_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "trajectory.csv"


    df = pd.DataFrame(traj, columns=COLUMNS)
    df.to_csv(output_path, index=False)
    print(f"Trajectory saved to {output_path}")
