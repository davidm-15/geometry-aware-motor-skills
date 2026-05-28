import argparse
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

# python -m scripts.plot_velocity_profile --csv_path datasets/windows-v2/test.csv

def load_csv_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    df = df[df["window_id"] == "15_wr1fr_1"]

    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Columns: {df.columns.tolist()}")
    return df


def quaternion_to_euler(qx, qy, qz, qw):
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)


def plot_velocity_profile(csv_path: str) -> None:
    df = load_csv_data(csv_path)
    
    if "time(s)" not in df.columns or "velocity" not in df.columns:
        raise ValueError("CSV must contain 'time(s)' and 'velocity' columns")

    time = df["time(s)"]
    velocity = df["velocity"]

    plt.figure(figsize=(8, 4.5))
    plt.plot(time, velocity, linewidth=1.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity")
    plt.title("Velocity Profile vs Time")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_angle_profile(csv_path: str) -> None:
    df = load_csv_data(csv_path)

    roll, pitch, yaw = quaternion_to_euler(df["qx"], df["qy"], df["qz"], df["qw"])
    df["roll"] = roll
    df["pitch"] = pitch
    df["yaw"] = yaw


    if "time(s)" not in df.columns or "roll" not in df.columns:
        raise ValueError("CSV must contain 'time(s)' and 'roll' columns")

    time = df["time(s)"]
    angle = df["roll"]

    plt.figure(figsize=(8, 4.5))
    plt.plot(time, angle, linewidth=1.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Angle")
    plt.title("Angle Profile vs Time")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot velocity profile from trajectory CSV")
    parser.add_argument(
        "--csv_path",
        nargs="?",
        default="outputs/test_load_data/trajectory.csv",
        help="Path to trajectory CSV file",
    )
    parser.add_argument(
        "--plot_angle",
        action="store_true",
        help="Plot angle profile instead of velocity profile",
    )

    args = parser.parse_args()
    if args.plot_angle:
        plot_angle_profile(args.csv_path)
    else:
        plot_velocity_profile(args.csv_path)


if __name__ == "__main__":
    main()











