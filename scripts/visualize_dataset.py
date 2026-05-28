import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# python -m scripts.visualize_dataset

def visualize_velocities(csv_path='datasets/L_shape/train.csv'):
    df = pd.read_csv(csv_path)

    
    segment_type = df['rule_vel_scale_geom']


    # Determine the range of rule_vel_scale to create bins of size 0.2
    min_scale = df['rule_vel_scale'].min()
    max_scale = df['rule_vel_scale'].max()
    bins = np.arange(np.floor(min_scale / 0.2) * 0.2, max_scale + 0.2, 0.2)

    for i in range(len(bins) - 1):
        lower, upper = bins[i], bins[i+1]
        # Filter data within the current range
        mask = (df['rule_vel_scale'] >= lower) & (df['rule_vel_scale'] < upper) & (segment_type == 'corner')
        subset = df[mask].copy()

        if subset.empty:
            continue

        # Sort by window_id and time to ensure proper ordering
        subset = subset.sort_values(['window_id', 'time(s)']).reset_index(drop=True)

        # Create unique trajectory IDs by detecting time resets
        subset['trajectory_id'] = 0
        trajectory_counter = 0
        prev_time = -1
        prev_window = None

        for idx in range(len(subset)):
            current_window = subset.loc[idx, 'window_id']
            current_time = subset.loc[idx, 'time(s)']

            # New trajectory if window changes or time jumps back to 0
            if current_window != prev_window or current_time < prev_time:
                trajectory_counter += 1

            subset.loc[idx, 'trajectory_id'] = trajectory_counter
            prev_time = current_time
            prev_window = current_window

        plt.figure(figsize=(12, 6))
        # Group by trajectory_id to plot individual trajectories
        unique_trajectories = subset['trajectory_id'].unique()
        for traj_id, group in subset.groupby('trajectory_id'):
            # Normalize time to start at 0 for comparison across trajectories
            relative_time = group['time(s)'] - group['time(s)'].iloc[0]
            label = f'Traj {traj_id}' if len(unique_trajectories) < 10 else None
            # plt.text(0.05, 0.95, f'Rule Vel Scale: {group["rule_vel_scale"].iloc[0]:.2f}', transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            plt.plot(relative_time, group['velocity'], alpha=0.6, label=label)
            # plt.grid(True, linestyle='--', alpha=0.7)
            # plt.show()

        plt.title(f'Velocity Profiles (rule_vel_scale: [{lower:.1f}, {upper:.1f}))')
        plt.xlabel('Time (s) from start of segment')
        plt.ylabel('Velocity')
        plt.grid(True, linestyle='--', alpha=0.7)
        if len(unique_trajectories) < 10:
            plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    visualize_velocities()
