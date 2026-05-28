import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
import numpy as np
import pandas as pd
import pathlib
import random
import sys
import subprocess
import os
"""
    Visualize generated CSV trajectory datasets in RViz2.
    Works with any shape dataset (L-shape, I-shape, window/cross, …).
"""

# python3 -m ros2.visualize_trajectory
# python3 -m ros2.visualize_trajectory datasets/L_shape/test.csv
# python3 -m ros2.visualize_trajectory outputs/test_load_data/trajectory.csv
# python3 -m ros2.visualize_trajectory datasets/window_cross/test.csv
# python3 -m ros2.visualize_trajectory datasets/windows-v2/train.csv --playback_speed 100 --hold_time 0.01


class TrajectoryVisualizer(Node):
    def __init__(self, csv_file, dt=0.01, playback_speed=1.0, hold_time=2.0):
        super().__init__('trajectory_visualizer')
        
        # Publishers
        self.path_pub = self.create_publisher(Marker, 'reference_path_colored', 10)
        self.effector_pub = self.create_publisher(PoseStamped, 'effector_pose', 10)
        self.mesh_pub = self.create_publisher(Marker, 'effector_mesh', 10)
        
        self.dt = dt
        self.scaling_multiplier = 0.005
        self.hold_time = hold_time
        
        # 1.0 = Real time, 0.5 = Half speed (slow motion), 2.0 = Double speed
        self.playback_speed = playback_speed 
        
        # Load the CSV and group by stroke_id instead of window_id!
        self.get_logger().info(f"Loading data from {csv_file}...")
        self.df = pd.read_csv(csv_file)
        self.grouped = self.df.groupby('demonstration_id')
        self.demo_ids = list(self.grouped.groups.keys())
        self.get_logger().info(f"Loaded {len(self.demo_ids)} unique demonstrations.")

        self.base_dir = pathlib.Path(csv_file).parent
        
        self.timer = self.create_timer(self.dt, self.timer_callback)
        
        self.hold_timer_start = None
        self.demo_strokes = []
        self.demo_group = None
        self.current_stroke_idx = 0
        self.load_random_demonstration()

    def load_random_demonstration(self):
        demo_id = random.choice(self.demo_ids)
        demo_group = self.grouped.get_group(demo_id)

        window_id = demo_group['window_id'].iloc[0]
        self.get_logger().info(f"Selected demonstration_id: {demo_id} (Window: {window_id})")

        # Build static path marker for all strokes of this demonstration
        demo_pos = demo_group[['x', 'y', 'z']].values * self.scaling_multiplier
        demo_seg = demo_group['segment_type'].values
        demo_stroke_ids = demo_group['stroke_id'].values
        self.ref_path_msg = self.create_colored_path_marker(demo_pos, demo_seg, demo_stroke_ids)

        # Store strokes in order for sequential playback
        self.demo_group = demo_group
        self.demo_strokes = sorted(demo_group['stroke_id'].unique())
        self.current_stroke_idx = 0

        mesh_path = self.base_dir / str(window_id) / f"{window_id}.obj"
        if mesh_path.exists():
            self.mesh_resource_uri = f"file://{mesh_path.absolute()}"
            self.get_logger().info(f"Found mesh: {self.mesh_resource_uri}")
        else:
            self.get_logger().warn(f"Mesh not found at {mesh_path}")
            self.mesh_resource_uri = ""

        self._load_stroke(0)

    def _load_stroke(self, idx):
        stroke_id = self.demo_strokes[idx]
        stroke_group = self.demo_group[self.demo_group['stroke_id'] == stroke_id]
        n = len(self.demo_strokes)
        self.get_logger().info(f"Playing stroke {idx + 1}/{n} (stroke_id={stroke_id})")

        self.time_stamps = stroke_group['time(s)'].values.copy()
        self.time_stamps -= self.time_stamps[0]

        self.positions = stroke_group[['x', 'y', 'z']].values.copy() * self.scaling_multiplier
        self.quats = stroke_group[['qx', 'qy', 'qz', 'qw']].values.copy()

        self.start_replay_time = None
        self.replay_finished = False

    def create_colored_path_marker(self, positions, segments, stroke_ids=None):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.ns = 'colored_path'
        marker.id = 1
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.03  # Line width
        marker.pose.orientation.w = 1.0
        
        color_straight = ColorRGBA(r=0.1, g=0.9, b=0.1, a=1.0) # Green
        color_corner = ColorRGBA(r=0.9, g=0.1, b=0.1, a=1.0)   # Red

        for i in range(len(positions) - 1):
            if stroke_ids is not None and stroke_ids[i] != stroke_ids[i + 1]:
                continue

            p1 = Point(x=float(positions[i][0]), y=float(positions[i][1]), z=float(positions[i][2]))
            p2 = Point(x=float(positions[i+1][0]), y=float(positions[i+1][1]), z=float(positions[i+1][2]))
            
            c = color_straight if segments[i] == 'straight' else color_corner
            
            marker.points.append(p1)
            marker.points.append(p2)
            marker.colors.append(c)
            marker.colors.append(c)
            
        return marker

    def timer_callback(self):
        now_time = self.get_clock().now()
        now_msg = now_time.to_msg()
        
        # 1. Publish reference path
        self.ref_path_msg.header.stamp = now_msg
        self.path_pub.publish(self.ref_path_msg)
        
        # 2. Publish mesh marker
        if self.mesh_resource_uri:
            mesh_marker = Marker()
            mesh_marker.header.frame_id = 'map'
            mesh_marker.header.stamp = now_msg
            mesh_marker.ns = 'effector_mesh'
            mesh_marker.id = 0
            mesh_marker.type = Marker.MESH_RESOURCE
            mesh_marker.action = Marker.ADD
            mesh_marker.pose.position.x = 0.0
            mesh_marker.pose.position.y = 0.0
            mesh_marker.pose.position.z = 0.0
            mesh_marker.pose.orientation.w = 1.0
            mesh_marker.scale.x = self.scaling_multiplier
            mesh_marker.scale.y = self.scaling_multiplier
            mesh_marker.scale.z = self.scaling_multiplier
            mesh_marker.color = ColorRGBA(r=0.6, g=0.6, b=0.6, a=0.8)
            mesh_marker.mesh_resource = self.mesh_resource_uri
            self.mesh_pub.publish(mesh_marker)

        # 3. Handle Hold State
        if self.hold_timer_start is not None:
            elapsed = (now_time - self.hold_timer_start).nanoseconds / 1e9
            if elapsed > self.hold_time: # 2 seconds hold
                self.hold_timer_start = None
                self.current_stroke_idx += 1
                if self.current_stroke_idx < len(self.demo_strokes):
                    self._load_stroke(self.current_stroke_idx)
                else:
                    self.load_random_demonstration()
            return

        if self.replay_finished:
            return

        # 4. Determine elapsed time with playback speed applied
        if self.start_replay_time is None:
            self.start_replay_time = now_time
            
        elapsed_replay = ((now_time - self.start_replay_time).nanoseconds / 1e9) * self.playback_speed
        
        # Find index where time_stamps is closest to elapsed
        idx = np.searchsorted(self.time_stamps, elapsed_replay)
        
        if idx == 0:
            current_pos = self.positions[0]
            current_quat = self.quats[0]
        elif idx >= len(self.time_stamps):
            idx = len(self.time_stamps) - 1
            current_pos = self.positions[idx]
            current_quat = self.quats[idx]
            
            self.get_logger().info("Reached end of replay. Holding for 2 seconds.")
            self.hold_timer_start = now_time
            self.replay_finished = True
        else:
            # INTERPOLATION: Smooth out the movement between points based on exact time
            t0 = self.time_stamps[idx - 1]
            t1 = self.time_stamps[idx]
            p0 = self.positions[idx - 1]
            p1 = self.positions[idx]
            q0 = self.quats[idx - 1]
            q1 = self.quats[idx]
            
            # Protect against division by zero if timestamps are identical
            ratio = (elapsed_replay - t0) / (t1 - t0) if t1 > t0 else 1.0
            
            # Position LERP
            current_pos = p0 + ratio * (p1 - p0)
            
            # Quaternion N-LERP (Linear interpolation + normalization)
            current_quat = q0 + ratio * (q1 - q0)
            current_quat /= np.linalg.norm(current_quat)

        # 5. Publish current pose
        effector_msg = PoseStamped()
        effector_msg.header.frame_id = 'map'
        effector_msg.header.stamp = now_msg
        effector_msg.pose.position.x = float(current_pos[0])
        effector_msg.pose.position.y = float(current_pos[1])
        effector_msg.pose.position.z = float(current_pos[2])
        effector_msg.pose.orientation.x = float(current_quat[0])
        effector_msg.pose.orientation.y = float(current_quat[1])
        effector_msg.pose.orientation.z = float(current_quat[2])
        effector_msg.pose.orientation.w = float(current_quat[3])
        self.effector_pub.publish(effector_msg)

def main(args=None):
    import argparse

    parser = argparse.ArgumentParser(description='Visualize trajectory CSV in RViz2')
    parser.add_argument('csv_path', nargs='?', default="datasets/L_shape/test_samples.csv",
                        help='Path to trajectory CSV file')
    parser.add_argument('--playback_speed', type=float, default=3.0,
                        help='Playback speed multiplier (e.g. 4.0)')
    parser.add_argument('--hold_time', type=float, default=2.0,
                        help='Time to hold at the end of each stroke (seconds)')
    parsed = parser.parse_args()

    rclpy.init(args=args)

    csv_path = parsed.csv_path
    playback_speed = parsed.playback_speed
    hold_time = parsed.hold_time

    node = TrajectoryVisualizer(csv_file=csv_path, dt=0.01, playback_speed=playback_speed, hold_time=hold_time)


    rviz_config_path = os.path.abspath("ros2/setups/visualize_l_shape.rviz")
    
    node.get_logger().info(f"Starting RViz2 with config: {rviz_config_path}")
    rviz_process = subprocess.Popen(['rviz2', '-d', rviz_config_path])
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down, closing RViz2...")
        rviz_process.terminate()
        rviz_process.wait()

        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()