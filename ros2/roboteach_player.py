import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import csv
import os
import time
import numpy as np
import subprocess  # Added for RViz automatic opening

# python3 -m ros2.roboteach_player

class RoboTeachPlayer(Node):
    def __init__(self):
        super().__init__('roboteach_player')
        self.window_pub = self.create_publisher(Marker, 'window_mesh', 10)
        self.path_pub = self.create_publisher(Marker, 'recording_path', 10)
        self.tool_pub = self.create_publisher(Marker, 'current_tool', 10)
        
        # Scaling to match existing visualizations
        self.multiplier_obj = 0.1  # For OBJ file in mm
        self.multiplier_csv = 10   # For CSV file in m
        self.y_offset = 0.001
        
        self.recordings_metadata_path = '/home/davidm15/Projects/SkillTrace2/measurements/Robotwin trajectories - recordings.csv'
        self.recordings_dir = '/home/davidm15/Projects/SkillTrace2/datasets/RoboTeach/2026_03_26'
        self.measurements_dir = '/home/davidm15/Projects/SkillTrace2/measurements'
        
        self.recordings_metadata = self.load_metadata()
        self.get_logger().info(f"Loaded {len(self.recordings_metadata)} recordings from metadata.")
        
        self.timer = self.create_timer(0.01, self.run_simulation)
        
        self.current_rec_idx = 0
        self.current_point_idx = 0
        self.current_rec_data = []
        self.processed_points = []
        
        self.state = 'LOAD_NEXT' # LOAD_NEXT, PLAYING, WAITING
        self.wait_start_time = None
        self.playback_speed = 1.0 # Adjust playback speed if needed

    def load_metadata(self):
        metadata = []
        with open(self.recordings_metadata_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                metadata.append(row)
        return metadata

    def load_recording_data(self, recording_id):
        filepath = os.path.join(self.recordings_dir, f"{recording_id}.csv")
        if not os.path.exists(filepath):
            self.get_logger().error(f"Recording file not found: {filepath}")
            return []
        
        data = []
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append({
                    't': float(row['time(s)']),
                    'x': float(row['x(m)']) * self.multiplier_csv,
                    'y': float(row['y(m)']) * self.multiplier_csv + self.y_offset,
                    'z': float(row['z(m)']) * self.multiplier_csv
                })
        return data

    def create_window_marker(self, window_id):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.ns = 'window'
        marker.id = 0
        marker.type = Marker.MESH_RESOURCE
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = self.multiplier_obj
        marker.scale.y = self.multiplier_obj
        marker.scale.z = self.multiplier_obj
        marker.color = ColorRGBA(r=0.7, g=0.7, b=0.7, a=0.3)
        
        obj_file = f"window_{window_id}.obj"
        obj_path = os.path.join(self.measurements_dir, obj_file)
        if not os.path.exists(obj_path):
            self.get_logger().warning(f"Window mesh not found: {obj_path}. Searching for fallback...")
            marker.action = Marker.DELETE
            return marker
            
        marker.mesh_resource = f"file://{obj_path}"
        return marker

    def create_path_marker(self):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.ns = 'path'
        marker.id = 1
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.02
        marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
        return marker

    def create_tool_marker(self, x, y, z):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.ns = 'tool'
        marker.id = 2
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
        return marker

    def run_simulation(self):
        if self.state == 'LOAD_NEXT':
            if self.current_rec_idx >= len(self.recordings_metadata):
                self.get_logger().info("Finished all recordings. Restarting...")
                self.current_rec_idx = 0
            
            self.metadata = self.recordings_metadata[self.current_rec_idx]
            self.get_logger().info(f"Playing recording {self.current_rec_idx + 1}/{len(self.recordings_metadata)}: {self.metadata['recording_id']}")
            
            self.current_rec_data = self.load_recording_data(self.metadata['recording_id'])
            if not self.current_rec_data:
                self.get_logger().warning(f"Skipping {self.metadata['recording_id']} due to missing data.")
                self.current_rec_idx += 1
                return

            self.window_marker = self.create_window_marker(self.metadata['window_id'])
            self.processed_points = []
            self.current_point_idx = 0
            self.state = 'PLAYING'
            self.start_time = self.get_clock().now()
            self.initial_data_time = self.current_rec_data[0]['t']

        elif self.state == 'PLAYING':
            now = self.get_clock().now()
            elapsed = (now - self.start_time).nanoseconds / 1e9 * self.playback_speed
            
            # Find points that should have been played by now (for the path line)
            while (self.current_point_idx < len(self.current_rec_data) and 
                   (self.current_rec_data[self.current_point_idx]['t'] - self.initial_data_time) <= elapsed):
                pt = self.current_rec_data[self.current_point_idx]
                self.processed_points.append(Point(x=pt['x'], y=pt['y'], z=pt['z']))
                self.current_point_idx += 1
            
            # Interpolate for the current tool position (smoothness)
            current_tool_pos = None
            if self.current_point_idx > 0:
                if self.current_point_idx < len(self.current_rec_data):
                    # Between two points
                    p1 = self.current_rec_data[self.current_point_idx - 1]
                    p2 = self.current_rec_data[self.current_point_idx]
                    t1 = p1['t'] - self.initial_data_time
                    t2 = p2['t'] - self.initial_data_time
                    
                    if t2 > t1:
                        alpha = (elapsed - t1) / (t2 - t1)
                        # Clamp alpha to [0, 1] just in case
                        alpha = max(0.0, min(1.0, alpha))
                        ix = p1['x'] + alpha * (p2['x'] - p1['x'])
                        iy = p1['y'] + alpha * (p2['y'] - p1['y'])
                        iz = p1['z'] + alpha * (p2['z'] - p1['z'])
                        current_tool_pos = Point(x=ix, y=iy, z=iz)
                    else:
                        current_tool_pos = Point(x=p2['x'], y=p2['y'], z=p2['z'])
                else:
                    # Last point
                    p_last = self.current_rec_data[-1]
                    current_tool_pos = Point(x=p_last['x'], y=p_last['y'], z=p_last['z'])
            
            # Publish
            stamp = now.to_msg()
            
            # Window
            if self.window_marker:
                self.window_marker.header.stamp = stamp
                self.window_pub.publish(self.window_marker)
            
            # Path
            path_marker = self.create_path_marker()
            path_marker.header.stamp = stamp
            path_marker.points = list(self.processed_points)
            if current_tool_pos:
                path_marker.points.append(current_tool_pos)
            self.path_pub.publish(path_marker)
            
            # Tool
            if current_tool_pos:
                tool_marker = self.create_tool_marker(current_tool_pos.x, current_tool_pos.y, current_tool_pos.z)
                tool_marker.header.stamp = stamp
                self.tool_pub.publish(tool_marker)

            if self.current_point_idx >= len(self.current_rec_data):
                self.get_logger().info(f"Trajectory {self.metadata['recording_id']} finished. Waiting 2 seconds...")
                self.state = 'WAITING'
                self.wait_start_time = self.get_clock().now()

        elif self.state == 'WAITING':
            # Still publish window and full path while waiting
            now = self.get_clock().now()
            stamp = now.to_msg()
            if self.window_marker:
                self.window_marker.header.stamp = stamp
                self.window_pub.publish(self.window_marker)
            
            path_marker = self.create_path_marker()
            path_marker.header.stamp = stamp
            path_marker.points = self.processed_points
            self.path_pub.publish(path_marker)

            # Tool at final position
            if self.processed_points:
                last_pt = self.processed_points[-1]
                tool_marker = self.create_tool_marker(last_pt.x, last_pt.y, last_pt.z)
                tool_marker.header.stamp = stamp
                self.tool_pub.publish(tool_marker)

            elapsed_wait = (now - self.wait_start_time).nanoseconds / 1e9
            if elapsed_wait >= 2.0:
                delete_marker = self.create_path_marker()
                delete_marker.action = Marker.DELETE
                self.path_pub.publish(delete_marker)

                self.current_rec_idx += 1
                self.state = 'LOAD_NEXT'


def main(args=None):
    rclpy.init(args=args)
    node = RoboTeachPlayer()
    
    # Launch RViz automatically
    rviz_config_path = os.path.abspath("ros2/setups/roboteach_player.rviz")
    node.get_logger().info(f"Starting RViz2 with config: {rviz_config_path}")
    rviz_process = subprocess.Popen(['rviz2', '-d', rviz_config_path])

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Shutdown RViz process cleanly on exit
        node.get_logger().info("Shutting down, closing RViz2...")
        rviz_process.terminate()
        rviz_process.wait()
        
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()