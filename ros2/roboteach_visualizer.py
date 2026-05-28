import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import csv
import os
import glob
import numpy as np

# python3 -m ros2.roboteach_visualizer

class RoboTeachVisualizer(Node):
    def __init__(self):
        super().__init__('roboteach_visualizer')
        self.recordings_pub = self.create_publisher(MarkerArray, 'recordings', 10)
        self.window_pub = self.create_publisher(Marker, 'window_mesh', 10)
        self.positions_pub = self.create_publisher(Marker, 'positions', 10)
        
        self.timer = self.create_timer(1.0, self.timer_callback)
        
        # Scaling to match existing visualizations (e.g., path_visualizer.py)
        self.multiplier_obj = 0.1  # For OBJ file in mm
        self.multiplier_csv = 10   # For CSV file in m (1m * 5.0 = 5.0m, matches 1000mm * 0.005 = 5.0m)
        self.y_offset = 0.001
        
        self.recordings_path = '/home/davidm15/Projects/SkillTrace2/datasets/RoboTeach/2026_03_26'
        self.obj_path = 'file:///home/davidm15/Projects/SkillTrace2/measurements/window_87.obj'
        
        self.get_logger().info(f"Loading recordings from {self.recordings_path}")
        self.marker_array = self.load_recordings()
        self.get_logger().info(f"Loaded {len(self.marker_array.markers)} recordings.")
        
        self.window_marker = self.create_window_marker()

        self.positions = np.array([[0.0, 0.0, 0.0], [5.5, 0, 0.0], [0.0, 2.5, 0.0], [5.5, 2.5, 0.0]])
        self.positions_marker = self.create_positions_marker()

    def load_recordings(self):
        marker_array = MarkerArray()
        csv_files = sorted(glob.glob(os.path.join(self.recordings_path, "*.csv")))
        
        # Color palette
        colors = [
            ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0), # Red
            ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0), # Green
            ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0), # Blue
            ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0), # Yellow
            ColorRGBA(r=1.0, g=0.0, b=1.0, a=1.0), # Magenta
            ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0), # Cyan
            ColorRGBA(r=1.0, g=0.5, b=0.0, a=1.0), # Orange
            ColorRGBA(r=0.5, g=0.0, b=1.0, a=1.0), # Purple
        ]
        
        for i, csv_file in enumerate(csv_files):
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.ns = 'recordings'
            marker.id = i
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.01 # Line width
            marker.color = colors[i % len(colors)]
            
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    p = Point()
                    # Directly use column names from the CSV file
                    p.x = float(row['x(m)']) * self.multiplier_csv
                    p.y = float(row['y(m)']) * self.multiplier_csv + self.y_offset
                    p.z = float(row['z(m)']) * self.multiplier_csv
                    marker.points.append(p)
            
            marker_array.markers.append(marker)
        return marker_array

    def create_window_marker(self):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.ns = 'window'
        marker.id = 0
        marker.type = Marker.MESH_RESOURCE
        marker.action = Marker.ADD
        
        # Position and Orientation
        marker.pose.position.x = 0.0
        marker.pose.position.y = self.y_offset
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        
        # Scale
        marker.scale.x = self.multiplier_obj
        marker.scale.y = self.multiplier_obj
        marker.scale.z = self.multiplier_obj
        
        # Color (semi-transparent grey)
        marker.color = ColorRGBA(r=0.7, g=0.7, b=0.7, a=0.3)
        
        marker.mesh_resource = self.obj_path
        return marker

    def timer_callback(self):
        now = self.get_clock().now().to_msg()
        for marker in self.marker_array.markers:
            marker.header.stamp = now
        self.window_marker.header.stamp = now
        
        self.recordings_pub.publish(self.marker_array)
        self.window_pub.publish(self.window_marker)
        self.positions_pub.publish(self.positions_marker)


    def create_positions_marker(self):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.ns = 'positions'
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        
        # Position and Orientation
        marker.pose.position.x = 0.0
        marker.pose.position.y = self.y_offset
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        
        # Scale
        marker.scale.x = self.multiplier_obj
        marker.scale.y = self.multiplier_obj
        marker.scale.z = self.multiplier_obj
        
        colors = [ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0), ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0), ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0), ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)]
        
        # Color (semi-transparent grey)
        marker.colors = colors  
        
        marker.points = [Point(x=p[0], y=p[1], z=p[2]) for p in self.positions]
        return marker

def main(args=None):
    rclpy.init(args=args)
    node = RoboTeachVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
