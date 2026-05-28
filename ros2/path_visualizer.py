import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import math
from data_generation.load_data import load_path
from scipy.spatial.transform import Rotation as R
import numpy as np

# python3 -m ros2.path_visualizer

class RVizVisualizer(Node):
    def __init__(self):
        super().__init__('rviz_visualizer')
        
        # Publishers for the Path and the End-Effector Pose
        self.path_pub = self.create_publisher(Path, 'reference_path', 10)
        self.pose_pub = self.create_publisher(PoseStamped, 'effector_pose', 10)
        
        self.timer = self.create_timer(0.05, self.timer_callback)
        self.i = 0
        
        self.path_msg = self.create_path()

    def create_path(self):
        """Generates a static 3D spiral path."""
        path = Path()
        path.header.frame_id = 'map'

        loaded_path = load_path("datasets/windows-v2/1_wr1fr_1/trajectory.txt")

        multiplier = 0.005
        for point in loaded_path:
            point_pose = point[0:3]
            r = R.from_euler('xyz', point[3:6], degrees=True) 
            quats = r.as_quat()
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.pose.position.x = point_pose[0]*multiplier
            pose.pose.position.y = point_pose[1]*multiplier+2.5
            pose.pose.position.z = point_pose[2]*multiplier
            pose.pose.orientation.w = quats[0]
            pose.pose.orientation.x = quats[1]
            pose.pose.orientation.y = quats[2]
            pose.pose.orientation.z = quats[3]
            path.poses.append(pose)
        return path

    def timer_callback(self):
        now = self.get_clock().now().to_msg()
        
        # 1. Publish the static path continuously
        self.path_msg.header.stamp = now
        self.path_pub.publish(self.path_msg)
        
        # 2. Publish the moving end-effector
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = 'map'
        pose_msg.header.stamp = now
        
        pose_msg.pose.position.x = self.path_msg.poses[self.i % len(self.path_msg.poses)].pose.position.x
        pose_msg.pose.position.y = self.path_msg.poses[self.i % len(self.path_msg.poses)].pose.position.y
        pose_msg.pose.position.z = self.path_msg.poses[self.i % len(self.path_msg.poses)].pose.position.z
        pose_msg.pose.orientation = self.path_msg.poses[self.i % len(self.path_msg.poses)].pose.orientation
        self.pose_pub.publish(pose_msg)
        self.i += 1
        
        self.pose_pub.publish(pose_msg)
        
def main(args=None):
    rclpy.init(args=args)
    node = RVizVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()