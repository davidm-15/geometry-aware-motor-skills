import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
import numpy as np
import pathlib

# python3 -m ros2.visualize_end_effector
# python3 -m ros2.visualize_end_effector outputs/test_load_data/trajectory.txt

from data_generation.path_parser import parse_trajectory_file
from data_generation.simulation import VirtualEndEffector
from data_generation.pure_pursuit import PurePursuitController
from scipy.spatial.transform import Rotation as R

class VirtualEffectorVisualizer(Node):
    def __init__(self, data_file, dt=0.02):
        super().__init__('virtual_end_effector_visualizer')
        
        # Publishers
        self.path_pub = self.create_publisher(Path, 'reference_path', 10)
        self.effector_pub = self.create_publisher(PoseStamped, 'effector_pose', 10)
        self.target_pub = self.create_publisher(PoseStamped, 'target_pose', 10)
        self.mesh_pub = self.create_publisher(Marker, 'effector_mesh', 10)
        
        self.dt = dt
        self.timer = self.create_timer(self.dt, self.timer_callback)
        
        # Load the trajectory and setup simulation
        self.get_logger().info(f"Loading trajectory from {data_file}...")
        subpaths = parse_trajectory_file(data_file)
        
        data_dir = pathlib.Path(data_file).absolute().parent
        obj_files = list(data_dir.glob("*.obj"))
        self.mesh_resource_uri = f"file://{obj_files[0]}" if obj_files else ""
        print(self.mesh_resource_uri)
        if self.mesh_resource_uri:
            self.get_logger().info(f"Found mesh: {self.mesh_resource_uri}")
        
        if not subpaths:
            self.get_logger().error("No valid paths found in the trajectory file.")
            raise ValueError("No paths to simulate.")
            
        self.all_scaled_paths = []
        multiplier = 0.005
        
        for subpath in subpaths:
            raw_pos = subpath['positions']
            scaled_pos = []
            for p in raw_pos:
                scaled_p = [p[0] * multiplier, p[1] * multiplier, p[2] * multiplier]
                scaled_pos.append(np.array(scaled_p))
            
            self.all_scaled_paths.append({
                'positions': np.array(scaled_pos),
                'quats': subpath['quaternions']
            })
            
        self.current_path_idx = 0
        self.load_path_by_idx(self.current_path_idx)
        
        self.speed_pub = self.create_publisher(Path, 'speed_plot', 10)
        self.accel_pub = self.create_publisher(Path, 'accel_plot', 10)
        self.all_paths_pub = self.create_publisher(MarkerArray, 'all_strokes', 10)
        
        self.plot_history_len = 500
        self.speed_history = []
        self.accel_history = []
        self.prev_v_norm = 0.0
        self.hold_timer_start = None

        self.clear_mesh_marker()

    def clear_mesh_marker(self):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.ns = 'effector_mesh'
        marker.id = 0
        marker.action = Marker.DELETE
        self.mesh_pub.publish(marker)

    def load_path_by_idx(self, idx):
        path_data = self.all_scaled_paths[idx]
        self.positions = path_data['positions']
        self.quats = path_data['quats']
        
        multiplier = 0.005
        
        self.simulator = VirtualEndEffector(dt=self.dt)
        self.simulator.controller.lookahead_distance *= multiplier
        self.simulator.reset(self.positions[0], self.quats[0])
        self.get_logger().info(f"Initialized simulation at position: {self.positions[0]}")
        
        self.ref_path_msg = self.create_path_msg(self.positions, self.quats)
        self.path_finished = False
        self.end_threshold = 1.0 * multiplier
        
        self.plot_history_len = 500
        self.speed_history = []
        self.accel_history = []
        self.prev_v_norm = 0.0

    def create_path_msg(self, positions, quats):
        path = Path()
        path.header.frame_id = 'map'
        for i in range(len(positions)):
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.pose.position.x = positions[i][0]
            pose.pose.position.y = positions[i][1]
            pose.pose.position.z = positions[i][2]
            pose.pose.orientation.x = quats[i][0]
            pose.pose.orientation.y = quats[i][1]
            pose.pose.orientation.z = quats[i][2]
            pose.pose.orientation.w = quats[i][3]
            path.poses.append(pose)
        return path

    def publish_all_paths(self, now):
        marker_array = MarkerArray()
        for idx, path_data in enumerate(self.all_scaled_paths):
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = now
            marker.ns = 'strokes'
            marker.id = idx
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.05 if idx == self.current_path_idx else 0.01
            
            if idx == self.current_path_idx:
                marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0) # Green
            else:
                marker.color = ColorRGBA(r=0.4, g=0.4, b=0.4, a=0.5) # Greyish
                
            for p in path_data['positions']:
                pt = Point()
                pt.x = float(p[0])
                pt.y = float(p[1])
                pt.z = float(p[2])
                marker.points.append(pt)
                
            marker_array.markers.append(marker)
        self.all_paths_pub.publish(marker_array)

    def timer_callback(self):
        now = self.get_clock().now().to_msg()
        
        # 1. Publish reference path
        self.ref_path_msg.header.stamp = now
        self.path_pub.publish(self.ref_path_msg)
        
        # 1.5 Publish all subpaths
        self.publish_all_paths(now)
        
        # Hold simulation state logic
        if self.hold_timer_start is not None:
            # We are holding at the end of a path
            # We continue to simulate physics, so the end-effector demonstrates braking
            elapsed = self.get_clock().now().nanoseconds - self.hold_timer_start
            if elapsed > 3.0 * 1e9:  # 3 seconds hold time
                # Simulation hold is over, jump to the next path
                self.hold_timer_start = None
                self.current_path_idx = (self.current_path_idx + 1) % len(self.all_scaled_paths)
                self.get_logger().info(f"Switching to path idx: {self.current_path_idx}")
                self.load_path_by_idx(self.current_path_idx)
                self.speed_history.clear()
                self.accel_history.clear()
                self.prev_v_norm = 0.0
                
        if self.path_finished:
            return
            
        # 2. Compute control
        target_pos, target_quat, target_idx = self.simulator.controller.get_lookahead_point(
            self.simulator.p, self.positions, self.quats
        )
        
        f_ctrl, tau_ctrl = self.simulator.controller.compute_control(
            self.simulator.p, self.simulator.q, target_pos, target_quat
        )
        
        # 3. Simulate step
        t, p, q, v_norm, w_norm = self.simulator.step(f_ctrl, tau_ctrl)
        
        # 4. Calculate acceleration and update histories
        accel = (v_norm - self.prev_v_norm) / self.dt
        self.prev_v_norm = v_norm
        
        self.speed_history.append(v_norm)
        self.accel_history.append(accel)
        if len(self.speed_history) > self.plot_history_len:
            self.speed_history.pop(0)
            self.accel_history.pop(0)
            
        # 5. Publish 2D Plots as Paths
        speed_path = Path()
        speed_path.header.frame_id = 'map'
        speed_path.header.stamp = now
        
        accel_path = Path()
        accel_path.header.frame_id = 'map'
        accel_path.header.stamp = now
        
        plot_x_scale = 0.01 
        plot_y_offset_speed = 3.0
        plot_y_offset_accel = 5.0
        x_offset = -(self.plot_history_len * plot_x_scale / 2.0)
        
        for i, (s, a) in enumerate(zip(self.speed_history, self.accel_history)):
            ps = PoseStamped()
            ps.header.frame_id = 'map'
            ps.pose.position.x = float(i) * plot_x_scale + x_offset
            ps.pose.position.y = float(s) + plot_y_offset_speed
            ps.pose.position.z = 0.0
            ps.pose.orientation.w = 1.0
            speed_path.poses.append(ps)
            
            pa = PoseStamped()
            pa.header.frame_id = 'map'
            pa.pose.position.x = float(i) * plot_x_scale + x_offset
            pa.pose.position.y = float(a) * 0.05 + plot_y_offset_accel  # Scale mostly high accel
            pa.pose.position.z = 0.0
            pa.pose.orientation.w = 1.0
            accel_path.poses.append(pa)
            
        self.speed_pub.publish(speed_path)
        self.accel_pub.publish(accel_path)
        
        # 6. Publish target lookahead pose
        target_msg = PoseStamped()
        target_msg.header.frame_id = 'map'
        target_msg.header.stamp = now
        target_msg.pose.position.x = target_pos[0]
        target_msg.pose.position.y = target_pos[1]
        target_msg.pose.position.z = target_pos[2]
        target_msg.pose.orientation.x = target_quat[0]
        target_msg.pose.orientation.y = target_quat[1]
        target_msg.pose.orientation.z = target_quat[2]
        target_msg.pose.orientation.w = target_quat[3]
        self.target_pub.publish(target_msg)
        
        # 7. Publish effector pose
        effector_msg = PoseStamped()
        effector_msg.header.frame_id = 'map'
        effector_msg.header.stamp = now
        effector_msg.pose.position.x = p[0]
        effector_msg.pose.position.y = p[1]
        effector_msg.pose.position.z = p[2]
        effector_msg.pose.orientation.x = q[0]
        effector_msg.pose.orientation.y = q[1]
        effector_msg.pose.orientation.z = q[2]
        effector_msg.pose.orientation.w = q[3]
        self.effector_pub.publish(effector_msg)
        

        if self.mesh_resource_uri:
            mesh_marker = Marker()
            mesh_marker.header.frame_id = 'map'
            mesh_marker.header.stamp = now
            mesh_marker.ns = 'effector_mesh'
            mesh_marker.id = 0
            mesh_marker.type = Marker.MESH_RESOURCE
            mesh_marker.action = Marker.ADD

            mesh_marker.pose.position.x = 0.0
            mesh_marker.pose.position.y = 0.0
            mesh_marker.pose.position.z = 0.0

            mesh_marker.pose.orientation.x = 0.0
            mesh_marker.pose.orientation.y = 0.0
            mesh_marker.pose.orientation.z = 0.0
            mesh_marker.pose.orientation.w = 1.0

            mesh_marker.scale.x = 0.005
            mesh_marker.scale.y = 0.005
            mesh_marker.scale.z = 0.005

            mesh_marker.color = ColorRGBA(r=0.6, g=0.6, b=0.6, a=0.8)
            mesh_marker.mesh_resource = self.mesh_resource_uri
            # mesh_marker.mesh_resource = "file:///home/davidm15/Projects/SkillTrace2/data_generation/L_shape.obj"
            # mesh_marker.mesh_resource = "file:///home/davidm15/Projects/SkillTrace2/datasets/L_shape/0_L_shape_source/L_shape.obj"
            # mesh_marker.mesh_resource = "file:///home/davidm15/Projects/SkillTrace2/datasets/L_shape/00_source/L_shape.obj"

            self.mesh_pub.publish(mesh_marker)



        # 8. Check if we reached the end
        if self.hold_timer_start is None:
            dist_to_end = np.linalg.norm(self.simulator.p - self.positions[-1])
            if dist_to_end < self.end_threshold: # Close enough to the end
                self.get_logger().info("Reached end of path. Holding for 3 seconds before next path.")
                self.hold_timer_start = self.get_clock().now().nanoseconds
                # We don't reset until the hold is finished


import sys

def main(args=None):
    rclpy.init(args=args)
    # Using a dataset file path - adjust if needed
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    else:
        data_file = "datasets/windows-v2/1_wr1fr_1/trajectory.txt"
    node = VirtualEffectorVisualizer(data_file=data_file, dt=0.02)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
