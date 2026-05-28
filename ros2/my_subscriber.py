import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int16

# python3 -m ros2.my_subscriber

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('my_first_subscriber')
        
        # Create a Subscriber matching the exact same Topic name and Type
        self.subscription = self.create_subscription(
            Int16,
            'robot_chatter',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{str(msg.data)}"')

def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = MinimalSubscriber()
    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()