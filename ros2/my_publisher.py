import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int16
import requests
from lxml import html

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('my_publisher')

        self.publisher_ = self.create_publisher(Int16, "robot_chatter", 10)

        self.timer = self.create_timer(2, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = Int16()
        msg.data = get_num_people()
        self.publisher_.publish(msg)
        self.get_logger().info(f"Publishing: '{str(msg.data)}'")
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()



def get_num_people() -> int:
    """
    Scrapes the Techlib website to get the current number of people 
    using the provided XPath and returns it as an integer.
    """
    url = "https://www.techlib.cz/cs/"
    xpath = '//*[@id="content"]/div/div/div[3]/div/div[2]/span'
    
    try:
        # Fetch the webpage
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status() # Check for HTTP errors
        
        # Parse the HTML content
        tree = html.fromstring(response.content)
        
        # Find the element using the provided XPath
        elements = tree.xpath(xpath)
        
        if elements:
            # Extract text, clean up whitespace, and convert to integer
            number_str = elements[0].text_content().strip()
            return int(number_str)
        else:
            print("Error: Element not found at the specified XPath.")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Network error occurred: {e}")
        return None
    except ValueError as e:
        print(f"Could not convert the scraped text to an integer: {e}")
        return None



if __name__ == "__main__":
    main()
    # print(get_num_people())