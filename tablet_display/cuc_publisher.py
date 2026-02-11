#!/usr/bin/env python3
import argparse
import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class CUCPublisher(Node):
    def __init__(self, text_topic: str, image_topic: str):
        super().__init__("cuc_publisher")
        self.text_pub = self.create_publisher(String, text_topic, 10)
        self.image_pub = self.create_publisher(Image, image_topic, 10)
        self.bridge = CvBridge()

    def publish_text(self, s: str):
        msg = String()
        msg.data = s
        self.text_pub.publish(msg)
        self.get_logger().info(f"Published text ({len(s)} chars)")

    def publish_image_file(self, path: str):
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Could not read image: {path}")
        msg = self.bridge.cv2_to_imgmsg(bgr, encoding="bgr8")
        self.image_pub.publish(msg)
        self.get_logger().info(f"Published image: {path} ({bgr.shape[1]}x{bgr.shape[0]})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-topic", default="/tablet/text")
    ap.add_argument("--image-topic", default="/tablet/image")
    ap.add_argument("--text", default=None, help="Text to show on tablet")
    ap.add_argument("--image", default=None, help="Image filepath to show on tablet")
    args = ap.parse_args()

    rclpy.init()
    node = CUCPublisher(args.text_topic, args.image_topic)

    try:
        if args.text is not None:
            node.publish_text(args.text)
        if args.image is not None:
            node.publish_image_file(args.image)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()