#!/usr/bin/env python3
import argparse
import time
from typing import Tuple

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


def _letterbox_fit(img_bgr, win_w: int, win_h: int) -> Tuple[object, float]:
    """Letterbox-fit image into (win_w, win_h) keeping aspect ratio."""
    h, w = img_bgr.shape[:2]
    if h <= 0 or w <= 0:
        canvas = cv2.cvtColor(cv2.UMat(win_h, win_w, cv2.CV_8UC1), cv2.COLOR_GRAY2BGR)
        return canvas, 1.0

    scale = min(win_w / w, win_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = cv2.copyMakeBorder(
        resized,
        top=(win_h - new_h) // 2,
        bottom=win_h - new_h - (win_h - new_h) // 2,
        left=(win_w - new_w) // 2,
        right=win_w - new_w - (win_w - new_w) // 2,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    return canvas, scale


class CUCPublisher(Node):
    def __init__(self, text_topic: str, image_topic: str, reliable: bool = True):
        super().__init__("cuc_publisher")

        reliability = QoSReliabilityPolicy.RELIABLE if reliable else QoSReliabilityPolicy.BEST_EFFORT
        qos = QoSProfile(
            reliability=reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=5,
        )

        self.text_pub = self.create_publisher(String, text_topic, qos)
        self.image_pub = self.create_publisher(Image, image_topic, qos)
        self.bridge = CvBridge()

        self._text_topic = text_topic
        self._image_topic = image_topic

    def publish_text(self, s: str):
        msg = String()
        msg.data = s
        self.text_pub.publish(msg)
        self.get_logger().info(f"Published text to {self._text_topic} ({len(s)} chars)")

    def publish_image_file(self, path: str, fit: bool, win_w: int, win_h: int):
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Could not read image: {path}")

        if fit:
            bgr_fit, scale = _letterbox_fit(bgr, win_w, win_h)
            msg = self.bridge.cv2_to_imgmsg(bgr_fit, encoding="bgr8")
            self.get_logger().info(
                f"Published image to {self._image_topic}: {path} "
                f"(orig {bgr.shape[1]}x{bgr.shape[0]} -> sent {win_w}x{win_h}, scale={scale:.3f})"
            )
        else:
            msg = self.bridge.cv2_to_imgmsg(bgr, encoding="bgr8")
            self.get_logger().info(
                f"Published image to {self._image_topic}: {path} ({bgr.shape[1]}x{bgr.shape[0]})"
            )

        msg.header.frame_id = "tablet"
        self.image_pub.publish(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-topic", default="/tablet/text")
    ap.add_argument("--image-topic", default="/tablet/image")
    ap.add_argument("--text", default=None, help="Text to show on tablet")
    ap.add_argument("--image", default=None, help="Image filepath to show on tablet")

    # New options (safe defaults)
    ap.add_argument("--fit", action="store_true", help="Letterbox-fit image into tablet size (default ON)")
    ap.add_argument("--no-fit", dest="fit", action="store_false", help="Send original image size")
    ap.set_defaults(fit=True)

    ap.add_argument("--win-w", type=int, default=800, help="Target width when --fit is enabled")
    ap.add_argument("--win-h", type=int, default=480, help="Target height when --fit is enabled")

    ap.add_argument("--best-effort", action="store_true", help="Use BEST_EFFORT QoS instead of RELIABLE")
    ap.add_argument("--settle-ms", type=int, default=400, help="Time to spin after publish (ms)")

    args = ap.parse_args()

    rclpy.init()
    node = CUCPublisher(args.text_topic, args.image_topic, reliable=(not args.best_effort))

    try:
        if args.text is not None:
            node.publish_text(args.text)

        if args.image is not None:
            node.publish_image_file(args.image, fit=args.fit, win_w=args.win_w, win_h=args.win_h)

        t_end = time.time() + (args.settle_ms / 1000.0)
        while time.time() < t_end and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)

    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()