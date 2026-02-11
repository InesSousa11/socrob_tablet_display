#!/usr/bin/env python3
import textwrap
from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.lifecycle import LifecycleState
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


def _fit_to_window(img_bgr: np.ndarray, win_w: int, win_h: int) -> np.ndarray:
    """Letterbox-fit an image into (win_w, win_h) keeping aspect ratio."""
    if img_bgr is None:
        return np.zeros((win_h, win_w, 3), dtype=np.uint8)

    h, w = img_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((win_h, win_w, 3), dtype=np.uint8)

    scale = min(win_w / w, win_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
    x0 = (win_w - new_w) // 2
    y0 = (win_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def _render_text_canvas(
    text: str,
    win_w: int,
    win_h: int,
    font_scale: float,
    thickness: int,
    margin: int,
) -> np.ndarray:
    """Render wrapped text centered on a black canvas."""
    canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)

    text = (text or "").strip()
    if not text:
        return canvas

    # Wrap width heuristic: depends on font scale + screen size
    # Adjust if you want longer/shorter lines.
    approx_chars_per_line = max(10, int(win_w / (12 * max(0.1, font_scale))))
    lines = []
    for para in text.split("\n"):
        lines.extend(textwrap.wrap(para, width=approx_chars_per_line) or [""])

    font = cv2.FONT_HERSHEY_SIMPLEX
    line_h = int(round(30 * font_scale)) + 10
    total_h = len(lines) * line_h

    y = max(margin + line_h, (win_h - total_h) // 2 + line_h)
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, font, font_scale, thickness)
        x = max(margin, (win_w - tw) // 2)
        cv2.putText(canvas, line, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += line_h
        if y > win_h - margin:
            break

    return canvas


class TabletDisplayLifecycleNode(LifecycleNode):
    """
    Lifecycle node that shows either:
      - last received text (std_msgs/String)
      - last received image (sensor_msgs/Image)

    It renders to an OpenCV window sized for the tablet screen.
    """

    def __init__(self):
        super().__init__("tablet_display")

        # ---- parameters ----
        self.declare_parameter("text_topic", "/tablet/text")
        self.declare_parameter("image_topic", "/tablet/image")
        self.declare_parameter("window_name", "Robot Tablet")
        self.declare_parameter("fullscreen", True)

        # If fullscreen is False, these define the tablet window size.
        self.declare_parameter("window_width", 800)
        self.declare_parameter("window_height", 480)

        # Useful when the tablet is a smaller monitor next to a bigger one:
        # set x/y so the window appears on the tablet monitor.
        self.declare_parameter("window_x", 0)
        self.declare_parameter("window_y", 0)

        # Text rendering knobs
        self.declare_parameter("font_scale", 1.2)
        self.declare_parameter("font_thickness", 2)
        self.declare_parameter("text_margin", 30)

        # Update rate for GUI (Hz)
        self.declare_parameter("ui_rate_hz", 30.0)

        # QoS (same idea as your wrapper)
        self.declare_parameter("reliability", 1)  # 1=RELIABLE, 2=BEST_EFFORT

        # ---- runtime ----
        self.bridge = CvBridge()
        self.text_sub = None
        self.image_sub = None
        self.ui_timer = None

        self.last_text: str = ""
        self.last_img_bgr: Optional[np.ndarray] = None
        self.mode: str = "text"  # "text" or "image"

        self._window_created = False

    # ---------------- Lifecycle hooks ----------------

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        try:
            self.get_logger().info("[tablet_display] Configuring...")

            self.text_topic = self.get_parameter("text_topic").value
            self.image_topic = self.get_parameter("image_topic").value
            self.window_name = self.get_parameter("window_name").value
            self.fullscreen = bool(self.get_parameter("fullscreen").value)
            self.win_w = int(self.get_parameter("window_width").value)
            self.win_h = int(self.get_parameter("window_height").value)
            self.win_x = int(self.get_parameter("window_x").value)
            self.win_y = int(self.get_parameter("window_y").value)

            self.font_scale = float(self.get_parameter("font_scale").value)
            self.font_thickness = int(self.get_parameter("font_thickness").value)
            self.text_margin = int(self.get_parameter("text_margin").value)

            self.ui_rate_hz = float(self.get_parameter("ui_rate_hz").value)

            reliability = int(self.get_parameter("reliability").value)
            if reliability == 2:
                rel = QoSReliabilityPolicy.BEST_EFFORT
            else:
                rel = QoSReliabilityPolicy.RELIABLE

            self.qos = QoSProfile(
                reliability=rel,
                history=QoSHistoryPolicy.KEEP_LAST,
                durability=QoSDurabilityPolicy.VOLATILE,
                depth=5,
            )

            self.get_logger().info(
                f"[tablet_display] topics: text={self.text_topic} image={self.image_topic}"
            )
            self.get_logger().info(
                f"[tablet_display] window: fullscreen={self.fullscreen} size={self.win_w}x{self.win_h} pos=({self.win_x},{self.win_y})"
            )

            return TransitionCallbackReturn.SUCCESS

        except Exception as e:
            self.get_logger().error(f"[tablet_display] configure failed: {repr(e)}")
            return TransitionCallbackReturn.FAILURE

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        try:
            self.get_logger().info("[tablet_display] Activating...")

            # Create window and move it to tablet screen coordinates (important with multi-monitor)
            self._create_window()

            self.text_sub = self.create_subscription(String, self.text_topic, self._on_text, self.qos)
            self.image_sub = self.create_subscription(Image, self.image_topic, self._on_image, self.qos)

            period = 1.0 / max(1e-3, self.ui_rate_hz)
            self.ui_timer = self.create_timer(period, self._ui_tick)

            self.get_logger().info("[tablet_display] Active.")
            return TransitionCallbackReturn.SUCCESS

        except Exception as e:
            self.get_logger().error(f"[tablet_display] activate failed: {repr(e)}")
            return TransitionCallbackReturn.FAILURE

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        try:
            self.get_logger().info("[tablet_display] Deactivating...")

            if self.ui_timer is not None:
                self.ui_timer.cancel()
                self.ui_timer = None

            if self.text_sub is not None:
                self.destroy_subscription(self.text_sub)
                self.text_sub = None

            if self.image_sub is not None:
                self.destroy_subscription(self.image_sub)
                self.image_sub = None

            self.get_logger().info("[tablet_display] Deactivated.")
            return TransitionCallbackReturn.SUCCESS

        except Exception as e:
            self.get_logger().error(f"[tablet_display] deactivate failed: {repr(e)}")
            return TransitionCallbackReturn.FAILURE

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        try:
            self.get_logger().info("[tablet_display] Cleaning up...")

            self.last_text = ""
            self.last_img_bgr = None
            self.mode = "text"

            self._destroy_window()

            self.get_logger().info("[tablet_display] Cleaned.")
            return TransitionCallbackReturn.SUCCESS

        except Exception as e:
            self.get_logger().error(f"[tablet_display] cleanup failed: {repr(e)}")
            return TransitionCallbackReturn.FAILURE

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        try:
            self.get_logger().info("[tablet_display] Shutting down...")
            self._destroy_window()
            return TransitionCallbackReturn.SUCCESS
        except Exception:
            return TransitionCallbackReturn.SUCCESS

    # ---------------- Sub callbacks ----------------

    def _on_text(self, msg: String) -> None:
        self.last_text = msg.data
        self.mode = "text"

    def _on_image(self, msg: Image) -> None:
        try:
            # Convert ROS image to OpenCV BGR
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.last_img_bgr = frame
            self.mode = "image"
        except Exception as e:
            self.get_logger().warn(f"[tablet_display] image convert failed: {repr(e)}")

    # ---------------- UI ----------------

    def _create_window(self) -> None:
        if self._window_created:
            return

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        if self.fullscreen:
            # Fullscreen tends to be best on tablets
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.resizeWindow(self.window_name, self.win_w, self.win_h)

        # Move to target monitor coords (tablet screen)
        # You may need to tune window_x/window_y for your monitor layout.
        cv2.moveWindow(self.window_name, self.win_x, self.win_y)

        self._window_created = True

    def _destroy_window(self) -> None:
        if self._window_created:
            try:
                cv2.destroyWindow(self.window_name)
            except Exception:
                pass
        self._window_created = False

    def _ui_tick(self) -> None:
        if not self._window_created:
            return

        # Decide target render size (if fullscreen: we still draw into win_w x win_h)
        # If you want “true fullscreen size”, set window_width/window_height to tablet’s resolution.
        win_w, win_h = self.win_w, self.win_h

        if self.mode == "image" and self.last_img_bgr is not None:
            canvas = _fit_to_window(self.last_img_bgr, win_w, win_h)
        else:
            canvas = _render_text_canvas(
                self.last_text,
                win_w, win_h,
                font_scale=self.font_scale,
                thickness=self.font_thickness,
                margin=self.text_margin,
            )

        cv2.imshow(self.window_name, canvas)
        # keep window responsive
        cv2.waitKey(1)


def main():
    rclpy.init()
    node = TabletDisplayLifecycleNode()

    # Auto-transition like your wrapper (configure -> activate)
    node.trigger_configure()
    node.trigger_activate()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()


if __name__ == "__main__":
    main()