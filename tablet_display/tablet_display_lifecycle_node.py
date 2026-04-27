#!/usr/bin/env python3
import json
from typing import Optional

import cv2
import numpy as np
import rclpy
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.lifecycle import LifecycleState
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
)

from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


# --------------------------- helpers ---------------------------

def _fit_to_window(img_bgr: np.ndarray, win_w: int, win_h: int) -> np.ndarray:
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


def _wrap_text_by_pixels(text: str, win_w: int, font, font_scale: float, thickness: int, margin: int):
    max_px = max(10, win_w - 2 * margin)
    lines_out = []

    for para in (text or "").split("\n"):
        words = para.split(" ")
        if not words:
            lines_out.append("")
            continue

        cur = ""
        for w in words:
            candidate = w if cur == "" else (cur + " " + w)
            (tw, _), _ = cv2.getTextSize(candidate, font, font_scale, thickness)

            if tw <= max_px:
                cur = candidate
            else:
                if cur != "":
                    lines_out.append(cur)
                    cur = w
                else:
                    chunk = ""
                    for ch in w:
                        cand2 = chunk + ch
                        (t2, _), _ = cv2.getTextSize(cand2, font, font_scale, thickness)
                        if t2 <= max_px:
                            chunk = cand2
                        else:
                            if chunk:
                                lines_out.append(chunk)
                            chunk = ch
                    cur = chunk

        lines_out.append(cur)

    return [ln for ln in lines_out if ln is not None]


def _render_text_canvas(text: str, win_w: int, win_h: int, font_scale: float, thickness: int, margin: int) -> np.ndarray:
    canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
    text = (text or "").strip()
    if not text:
        return canvas

    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = _wrap_text_by_pixels(text, win_w, font, font_scale, thickness, margin)

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


def _draw_round_rect(img, x1, y1, x2, y2, color, r=14, thickness=-1):
    r = int(max(0, min(r, (x2 - x1) // 2, (y2 - y1) // 2)))

    if thickness < 0:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        cv2.circle(img, (x1 + r, y1 + r), r, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (x2 - r, y1 + r), r, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (x1 + r, y2 - r), r, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (x2 - r, y2 - r), r, color, -1, lineType=cv2.LINE_AA)
        return

    overlay = img.copy()
    cv2.line(overlay, (x1 + r, y1), (x2 - r, y1), color, thickness, lineType=cv2.LINE_AA)
    cv2.line(overlay, (x1 + r, y2), (x2 - r, y2), color, thickness, lineType=cv2.LINE_AA)
    cv2.line(overlay, (x1, y1 + r), (x1, y2 - r), color, thickness, lineType=cv2.LINE_AA)
    cv2.line(overlay, (x2, y1 + r), (x2, y2 - r), color, thickness, lineType=cv2.LINE_AA)

    cv2.ellipse(overlay, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, lineType=cv2.LINE_AA)
    cv2.ellipse(overlay, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, lineType=cv2.LINE_AA)
    cv2.ellipse(overlay, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness, lineType=cv2.LINE_AA)
    cv2.ellipse(overlay, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness, lineType=cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.95, img, 0.05, 0.0, dst=img)


def _draw_shadow(img, x1, y1, x2, y2, r=14, dx=0, dy=4, alpha=0.18):
    overlay = img.copy()
    _draw_round_rect(overlay, x1 + dx, y1 + dy, x2 + dx, y2 + dy, (0, 0, 0), r=r, thickness=-1)
    cv2.addWeighted(overlay, float(alpha), img, 1.0 - float(alpha), 0.0, dst=img)


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


# --------------------------- node ---------------------------

class TabletDisplayLifecycleNode(LifecycleNode):
    def __init__(self):
        super().__init__("tablet_display")

        self.declare_parameter("text_topic", "/tablet/text")
        self.declare_parameter("image_topic", "/tablet/image")
        self.declare_parameter("question_topic", "/tablet/question")
        self.declare_parameter("answer_topic", "/tablet/answer")

        self.declare_parameter("window_name", "Robot Tablet")
        self.declare_parameter("fullscreen", True)
        self.declare_parameter("window_width", 800)
        self.declare_parameter("window_height", 480)
        self.declare_parameter("window_x", 0)
        self.declare_parameter("window_y", 0)

        self.declare_parameter("font_scale", 1.2)
        self.declare_parameter("font_thickness", 2)
        self.declare_parameter("text_margin", 30)

        self.declare_parameter("idle_text", "")
        self.declare_parameter("ui_rate_hz", 30.0)
        self.declare_parameter("reliability", 1)

        self.bridge = CvBridge()

        self.text_sub = None
        self.image_sub = None
        self.question_sub = None
        self.answer_pub = None
        self.ui_timer = None

        self.mode: str = "idle"
        self.idle_text: str = ""
        self.last_text: str = ""
        self.last_img_bgr: Optional[np.ndarray] = None

        self.current_question_id = ""
        self.current_question = ""
        self.current_options = []
        self.current_multi = True
        self.selected = set()

        self._buttons = []

        # scrolling state
        self._scroll_px = 0
        self._dragging = False
        self._drag_start_y = 0
        self._scroll_start_px = 0
        self._options_view_rect = None  # (x1,y1,x2,y2)
        self._content_h = 0

        self._window_created = False

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        try:
            self.text_topic = self.get_parameter("text_topic").value
            self.image_topic = self.get_parameter("image_topic").value
            self.question_topic = self.get_parameter("question_topic").value
            self.answer_topic = self.get_parameter("answer_topic").value

            self.window_name = self.get_parameter("window_name").value
            self.fullscreen = bool(self.get_parameter("fullscreen").value)
            self.win_w = int(self.get_parameter("window_width").value)
            self.win_h = int(self.get_parameter("window_height").value)
            self.win_x = int(self.get_parameter("window_x").value)
            self.win_y = int(self.get_parameter("window_y").value)

            self.font_scale = float(self.get_parameter("font_scale").value)
            self.font_thickness = int(self.get_parameter("font_thickness").value)
            self.text_margin = int(self.get_parameter("text_margin").value)

            self.idle_text = str(self.get_parameter("idle_text").value)
            self.ui_rate_hz = float(self.get_parameter("ui_rate_hz").value)

            reliability = int(self.get_parameter("reliability").value)
            rel = QoSReliabilityPolicy.BEST_EFFORT if reliability == 2 else QoSReliabilityPolicy.RELIABLE

            self.qos = QoSProfile(
                reliability=rel,
                history=QoSHistoryPolicy.KEEP_LAST,
                durability=QoSDurabilityPolicy.VOLATILE,
                depth=5,
            )

            self._set_idle()
            return TransitionCallbackReturn.SUCCESS
        except Exception as e:
            self.get_logger().error(f"[tablet_display] configure failed: {repr(e)}")
            return TransitionCallbackReturn.FAILURE

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        try:
            self._create_window()
            cv2.setMouseCallback(self.window_name, self._on_mouse)

            self.text_sub = self.create_subscription(String, self.text_topic, self._on_text, self.qos)
            self.image_sub = self.create_subscription(Image, self.image_topic, self._on_image, self.qos)
            self.question_sub = self.create_subscription(String, self.question_topic, self._on_question, self.qos)
            self.answer_pub = self.create_publisher(String, self.answer_topic, self.qos)

            period = 1.0 / max(1e-3, self.ui_rate_hz)
            self.ui_timer = self.create_timer(period, self._ui_tick)
            return TransitionCallbackReturn.SUCCESS
        except Exception as e:
            self.get_logger().error(f"[tablet_display] activate failed: {repr(e)}")
            return TransitionCallbackReturn.FAILURE

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        try:
            if self.ui_timer is not None:
                self.ui_timer.cancel()
                self.ui_timer = None

            if self.text_sub is not None:
                self.destroy_subscription(self.text_sub)
                self.text_sub = None

            if self.image_sub is not None:
                self.destroy_subscription(self.image_sub)
                self.image_sub = None

            if self.question_sub is not None:
                self.destroy_subscription(self.question_sub)
                self.question_sub = None

            if self.answer_pub is not None:
                self.destroy_publisher(self.answer_pub)
                self.answer_pub = None

            self._set_idle()
            self._destroy_window()
            return TransitionCallbackReturn.SUCCESS
        except Exception as e:
            self.get_logger().error(f"[tablet_display] deactivate failed: {repr(e)}")
            return TransitionCallbackReturn.FAILURE

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self._set_idle()
        self._destroy_window()
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        self._set_idle()
        self._destroy_window()
        return TransitionCallbackReturn.SUCCESS

    # ---------------- state ----------------

    def _set_idle(self):
        self.mode = "idle"
        self.last_text = self.idle_text
        self.last_img_bgr = None

        self.current_question_id = ""
        self.current_question = ""
        self.current_options = []
        self.current_multi = True
        self.selected = set()
        self._buttons = []

        self._scroll_px = 0
        self._dragging = False
        self._options_view_rect = None
        self._content_h = 0

    def _max_scroll(self) -> int:
        if not self._options_view_rect:
            return 0
        _, y1, _, y2 = self._options_view_rect
        view_h = max(1, y2 - y1)
        return max(0, int(self._content_h - view_h))

    def _set_scroll(self, px: int) -> None:
        self._scroll_px = _clamp(int(px), 0, self._max_scroll())

    def _in_rect(self, x: int, y: int, rect) -> bool:
        x1, y1, x2, y2 = rect
        return (x1 <= x <= x2) and (y1 <= y <= y2)

    # ---------------- callbacks ----------------

    def _on_text(self, msg: String) -> None:
        self.last_text = msg.data
        self.last_img_bgr = None
        self.mode = "text"

    def _on_image(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.last_img_bgr = frame
            self.mode = "image"
        except Exception as e:
            self.get_logger().warn(f"[tablet_display] image convert failed: {repr(e)}")

    def _on_question(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
            qid = str(data.get("id", ""))
            q = str(data.get("question", "")).strip()
            opts = data.get("options", [])
            multi = bool(data.get("multi", True))

            if not q or (not isinstance(opts, list)) or len(opts) == 0:
                self._set_idle()
                return

            self.current_question_id = qid
            self.current_question = q
            self.current_options = [str(x) for x in opts]
            self.current_multi = multi
            self.selected = set()
            self.mode = "question"

            self._scroll_px = 0
            self._dragging = False
        except Exception:
            self._set_idle()

    def _on_mouse(self, event, x, y, flags, param):
        if self.mode != "question":
            return

        # wheel for desktop
        if event == cv2.EVENT_MOUSEWHEEL:
            step = 70
            if flags > 0:
                self._set_scroll(self._scroll_px - step)
            else:
                self._set_scroll(self._scroll_px + step)
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            # start drag only inside viewport
            if self._options_view_rect and self._in_rect(x, y, self._options_view_rect):
                self._dragging = True
                self._drag_start_y = y
                self._scroll_start_px = self._scroll_px

            # click handlers
            for b in self._buttons:
                if self._in_rect(x, y, b["rect"]):
                    if b["kind"] == "opt":
                        idx = b["idx"]
                        if not self.current_multi:
                            self.selected = {idx}
                        else:
                            if idx in self.selected:
                                self.selected.remove(idx)
                            else:
                                self.selected.add(idx)
                    elif b["kind"] == "done":
                        self._publish_answer_and_idle()
                    return

        elif event == cv2.EVENT_MOUSEMOVE:
            if self._dragging:
                dy = y - self._drag_start_y
                self._set_scroll(self._scroll_start_px - dy)

        elif event == cv2.EVENT_LBUTTONUP:
            self._dragging = False

    def _publish_answer_and_idle(self):
        if self.answer_pub is None:
            return

        selected_sorted = sorted(list(self.selected))
        payload = {
            "id": self.current_question_id,
            "selected_indices": selected_sorted,
            "selected_options": [
                self.current_options[i]
                for i in selected_sorted
                if 0 <= i < len(self.current_options)
            ],
        }
        out = String()
        out.data = json.dumps(payload, ensure_ascii=False)
        self.answer_pub.publish(out)
        self._set_idle()

    # ---------------- render ----------------

    def _render_question_canvas(self, win_w: int, win_h: int) -> np.ndarray:
        canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)

        # palette (BGR)
        bg = (20, 20, 20)
        card_bg = (28, 28, 28)
        option_bg = (52, 52, 52)
        option_selected_bg = (125, 68, 92)
        done_bg = (30, 125, 30)

        text_color = (245, 245, 245)
        subtext_color = (175, 175, 175)
        border_soft = (90, 90, 90)
        border_selected = (220, 150, 180)
        border_done = (150, 220, 150)

        canvas[:] = bg
        self._buttons = []

        margin = 24
        font = cv2.FONT_HERSHEY_SIMPLEX

        q_scale = 0.95
        o_scale = 0.90
        small_scale = 0.60
        thick = 2

        # --- header card ---
        qx1 = margin
        qx2 = win_w - margin
        qy1 = margin
        qy2 = 118

        _draw_shadow(canvas, qx1, qy1, qx2, qy2, r=16)
        _draw_round_rect(canvas, qx1, qy1, qx2, qy2, card_bg, r=16, thickness=-1)
        _draw_round_rect(canvas, qx1, qy1, qx2, qy2, border_soft, r=16, thickness=1)

        q_pad = 18
        q_lines = _wrap_text_by_pixels(self.current_question, qx2 - qx1, font, q_scale, thick, q_pad)[:2]
        ytxt = qy1 + 36
        for line in q_lines:
            cv2.putText(canvas, line, (qx1 + q_pad, ytxt), font, q_scale, text_color, thick, cv2.LINE_AA)
            ytxt += 34

        helper = "Select one option" if not self.current_multi else "Select one or more options, then press Done"
        cv2.putText(canvas, helper, (qx1 + q_pad, qy2 - 12), font, small_scale, subtext_color, 1, cv2.LINE_AA)

        # --- geometry ---
        start_y = qy2 + 18
        done_h = 62
        gap = 12
        opt_h = 58

        ox1 = margin
        ox2 = win_w - margin

        done_y2 = win_h - margin
        done_y1 = done_y2 - done_h

        view_x1 = ox1
        view_x2 = ox2
        view_y1 = start_y
        view_y2 = done_y1 - 16

        self._options_view_rect = (view_x1, view_y1, view_x2, view_y2)
        view_h = max(1, view_y2 - view_y1)

        n_opts = len(self.current_options)
        self._content_h = n_opts * opt_h + max(0, n_opts - 1) * gap
        self._set_scroll(self._scroll_px)

        # --- CLIPPED options rendering ---
        # draw options on overlay, then copy only viewport slice back
        opt_layer = canvas.copy()

        for i, opt in enumerate(self.current_options):
            cy1 = i * (opt_h + gap)
            cy2 = cy1 + opt_h

            y1 = view_y1 + (cy1 - self._scroll_px)
            y2 = view_y1 + (cy2 - self._scroll_px)

            if y2 < view_y1 - 2 or y1 > view_y2 + 2:
                continue

            selected = i in self.selected
            fill = option_selected_bg if selected else option_bg
            border = border_selected if selected else border_soft

            _draw_shadow(opt_layer, ox1, y1, ox2, y2, r=14)
            _draw_round_rect(opt_layer, ox1, y1, ox2, y2, fill, r=14, thickness=-1)
            _draw_round_rect(opt_layer, ox1, y1, ox2, y2, border, r=14, thickness=1)

            # checkbox
            box_size = 24
            bx1 = ox1 + 16
            by1 = y1 + (opt_h - box_size) // 2
            bx2 = bx1 + box_size
            by2 = by1 + box_size

            _draw_round_rect(opt_layer, bx1, by1, bx2, by2, (35, 35, 35), r=6, thickness=-1)
            _draw_round_rect(opt_layer, bx1, by1, bx2, by2, (210, 210, 210), r=6, thickness=1)

            if selected:
                cv2.line(opt_layer, (bx1 + 5, by1 + 13), (bx1 + 10, by2 - 6), (255, 255, 255), 2, cv2.LINE_AA)
                cv2.line(opt_layer, (bx1 + 10, by2 - 6), (bx2 - 5, by1 + 6), (255, 255, 255), 2, cv2.LINE_AA)

            label_x = bx2 + 14
            cv2.putText(opt_layer, opt, (label_x, y1 + 38), font, o_scale, text_color, thick, cv2.LINE_AA)

            self._buttons.append({"kind": "opt", "idx": i, "rect": (ox1, y1, ox2, y2)})

        # copy only the viewport slice (hard clip!)
        canvas[view_y1:view_y2, view_x1:view_x2] = opt_layer[view_y1:view_y2, view_x1:view_x2]

        # viewport subtle mask line (optional)
        cv2.rectangle(canvas, (view_x1, view_y1), (view_x2, view_y2), (0, 0, 0), 1)

        # scrollbar (only if needed)
        max_scroll = self._max_scroll()
        if max_scroll > 0:
            rail_w = 6
            rail_x2 = view_x2 - 6
            rail_x1 = rail_x2 - rail_w
            rail_y1 = view_y1 + 6
            rail_y2 = view_y2 - 6
            cv2.rectangle(canvas, (rail_x1, rail_y1), (rail_x2, rail_y2), (45, 45, 45), -1)

            rail_h = max(1, rail_y2 - rail_y1)
            thumb_h = max(24, int(rail_h * (view_h / max(1, self._content_h))))
            t = float(self._scroll_px) / float(max_scroll)
            thumb_y1 = int(rail_y1 + t * (rail_h - thumb_h))
            thumb_y2 = thumb_y1 + thumb_h
            cv2.rectangle(canvas, (rail_x1, thumb_y1), (rail_x2, thumb_y2), (120, 120, 120), -1)

        # --- Done button ---
        _draw_shadow(canvas, ox1, done_y1, ox2, done_y2, r=16)
        _draw_round_rect(canvas, ox1, done_y1, ox2, done_y2, done_bg, r=16, thickness=-1)
        _draw_round_rect(canvas, ox1, done_y1, ox2, done_y2, border_done, r=16, thickness=1)

        done_label = "Done"
        (tw, th), _ = cv2.getTextSize(done_label, font, o_scale, thick)
        tx = ox1 + (ox2 - ox1 - tw) // 2
        ty = done_y1 + (done_h + th) // 2 - 4
        cv2.putText(canvas, done_label, (tx, ty), font, o_scale, text_color, thick, cv2.LINE_AA)

        if self.current_multi:
            info = f"{len(self.selected)} selected"
            (tw2, _), _ = cv2.getTextSize(info, font, 0.55, 1)
            cv2.putText(canvas, info, (ox2 - tw2 - 16, done_y2 - 14), font, 0.55, (235, 235, 235), 1, cv2.LINE_AA)

        self._buttons.append({"kind": "done", "idx": -1, "rect": (ox1, done_y1, ox2, done_y2)})

        return canvas

    # ---------------- window ----------------

    def _create_window(self) -> None:
        if self._window_created:
            return

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        if self.fullscreen:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.resizeWindow(self.window_name, self.win_w, self.win_h)
        cv2.moveWindow(self.window_name, self.win_x, self.win_y)
        self._window_created = True

    def _destroy_window(self) -> None:
        if not self._window_created:
            return
        try:
            cv2.destroyWindow(self.window_name)
            for _ in range(10):
                cv2.waitKey(1)
        except Exception:
            pass
        self._window_created = False

    def _ui_tick(self) -> None:
        if not self._window_created:
            return

        if self.mode == "question":
            canvas = self._render_question_canvas(self.win_w, self.win_h)
        elif self.mode == "image" and self.last_img_bgr is not None:
            canvas = _fit_to_window(self.last_img_bgr, self.win_w, self.win_h)
        else:
            canvas = _render_text_canvas(
                self.last_text,
                self.win_w, self.win_h,
                font_scale=self.font_scale,
                thickness=self.font_thickness,
                margin=self.text_margin,
            )

        cv2.imshow(self.window_name, canvas)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = TabletDisplayLifecycleNode()

    node.trigger_configure()
    node.trigger_activate()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()