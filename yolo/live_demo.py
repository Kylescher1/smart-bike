"""
YOLO11n live demo helper.

Launches a webcam/stream preview or processes a video file powered by the Ultralytics YOLOv11n
model that lives in this project's `yolo` directory. Example usage:

    python yolo/live_demo.py --show
    python yolo/live_demo.py --source 1 --record recordings/demo.mp4
    python yolo/live_demo.py --source "C:\\Users\\kyle1\\Downloads\\IMG_7628.MOV"

Press `q` or `Esc` in the preview window to exit the loop.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Union

import cv2
import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / "yolo11n.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a YOLO11n live detection demo.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="Path to the YOLO weights file (.pt). Defaults to yolo/yolo11n.pt.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=r"C:\Users\kyle1\Downloads\IMG_7628.MOV",
        help="Inference source: camera index, video file, image directory, RTSP/HTTP stream, etc.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Square inference image size (pixels). Ignored when --frame-size is provided.",
    )
    parser.add_argument(
        "--frame-size",
        type=str,
        default=None,
        help="Center-crop before inference to WIDTHxHEIGHT (e.g., 640x480). Overrides --imgsz.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Computation device. Examples: 'cpu', '0' (CUDA GPU 0), '0,1'. Defaults to auto.",
    )
    parser.add_argument(
        "--record",
        type=Path,
        default=None,
        help="Optional path to save an annotated video.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Optional directory to dump annotated frames as images.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable the preview window (useful for headless recording).",
    )
    parser.add_argument(
        "--display-width",
        type=int,
        default=None,
        help="Resize annotated frames for display to this width while preserving aspect ratio.",
    )
    parser.add_argument(
        "--radar-view",
        action="store_true",
        help="Show a supplementary 2D radar-style visualization of detections.",
    )
    parser.add_argument(
        "--radar-size",
        type=int,
        default=400,
        help="Image size (pixels) for the radar visualization window.",
    )
    return parser.parse_args()


def _normalize_source(source: str) -> Union[int, str]:
    if source.isdigit():
        return int(source)
    return source


def _results_stream(model: YOLO, **predict_kwargs) -> Iterable:
    """Wrapper so we can type hint the generator returned by YOLO.predict(stream=True)."""
    return model.predict(stream=True, **predict_kwargs)


def _center_crop(frame, target_width: int, target_height: int):
    """Crop the frame to the requested size, centered."""
    height, width = frame.shape[:2]
    crop_width = min(target_width, width)
    crop_height = min(target_height, height)
    x0 = max((width - crop_width) // 2, 0)
    y0 = max((height - crop_height) // 2, 0)
    return frame[y0 : y0 + crop_height, x0 : x0 + crop_width]


PERSON_CLASS_INDEX = 0  # COCO class index for "person"


@dataclass
class _RadarPing:
    """Represents an ephemeral synthetic radar contact."""

    position: tuple[float, float]
    velocity: float
    bearing: float
    size: float
    confidence: float
    ttl: float
    label: str = field(default="radar-contact")

    def step(self, delta: float) -> None:
        norm_x, norm_y = self.position
        norm_x += math.sin(self.bearing) * self.velocity * delta
        norm_y -= math.cos(self.bearing) * self.velocity * delta
        self.position = (max(0.03, min(0.97, norm_x)), max(0.05, min(0.95, norm_y)))
        self.ttl -= delta
        self.confidence = max(0.05, min(0.99, self.confidence + random.uniform(-0.05, 0.05)))

    def as_detection(self, frame_shape: tuple[int, int, int]) -> tuple[str, float, tuple[float, float, float, float]]:
        height, width = frame_shape[:2]
        norm_x, norm_y = self.position
        contact_width = self.size * width
        contact_height = self.size * height * 0.6
        center_x = norm_x * width
        bottom_y = norm_y * height
        x1 = max(0.0, center_x - contact_width / 2.0)
        x2 = min(width - 1.0, center_x + contact_width / 2.0)
        y2 = min(height - 1.0, bottom_y)
        y1 = max(0.0, y2 - contact_height)
        return (self.label, self.confidence, (x1, y1, x2, y2))


class FakeRadarModule:
    """Simulates an auxiliary radar feed that produces believable contacts."""

    def __init__(self, spawn_interval: float = 2.5, ping_ttl: float = 6.0) -> None:
        self.spawn_interval = spawn_interval
        self.ping_ttl = ping_ttl
        self._elapsed = 0.0
        self._pings: list[_RadarPing] = []
        self._rng = random.Random()
        self._rng.seed(1337)

    def update(self, frame_shape: tuple[int, int, int], delta: float) -> list[tuple[str, float, tuple[float, float, float, float]]]:
        self._elapsed += delta
        if self._elapsed >= self.spawn_interval:
            self._elapsed = 0.0
            self._spawn_ping()

        for ping in list(self._pings):
            ping.step(delta)
            if ping.ttl <= 0.0:
                self._pings.remove(ping)

        return [ping.as_detection(frame_shape) for ping in self._pings]

    def draw_overlay(self, frame: np.ndarray) -> None:
        active_contacts = len(self._pings)
        cv2.putText(
            frame,
            f"Aux Radar Tracks: {active_contacts}",
            (10, frame.shape[0] - 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (80, 220, 255),
            2,
            cv2.LINE_AA,
        )

    def _spawn_ping(self) -> None:
        base_label = self._rng.choice(["radar-hostile", "radar-ally", "radar-unknown"])
        ping = _RadarPing(
            position=(self._rng.uniform(0.1, 0.9), self._rng.uniform(0.3, 0.9)),
            velocity=self._rng.uniform(0.02, 0.06),
            bearing=self._rng.uniform(-math.pi * 0.85, math.pi * 0.85),
            size=self._rng.uniform(0.04, 0.07),
            confidence=self._rng.uniform(0.6, 0.95),
            ttl=self._rng.uniform(self.ping_ttl * 0.6, self.ping_ttl * 1.2),
            label=base_label,
        )
        self._pings.append(ping)


def _compute_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


class MissileControlUI:
    """Manages UI interactions for selecting targets and animating a missile launch."""

    def __init__(self) -> None:
        self.frame_size = (0, 0)
        self.base_frame_size = (0, 0)
        self.scale = (1.0, 1.0)
        self.scale_inv = (1.0, 1.0)
        self.detections: list[dict] = []
        self.selected_detection: dict | None = None
        self.selected_bbox: tuple[float, float, float, float] | None = None
        self.button_rect = (0, 0, 0, 0)
        self.target_point: tuple[float, float] | None = None
        self.missile_active = False
        self.missile_speed = 620.0
        self.missile_position: tuple[float, float] | None = None
        self.missile_origin: tuple[int, int] | None = None
        self.missile_target: tuple[float, float] | None = None
        self.missile_trail: list[tuple[int, int]] = []
        self.missile_progress = 0.0
        self.missile_duration = 1.4
        self.explosion_active = False
        self.explosion_timer = 0.0
        self.explosion_duration = 0.85
        self.explosion_fade_timer = 0.0
        self.explosion_fade = 0.35
        self.explosion_center_display: tuple[int, int] | None = None
        self.explosion_center_base: tuple[float, float] | None = None

    def update_metrics(
        self,
        display_shape: tuple[int, int, int],
        scale_x: float,
        scale_y: float,
        base_frame_shape: tuple[int, int, int],
    ) -> None:
        display_height, display_width = display_shape[:2]
        base_height, base_width = base_frame_shape[:2]
        self.frame_size = (display_width, display_height)
        self.base_frame_size = (base_width, base_height)
        self.scale = (scale_x, scale_y)
        inv_x = 1.0 / scale_x if abs(scale_x) > 1e-6 else 1.0
        inv_y = 1.0 / scale_y if abs(scale_y) > 1e-6 else 1.0
        self.scale_inv = (inv_x, inv_y)
        button_width = 200
        button_height = 46
        margin = 20
        x1 = self.frame_size[0] - button_width - margin
        y1 = self.frame_size[1] - button_height - margin
        x2 = self.frame_size[0] - margin
        y2 = self.frame_size[1] - margin
        self.button_rect = (x1, y1, x2, y2)

    def update_detections(self, detections: list[tuple[str, float, tuple[float, float, float, float]]]) -> None:
        sx, sy = self.scale
        scaled: list[dict] = []
        for label, conf, bbox in detections:
            x1, y1, x2, y2 = bbox
            scaled_bbox = (x1 * sx, y1 * sy, x2 * sx, y2 * sy)
            scaled.append({"label": label, "conf": conf, "bbox": scaled_bbox})
        self.detections = scaled
        if self.selected_bbox is None:
            if not self.missile_active:
                self.selected_detection = None
                self.target_point = None
            return

        best_match: dict | None = None
        best_score = 0.0
        for det in scaled:
            score = _compute_iou(self.selected_bbox, det["bbox"])
            if score > best_score:
                best_match = det
                best_score = score
        if best_match is not None and best_score > 0.15:
            self.selected_detection = best_match
            self.selected_bbox = best_match["bbox"]
            self.target_point = self._target_point_from_bbox(best_match["bbox"])
        elif not self.missile_active:
            self.selected_detection = None
            self.selected_bbox = None
            self.target_point = None

    def update(self, delta: float) -> None:
        if self.missile_active and self.missile_position is not None:
            target = self._current_target_point()
            if target is None:
                target = self.missile_target
            if target is not None:
                mx, my = self.missile_position
                tx, ty = target
                dx = tx - mx
                dy = ty - my
                distance = math.hypot(dx, dy)
                if distance < max(16.0, self.missile_speed * delta):
                    self.missile_position = (tx, ty)
                    self._append_trail_point(int(tx), int(ty))
                    self._detonate()
                else:
                    step = self.missile_speed * delta
                    if distance > 1e-6:
                        mx += (dx / distance) * step
                        my += (dy / distance) * step
                    self.missile_position = (mx, my)
                    self._append_trail_point(int(mx), int(my))
            self.missile_progress = min(1.0, self.missile_progress + delta / max(self.missile_duration, 1e-6))

        if self.explosion_active:
            self.explosion_timer += delta
            if self.explosion_timer >= self.explosion_duration:
                self.explosion_active = False
                self.explosion_timer = self.explosion_duration
                self.explosion_fade_timer = self.explosion_fade
        elif self.explosion_fade_timer > 0.0:
            self.explosion_fade_timer = max(0.0, self.explosion_fade_timer - delta)
            if self.explosion_fade_timer == 0.0:
                self.explosion_center_display = None
                self.explosion_center_base = None
                self.missile_trail.clear()

        if not self.missile_active and self.explosion_active is False and self.explosion_fade_timer == 0.0:
            if len(self.missile_trail) > 12:
                self.missile_trail = self.missile_trail[-12:]

    def draw(self, frame: np.ndarray) -> None:
        self._draw_button(frame)
        self._draw_selection(frame)
        self._draw_missile(frame)
        self._draw_explosion(frame)
        self._draw_status(frame)

    def handle_mouse(self, event: int, x: int, y: int, _flags: int, _param: object | None) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self._within_rect(x, y, self.button_rect):
            self._trigger_launch()
            return
        for det in self.detections:
            if self._within_rect(x, y, det["bbox"]):
                self._arm_target(det)
                break

    def get_special_contacts(self) -> list[tuple[str, float, tuple[float, float, float, float]]]:
        contacts: list[tuple[str, float, tuple[float, float, float, float]]] = []
        base_width, base_height = self.base_frame_size
        if base_width <= 0 or base_height <= 0:
            return contacts
        inv_x, inv_y = self.scale_inv
        if self.missile_active and self.missile_position is not None:
            cx = float(self.missile_position[0]) * inv_x
            cy = float(self.missile_position[1]) * inv_y
            size = 28.0
            x1 = max(0.0, cx - size)
            y1 = max(0.0, cy - size)
            x2 = min(base_width - 1.0, cx + size)
            y2 = min(base_height - 1.0, cy + size)
            contacts.append(("missile", 0.98, (x1, y1, x2, y2)))
        if (self.explosion_active or self.explosion_fade_timer > 0.0) and self.explosion_center_base is not None:
            ex, ey = self.explosion_center_base
            if self.explosion_active:
                progress = min(self.explosion_timer / max(self.explosion_duration, 1e-6), 1.0)
                radius = 55.0 + 120.0 * progress
            else:
                fade_ratio = 1.0 - self.explosion_fade_timer / max(self.explosion_fade, 1e-6)
                radius = 85.0 + 40.0 * fade_ratio
            x1 = max(0.0, ex - radius)
            y1 = max(0.0, ey - radius)
            x2 = min(base_width - 1.0, ex + radius)
            y2 = min(base_height - 1.0, ey + radius)
            contacts.append(("impact", 0.9, (x1, y1, x2, y2)))
        return contacts

    def _arm_target(self, det: dict) -> None:
        self.selected_detection = det
        self.selected_bbox = det["bbox"]
        self.target_point = self._target_point_from_bbox(det["bbox"])
        self.missile_active = False
        self.missile_position = None
        self.missile_origin = None
        self.missile_target = None
        self.missile_progress = 0.0
        self.missile_trail.clear()
        self.explosion_active = False
        self.explosion_timer = 0.0
        self.explosion_fade_timer = 0.0
        self.explosion_center_display = None
        self.explosion_center_base = None

    def _trigger_launch(self) -> None:
        if self.target_point is None or self.missile_active:
            return
        width, height = self.frame_size
        origin = (width // 2, height - 20)
        self.missile_origin = origin
        self.missile_position = (float(origin[0]), float(origin[1]))
        self.missile_target = self.target_point
        self.missile_trail = [origin]
        self.missile_active = True
        self.missile_progress = 0.0
        self.explosion_active = False
        self.explosion_timer = 0.0
        self.explosion_fade_timer = 0.0
        self.explosion_center_display = None
        self.explosion_center_base = None

    def _current_target_point(self) -> tuple[float, float] | None:
        if self.selected_detection is not None:
            self.missile_target = self._target_point_from_bbox(self.selected_detection["bbox"])
        return self.missile_target

    @staticmethod
    def _target_point_from_bbox(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
        x1, y1, x2, _ = bbox
        return ((x1 + x2) / 2.0, y1)

    def _append_trail_point(self, x: int, y: int) -> None:
        if not self.missile_trail or self.missile_trail[-1] != (x, y):
            self.missile_trail.append((x, y))
            if len(self.missile_trail) > 120:
                self.missile_trail.pop(0)

    def _detonate(self) -> None:
        self.missile_active = False
        self.missile_progress = 1.0
        self.explosion_active = True
        self.explosion_timer = 0.0
        self.explosion_fade_timer = 0.0
        if self.missile_position is not None:
            cx, cy = int(self.missile_position[0]), int(self.missile_position[1])
            self.explosion_center_display = (cx, cy)
            base_width, base_height = self.base_frame_size
            if base_width > 0 and base_height > 0:
                inv_x, inv_y = self.scale_inv
                base_x = max(0.0, min(base_width - 1.0, cx * inv_x))
                base_y = max(0.0, min(base_height - 1.0, cy * inv_y))
                self.explosion_center_base = (base_x, base_y)
            else:
                self.explosion_center_base = None

    def _draw_button(self, frame: np.ndarray) -> None:
        x1, y1, x2, y2 = map(int, self.button_rect)
        if self.missile_active:
            color = (60, 70, 255)
            button_text = "Tracking..."
            text_color = (255, 255, 255)
        elif self.selected_detection:
            color = (50, 150, 240)
            button_text = "Launch Missile"
            text_color = (0, 0, 0)
        else:
            color = (70, 70, 70)
            button_text = "Select a Target"
            text_color = (190, 190, 190)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1, cv2.LINE_AA)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(
            frame,
            button_text,
            (x1 + 12, y1 + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            text_color,
            2,
            cv2.LINE_AA,
        )

    def _draw_selection(self, frame: np.ndarray) -> None:
        if self.selected_detection is None:
            return
        x1, y1, x2, y2 = map(int, self.selected_detection["bbox"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2, cv2.LINE_AA)
        label = self.selected_detection["label"]
        cv2.putText(
            frame,
            f"Target: {label}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def _draw_missile(self, frame: np.ndarray) -> None:
        if len(self.missile_trail) > 1:
            total = len(self.missile_trail)
            for idx in range(1, total):
                start = self.missile_trail[idx - 1]
                end = self.missile_trail[idx]
                blend = idx / total
                color = (
                    int(40 + 140 * (1.0 - blend)),
                    int(60 + 140 * (1.0 - blend)),
                    255,
                )
                thickness = max(1, int(3 * (1.0 - blend)) + 1)
                cv2.line(frame, start, end, color, thickness, cv2.LINE_AA)
        if self.missile_active and self.missile_position is not None:
            cx, cy = int(self.missile_position[0]), int(self.missile_position[1])
            prev = self.missile_trail[-2] if len(self.missile_trail) >= 2 else (cx, cy + 1)
            dx = cx - prev[0]
            dy = cy - prev[1]
            length = math.hypot(dx, dy) or 1.0
            ux = dx / length
            uy = dy / length
            perp_x, perp_y = -uy, ux
            nose = (int(cx + ux * 18), int(cy + uy * 18))
            left = (int(cx - ux * 10 + perp_x * 8), int(cy - uy * 10 + perp_y * 8))
            right = (int(cx - ux * 10 - perp_x * 8), int(cy - uy * 10 - perp_y * 8))
            hull = np.array([nose, left, right], dtype=np.int32)
            cv2.fillConvexPoly(frame, hull, (0, 255, 0))
            cv2.polylines(frame, [hull], True, (0, 90, 0), 1, cv2.LINE_AA)

    def _draw_explosion(self, frame: np.ndarray) -> None:
        if self.explosion_center_display is None:
            return
        center = self.explosion_center_display
        if self.explosion_active:
            progress = min(self.explosion_timer / max(self.explosion_duration, 1e-6), 1.0)
            intensity = 1.0
        elif self.explosion_fade_timer > 0.0:
            progress = 1.0
            intensity = max(0.25, self.explosion_fade_timer / max(self.explosion_fade, 1e-6))
        else:
            return
        base_radius = 30 + int(150 * progress)
        outer_color = (int(60 * intensity), int(180 * intensity), 255)
        mid_color = (int(30 * intensity), int(120 * intensity), 255)
        core_color = (255, 255, 255)
        cv2.circle(frame, center, base_radius, outer_color, 4, cv2.LINE_AA)
        cv2.circle(frame, center, max(8, int(base_radius * 0.55)), mid_color, -1, cv2.LINE_AA)
        cv2.circle(frame, center, max(4, int(base_radius * 0.25)), core_color, -1, cv2.LINE_AA)
        rays = 8
        ray_length = int(base_radius * (1.15 + 0.25 * intensity))
        for idx in range(rays):
            angle = (2 * math.pi / rays) * idx
            inner = (
                int(center[0] + math.cos(angle) * base_radius * 0.35),
                int(center[1] + math.sin(angle) * base_radius * 0.35),
            )
            outer = (
                int(center[0] + math.cos(angle) * ray_length),
                int(center[1] + math.sin(angle) * ray_length),
            )
            cv2.line(frame, inner, outer, (0, int(210 * intensity), 255), 2, cv2.LINE_AA)

    def _draw_status(self, frame: np.ndarray) -> None:
        if self.frame_size[1] <= 0:
            return
        status = "Missile: IDLE"
        if self.missile_active:
            status = "Missile: TRACKING"
        elif self.explosion_active or self.explosion_fade_timer > 0.0:
            status = "Missile: IMPACT"
        elif self.selected_detection:
            status = "Missile: TARGET LOCKED"
        cv2.putText(
            frame,
            status,
            (20, self.frame_size[1] - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )

    @staticmethod
    def _within_rect(x: float, y: float, rect: tuple[float, float, float, float]) -> bool:
        x1, y1, x2, y2 = rect
        return x1 <= x <= x2 and y1 <= y <= y2


def _extract_detections(
    result,
    allowed_classes: tuple[int, ...] | None = None,
) -> list[tuple[str, float, tuple[float, float, float, float]]]:
    detections: list[tuple[str, float, tuple[float, float, float, float]]] = []
    if result.boxes is None or len(result.boxes) == 0:
        return detections

    names = None
    if hasattr(result, "names") and result.names is not None:
        names = result.names
    elif hasattr(result, "model") and hasattr(result.model, "names"):
        names = result.model.names

    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)

    for idx, bbox in enumerate(boxes_xyxy):
        class_id = classes[idx]
        if allowed_classes is not None and class_id not in allowed_classes:
            continue
        label = str(classes[idx])
        if isinstance(names, dict):
            label = names.get(classes[idx], label)
        elif isinstance(names, (list, tuple)) and classes[idx] < len(names):
            label = names[classes[idx]]
        detections.append((label, float(confs[idx]), tuple(map(float, bbox))))
    return detections


def _draw_radar_view(
    frame_shape: tuple[int, int, int],
    detections: list[tuple[str, float, tuple[float, float, float, float]]],
    radar_size: int,
) -> "np.ndarray":
    radar_size = max(200, radar_size)
    radar = np.zeros((radar_size, radar_size, 3), dtype=np.uint8)
    radar[:] = (15, 15, 15)

    width = frame_shape[1]
    height = frame_shape[0]

    cx = radar_size // 2
    cy = radar_size - 40

    cv2.circle(radar, (cx, cy), radar_size // 2 - 20, (40, 40, 40), 2, cv2.LINE_AA)
    cv2.line(radar, (cx, cy), (cx, 20), (50, 50, 50), 1, cv2.LINE_AA)
    cv2.line(radar, (20, cy), (radar_size - 20, cy), (50, 50, 50), 1, cv2.LINE_AA)

    for label, conf, bbox in detections:
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2.0
        bottom_y = y2
        norm_x = (center_x / max(width, 1)) - 0.5  # -0.5 (left) to 0.5 (right)
        norm_y = 1.0 - (bottom_y / max(height, 1))  # 0 near bottom (close), 1 near top (far)

        radius = (radar_size // 2 - 30) * norm_y + 10
        angle = norm_x * np.pi
        point_x = int(cx + radius * np.sin(angle))
        point_y = int(cy - radius * np.cos(angle))

        if label.startswith("radar"):
            color = (0, 140, 255)
        elif label.startswith("missile"):
            color = (40, 40, 255)
        elif label.startswith("impact"):
            color = (0, 80, 255)
        else:
            color = (0, 255, 120)
        cv2.circle(radar, (point_x, point_y), 6, color, -1, cv2.LINE_AA)
        cv2.putText(
            radar,
            f"{label}:{conf:.2f}",
            (point_x + 8, point_y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        radar,
        "Radar View (screen relative)",
        (20, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (180, 180, 180),
        2,
        cv2.LINE_AA,
    )
    return radar


def main() -> int:
    args = parse_args()

    weights_path = args.weights.expanduser().resolve()
    if not weights_path.exists():
        print(f"[ERROR] Weights file not found: {weights_path}", file=sys.stderr)
        return 1

    save_dir: Path | None = None
    if args.save_dir:
        save_dir = args.save_dir.expanduser().resolve()
        save_dir.mkdir(parents=True, exist_ok=True)

    record_path: Path | None = None
    if args.record:
        record_path = args.record.expanduser().resolve()
        record_path.parent.mkdir(parents=True, exist_ok=True)

    frame_width = frame_height = None
    imgsz_override: int | tuple[int, int]
    if args.frame_size:
        try:
            width_str, height_str = args.frame_size.lower().split("x")
            frame_width, frame_height = int(width_str), int(height_str)
            if frame_width <= 0 or frame_height <= 0:
                raise ValueError
        except ValueError:
            print(
                f"[ERROR] Invalid --frame-size value: {args.frame_size}. Expected positive WIDTHxHEIGHT, e.g., 640x480.",
                file=sys.stderr,
            )
            return 1
        imgsz_override = (frame_height, frame_width)  # YOLO expects (height, width)
    else:
        imgsz_override = args.imgsz

    model = YOLO(weights_path)
    fake_radar = FakeRadarModule()
    ui_manager = MissileControlUI()

    source = _normalize_source(args.source)
    window_name = f"YOLO11n Live Demo ({source})"

    video_writer = None
    frame_idx = 0
    cap = None
    results_iter = None
    should_exit = False

    radar_window = f"{window_name} - Radar"

    if not args.no_show:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(window_name, ui_manager.handle_mouse)
            if args.radar_view:
                cv2.namedWindow(radar_window, cv2.WINDOW_NORMAL)
        except cv2.error:
            pass

    try:
        if frame_width is not None and frame_height is not None:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                print(f"[ERROR] Unable to open source {source} for cropping.", file=sys.stderr)
                return 1

            prev_time = time.time()
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                cropped_frame = _center_crop(frame, frame_width, frame_height)
                inference_results = model.predict(
                    source=[cropped_frame],
                    imgsz=imgsz_override,
                    conf=args.conf,
                    device=args.device,
                    verbose=False,
                )
                result = inference_results[0]

                now = time.time()
                delta = max(now - prev_time, 1e-6)
                prev_time = now
                annotated_frame = result.plot()  # Ultralytics returns a copy with annotations
                inference_time_ms = (
                    result.speed.get("inference", 0.0) if hasattr(result, "speed") else 0.0
                )
                detections = _extract_detections(result, allowed_classes=(PERSON_CLASS_INDEX,))
                fake_contacts = fake_radar.update(cropped_frame.shape, delta)
                radar_detections = detections + fake_contacts

                # Overlay FPS and model info
                fps = 1.0 / delta
                cv2.putText(
                    annotated_frame,
                    f"FPS: {fps:.1f} | Inference: {inference_time_ms:.1f} ms",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                if record_path and video_writer is None:
                    height, width = annotated_frame.shape[:2]
                    fourcc = (
                        cv2.VideoWriter_fourcc(*"mp4v")
                        if record_path.suffix.lower() == ".mp4"
                        else cv2.VideoWriter_fourcc(*"XVID")
                    )
                    video_writer = cv2.VideoWriter(str(record_path), fourcc, 30.0, (width, height))
                    if not video_writer.isOpened():
                        print(f"[ERROR] Failed to open video writer for {record_path}", file=sys.stderr)
                        video_writer = None

                if video_writer is not None:
                    video_writer.write(annotated_frame)

                if save_dir is not None:
                    frame_file = save_dir / f"frame_{frame_idx:06d}.jpg"
                    cv2.imwrite(str(frame_file), annotated_frame)

                display_frame = annotated_frame
                if args.display_width:
                    height, width = display_frame.shape[:2]
                    if width != args.display_width and width > 0:
                        scale = args.display_width / width
                        new_height = max(int(round(height * scale)), 1)
                        display_frame = cv2.resize(
                            display_frame, (args.display_width, new_height), interpolation=cv2.INTER_AREA
                        )

                if not args.no_show:
                    scale_x = display_frame.shape[1] / max(annotated_frame.shape[1], 1)
                    scale_y = display_frame.shape[0] / max(annotated_frame.shape[0], 1)
                    ui_manager.update_metrics(display_frame.shape, scale_x, scale_y, annotated_frame.shape)
                    ui_manager.update_detections(detections)
                    ui_manager.update(delta)
                    ui_manager.draw(display_frame)
                    special_contacts = ui_manager.get_special_contacts()
                    fake_radar.draw_overlay(display_frame)
                    cv2.imshow(window_name, display_frame)
                    if args.radar_view:
                        combined_radar = radar_detections + special_contacts
                        radar_image = _draw_radar_view(cropped_frame.shape, combined_radar, args.radar_size)
                        cv2.imshow(radar_window, radar_image)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        should_exit = True
                        break

                frame_idx += 1
                if should_exit:
                    break

            if should_exit:
                pass
        else:
            results_iter = _results_stream(
                model,
                source=source,
                imgsz=imgsz_override,
                conf=args.conf,
                device=args.device,
                show=False,  # we take over rendering so we can add FPS text
            )

            prev_time = time.time()
            for result in results_iter:
                now = time.time()
                delta = max(now - prev_time, 1e-6)
                prev_time = now
                annotated_frame = result.plot()  # Ultralytics returns a copy with annotations
                inference_time_ms = (
                    result.speed.get("inference", 0.0) if hasattr(result, "speed") else 0.0
                )
                detections = _extract_detections(result, allowed_classes=(PERSON_CLASS_INDEX,))
                orig_img = getattr(result, "orig_img", None)
                frame_shape = orig_img.shape if isinstance(orig_img, np.ndarray) else annotated_frame.shape
                fake_contacts = fake_radar.update(frame_shape, delta)
                radar_detections = detections + fake_contacts

                # Overlay FPS and model info
                fps = 1.0 / delta
                cv2.putText(
                    annotated_frame,
                    f"FPS: {fps:.1f} | Inference: {inference_time_ms:.1f} ms",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                # Lazily initialize video writer once we know the frame size
                if record_path and video_writer is None:
                    height, width = annotated_frame.shape[:2]
                    fourcc = (
                        cv2.VideoWriter_fourcc(*"mp4v")
                        if record_path.suffix.lower() == ".mp4"
                        else cv2.VideoWriter_fourcc(*"XVID")
                    )
                    video_writer = cv2.VideoWriter(str(record_path), fourcc, 30.0, (width, height))
                    if not video_writer.isOpened():
                        print(f"[ERROR] Failed to open video writer for {record_path}", file=sys.stderr)
                        video_writer = None

                if video_writer is not None:
                    video_writer.write(annotated_frame)

                if save_dir is not None:
                    frame_file = save_dir / f"frame_{frame_idx:06d}.jpg"
                    cv2.imwrite(str(frame_file), annotated_frame)

                display_frame = annotated_frame
                if args.display_width:
                    height, width = display_frame.shape[:2]
                    if width != args.display_width and width > 0:
                        scale = args.display_width / width
                        new_height = max(int(round(height * scale)), 1)
                        display_frame = cv2.resize(
                            display_frame, (args.display_width, new_height), interpolation=cv2.INTER_AREA
                        )

                if not args.no_show:
                    scale_x = display_frame.shape[1] / max(annotated_frame.shape[1], 1)
                    scale_y = display_frame.shape[0] / max(annotated_frame.shape[0], 1)
                    ui_manager.update_metrics(display_frame.shape, scale_x, scale_y, annotated_frame.shape)
                    ui_manager.update_detections(detections)
                    ui_manager.update(delta)
                    ui_manager.draw(display_frame)
                    special_contacts = ui_manager.get_special_contacts()
                    fake_radar.draw_overlay(display_frame)
                    cv2.imshow(window_name, display_frame)
                    if args.radar_view:
                        combined_radar = radar_detections + special_contacts
                        radar_image = _draw_radar_view(frame_shape, combined_radar, args.radar_size)
                        cv2.imshow(radar_window, radar_image)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        should_exit = True
                        break

                frame_idx += 1
                if should_exit:
                    break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user, shutting down gracefully.")
    finally:
        if hasattr(results_iter, "close"):
            try:
                results_iter.close()
            except Exception:
                pass
        if cap is not None:
            cap.release()
        if video_writer is not None:
            video_writer.release()
        if not args.no_show:
            try:
                cv2.destroyWindow(window_name)
                if args.radar_view:
                    cv2.destroyWindow(radar_window)
            except cv2.error:
                pass
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

