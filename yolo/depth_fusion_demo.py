"""
Depth fusion demo using Ultralytics YOLO detections plus the stereo depth map produced
by `src.hal.VISION.VISION`.

Requirements:
  - A calibrated stereo configuration stored in a dill config (see `config_setup.py`)
    with rectification maps and Q matrix populated.
  - The Ultralytics `ultralytics` package available (same as `live_demo.py`).
  - A YOLO model checkpoint (detection or segmentation). Segmentation masks are used
    when available; otherwise per-box depth statistics fall back to the bounding box.

Example usage:

    python yolo/depth_fusion_demo.py \\
        --config config.dill \\
        --weights yolo/yolo11n.pt \\
        --imgsz 640 --conf 0.35

Press `q` or `Esc` in the display window to exit.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import dill
import numpy as np
from ultralytics import YOLO

# Ensure we can import the HAL modules when this script is invoked directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.hal.VISION.VISION import VISION  # noqa: E402


@dataclass
class DetectionDepthStats:
    label: str
    confidence: float
    mean: float
    median: float
    min_depth: float
    max_depth: float
    valid_samples: int
    box: Tuple[int, int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuse YOLO detections with stereo depth from VISION."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a dill config file containing the calibrated camera parameters.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("yolo11n.pt"),
        help="YOLO weights file (.pt).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Square inference image size passed to YOLO.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Computation device (e.g., 'cpu', '0'). Defaults to auto.",
    )
    parser.add_argument(
        "--display-width",
        type=int,
        default=None,
        help="Optional display width for annotated frames (maintains aspect ratio).",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="If set, annotated frames and depth overlays are saved to this directory.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable on-screen visualization.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional limit on number of frames to process.",
    )
    parser.add_argument(
        "--stats-every",
        type=int,
        default=30,
        help="How often (in frames) to print aggregated detection depth stats. Set <=0 to disable.",
    )
    return parser.parse_args()


def load_camera_config(config_path: Path) -> Dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("rb") as f:
        config = dill.load(f)
    if "camera" not in config:
        raise KeyError("Config file missing 'camera' section required by VISION.")
    return config["camera"]


def ensure_save_dir(path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def prepare_rectified_depth(
    vision: VISION,
    left_frame: np.ndarray,
    right_frame: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Mirrors the DepthProcessor.process_frames workflow but also returns the rectified left frame.
    """
    processor = vision.depth_processor
    if processor is None:
        raise RuntimeError("Depth processor is not initialized. Did you call vision.start()?")

    # Copy frames to avoid mutating upstream sources when downsampling.
    left = left_frame.copy()
    right = right_frame.copy()

    if processor.downSample > 0:
        scale_percent = max(10, 100 - processor.downSample)
        scale_factor = scale_percent / 100.0
        new_width = int(round(left.shape[1] * scale_factor))
        new_height = int(round(left.shape[0] * scale_factor))
        if new_width < 8 or new_height < 8:
            raise ValueError("Downsample settings shrink images below workable size.")
        left = cv2.resize(left, (new_width, new_height), interpolation=cv2.INTER_AREA)
        right = cv2.resize(right, (new_width, new_height), interpolation=cv2.INTER_AREA)

    rect_left, rect_right = processor.rectify(left, right)
    disparity = processor.compute_disparity(rect_left, rect_right)
    depth_map = processor.disparity_to_depth(disparity)

    if processor._smooth_kernel >= 3:  # pylint: disable=protected-access
        depth_map = cv2.GaussianBlur(
            depth_map, (processor._smooth_kernel, processor._smooth_kernel), 0  # pylint: disable=protected-access
        )

    metadata = {
        "timestamp": time.time(),
        "num_disparities": int(
            processor.stereo_matcher.getNumDisparities() if processor.stereo_matcher else 0
        ),
    }
    return rect_left, depth_map, metadata


def compute_depth_stats(
    depth_map: np.ndarray, mask: np.ndarray
) -> Tuple[float, float, float, float, int]:
    valid_depths = depth_map[mask]
    valid_depths = valid_depths[np.isfinite(valid_depths)]
    valid_depths = valid_depths[valid_depths > 0]

    if valid_depths.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), 0

    return (
        float(np.mean(valid_depths)),
        float(np.median(valid_depths)),
        float(np.min(valid_depths)),
        float(np.max(valid_depths)),
        int(valid_depths.size),
    )


def extract_stats_for_detections(
    depth_map: np.ndarray,
    result,
    label_lookup: Dict[int, str],
) -> Tuple[list[DetectionDepthStats], list[np.ndarray]]:
    stats_list: list[DetectionDepthStats] = []
    masks: list[np.ndarray] = []

    if result.boxes is None or len(result.boxes) == 0:
        return stats_list, masks

    h, w = depth_map.shape[:2]
    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)

    mask_data = None
    if result.masks is not None and len(result.masks.data) > 0:
        mask_data = result.masks.data.cpu().numpy()

    for idx, box in enumerate(boxes_xyxy):
        x1 = max(int(np.floor(box[0])), 0)
        y1 = max(int(np.floor(box[1])), 0)
        x2 = min(int(np.ceil(box[2])), w)
        y2 = min(int(np.ceil(box[3])), h)

        if x2 <= x1 or y2 <= y1:
            continue

        if mask_data is not None and idx < mask_data.shape[0]:
            mask = mask_data[idx]
            if mask.shape != (h, w):
                mask = cv2.resize(
                    mask.astype(np.float32),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST,
                )
            mask = mask > 0.5
        else:
            mask = np.zeros((h, w), dtype=bool)
            mask[y1:y2, x1:x2] = True

        mean, median, min_depth, max_depth, count = compute_depth_stats(depth_map, mask)
        stats_list.append(
            DetectionDepthStats(
                label_lookup.get(classes[idx], str(classes[idx])),
                float(confidences[idx]),
                mean,
                median,
                min_depth,
                max_depth,
                count,
                (x1, y1, x2, y2),
            )
        )
        masks.append(mask)

    return stats_list, masks


def annotate_frame(
    frame: np.ndarray,
    stats_list: list[DetectionDepthStats],
    masks: list[np.ndarray],
) -> np.ndarray:
    annotated = frame.copy()

    colors = [
        (0, 255, 0),
        (255, 0, 0),
        (0, 140, 255),
        (128, 0, 255),
        (0, 255, 255),
    ]

    for idx, stats in enumerate(stats_list):
        color = colors[idx % len(colors)]
        mask = masks[idx] if idx < len(masks) else None

        if mask is not None:
            mask_uint8 = (mask.astype(np.uint8) * 255)
            colored_mask = cv2.applyColorMap(mask_uint8, cv2.COLORMAP_TURBO)
            overlay = cv2.addWeighted(annotated, 0.7, colored_mask, 0.3, 0)
            annotated = np.where(mask[..., None], overlay, annotated)

        x1, y1, x2, y2 = stats.box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        depth_text = (
            f"{stats.mean:.2f}m μ | {stats.median:.2f}m med"
            if np.isfinite(stats.mean)
            else "depth: N/A"
        )
        text = f"{stats.label} {stats.confidence:.2f} {depth_text}".strip()
        text_pos = (x1, max(y1 - 10, 20 + idx * 24))
        cv2.putText(
            annotated,
            text,
            text_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return annotated


def run():
    args = parse_args()
    save_dir = ensure_save_dir(args.save_dir)

    camera_cfg = load_camera_config(args.config)
    vision = VISION(name="DepthFusion", **camera_cfg)

    model = YOLO(args.weights)
    if hasattr(model.model, "names"):
        label_lookup = model.model.names
    else:
        label_lookup = model.names
    if not isinstance(label_lookup, dict):
        label_lookup = {idx: name for idx, name in enumerate(label_lookup)}

    frame_idx = 0
    window_name = "YOLO + Depth Fusion"
    depth_window = "Depth Map"

    try:
        vision.start()
        if not args.no_show:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.namedWindow(depth_window, cv2.WINDOW_NORMAL)

        while True:
            if args.max_frames is not None and frame_idx >= args.max_frames:
                break

            left_frame = vision.left_camera.read_frame() if vision.left_camera else None
            right_frame = vision.right_camera.read_frame() if vision.right_camera else None

            if left_frame is None or right_frame is None:
                print("[WARN] Failed to grab stereo frames. Retrying...")
                continue

            rect_left, depth_map, metadata = prepare_rectified_depth(vision, left_frame, right_frame)

            results = model.predict(
                source=[rect_left],
                imgsz=args.imgsz,
                conf=args.conf,
                device=args.device,
                verbose=False,
            )
            result = results[0]

            stats_list, masks = extract_stats_for_detections(depth_map, result, label_lookup)

            if stats_list and args.stats_every > 0 and frame_idx % args.stats_every == 0:
                timestamp = time.strftime("%H:%M:%S", time.localtime(metadata["timestamp"]))
                for stats in stats_list:
                    print(
                        f"[{timestamp}] {stats.label} "
                        f"conf={stats.confidence:.2f} "
                        f"mean={stats.mean:.2f}m median={stats.median:.2f}m "
                        f"samples={stats.valid_samples}"
                    )

            annotated = annotate_frame(rect_left, stats_list, masks)

            if save_dir is not None:
                frame_path = save_dir / f"fusion_frame_{frame_idx:06d}.jpg"
                depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
                depth_vis = depth_vis.astype(np.uint8)
                depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                depth_path = save_dir / f"depth_colored_{frame_idx:06d}.jpg"
                cv2.imwrite(str(frame_path), annotated)
                cv2.imwrite(str(depth_path), depth_colored)

            if not args.no_show:
                display_frame = annotated
                if args.display_width:
                    h, w = display_frame.shape[:2]
                    if w > 0 and w != args.display_width:
                        scale = args.display_width / float(w)
                        new_height = max(int(round(h * scale)), 1)
                        display_frame = cv2.resize(
                            display_frame,
                            (args.display_width, new_height),
                            interpolation=cv2.INTER_LINEAR,
                        )
                cv2.imshow(window_name, display_frame)

                depth_visual = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
                depth_visual = depth_visual.astype(np.uint8)
                depth_visual = cv2.applyColorMap(depth_visual, cv2.COLORMAP_HOT)
                cv2.imshow(depth_window, depth_visual)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            frame_idx += 1

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user; shutting down.")
    finally:
        if vision.connected:
            vision.stop()
        if not args.no_show:
            cv2.destroyWindow(window_name)
            cv2.destroyWindow(depth_window)


if __name__ == "__main__":
    run()

