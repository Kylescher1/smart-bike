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
import sys
import time
from pathlib import Path
from typing import Iterable, Union

import cv2
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

    source = _normalize_source(args.source)
    window_name = f"YOLO11n Live Demo ({source})"

    video_writer = None
    frame_idx = 0
    cap = None

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
                    cv2.imshow(window_name, display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break

                frame_idx += 1
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
                    cv2.imshow(window_name, display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break

                frame_idx += 1

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user, shutting down gracefully.")
    finally:
        if cap is not None:
            cap.release()
        if video_writer is not None:
            video_writer.release()
        if not args.no_show:
            try:
                cv2.destroyWindow(window_name)
            except cv2.error:
                pass
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

