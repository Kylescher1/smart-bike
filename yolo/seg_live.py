"""
RKNN YOLO detection with depth overlay.

Runs RKNN YOLO models on NPU with stereo depth overlay.
Uses VISION class to get depth maps and overlays depth on detected objects.
Press 'q' or Esc to exit.

Usage:
    python yolo/seg_live.py --model yolo/models/yolo11n.rknn
    python yolo/seg_live.py --model yolo/models/yolo11n.rknn --config custom.dill
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Tuple, List, Dict

# Add system dist-packages to path for rknnlite (system package)
import site
system_packages = '/usr/lib/python3/dist-packages'
if system_packages not in sys.path:
    sys.path.insert(0, system_packages)

import cv2
import dill
import numpy as np
from rknnlite.api import RKNNLite

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.hal.VISION.VISION import VISION  # noqa: E402

# Import RKNN processing functions
from yolo.rknn_inference import (  # noqa: E402
    letterbox,
    nms,
    dfl,
    box_process_yolov8,
    post_process_yolov8,
    process_output,
    COCO_CLASSES,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "yolo11n.rknn"
DEFAULT_CONFIG = PROJECT_ROOT / "config.dill"


def parse_args():
    parser = argparse.ArgumentParser(description="RKNN YOLO detection with depth overlay")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"Path to RKNN model file. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Path to config.dill file. Default: {DEFAULT_CONFIG}",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (0.0-1.0). Default: 0.25",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (pixels). Smaller = faster (e.g., 320, 416, 640). Default: 640",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Target platform (RK3562/RK3566/RK3568/RK3588). Use 'None' for on-device NPU. Default: None",
    )
    parser.add_argument(
        "--core",
        type=int,
        default=0,
        help="NPU core mask (0=auto, 1=core0, 2=core1, 4=core2, 3=core0+1, 7=all). Default: 0 (auto)",
    )
    parser.add_argument(
        "--skip-depth",
        type=int,
        default=0,
        help="Skip depth processing every N frames (0=process all, 1=skip every other, etc.). Higher = faster.",
    )
    parser.add_argument(
        "--fast-depth",
        action="store_true",
        help="Enable fast depth mode: disable WLS, morphological filters, and smoothing",
    )
    parser.add_argument(
        "--depth-downsample",
        type=int,
        default=None,
        help="Override depth downsampling percentage (0-90). Higher = faster but lower quality. Default uses config.",
    )
    parser.add_argument(
        "--fast-stereo",
        action="store_true",
        help="Use optimized stereo matcher mode (SGBM_3WAY) - faster than regular SGBM",
    )
    parser.add_argument(
        "--use-bm",
        action="store_true",
        help="Use BM (Block Matching) instead of SGBM - much faster but less accurate",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=0,
        help="Skip N frames between RKNN inferences (0=process all frames). Default: 0",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    """Load camera configuration from dill file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("rb") as f:
        config = dill.load(f)
    if "camera" not in config:
        raise KeyError("Config file missing 'camera' section")
    return config["camera"]


def compute_depth_stats(depth_map: np.ndarray, mask: np.ndarray) -> Tuple[float, int]:
    """Compute mean depth and valid sample count from a mask region."""
    valid_depths = depth_map[mask]
    valid_depths = valid_depths[np.isfinite(valid_depths)]
    valid_depths = valid_depths[valid_depths > 0]
    
    if valid_depths.size == 0:
        return float("nan"), 0
    
    return float(np.mean(valid_depths)), int(valid_depths.size)


def overlay_depth_on_detections(
    frame: np.ndarray,
    depth_map: np.ndarray,
    detections: List[Dict],
    mask_opacity: float = 0.3,
) -> np.ndarray:
    """Overlay depth information on RKNN detections."""
    annotated = frame.copy()
    frame_h, frame_w = frame.shape[:2]
    depth_h, depth_w = depth_map.shape[:2]
    
    # Resize depth map to match frame if needed
    if (depth_h, depth_w) != (frame_h, frame_w):
        depth_map = cv2.resize(depth_map, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)
    
    if not detections:
        return annotated
    
    h, w = frame.shape[:2]
    
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 140, 255),  # Orange
        (128, 0, 255),  # Purple
        (0, 255, 255),  # Yellow
    ]
    
    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = det['bbox']
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w, int(x2))
        y2 = min(h, int(y2))
        
        if x2 <= x1 or y2 <= y1:
            continue
        
        # Create mask from bounding box (RKNN detection doesn't have segmentation masks)
        mask = np.zeros((h, w), dtype=bool)
        mask[y1:y2, x1:x2] = True
        
        # Compute depth statistics
        mean_depth, valid_count = compute_depth_stats(depth_map, mask)
        
        # Draw mask overlay (semi-transparent)
        color = colors[idx % len(colors)]
        colored_mask = np.zeros_like(annotated)
        colored_mask[mask] = color
        # mask_opacity controls overlay intensity (0.0 = no overlay, 1.0 = full overlay)
        frame_opacity = 1.0 - mask_opacity
        annotated = cv2.addWeighted(annotated, frame_opacity, colored_mask, mask_opacity, 0)
        
        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Add label with depth info
        label = det.get('class_name', f"class_{det['class_id']}")
        conf = det['score']
        
        if np.isfinite(mean_depth):
            # Convert from mm to inches (1 inch = 25.4 mm)
            depth_inches = mean_depth / 25.4
            depth_text = f"{depth_inches:.1f}\""
        else:
            depth_text = "N/A"
        
        text = f"{label} {conf:.2f} | {depth_text}"
        text_pos = (x1, max(y1 - 10, 20))
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


def main():
    args = parse_args()
    
    # Load RKNN model
    model_path = args.model.expanduser().resolve()
    if not model_path.exists():
        print(f"[ERROR] Model file not found: {model_path}", file=sys.stderr)
        return 1
    
    # Initialize RKNN
    rknn = RKNNLite(verbose=False)
    
    # Load model
    ret = rknn.load_rknn(str(model_path))
    if ret != 0:
        print(f"[ERROR] Failed to load RKNN model: {ret}", file=sys.stderr)
        rknn.release()
        return 1
    
    # Initialize runtime
    if args.target is None or (isinstance(args.target, str) and (args.target.lower() == 'none' or args.target == '')):
        target = None
    else:
        target = args.target
    
    ret = rknn.init_runtime(target=target, core_mask=args.core)
    if ret != 0:
        print(f"[ERROR] Failed to initialize runtime: {ret}", file=sys.stderr)
        rknn.release()
        return 1
    
    # Load config and initialize VISION
    try:
        camera_config = load_config(args.config)
    except Exception as e:
        print(f"[ERROR] Failed to load config: {e}", file=sys.stderr)
        rknn.release()
        return 1
    
    # Apply fast depth optimizations if requested
    if args.fast_depth:
        original_wls = camera_config.get("useWLS", False)
        original_morph = camera_config.get("useMorph", False)
        original_smooth = camera_config.get("smoothingKernel", 0)
        camera_config["useWLS"] = False
        camera_config["useMorph"] = False
        camera_config["smoothingKernel"] = 0
    
    # Override downsampling if specified
    if args.depth_downsample is not None:
        camera_config["downSample"] = max(0, min(90, args.depth_downsample))
    
    # Use faster stereo mode if requested
    if args.fast_stereo:
        camera_config["sgbmMode"] = 2  # SGBM_3WAY mode
    
    vision = VISION(name="RKNNDepth", **camera_config)
    use_bm = args.use_bm
    
    window_name = "RKNN Detection + Depth"
    depth_window = "Depth Map"
    
    # Trackbar for mask opacity (0-100, default 30 for 0.3 opacity)
    mask_opacity_value = [30]  # Use list to allow modification in callback
    
    def on_opacity_change(val):
        mask_opacity_value[0] = val
    
    try:
        vision.start()
        
        # Replace SGBM with BM if requested
        if use_bm:
            if vision.depth_processor:
                vision.depth_processor.useWLS = False
                vision.depth_processor.wls_filter = None
                vision.depth_processor.right_matcher = None
            
            block_size = camera_config.get("blockSize", 11)
            block_size = block_size if block_size % 2 == 1 else block_size + 1
            num_disparities = max(16, 16 * camera_config.get("numDisparitiesK", 2))
            num_disparities_bm = (num_disparities // 16) * 16
            if num_disparities_bm < 16:
                num_disparities_bm = 16
            
            bm_matcher = cv2.StereoBM_create(
                numDisparities=num_disparities_bm,
                blockSize=max(5, block_size)
            )
            bm_matcher.setPreFilterType(1)
            prefilter_size = max(5, min(255, camera_config.get("preFilterCap", 43)))
            if prefilter_size % 2 == 0:
                prefilter_size += 1
            bm_matcher.setPreFilterSize(prefilter_size)
            bm_matcher.setPreFilterCap(31)
            bm_matcher.setTextureThreshold(10)
            bm_matcher.setUniquenessRatio(camera_config.get("uniquenessRatio", 15))
            bm_matcher.setSpeckleWindowSize(camera_config.get("speckleWindowSize", 0))
            bm_matcher.setSpeckleRange(camera_config.get("speckleRange", 0))
            bm_matcher.setDisp12MaxDiff(-1)
            
            vision.stereo = bm_matcher
            if vision.depth_processor:
                vision.depth_processor.update_matcher(bm_matcher)
        
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.namedWindow(depth_window, cv2.WINDOW_NORMAL)
        
        # Create trackbar for opacity control (0-100, default 30)
        cv2.createTrackbar("Mask Opacity", window_name, 30, 100, on_opacity_change)
        
        prev_time = time.time()
        frame_counter = 0
        depth_map = None  # Keep previous depth map when skipping
        
        # Pre-allocate buffers for better performance
        img_input_buffer = None
        
        while True:
            frame_counter += 1
            
            # Capture frames from both cameras simultaneously
            left_frame = vision.left_camera.read_frame() if vision.left_camera else None
            right_frame = vision.right_camera.read_frame() if vision.right_camera else None
            
            if left_frame is None or right_frame is None:
                time.sleep(0.1)
                continue
            
            # Process depth using the depth processor directly
            if vision.depth_processor is None:
                print("[ERROR] Depth processor not initialized", file=sys.stderr)
                break
            
            # Skip depth processing if requested (use previous depth map)
            should_process_depth = (args.skip_depth == 0) or (frame_counter % (args.skip_depth + 1) == 1)
            
            if should_process_depth:
                try:
                    left_rect, right_rect = vision.depth_processor.rectify(left_frame, right_frame)
                    disparity = vision.depth_processor.compute_disparity(left_rect, right_rect)
                    depth_map = vision.depth_processor.disparity_to_depth(disparity)
                    
                    # Apply smoothing if enabled
                    if vision.depth_processor._smooth_kernel >= 3:
                        depth_map = cv2.GaussianBlur(
                            depth_map,
                            (vision.depth_processor._smooth_kernel, vision.depth_processor._smooth_kernel),
                            0
                        )
                except Exception as e:
                    print(f"[WARN] Error processing depth: {e}", file=sys.stderr)
                    continue
                
                if depth_map is None or depth_map.size == 0:
                    continue
                
                # Check if depth map has valid values
                valid_depth_count = np.sum(depth_map > 0)
                if valid_depth_count == 0:
                    continue
            else:
                # Use previous rectified frame (need to rectify for RKNN anyway)
                try:
                    left_rect, right_rect = vision.depth_processor.rectify(left_frame, right_frame)
                except Exception as e:
                    continue
            
            # Skip RKNN inference if requested
            if args.skip_frames > 0 and frame_counter % (args.skip_frames + 1) != 0:
                # Still display previous frame if available
                if depth_map is not None and depth_map.size > 0:
                    depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
                    depth_display = depth_normalized.astype(np.uint8)
                    depth_colored = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
                    cv2.imshow(depth_window, depth_colored)
                cv2.waitKey(1)
                continue
            
            # Preprocess frame for RKNN
            img_resized, ratio, (dw, dh) = letterbox(left_rect, new_shape=(args.imgsz, args.imgsz))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            
            # Pre-allocate buffer if not exists, or reuse if same size
            if img_input_buffer is None or img_input_buffer.shape != (1, args.imgsz, args.imgsz, 3):
                img_input_buffer = np.zeros((1, args.imgsz, args.imgsz, 3), dtype=np.uint8)
            img_input_buffer[0] = img_rgb.astype(np.uint8)
            img_input = img_input_buffer
            
            # Run RKNN inference
            try:
                outputs = rknn.inference([img_input])
            except Exception as e:
                print(f"[ERROR] Inference failed: {e}", file=sys.stderr)
                continue
            
            if outputs is None:
                continue
            
            # Process output
            detections = []
            try:
                detections = process_output(outputs, conf_threshold=args.conf, img_shape=(args.imgsz, args.imgsz))
            except Exception as e:
                print(f"[ERROR] Failed to process output: {e}", file=sys.stderr)
                detections = []
            
            # Scale boxes back to original image size (vectorized for speed)
            if detections:
                h_orig, w_orig = left_rect.shape[:2]
                scale = min(args.imgsz / w_orig, args.imgsz / h_orig)
                new_w = int(w_orig * scale)
                new_h = int(h_orig * scale)
                pad_x = (args.imgsz - new_w) / 2
                pad_y = (args.imgsz - new_h) / 2
                
                # Vectorized box scaling
                boxes = np.array([det['bbox'] for det in detections], dtype=np.float32)
                boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
                boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
                boxes = boxes.astype(np.int32)
                
                for i, det in enumerate(detections):
                    det['bbox'] = boxes[i].tolist()
            
            # Get current opacity value from trackbar (0-100 -> 0.0-1.0)
            current_opacity = mask_opacity_value[0] / 100.0
            
            # Overlay depth on detections (both are rectified, so dimensions match)
            if depth_map is not None and depth_map.size > 0:
                annotated_frame = overlay_depth_on_detections(
                    left_rect,
                    depth_map,
                    detections,
                    mask_opacity=current_opacity,
                )
            else:
                # Fallback: draw detections without depth
                annotated_frame = left_rect.copy()
                for det in detections:
                    x1, y1, x2, y2 = det['bbox']
                    label = det.get('class_name', f"class_{det['class_id']}")
                    conf = det['score']
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Calculate FPS
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            
            # Add FPS text and performance info
            fps_text = f"FPS: {fps:.1f} | Detections: {len(detections)}"
            if args.skip_depth > 0:
                fps_text += f" | Depth: {100 // (args.skip_depth + 1)}%"
            cv2.putText(
                annotated_frame,
                fps_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            
            # Normalize depth map for visualization
            if depth_map is not None and depth_map.size > 0:
                depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
                depth_display = depth_normalized.astype(np.uint8)
                depth_colored = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
                
                cv2.imshow(window_name, annotated_frame)
                cv2.imshow(depth_window, depth_colored)
            else:
                cv2.imshow(window_name, annotated_frame)
            
            # Check for exit
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
                
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    finally:
        if vision.connected:
            vision.stop()
        cv2.destroyAllWindows()
        rknn.release()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
