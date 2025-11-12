"""
Lightweight YOLO segmentation live preview with depth overlay.

Simple script to run YOLOv8n-seg segmentation model with stereo depth overlay.
Uses VISION class to get depth maps and overlays depth on detected objects.
Press 'q' or Esc to exit.

Usage:
    python yolo/seg_live.py                    # Use config.dill for stereo cameras
    python yolo/seg_live.py --config custom.dill
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Tuple

import cv2
import dill
import numpy as np
from ultralytics import YOLO

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.hal.VISION.VISION import VISION  # noqa: E402


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "yolov3-tinyu.pt"
DEFAULT_CONFIG = PROJECT_ROOT / "config.dill"


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO segmentation with depth overlay")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"Path to segmentation model. Default: {DEFAULT_MODEL}",
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
        help="Confidence threshold (0.0-1.0)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (pixels). Smaller = faster (e.g., 320, 416, 640)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for YOLO inference. Default: 'cpu' (CPU-only mode)",
    )
    parser.add_argument(
        "--skip-depth",
        type=int,
        default=0,
        help="Skip depth processing every N frames (0=process all, 1=skip every other, etc.). Higher = faster.",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="[GPU only] Use FP16 half precision. Ignored in CPU mode.",
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
    result,
    label_lookup: dict,
    mask_opacity: float = 0.3,
) -> np.ndarray:
    """Overlay depth information on YOLO detections."""
    annotated = frame.copy()
    frame_h, frame_w = frame.shape[:2]
    depth_h, depth_w = depth_map.shape[:2]
    
    # Resize depth map to match frame if needed
    if (depth_h, depth_w) != (frame_h, frame_w):
        depth_map = cv2.resize(depth_map, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)
    
    if result.boxes is None or len(result.boxes) == 0:
        return annotated
    
    h, w = frame.shape[:2]
    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    
    # Get segmentation masks if available
    mask_data = None
    if result.masks is not None and len(result.masks.data) > 0:
        mask_data = result.masks.data.cpu().numpy()
    
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 140, 255),  # Orange
        (128, 0, 255),  # Purple
        (0, 255, 255),  # Yellow
    ]
    
    for idx, box in enumerate(boxes_xyxy):
        x1 = max(int(np.floor(box[0])), 0)
        y1 = max(int(np.floor(box[1])), 0)
        x2 = min(int(np.ceil(box[2])), w)
        y2 = min(int(np.ceil(box[3])), h)
        
        if x2 <= x1 or y2 <= y1:
            continue
        
        # Create mask from segmentation or bounding box
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
        label = label_lookup.get(classes[idx], str(classes[idx]))
        conf = float(confidences[idx])
        
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
    
    # Load model
    model_path = args.model.expanduser().resolve()
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}", file=sys.stderr)
        return 1
    
    print(f"[INFO] Loading model: {model_path}")
    model = YOLO(str(model_path))
    
    # Set device for inference (CPU only)
    if args.device:
        device = args.device
    else:
        device = "cpu"  # Force CPU mode
    
    print(f"[INFO] Using device: {device} (CPU-only mode)")
    if args.half:
        print("[WARN] FP16 half precision requires GPU - ignoring --half flag")
        args.half = False
    
    # Optimize OpenCV for CPU (use all available threads)
    try:
        import os
        # Set OpenCV to use all available CPU threads
        num_threads = os.cpu_count() or 4
        cv2.setNumThreads(num_threads)
        print(f"[INFO] OpenCV using {num_threads} threads for CPU processing")
    except Exception:
        pass
    
    # Get label lookup
    if hasattr(model.model, "names"):
        label_lookup = model.model.names
    else:
        label_lookup = model.names
    if not isinstance(label_lookup, dict):
        label_lookup = {idx: name for idx, name in enumerate(label_lookup)}
    
    # Load config and initialize VISION
    try:
        camera_config = load_config(args.config)
        print(f"[INFO] Loaded config from: {args.config}")
    except Exception as e:
        print(f"[ERROR] Failed to load config: {e}", file=sys.stderr)
        return 1
    
    # Apply fast depth optimizations if requested
    if args.fast_depth:
        print("[INFO] Fast depth mode: disabling expensive filters")
        # Temporarily override expensive filters
        original_wls = camera_config.get("useWLS", False)
        original_morph = camera_config.get("useMorph", False)
        original_smooth = camera_config.get("smoothingKernel", 0)
        camera_config["useWLS"] = False
        camera_config["useMorph"] = False
        camera_config["smoothingKernel"] = 0
        print(f"[INFO] Disabled: WLS={original_wls}, Morph={original_morph}, Smooth={original_smooth}")
    
    # Override downsampling if specified
    if args.depth_downsample is not None:
        camera_config["downSample"] = max(0, min(90, args.depth_downsample))
        print(f"[INFO] Depth downsampling set to {camera_config['downSample']}%")
    
    # Use faster stereo mode if requested
    if args.fast_stereo:
        original_mode = camera_config.get("sgbmMode", 2)
        camera_config["sgbmMode"] = 2  # SGBM_3WAY mode (optimized, faster than regular SGBM)
        print(f"[INFO] Using optimized stereo mode: SGBM_3WAY (was mode {original_mode})")
    
    vision = VISION(name="YOLODepth", **camera_config)
    
    # Replace SGBM with BM if requested (after start, we'll replace the matcher)
    use_bm = args.use_bm
    
    window_name = "YOLO Segmentation + Depth"
    depth_window = "Depth Map"
    
    # Trackbar for mask opacity (0-100, default 30 for 0.3 opacity)
    mask_opacity_value = [30]  # Use list to allow modification in callback
    
    def on_opacity_change(val):
        mask_opacity_value[0] = val
    
    print("[INFO] Starting vision system...")
    print("[INFO] Press 'q' or Esc to exit")
    print("[INFO] Use the 'Mask Opacity' slider to adjust overlay brightness")
    if args.skip_depth > 0:
        print(f"[INFO] Depth processing: every {args.skip_depth + 1} frames (faster mode)")
    if args.imgsz <= 416:
        print(f"[INFO] Using smaller inference size ({args.imgsz}) optimized for CPU")
    if args.fast_depth:
        print("[INFO] Fast depth mode enabled - expensive filters disabled")
    if args.fast_stereo:
        print("[INFO] Fast stereo mode enabled - using optimized SGBM_3WAY")
    if args.use_bm:
        print("[INFO] BM (Block Matching) mode enabled - much faster than SGBM")
    print("[INFO] CPU-only mode - using optimized settings for CPU performance")
    
    try:
        vision.start()
        
        # Replace SGBM with BM if requested (BM is faster but less accurate)
        if use_bm:
            print("[INFO] Replacing SGBM with BM (Block Matching) for faster processing")
            # Disable WLS when using BM (BM doesn't support WLS)
            if vision.depth_processor:
                vision.depth_processor.useWLS = False
                vision.depth_processor.wls_filter = None
                vision.depth_processor.right_matcher = None
            
            block_size = camera_config.get("blockSize", 11)
            block_size = block_size if block_size % 2 == 1 else block_size + 1
            num_disparities = max(16, 16 * camera_config.get("numDisparitiesK", 2))
            
            # Create StereoBM matcher (faster than SGBM)
            bm_matcher = cv2.StereoBM_create(
                numDisparities=num_disparities,
                blockSize=max(5, block_size)  # BM requires blockSize >= 5
            )
            bm_matcher.setPreFilterType(1)  # 0 = PREFILTER_NORMALIZED_RESPONSE, 1 = PREFILTER_XSOBEL
            bm_matcher.setPreFilterSize(camera_config.get("preFilterCap", 43))
            bm_matcher.setPreFilterCap(31)
            bm_matcher.setTextureThreshold(10)
            bm_matcher.setUniquenessRatio(camera_config.get("uniquenessRatio", 1))
            bm_matcher.setSpeckleWindowSize(camera_config.get("speckleWindowSize", 196))
            bm_matcher.setSpeckleRange(camera_config.get("speckleRange", 34))
            bm_matcher.setDisp12MaxDiff(camera_config.get("disp12MaxDiff", 18))
            
            # Replace the stereo matcher
            vision.stereo = bm_matcher
            # Update depth processor with new matcher
            if vision.depth_processor:
                vision.depth_processor.update_matcher(bm_matcher)
            print(f"[INFO] BM matcher created: numDisparities={num_disparities}, blockSize={block_size}")
            print("[INFO] WLS filtering disabled (not supported by BM)")
        
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.namedWindow(depth_window, cv2.WINDOW_NORMAL)
        
        # Create trackbar for opacity control (0-100, default 30)
        cv2.createTrackbar("Mask Opacity", window_name, 30, 100, on_opacity_change)
        
        prev_time = time.time()
        frame_counter = 0
        depth_map = None  # Keep previous depth map when skipping
        
        while True:
            frame_counter += 1
            
            # Capture frames from both cameras simultaneously
            left_frame = vision.left_camera.read_frame() if vision.left_camera else None
            right_frame = vision.right_camera.read_frame() if vision.right_camera else None
            
            if left_frame is None or right_frame is None:
                print("[WARN] Failed to grab stereo frames. Retrying...")
                time.sleep(0.1)
                continue
            
            # Process depth using the depth processor directly
            if vision.depth_processor is None:
                print("[ERROR] Depth processor not initialized")
                break
            
            # Skip depth processing if requested (use previous depth map)
            should_process_depth = (args.skip_depth == 0) or (frame_counter % (args.skip_depth + 1) == 1)
            
            if should_process_depth:
                # Process frames through depth pipeline
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
                    print(f"[WARN] Error processing depth: {e}")
                    continue
                
                if depth_map is None or depth_map.size == 0:
                    print("[WARN] No depth data available")
                    continue
            else:
                # Use previous rectified frame (need to rectify for YOLO anyway)
                try:
                    left_rect, right_rect = vision.depth_processor.rectify(left_frame, right_frame)
                except Exception as e:
                    print(f"[WARN] Error rectifying frames: {e}")
                    continue
            
            # Use rectified left frame for YOLO (matches depth map dimensions)
            # Run YOLO inference on rectified left frame
            predict_kwargs = {
                "source": left_rect,
                "imgsz": args.imgsz,
                "conf": args.conf,
                "verbose": False,
                "device": device,
            }
            if args.half:
                predict_kwargs["half"] = True
            
            yolo_results = model.predict(**predict_kwargs)
            result = yolo_results[0]
            
            # Get current opacity value from trackbar (0-100 -> 0.0-1.0)
            current_opacity = mask_opacity_value[0] / 100.0
            
            # Overlay depth on detections (both are rectified, so dimensions match)
            # Only overlay if we have a valid depth map
            if depth_map is not None and depth_map.size > 0:
                annotated_frame = overlay_depth_on_detections(
                    left_rect,
                    depth_map,
                    result,
                    label_lookup,
                    mask_opacity=current_opacity,
                )
            else:
                # Fallback: use YOLO's built-in plotting if no depth
                annotated_frame = result.plot()
            
            # Calculate FPS
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            
            # Add FPS text and performance info
            fps_text = f"FPS: {fps:.1f}"
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
            
            # Normalize depth map for visualization (use previous if skipping)
            if depth_map is not None and depth_map.size > 0:
                depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
                depth_display = depth_normalized.astype(np.uint8)
                depth_colored = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
                
                # Show frames
                cv2.imshow(window_name, annotated_frame)
                cv2.imshow(depth_window, depth_colored)
            else:
                # Show only annotated frame if no depth map
                cv2.imshow(window_name, annotated_frame)
            
            # Check for exit
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                print("[INFO] Exiting...")
                break
                
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    finally:
        if vision.connected:
            vision.stop()
        cv2.destroyAllWindows()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

