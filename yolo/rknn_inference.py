"""
RKNN Model Inference Script

This script loads and runs RKNN models using rknnlite for inference on the NPU.
It supports YOLO models and can process camera feeds or image files.

Usage:
    python yolo/rknn_inference.py --model yolo/models/yolo11n.rknn --source 0
    python yolo/rknn_inference.py --model yolo/models/yolov8n.rknn --source /path/to/image.jpg
"""

import argparse
import sys
import time
from pathlib import Path

# Add system dist-packages to path for rknnlite (system package)
import site
system_packages = '/usr/lib/python3/dist-packages'
if system_packages not in sys.path:
    sys.path.insert(0, system_packages)

import cv2
import numpy as np
from rknnlite.api import RKNNLite

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "yolo11n.rknn"

# COCO class names (YOLO models typically use COCO dataset)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run RKNN model inference")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"Path to RKNN model file. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Input source: camera index (e.g., 0), image file, or video file",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (width/height). Default: 640",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (0.0-1.0). Default: 0.25",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Target platform (RK3562/RK3566/RK3568/RK3588). Use 'None' or leave empty for on-device NPU. Default: None (on-device)",
    )
    parser.add_argument(
        "--core",
        type=int,
        default=0,
        help="NPU core mask (0=auto, 1=core0, 2=core1, 4=core2, 3=core0+1, 7=all). Default: 0 (auto)",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=0,
        help="Skip N frames between inferences (0=process all frames). Default: 0",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable display window for maximum speed",
    )
    return parser.parse_args()


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """
    Resize image to new_shape while maintaining aspect ratio and padding with color.
    Returns: (resized_image, ratio, (dw, dh))
    """
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, r, (dw, dh)


def xywh2xyxy(x):
    """Convert boxes from [x, y, w, h] to [x1, y1, x2, y2] format."""
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # x1
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # y1
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # x2
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # y2
    return y


def nms(boxes, scores, iou_threshold=0.45):
    """Non-maximum suppression."""
    # Sort by score
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        # Calculate IoU
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1]) + \
              (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1]) - inter
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return np.array(keep)


def dfl(position):
    """Distribution Focal Loss (DFL) for YOLOv8/v11 box decoding."""
    # Use numpy implementation (matches official Radxa script)
    n, c, h, w = position.shape
    p_num = 4
    mc = c // p_num
    x = position.reshape(n, p_num, mc, h, w)
    # Softmax
    exp_x = np.exp(x - np.max(x, axis=2, keepdims=True))
    softmax_x = exp_x / np.sum(exp_x, axis=2, keepdims=True)
    # Weighted sum
    acc_metrix = np.arange(mc).reshape(1, 1, mc, 1, 1).astype(np.float32)
    y = np.sum(softmax_x * acc_metrix, axis=2)
    return y


def box_process_yolov8(position, img_size=(640, 640)):
    """Process YOLOv8/v11 box outputs."""
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([img_size[1] // grid_h, img_size[0] // grid_w]).reshape(1, 2, 1, 1)
    
    position = dfl(position)
    box_xy = grid + 0.5 - position[:, 0:2, :, :]
    box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
    xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)
    
    return xyxy


def post_process_yolov8(input_data, conf_threshold=0.25, iou_threshold=0.45, img_size=(640, 640)):
    """Post-process YOLOv8/v11 outputs (multi-branch format).
    
    Matches official Radxa script - expects 6 outputs (3 scales × 2: boxes + classes).
    """
    boxes, scores, classes_conf = [], [], []
    default_branch = 3
    
    # Official Radxa script expects 6 outputs (3 branches × 2 outputs each)
    # If we have 9 outputs, we need to filter out the objectness outputs
    # The pattern is: [boxes_0, classes_0, obj_0, boxes_1, classes_1, obj_1, boxes_2, classes_2, obj_2]
    # We only use: [boxes_0, classes_0, boxes_1, classes_1, boxes_2, classes_2]
    if len(input_data) == 9:
        # Filter to get only boxes and classes (skip objectness outputs)
        filtered_outputs = []
        for i in range(3):
            filtered_outputs.append(input_data[i * 3])      # boxes
            filtered_outputs.append(input_data[i * 3 + 1])  # classes
        input_data = filtered_outputs
        # print(f"[INFO] Filtered 9 outputs to 6 (removed objectness outputs)")
    elif len(input_data) != 6:
        raise ValueError(f"Unexpected number of outputs: {len(input_data)}. Expected 6 or 9.")
    
    pair_per_branch = len(input_data) // default_branch
    
    # Process each branch (scale) - matches official Radxa script
    for i in range(default_branch):
        boxes.append(box_process_yolov8(input_data[pair_per_branch * i], img_size))
        classes_conf.append(input_data[pair_per_branch * i + 1])
        # Use ones as placeholder (matches official script)
        scores.append(np.ones_like(input_data[pair_per_branch * i + 1][:, :1, :, :], dtype=np.float32))
    
    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0, 2, 3, 1)
        return _in.reshape(-1, ch)
    
    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]
    
    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)
    
    # Convert to float32 if needed (RKNN outputs might be int8)
    classes_conf = classes_conf.astype(np.float32)
    scores = scores.astype(np.float32)
    
    # DO NOT apply sigmoid to class scores - they're already probabilities or properly scaled
    # The official Radxa script uses them directly
    
    # Filter by confidence (matches official script)
    box_confidences = scores.reshape(-1)
    candidate, class_num = classes_conf.shape
    
    class_max_score = np.max(classes_conf, axis=-1)
    classes = np.argmax(classes_conf, axis=-1)
    
    # Combined confidence = objectness (ones) * class_confidence
    _class_pos = np.where(class_max_score * box_confidences >= conf_threshold)
    scores = (class_max_score * box_confidences)[_class_pos]
    
    boxes = boxes[_class_pos]
    classes = classes[_class_pos]
    
    if len(boxes) == 0:
        return [], [], []
    
    # NMS per class
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c_vals = classes[inds]
        s = scores[inds]
        keep = nms(b, s, iou_threshold)
        
        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c_vals[keep])
            nscores.append(s[keep])
    
    if not nclasses and not nscores:
        return [], [], []
    
    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)
    
    return boxes, classes, scores


def process_output(output, conf_threshold=0.25, iou_threshold=0.45, img_shape=(640, 640)):
    """
    Process RKNN model output to get detections.
    Supports YOLOv8/v11 (6 or 9 outputs) and older YOLO (single output) formats.
    """
    if not isinstance(output, list):
        output = [output]
    
    # Detect output format: YOLOv8/v11 has 6 or 9 outputs, older YOLO has 1 output
    if len(output) == 6 or len(output) == 9:
        # YOLOv8/v11 format (6 outputs = boxes+classes, 9 outputs = boxes+classes+objectness)
        # post_process_yolov8 will handle both cases
        boxes, classes, scores = post_process_yolov8(output, conf_threshold, iou_threshold, img_shape)
    elif len(output) == 1:
        # Older YOLO format: [batch, num_boxes, 85] where 85 = 4 (bbox) + 1 (objectness) + 80 (classes)
        output_data = output[0]
        if len(output_data.shape) == 3:
            output_data = output_data.reshape(-1, output_data.shape[-1])
        
        # Extract boxes, scores, classes
        boxes_norm = output_data[:, :4]  # [x_center, y_center, w, h] (normalized)
        objectness = output_data[:, 4:5]  # objectness score
        class_scores = output_data[:, 5:]  # class scores
        
        # Get class predictions
        class_ids = np.argmax(class_scores, axis=1)
        class_conf = np.max(class_scores, axis=1)
        
        # Combined confidence = objectness * class_confidence
        scores = objectness.flatten() * class_conf
        
        # Filter by confidence threshold
        valid = scores > conf_threshold
        boxes_norm = boxes_norm[valid]
        scores = scores[valid]
        class_ids = class_ids[valid]
        
        if len(boxes_norm) == 0:
            return []
        
        # Convert normalized [x_center, y_center, w, h] to pixel [x1, y1, x2, y2]
        h, w = img_shape
        boxes = np.zeros_like(boxes_norm)
        boxes[:, 0] = (boxes_norm[:, 0] - boxes_norm[:, 2] / 2) * w  # x1
        boxes[:, 1] = (boxes_norm[:, 1] - boxes_norm[:, 3] / 2) * h  # y1
        boxes[:, 2] = (boxes_norm[:, 0] + boxes_norm[:, 2] / 2) * w  # x2
        boxes[:, 3] = (boxes_norm[:, 1] + boxes_norm[:, 3] / 2) * h  # y2
        
        # Clip to image bounds
        boxes[:, 0] = np.clip(boxes[:, 0], 0, w)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, h)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, w)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, h)
        
        # Apply NMS
        keep = nms(boxes, scores, iou_threshold)
        boxes = boxes[keep]
        scores = scores[keep]
        classes = class_ids[keep]
    else:
        # print(f"[WARN] Unexpected number of outputs: {len(output)}. Trying to process as single output...")
        # Try to process first output
        output_data = output[0]
        if len(output_data.shape) == 3:
            output_data = output_data.reshape(-1, output_data.shape[-1])
        # Assume same format as older YOLO
        boxes_norm = output_data[:, :4]
        objectness = output_data[:, 4:5] if output_data.shape[1] > 4 else np.ones((len(output_data), 1))
        class_scores = output_data[:, 5:] if output_data.shape[1] > 5 else output_data[:, 4:]
        
        class_ids = np.argmax(class_scores, axis=1)
        class_conf = np.max(class_scores, axis=1)
        scores = objectness.flatten() * class_conf
        
        valid = scores > conf_threshold
        boxes_norm = boxes_norm[valid]
        scores = scores[valid]
        classes = class_ids[valid]
        
        if len(boxes_norm) == 0:
            return []
        
        h, w = img_shape
        boxes = np.zeros_like(boxes_norm)
        boxes[:, 0] = (boxes_norm[:, 0] - boxes_norm[:, 2] / 2) * w
        boxes[:, 1] = (boxes_norm[:, 1] - boxes_norm[:, 3] / 2) * h
        boxes[:, 2] = (boxes_norm[:, 0] + boxes_norm[:, 2] / 2) * w
        boxes[:, 3] = (boxes_norm[:, 1] + boxes_norm[:, 3] / 2) * h
        
        boxes[:, 0] = np.clip(boxes[:, 0], 0, w)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, h)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, w)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, h)
        
        keep = nms(boxes, scores, iou_threshold)
        boxes = boxes[keep]
        scores = scores[keep]
        classes = classes[keep]
    
    if len(boxes) == 0:
        return []
    
    # Format as list of detections
    detections = []
    for i in range(len(boxes)):
        detections.append({
            'bbox': boxes[i].astype(int),
            'score': float(scores[i]),
            'class_id': int(classes[i]),
            'class_name': COCO_CLASSES[classes[i]] if classes[i] < len(COCO_CLASSES) else f'class_{classes[i]}'
        })
    
    return detections


def draw_detections(img, detections):
    """Draw bounding boxes and labels on image."""
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        score = det['score']
        class_name = det['class_name']
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label = f"{class_name} {score:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - text_height - baseline - 5), (x1 + text_width, y1), (0, 255, 0), -1)
        cv2.putText(img, label, (x1, y1 - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return img


def main():
    args = parse_args()
    
    # Validate model file
    model_path = args.model.expanduser().resolve()
    if not model_path.exists():
        print(f"[ERROR] Model file not found: {model_path}", file=sys.stderr)
        return 1
    
    # print(f"[INFO] Loading RKNN model: {model_path}")
    
    # Initialize RKNN
    rknn = RKNNLite(verbose=False)
    
    # Load model
    ret = rknn.load_rknn(str(model_path))
    if ret != 0:
        print(f"[ERROR] Failed to load RKNN model: {ret}", file=sys.stderr)
        rknn.release()
        return 1
    
    # Initialize runtime
    # When running on-device, target should be None (uses internal NPU)
    # target is only needed when connecting via ADB to a remote device
    if args.target is None or (isinstance(args.target, str) and (args.target.lower() == 'none' or args.target == '')):
        target = None
        # print("[INFO] Initializing runtime for on-device NPU...")
    else:
        target = args.target
        # print(f"[INFO] Initializing runtime for {target}...")
    
    ret = rknn.init_runtime(target=target, core_mask=args.core)
    if ret != 0:
        print(f"[ERROR] Failed to initialize runtime: {ret}", file=sys.stderr)
        rknn.release()
        return 1
    
    # print("[INFO] Model loaded and runtime initialized successfully!")
    
    # Determine input source
    source = args.source if args.source is not None else "0"
    if source.isdigit():
        # Camera
        source = int(source)
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[ERROR] Failed to open camera {source}", file=sys.stderr)
            rknn.release()
            return 1
        
        # Set camera properties for better performance
        # Set camera to proper resolution (try highest first, fallback if needed)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
        cap.set(cv2.CAP_PROP_FPS, 60)
        
        # Verify actual resolution (camera might not support requested resolution)
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_width != 1920 or actual_height != 1200:
            # Try alternative common resolutions
            for w, h in [(1600, 1200), (1280, 1024), (1280, 720), (1024, 768)]:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if actual_width == w and actual_height == h:
                    break
        
        # Test if we can actually read a frame
        ret, test_frame = cap.read()
        if not ret or test_frame is None:
            print(f"[ERROR] Camera {source} opened but cannot read frames. Check camera connection.", file=sys.stderr)
            cap.release()
            rknn.release()
            return 1
        
        # print(f"[INFO] Using camera {source} (resolution: {actual_width}x{actual_height})")
        is_video = True
    else:
        # Image or video file
        source_path = Path(source)
        if not source_path.exists():
            print(f"[ERROR] Source file not found: {source}", file=sys.stderr)
            rknn.release()
            return 1
        
        if source_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            # Single image
            cap = None
            img = cv2.imread(str(source_path))
            if img is None:
                print(f"[ERROR] Failed to load image: {source}", file=sys.stderr)
                rknn.release()
                return 1
            # print(f"[INFO] Processing image: {source}")
            is_video = False
        else:
            # Video file
            cap = cv2.VideoCapture(str(source_path))
            if not cap.isOpened():
                print(f"[ERROR] Failed to open video: {source}", file=sys.stderr)
                rknn.release()
                return 1
            # print(f"[INFO] Processing video: {source}")
            is_video = True
    
    window_name = "RKNN Inference"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    frame_count = 0
    prev_time = time.time()
    
    try:
        while True:
            # Get frame
            if is_video:
                ret, frame = cap.read()
                if not ret:
                    # print("[WARN] Failed to read frame from camera. Retrying...")
                    time.sleep(0.1)
                    continue
                if frame is None:
                    # print("[WARN] Received empty frame. Retrying...")
                    time.sleep(0.1)
                    continue
            else:
                frame = img.copy()
            
            # Preprocess
            img_resized, ratio, (dw, dh) = letterbox(frame, new_shape=(args.imgsz, args.imgsz))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            
            # RKNN expects 4D input: (batch, height, width, channels) for NHWC format
            # Add batch dimension: (height, width, channels) -> (1, height, width, channels)
            img_input = np.expand_dims(img_rgb.astype(np.uint8), axis=0)
            
            # Run inference
            inference_start = time.time()
            try:
                outputs = rknn.inference([img_input])
            except Exception as e:
                print(f"[ERROR] Inference failed: {e}", file=sys.stderr)
                continue
            inference_time = (time.time() - inference_start) * 1000  # ms
            
            # Process output
            if outputs is None:
                # print("[WARN] Inference returned None, skipping frame")
                continue
            
            detections = []
            try:
                detections = process_output(outputs, conf_threshold=args.conf, img_shape=(args.imgsz, args.imgsz))
                # Debug: print detection stats occasionally
                # if frame_count % 30 == 0 and len(detections) == 0:
                #     # Check raw output stats
                #     if isinstance(outputs, list) and len(outputs) > 0:
                #         print(f"[DEBUG] Output shapes: {[o.shape for o in outputs]}")
                #         if len(outputs) >= 3:
                #             # Check objectness score range
                #             obj_scores = outputs[2]  # First objectness output
                #             print(f"[DEBUG] Objectness range: [{obj_scores.min():.3f}, {obj_scores.max():.3f}]")
            except Exception as e:
                print(f"[ERROR] Failed to process output: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()
                # Continue to display frame even if processing failed
                detections = []
            
            # Scale boxes back to original image size
            h_orig, w_orig = frame.shape[:2]
            scale = min(args.imgsz / w_orig, args.imgsz / h_orig)
            new_w = int(w_orig * scale)
            new_h = int(h_orig * scale)
            pad_x = (args.imgsz - new_w) / 2
            pad_y = (args.imgsz - new_h) / 2
            
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                # Remove padding and scale back
                x1 = int((x1 - pad_x) / scale)
                y1 = int((y1 - pad_y) / scale)
                x2 = int((x2 - pad_x) / scale)
                y2 = int((y2 - pad_y) / scale)
                det['bbox'] = [x1, y1, x2, y2]
            
            # Draw detections
            annotated = draw_detections(frame.copy(), detections)
            
            # Calculate FPS
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            
            # Add info text
            info_text = f"FPS: {fps:.1f} | Inference: {inference_time:.1f}ms | Detections: {len(detections)}"
            cv2.putText(annotated, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display
            cv2.imshow(window_name, annotated)
            
            frame_count += 1
            
            # Exit on 'q' or Esc
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                # print("[INFO] Exiting...")
                break
            
            # For single image, wait for key press
            if not is_video:
                # print(f"[INFO] Processed image. Found {len(detections)} detections")
                # print("[INFO] Press any key to exit...")
                cv2.waitKey(0)
                break
    
    except KeyboardInterrupt:
        # print("\n[INFO] Interrupted by user")
        pass
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        rknn.release()
        # print("[INFO] Released RKNN resources")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

