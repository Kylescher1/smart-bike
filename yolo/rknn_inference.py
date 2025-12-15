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
from collections import defaultdict

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

from src.hal.cam.Camera import Camera, ThreadedCamera, CAMERA_CONFIG

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


def post_process_yolov5(outputs, conf_threshold=0.25, iou_threshold=0.45, img_shape=(640, 640)):
    """
    Post-process YOLOv5 outputs (3 anchor-based outputs).
    
    YOLOv5 outputs 3 tensors with shape (1, 255, H, W) where:
    - 255 = 3 anchors × 85 (4 box + 1 obj + 80 classes)
    - H, W = 80, 40, 20 for strides 8, 16, 32
    """
    # YOLOv5 anchors (width, height) for each stride
    anchors = [
        [(10, 13), (16, 30), (33, 23)],      # stride 8
        [(30, 61), (62, 45), (59, 119)],     # stride 16
        [(116, 90), (156, 198), (373, 326)]  # stride 32
    ]
    strides = [8, 16, 32]
    
    all_boxes = []
    all_scores = []
    all_classes = []
    
    for i, output in enumerate(outputs):
        # output shape: (1, 255, H, W)
        _, channels, h, w = output.shape
        num_anchors = 3
        num_classes = channels // num_anchors - 5  # 80 for COCO
        
        # Reshape to (1, 3, 85, H, W) then transpose to (1, 3, H, W, 85)
        output = output.reshape(1, num_anchors, 5 + num_classes, h, w)
        output = output.transpose(0, 1, 3, 4, 2)  # (1, 3, H, W, 85)
        
        # Create grid
        grid_y, grid_x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        grid = np.stack([grid_x, grid_y], axis=-1).astype(np.float32)
        
        # Decode for each anchor
        for a in range(num_anchors):
            anchor_w, anchor_h = anchors[i][a]
            pred = output[0, a]  # (H, W, 85)
            
            # Box decoding (sigmoid already applied by model with relu approximation)
            # For YOLOv5, outputs are already sigmoid-activated
            xy = pred[..., :2]  # Already sigmoid
            wh = pred[..., 2:4]
            obj = pred[..., 4:5]
            cls = pred[..., 5:]
            
            # Decode boxes
            xy = (xy * 2 - 0.5 + grid) * strides[i]
            wh = (wh * 2) ** 2 * np.array([anchor_w, anchor_h])
            
            # Convert to x1y1x2y2
            x1y1 = xy - wh / 2
            x2y2 = xy + wh / 2
            boxes = np.concatenate([x1y1, x2y2], axis=-1)
            
            # Confidence = objectness * class_prob
            scores = obj * cls
            
            # Flatten spatial dimensions
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1, num_classes)
            
            # Get best class for each box
            class_ids = np.argmax(scores, axis=1)
            class_scores = np.max(scores, axis=1)
            
            # Filter by confidence
            mask = class_scores > conf_threshold
            if np.any(mask):
                all_boxes.append(boxes[mask])
                all_scores.append(class_scores[mask])
                all_classes.append(class_ids[mask])
    
    if not all_boxes:
        return [], [], []
    
    boxes = np.concatenate(all_boxes)
    scores = np.concatenate(all_scores)
    classes = np.concatenate(all_classes)
    
    # Clip boxes to image bounds
    h_img, w_img = img_shape
    boxes[:, 0] = np.clip(boxes[:, 0], 0, w_img)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, h_img)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, w_img)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, h_img)
    
    # NMS per class
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)[0]
        b = boxes[inds]
        s = scores[inds]
        keep = nms(b, s, iou_threshold)
        if len(keep) > 0:
            nboxes.append(b[keep])
            nclasses.append(np.full(len(keep), c))
            nscores.append(s[keep])
    
    if not nboxes:
        return [], [], []
    
    return np.concatenate(nboxes), np.concatenate(nclasses), np.concatenate(nscores)


def post_process_yolov5_seg(outputs, conf_threshold=0.25, iou_threshold=0.45, img_shape=(640, 640)):
    """
    Post-process YOLOv5-seg outputs (7 outputs: 3 det + 3 mask coeff + 1 proto).
    
    Returns: boxes, classes, scores, masks (list of binary masks per detection)
    """
    # Separate detection, mask coefficient, and prototype outputs
    det_outputs = [outputs[0], outputs[2], outputs[4]]  # (1, 255, H, W)
    mask_outputs = [outputs[1], outputs[3], outputs[5]]  # (1, 96, H, W) = 3 anchors × 32 coeffs
    proto = outputs[6]  # (1, 32, 160, 160)
    
    anchors = [
        [(10, 13), (16, 30), (33, 23)],
        [(30, 61), (62, 45), (59, 119)],
        [(116, 90), (156, 198), (373, 326)]
    ]
    strides = [8, 16, 32]
    
    all_boxes = []
    all_scores = []
    all_classes = []
    all_mask_coeffs = []
    
    for i, (det_out, mask_out) in enumerate(zip(det_outputs, mask_outputs)):
        _, det_channels, h, w = det_out.shape
        num_anchors = 3
        num_classes = det_channels // num_anchors - 5  # 80
        num_mask_coeffs = 32
        
        # Reshape detection output
        det_out = det_out.reshape(1, num_anchors, 5 + num_classes, h, w)
        det_out = det_out.transpose(0, 1, 3, 4, 2)  # (1, 3, H, W, 85)
        
        # Reshape mask coefficient output
        mask_out = mask_out.reshape(1, num_anchors, num_mask_coeffs, h, w)
        mask_out = mask_out.transpose(0, 1, 3, 4, 2)  # (1, 3, H, W, 32)
        
        grid_y, grid_x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        grid = np.stack([grid_x, grid_y], axis=-1).astype(np.float32)
        
        for a in range(num_anchors):
            anchor_w, anchor_h = anchors[i][a]
            pred = det_out[0, a]  # (H, W, 85)
            mask_pred = mask_out[0, a]  # (H, W, 32)
            
            xy = pred[..., :2]
            wh = pred[..., 2:4]
            obj = pred[..., 4:5]
            cls = pred[..., 5:]
            
            xy = (xy * 2 - 0.5 + grid) * strides[i]
            wh = (wh * 2) ** 2 * np.array([anchor_w, anchor_h])
            
            x1y1 = xy - wh / 2
            x2y2 = xy + wh / 2
            boxes = np.concatenate([x1y1, x2y2], axis=-1)
            
            scores = obj * cls
            
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1, num_classes)
            mask_coeffs = mask_pred.reshape(-1, num_mask_coeffs)
            
            class_ids = np.argmax(scores, axis=1)
            class_scores = np.max(scores, axis=1)
            
            mask = class_scores > conf_threshold
            if np.any(mask):
                all_boxes.append(boxes[mask])
                all_scores.append(class_scores[mask])
                all_classes.append(class_ids[mask])
                all_mask_coeffs.append(mask_coeffs[mask])
    
    if not all_boxes:
        return [], [], [], []
    
    boxes = np.concatenate(all_boxes)
    scores = np.concatenate(all_scores)
    classes = np.concatenate(all_classes)
    mask_coeffs = np.concatenate(all_mask_coeffs)
    
    h_img, w_img = img_shape
    boxes[:, 0] = np.clip(boxes[:, 0], 0, w_img)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, h_img)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, w_img)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, h_img)
    
    # NMS per class (keeping mask coefficients aligned)
    nboxes, nclasses, nscores, nmask_coeffs = [], [], [], []
    for c in set(classes):
        inds = np.where(classes == c)[0]
        b = boxes[inds]
        s = scores[inds]
        mc = mask_coeffs[inds]
        keep = nms(b, s, iou_threshold)
        if len(keep) > 0:
            nboxes.append(b[keep])
            nclasses.append(np.full(len(keep), c))
            nscores.append(s[keep])
            nmask_coeffs.append(mc[keep])
    
    if not nboxes:
        return [], [], [], []
    
    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)
    mask_coeffs = np.concatenate(nmask_coeffs)
    
    # Generate masks from prototypes: mask = sigmoid(coeffs @ protos)
    # proto shape: (1, 32, 160, 160) -> (32, 160*160)
    proto_h, proto_w = proto.shape[2], proto.shape[3]
    protos = proto[0].reshape(32, -1)  # (32, 25600)
    
    # mask_coeffs: (N, 32), protos: (32, 25600) -> masks: (N, 25600)
    masks_flat = mask_coeffs @ protos  # (N, 25600)
    masks_flat = 1 / (1 + np.exp(-masks_flat))  # sigmoid
    masks_proto = masks_flat.reshape(-1, proto_h, proto_w)  # (N, 160, 160)
    
    # Crop and resize masks to bounding boxes at original image size
    masks = []
    scale_x = proto_w / w_img
    scale_y = proto_h / h_img
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        # Scale box to proto size
        px1 = int(max(0, x1 * scale_x))
        py1 = int(max(0, y1 * scale_y))
        px2 = int(min(proto_w, x2 * scale_x))
        py2 = int(min(proto_h, y2 * scale_y))
        
        if px2 <= px1 or py2 <= py1:
            masks.append(np.zeros((int(y2 - y1), int(x2 - x1)), dtype=np.uint8))
            continue
        
        # Crop mask from proto resolution
        mask_crop = masks_proto[i, py1:py2, px1:px2]
        
        # Resize to box size
        box_h = max(1, int(y2 - y1))
        box_w = max(1, int(x2 - x1))
        mask_resized = cv2.resize(mask_crop, (box_w, box_h), interpolation=cv2.INTER_LINEAR)
        
        # Threshold to binary
        mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
        masks.append(mask_binary)
    
    return boxes, classes, scores, masks


def process_output(output, conf_threshold=0.25, iou_threshold=0.45, img_shape=(640, 640)):
    """
    Process RKNN model output to get detections.
    Supports YOLOv5 (3 outputs), YOLOv8/v11 (6 or 9 outputs), and single-output formats.
    """
    if not isinstance(output, list):
        output = [output]
    
    # Detect output format based on number and shape of outputs
    # YOLOv5: 3 outputs with shape (1, 255, H, W)
    # YOLOv5-seg: 7 outputs (3 det + 3 mask coeff + 1 proto)
    # YOLOv8/v11: 6 or 9 outputs
    masks = None  # Will be set for segmentation models
    
    if len(output) == 7 and output[0].shape[1] == 255:
        # YOLOv5-seg format: full segmentation with masks
        boxes, classes, scores, masks = post_process_yolov5_seg(output, conf_threshold, iou_threshold, img_shape)
    elif len(output) == 3 and output[0].shape[1] == 255:
        # YOLOv5 format (3 anchor-based outputs)
        boxes, classes, scores = post_process_yolov5(output, conf_threshold, iou_threshold, img_shape)
    elif len(output) == 6 or len(output) == 9:
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
        det = {
            'bbox': boxes[i].astype(int),
            'score': float(scores[i]),
            'class_id': int(classes[i]),
            'class_name': COCO_CLASSES[classes[i]] if classes[i] < len(COCO_CLASSES) else f'class_{classes[i]}'
        }
        # Include mask if available (segmentation model)
        if masks is not None and i < len(masks):
            det['mask'] = masks[i]
        detections.append(det)
    
    return detections


# Color palette for segmentation masks (different color per class)
MASK_COLORS = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29), (207, 210, 49),
    (72, 249, 10), (146, 204, 23), (61, 219, 134), (26, 147, 52), (0, 212, 187),
    (44, 153, 168), (0, 194, 255), (52, 69, 147), (100, 115, 255), (0, 24, 236),
    (132, 56, 255), (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199)
]


def draw_detections(img, detections):
    """Draw bounding boxes, labels, and segmentation masks on image."""
    # First draw masks (so boxes appear on top)
    overlay = img.copy()
    for det in detections:
        if 'mask' in det:
            x1, y1, x2, y2 = det['bbox']
            mask = det['mask']
            class_id = det['class_id']
            color = MASK_COLORS[class_id % len(MASK_COLORS)]
            
            # Create colored mask
            h, w = mask.shape[:2]
            if h > 0 and w > 0:
                # Ensure mask fits in image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
                actual_w, actual_h = x2 - x1, y2 - y1
                
                if actual_w > 0 and actual_h > 0:
                    # Resize mask if needed
                    if mask.shape[0] != actual_h or mask.shape[1] != actual_w:
                        mask = cv2.resize(mask, (actual_w, actual_h))
                    
                    # Apply colored mask where mask > 0
                    mask_region = overlay[y1:y2, x1:x2]
                    mask_bool = mask > 127
                    mask_region[mask_bool] = (
                        mask_region[mask_bool] * 0.5 + np.array(color) * 0.5
                    ).astype(np.uint8)
    
    # Blend overlay with original
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    
    # Then draw boxes and labels
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        score = det['score']
        class_name = det['class_name']
        class_id = det['class_id']
        color = MASK_COLORS[class_id % len(MASK_COLORS)] if 'mask' in det else (0, 255, 0)
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{class_name} {score:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - text_height - baseline - 5), (x1 + text_width, y1), color, -1)
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
    camera = None
    cap = None  # Keep for video files
    
    if source.isdigit():
        # Camera - use ThreadedCamera for async capture (eliminates capture wait time)
        source = int(source)
        
        # Optimized camera config: use 640x480 since we resize to 640x640 anyway
        # Lower resolution = faster MJPEG decode and less data transfer
        camera_config = CAMERA_CONFIG.copy()
        camera_config.update({
            "width": 640,
            "height": 480,
            "fps": 30,
            "fourcc": "MJPG"
        })
        
        try:
            # ThreadedCamera captures in background thread - read_frame() returns immediately
            camera = ThreadedCamera(index=source, config=camera_config)
            camera.open()
            
            # Test if we can actually read a frame
            test_frame = camera.read_frame()
            if test_frame is None:
                print(f"[ERROR] Camera {source} opened but cannot read frames. Check camera connection.", file=sys.stderr)
                camera.close()
                rknn.release()
                return 1
            
            print(f"[INFO] Using ThreadedCamera {source} (resolution: {camera.width}x{camera.height})", file=sys.stderr)
        except RuntimeError as e:
            print(f"[ERROR] {e}", file=sys.stderr)
            rknn.release()
            return 1
        
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
    if not args.no_display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    frame_count = 0
    prev_time = time.time()
    
    # Pre-allocate buffers for better performance
    img_input_buffer = None
    
    # Timing diagnostics
    total_time = 0
    capture_time = 0
    preprocess_time = 0
    inference_time_total = 0
    postprocess_time = 0
    display_time = 0
    
    try:
        while True:
            loop_start = time.time()
            
            # Get frame
            capture_start = time.time()
            if is_video:
                if camera is not None:
                    # Use Camera class
                    frame = camera.read_frame()
                    if frame is None:
                        # print("[WARN] Failed to read frame from camera. Retrying...")
                        time.sleep(0.1)
                        continue
                else:
                    # Fallback for video files
                    ret, frame = cap.read()
                    if not ret:
                        # print("[WARN] Failed to read frame from camera. Retrying...")
                        time.sleep(0.1)
                        continue
                    if frame is None:
                        # print("[WARN] Received empty frame. Retrying...")
                        time.sleep(0.1)
                        continue
                
                # Skip frames if requested (for faster processing)
                if args.skip_frames > 0 and frame_count % (args.skip_frames + 1) != 0:
                    frame_count += 1
                    continue
            else:
                frame = img.copy()
            capture_time += (time.time() - capture_start) * 1000
            
            # Preprocess (optimized: avoid unnecessary copies)
            preprocess_start = time.time()
            img_resized, ratio, (dw, dh) = letterbox(frame, new_shape=(args.imgsz, args.imgsz))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            
            # RKNN expects 4D input: (batch, height, width, channels) for NHWC format
            # Pre-allocate buffer if not exists, or reuse if same size
            if img_input_buffer is None or img_input_buffer.shape != (1, args.imgsz, args.imgsz, 3):
                img_input_buffer = np.zeros((1, args.imgsz, args.imgsz, 3), dtype=np.uint8)
            img_input_buffer[0] = img_rgb.astype(np.uint8)
            img_input = img_input_buffer
            preprocess_time += (time.time() - preprocess_start) * 1000
            
            # Run inference
            inference_start = time.time()
            try:
                outputs = rknn.inference([img_input])
            except Exception as e:
                print(f"[ERROR] Inference failed: {e}", file=sys.stderr)
                continue
            inference_time_ms = (time.time() - inference_start) * 1000  # ms
            inference_time_total += inference_time_ms
            
            # Process output
            if outputs is None:
                # print("[WARN] Inference returned None, skipping frame")
                continue
            
            postprocess_start = time.time()
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
            
            # Scale boxes back to original image size (vectorized for speed)
            if detections:
                h_orig, w_orig = frame.shape[:2]
                scale = min(args.imgsz / w_orig, args.imgsz / h_orig)
                new_w = int(w_orig * scale)
                new_h = int(h_orig * scale)
                pad_x = (args.imgsz - new_w) / 2
                pad_y = (args.imgsz - new_h) / 2
                
                # Vectorized box scaling (much faster than loop)
                boxes = np.array([det['bbox'] for det in detections], dtype=np.float32)
                boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
                boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
                boxes = boxes.astype(np.int32)
                
                for i, det in enumerate(detections):
                    det['bbox'] = boxes[i].tolist()
            postprocess_time += (time.time() - postprocess_start) * 1000
            
            # Draw detections and display (skip if no-display mode)
            display_start = time.time()
            if not args.no_display:
                annotated = draw_detections(frame.copy(), detections)
                
                # Calculate FPS
                now = time.time()
                fps = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now
                
                # Add info text with detailed timing
                info_text = f"FPS: {fps:.1f} | Inf: {inference_time_ms:.1f}ms | Det: {len(detections)}"
                cv2.putText(annotated, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display
                cv2.imshow(window_name, annotated)
                
                # Exit on 'q' or Esc (include waitKey in display timing)
                key = cv2.waitKey(1) & 0xFF
                display_time += (time.time() - display_start) * 1000
                if key in (27, ord('q')):
                    break
            else:
                # Still calculate FPS for monitoring
                now = time.time()
                fps = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now
                # In no-display mode, check for keyboard interrupt
                time.sleep(0.001)  # Small sleep to allow interrupt
                display_time += (time.time() - display_start) * 1000
            
            loop_time = (time.time() - loop_start) * 1000
            total_time += loop_time
            frame_count += 1
            
            # Print timing breakdown every 60 frames
            if frame_count % 60 == 0:
                avg_capture = capture_time / frame_count
                avg_preprocess = preprocess_time / frame_count
                avg_inference = inference_time_total / frame_count
                avg_postprocess = postprocess_time / frame_count
                avg_display = display_time / frame_count
                avg_total = total_time / frame_count
                avg_other = avg_total - avg_capture - avg_preprocess - avg_inference - avg_postprocess - avg_display
                print(f"[TIMING] Frame {frame_count}: Cap={avg_capture:.1f}ms | "
                      f"Pre={avg_preprocess:.1f}ms | Inf={avg_inference:.1f}ms | "
                      f"Post={avg_postprocess:.1f}ms | Disp={avg_display:.1f}ms | "
                      f"Other={avg_other:.1f}ms | Total={avg_total:.1f}ms ({1000/avg_total:.1f} FPS)", file=sys.stderr)
            
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
        # Print final timing summary
        if frame_count > 0:
            avg_capture = capture_time / frame_count
            avg_preprocess = preprocess_time / frame_count
            avg_inference = inference_time_total / frame_count
            avg_postprocess = postprocess_time / frame_count
            avg_display = display_time / frame_count
            avg_total = total_time / frame_count
            avg_other = avg_total - avg_capture - avg_preprocess - avg_inference - avg_postprocess - avg_display
            print(f"\n[TIMING SUMMARY] Processed {frame_count} frames:", file=sys.stderr)
            print(f"  Capture:     {avg_capture:.2f}ms ({100*avg_capture/avg_total:.1f}%)", file=sys.stderr)
            print(f"  Preprocess:  {avg_preprocess:.2f}ms ({100*avg_preprocess/avg_total:.1f}%)", file=sys.stderr)
            print(f"  Inference:   {avg_inference:.2f}ms ({100*avg_inference/avg_total:.1f}%)", file=sys.stderr)
            print(f"  Postprocess: {avg_postprocess:.2f}ms ({100*avg_postprocess/avg_total:.1f}%)", file=sys.stderr)
            print(f"  Display:     {avg_display:.2f}ms ({100*avg_display/avg_total:.1f}%)", file=sys.stderr)
            print(f"  Other:       {avg_other:.2f}ms ({100*avg_other/avg_total:.1f}%)", file=sys.stderr)
            print(f"  Total:       {avg_total:.2f}ms ({1000/avg_total:.2f} FPS)", file=sys.stderr)
        
        # Clean up camera resources
        if camera is not None:
            camera.close()
        if cap is not None:
            cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        rknn.release()
        # print("[INFO] Released RKNN resources")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

