import numpy as np
import cv2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def nms(boxes, scores, iou_thres=0.45):
    idxs = np.argsort(scores)[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1:
            break
        ious = bbox_iou(boxes[i], boxes[idxs[1:]])
        idxs = idxs[1:][ious < iou_thres]
    return keep

def bbox_iou(box1, boxes):
    x1 = np.maximum(box1[0], boxes[:, 0])
    y1 = np.maximum(box1[1], boxes[:, 1])
    x2 = np.minimum(box1[2], boxes[:, 2])
    y2 = np.minimum(box1[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    union = (box1[2]-box1[0])*(box1[3]-box1[1]) + \
            (boxes[:, 2]-boxes[:, 0])*(boxes[:, 3]-boxes[:, 1]) - inter
    return inter / np.clip(union, 1e-6, None)

def yolov8_post_process(outs, conf_thres=0.15, nms_thres=0.45, img_size=640):
    # strides correspond to 8, 16, 32
    strides = [8, 16, 32]
    grid_sizes = [80, 40, 20]
    boxes_all, scores_all, classes_all = [], [], []

    for i in range(3):  # each detection scale
        # feature maps
        box = outs[i * 3]        # (1,64,h,w)
        cls = outs[i * 3 + 1]    # (1,80,h,w)
        obj = outs[i * 3 + 2]    # (1,1,h,w)
        h, w = grid_sizes[i], grid_sizes[i]

        # reshape and transpose to (h*w, channels)
        box = box.reshape(64, h * w).T
        cls = cls.reshape(80, h * w).T
        obj = obj.reshape(1, h * w).T

        obj_conf = sigmoid(obj)
        cls_conf = sigmoid(cls)
        scores = obj_conf * cls_conf
        max_scores = np.max(scores, axis=1)
        cls_ids = np.argmax(scores, axis=1)
        mask = max_scores > conf_thres
        if not np.any(mask):
            continue

        box = box[mask]
        cls_ids = cls_ids[mask]
        max_scores = max_scores[mask]

        # decode bbox coordinates
        # decode bbox coordinates (RKNN YOLOv8 grid alignment)
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        grid = np.stack((grid_x, grid_y), axis=-1).reshape(-1, 2)
        grid = grid[mask]

        cx = (box[:, 0] + grid[:, 0]) * strides[i]
        cy = (box[:, 1] + grid[:, 1]) * strides[i]
        bw = np.exp(box[:, 2]) * strides[i]
        bh = np.exp(box[:, 3]) * strides[i]

        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2


        boxes = np.stack((x1, y1, x2, y2), axis=1)
        boxes_all.append(boxes)
        scores_all.append(max_scores)
        classes_all.append(cls_ids)

    if not boxes_all:
        return [], [], []

    boxes_all = np.concatenate(boxes_all)
    scores_all = np.concatenate(scores_all)
    classes_all = np.concatenate(classes_all)
    keep = nms(boxes_all, scores_all, nms_thres)
    boxes_all = boxes_all[keep]
    scores_all = scores_all[keep]
    classes_all = classes_all[keep]

    return boxes_all, classes_all, scores_all
