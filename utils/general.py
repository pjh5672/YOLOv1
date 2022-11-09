import os
import cv2
import torch
import numpy as np


def set_grid(grid_size):
    grid_y, grid_x = torch.meshgrid((torch.arange(grid_size), torch.arange(grid_size)), indexing="ij")
    return (grid_x, grid_y)


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def scale_to_original(boxes, scale_w, scale_h):
    boxes[:,[0,2]] *= scale_w
    boxes[:,[1,3]] *= scale_h
    return boxes.round(2)


def scale_to_norm(boxes, image_w, image_h):
    boxes[:,[0,2]] /= image_w
    boxes[:,[1,3]] /= image_h
    return boxes


def clip_box_coordinate(boxes):
    boxes = transform_xcycwh_to_x1y1x2y2(boxes)
    boxes = transform_x1y1x2y2_to_xcycwh(boxes)
    return boxes


def transform_x1y1x2y2_to_x1y1wh(boxes):
    x1y1 = boxes[:, :2]
    wh = boxes[:, 2:] - boxes[:, :2]
    return np.concatenate((x1y1, wh), axis=1)


def transform_xcycwh_to_x1y1wh(boxes):
    x1y1 = boxes[:, :2] - boxes[:, 2:] / 2
    wh = boxes[:, 2:]
    return np.concatenate((x1y1, wh), axis=1).clip(min=0)


def transform_xcycwh_to_x1y1x2y2(boxes, clip_max=None):
    x1y1 = boxes[:, :2] - boxes[:, 2:] / 2
    x2y2 = boxes[:, :2] + boxes[:, 2:] / 2
    x1y1x2y2 = np.concatenate((x1y1, x2y2), axis=1)
    return x1y1x2y2.clip(min=0, max=clip_max if clip_max is not None else 1)


def transform_x1y1x2y2_to_xcycwh(boxes):
    wh = boxes[:, 2:] - boxes[:, :2]
    xcyc = boxes[:, :2] + wh / 2
    return np.concatenate((xcyc, wh), axis=1)


def filter_confidence(prediction, conf_threshold=0.01):
    keep = (prediction[:, 0] > conf_threshold)
    conf = prediction[:, 0][keep]
    box = prediction[:, 1:5][keep]
    cls_id = prediction[:, 5][keep]
    return np.concatenate([cls_id[:, np.newaxis], box, conf[:, np.newaxis]], axis=-1)


def hard_NMS(boxes, scores, iou_threshold):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []                                             
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(1e-10, xx2 - xx1)
        h = np.maximum(1e-10, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


def run_NMS(prediction, iou_threshold, maxDets=100):
    keep = np.zeros(len(prediction), dtype=np.int)
    for cls_id in np.unique(prediction[:, 0]):
        inds = np.where(prediction[:, 0] == cls_id)[0]
        if len(inds) == 0:
            continue
        cls_boxes = prediction[inds, 1:5]
        cls_scores = prediction[inds, 5]
        cls_keep = hard_NMS(boxes=cls_boxes, scores=cls_scores, iou_threshold=iou_threshold)
        keep[inds[cls_keep]] = 1
    prediction = prediction[np.where(keep > 0)]
    order = prediction[:, 5].argsort()[::-1]
    return prediction[order[:maxDets]]


def imwrite(filename, img):
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img)

        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False