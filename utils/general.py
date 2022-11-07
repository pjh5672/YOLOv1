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


def hard_NMS(prediction, iou_threshold):
    x1 = prediction[:, 1]
    y1 = prediction[:, 2]
    x2 = prediction[:, 3]
    y2 = prediction[:, 4]
    areas = (x2 - x1) * (y2 - y1)
    order = prediction[:, -1].argsort()[::-1]

    pick = []
    while len(order) > 0:
        pick.append(order[0])
        if len(order) == 1:
            break
        xx1 = np.maximum(x1[order[0]], x1[order[1:]])
        yy1 = np.maximum(y1[order[0]], y1[order[1:]])
        xx2 = np.minimum(x2[order[0]], x2[order[1:]])
        yy2 = np.minimum(y2[order[0]], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        overlap = (w * h)
        ious = overlap / (areas[order[0]] + areas[order[1:]] - overlap + 1e-8)
        order = order[np.where(ious <= iou_threshold)[0] + 1]
    return pick


def run_NMS(prediction, iou_threshold, class_agnostic=False):
    if len(prediction) == 0:
        return []

    if class_agnostic:
        pick = hard_NMS(prediction=prediction, iou_threshold=iou_threshold)
        return prediction[pick]

    prediction_multi_class = []
    for cls_id in np.unique(prediction[:, 0]):
        pred_per_cls_id = prediction[prediction[:, 0] == cls_id]
        pick_per_cls_id = hard_NMS(prediction=pred_per_cls_id, iou_threshold=iou_threshold)
        prediction_multi_class.append(pred_per_cls_id[pick_per_cls_id])
    prediction_multi_class = np.concatenate(prediction_multi_class, axis=0)
    order = prediction_multi_class[:, -1].argsort()[::-1]
    return prediction_multi_class[order]


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