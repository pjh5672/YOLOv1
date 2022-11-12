import sys
import random
from pathlib import Path

import cv2
import torch
import numpy as np
import albumentations as A
import torchvision.transforms.functional as TF

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils import transform_xcycwh_to_x1y1x2y2, transform_x1y1x2y2_to_xcycwh, scale_to_original, scale_to_norm


MEAN = 0.485, 0.456, 0.406 # RGB
STD = 0.229, 0.224, 0.225 # RGB


class BasicTransform(object):
    def __init__(self, input_size):
        self.input_size = input_size

    def __call__(self, image, label):
        image = cv2.resize(image, (self.input_size, self.input_size)).astype(np.float32)
        tensor = to_tensor(image)
        return tensor, label


class AugmentTransform:
    def __init__(self, input_size):
        self.input_size = input_size
        self.flip = Albumentations(p_flipud=0.0, p_fliplr=0.5)
        self.gain_h = 0.015
        self.gain_s = 0.5
        self.gain_v = 0.5
        self.degrees = 0
        self.translate = 0.3
        self.scale = 0.3
        self.perspective = 0.0001
    

    def __call__(self, image, label):
        img_h, img_w = image.shape[:2]
        crop_size = random.randint(int(min(img_h, img_w) * 0.8), int(min(img_h, img_w) * 1.0))
        image, label = self.flip(image=image, label=label)
        image = augment_hsv(image, gain_h=self.gain_h, gain_s=self.gain_s, gain_v=self.gain_v)
        label[:, 1:5] = transform_xcycwh_to_x1y1x2y2(label[:, 1:5])
        label[:, 1:5] = scale_to_original(label[:, 1:5], scale_w=img_w, scale_h=img_h)
        image, label = random_crop(image, label, crop_size=crop_size, area_thres=0.25)
        image, label = random_perspective(image, label, size=crop_size, 
                                          degrees=self.degrees, translate=self.translate, 
                                          scale=self.scale, perspective=self.perspective)
        label[:, 1:5] = scale_to_norm(label[:, 1:5], image_w=crop_size, image_h=crop_size)
        label[:, 1:5] = transform_x1y1x2y2_to_xcycwh(label[:, 1:5])
        image = cv2.resize(image, dsize=(self.input_size, self.input_size))
        tensor = to_tensor(image)
        return tensor, label


def to_tensor(image, mean=MEAN, std=STD):
    image = np.ascontiguousarray(image.transpose(2, 0, 1))
    image = torch.from_numpy(image).float()
    tensor = TF.normalize(image / 255, mean, std)
    return tensor


def to_image(tensor, mean=MEAN, std=STD):
    denorm_tensor = tensor.clone()
    for t, m, s in zip(denorm_tensor, mean, std):
        t.mul_(s).add_(m)
    denorm_tensor.clamp_(min=0, max=1.)
    denorm_tensor *= 255
    image = denorm_tensor.permute(1,2,0).numpy().astype(np.uint8)
    return image


def to_square_image(image, label):
    pad_h, pad_w = 0, 0
    img_h, img_w, img_c = image.shape
    dtype = image.dtype
    max_size = max(img_h, img_w)

    if img_h < max_size:
        pad_h = max_size - img_h
    if img_w < max_size:
        pad_w = max_size - img_w
    
    square_image = np.zeros(shape=(img_h + pad_h, img_w + pad_w, img_c), dtype=dtype)
    square_image[0:img_h, 0:img_w, :] = image
    label[:, [1,3]] *= (img_w / (img_w + pad_w))
    label[:, [2,4]] *= (img_h / (img_h + pad_h))
    return square_image, label


class Albumentations:
    def __init__(self, p_flipud=0.0, p_fliplr=0.5):
        self.transform = A.Compose([
            A.Blur(p=0.01),
            A.MedianBlur(p=0.01),
            A.ToGray(p=0.01),
            A.CLAHE(p=0.01),
            A.VerticalFlip(p=p_flipud),
            A.HorizontalFlip(p=p_fliplr),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_ids']))


    def __call__(self, image, label):
        transform_data = self.transform(image=image, bboxes=label[:, 1:5], class_ids=label[:, 0])
        image = transform_data['image']
        bboxes = np.array(transform_data['bboxes'], dtype=np.float32)
        class_ids = np.array(transform_data['class_ids'], dtype=np.float32)
        label = np.concatenate((class_ids[:, np.newaxis], bboxes), axis=1)
        return image, label


def augment_hsv(image, gain_h=0.5, gain_s=0.5, gain_v=0.5):
    r = np.random.uniform(-1, 1, 3) * [gain_h, gain_s, gain_v] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))
    dtype = image.dtype  # uint8
    x = np.arange(0, 256, dtype=r.dtype)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
    image_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    return cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)


def random_crop(image, label, crop_size, area_thres=0.2):
    img_h, img_w, _ = image.shape
    x1 = random.randint(0, img_w - crop_size)
    y1 = random.randint(0, img_h - crop_size)
    x2 = x1 + crop_size
    y2 = y1 + crop_size
    
    image = image[y1:y2, x1:x2, :]
    boxes = label[:, 1:].copy()
    crop_x1 = np.clip(boxes[:, 0] - x1, a_min=0, a_max=None)
    crop_y1 = np.clip(boxes[:, 1] - y1, a_min=0, a_max=None)
    crop_x2 = np.clip(boxes[:, 2] - x1, a_min=0, a_max=crop_size)
    crop_y2 = np.clip(boxes[:, 3] - y1, a_min=0, a_max=crop_size)
    crop_boxes = np.stack((crop_x1, crop_y1, crop_x2, crop_y2), axis=1)
    idx = box_candidates(box1=boxes.T, box2=crop_boxes.T, area_thres=area_thres)
    
    label = label[idx]
    label[:, 1:5] = crop_boxes[idx]
    if len(label) == 0:
        label = np.array([[-1, 0.5, 0.5, 1.0, 1.0]], dtype=np.float32)
    return image, label


def random_perspective(image, label, size, degrees=0, translate=0.1, scale=0.1, perspective=0.0):
    img_h, img_w = image.shape[:2]
    # Center
    C = np.eye(3)
    C[0, 2] = -img_w / 2  # x translation (pixels)
    C[1, 2] = -img_h / 2  # y translation (pixels)
    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)
    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)
    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * img_w  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * img_h  # y translation (pixels)
    # Combined rotation matrix
    M = T @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    image = cv2.warpPerspective(image, M, dsize=(size, size), borderValue=(0, 0, 0))

    n = len(label)
    xy = np.ones((n * 4, 3))
    xy[:, :2] = label[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
    xy = xy @ M.T  # transform
    xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # perspective rescale or affine
    x = xy[:, [0, 2, 4, 6]]
    y = xy[:, [1, 3, 5, 7]]
    new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
    new[:, [0, 2]] = new[:, [0, 2]].clip(0, size)
    new[:, [1, 3]] = new[:, [1, 3]].clip(0, size)

    idx = box_candidates(box1=label[:, 1:5].T * s, box2=new.T, wh_thres=4)
    label = label[idx]
    label[:, 1:5] = new[idx]
    return image, label


def box_candidates(box1, box2, wh_thres=4, ar_thres=20, area_thres=0.05, eps=1e-16):
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thres) & (h2 > wh_thres) & (w2 * h2 / (w1 * h1 + eps) > area_thres) & (ar < ar_thres)  # candidates



if __name__ == "__main__":
    from dataset import Dataset
    from utils import visualize_target, generate_random_color, clip_box_coordinate

    yaml_path = ROOT / 'data' / 'toy.yaml'
    input_size = 448
    train_dataset = Dataset(yaml_path=yaml_path, phase='train')
    # transformer = BasicTransform(input_size=input_size)
    transformer = AugmentTransform(input_size=input_size)
    train_dataset.load_transformer(transformer=transformer)
    class_list = train_dataset.class_list
    color_list = generate_random_color(len(class_list))

    index = np.random.randint(0, len(train_dataset))
    filename, image, label = train_dataset.get_GT_item(index)
    input_tensor, label = transformer(image=image, label=label)
    image = to_image(input_tensor)
    image_with_bbox = visualize_target(image, label, class_list, color_list)
    print(filename, input_tensor.shape, label)
    cv2.imwrite('./asset/augment.jpg', image_with_bbox)
