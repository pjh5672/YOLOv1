import math
import random

import cv2
import torch
import numpy as np
import torchvision.transforms.functional as TF


IMAGENET_MEAN = 0.485, 0.456, 0.406  # IMAGENET MEAN on RGB space
IMAGENET_STD = 0.229, 0.224, 0.225  # IMAGENET STD on RGB space


def to_tensor(image, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    image = np.ascontiguousarray(image.transpose(2, 0, 1))
    image = torch.from_numpy(image).float()
    tensor = TF.normalize(image / 255, mean, std)
    return tensor


def to_image(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
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



class BasicTransform:
    def __init__(self, input_size):
        self.input_size = input_size


    def __call__(self, image, label):
        image, label = to_square_image(image, label)
        image = cv2.resize(image, dsize=(self.input_size, self.input_size))
        tensor = to_tensor(image)
        return tensor, label



if __name__ == "__main__":
    import sys
    from pathlib import Path
    from dataset import Dataset

    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[1]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    
    from utils import visualize_target, generate_random_color, clip_box_coordinate

    yaml_path = ROOT / 'data' / 'toy.yaml'
    input_size = 448
    train_dataset = Dataset(yaml_path=yaml_path, phase='train', input_size=input_size)
    transformer = BasicTransform(input_size=input_size)
    class_list = train_dataset.class_list
    color_list = generate_random_color(len(class_list))

    index = np.random.randint(0, len(train_dataset))
    filename, image, label = train_dataset.get_GT_item(index)
    input_tensor, label = transformer(image=image, label=label)
    image = to_image(input_tensor)
    image_with_bbox = visualize_target(image, label, class_list, color_list)
    print(filename, input_tensor.shape, label)
    cv2.imwrite('./asset/train-data.png', image_with_bbox)
