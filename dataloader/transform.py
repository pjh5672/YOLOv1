import cv2
import torch
import numpy as np


MEAN = 0.485, 0.456, 0.406 # RGB
STD = 0.229, 0.224, 0.225 # RGB


def to_tensor(image):
    image = np.ascontiguousarray(image.transpose(2, 0, 1))
    return torch.from_numpy(image).float()


def to_image(tensor, mean=MEAN, std=STD):
    denorm_tensor = tensor.clone()
    for t, m, s in zip(denorm_tensor, mean, std):
        t.mul_(s).add_(m)
    denorm_tensor.clamp_(min=0, max=1.)
    denorm_tensor *= 255
    image = denorm_tensor.permute(1,2,0).numpy().astype(np.uint8)
    return image


def denormalize(image, mean=MEAN, std=STD):
    image *= std
    image += mean
    image *= 255.
    return image.astype(np.uint8)


class LetterBox:
    def __init__(self, new_shape=(448, 448), color=(0, 0, 0)):
        self.new_shape = (new_shape, new_shape) if isinstance(new_shape, int) else new_shape
        self.color = color

    def __call__(self, image, boxes, labels=None):
        # Resize and pad image while meeting stride-multiple constraints
        shape = image.shape[:2]  # current shape [height, width]
        # Scale ratio (new / old)
        r = min(self.new_shape[0] / shape[0], self.new_shape[1] / shape[1])
        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r)) # [width, height]
        dw, dh = self.new_shape[1] - new_unpad[0], self.new_shape[0] - new_unpad[1]  # wh padding
        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.color)
        boxes[:, :2] = (boxes[:, :2] * (new_unpad[0], new_unpad[1]) + (left, top))
        boxes[:, :2] /= (image.shape[1], image.shape[0])
        boxes[:, 2:] /= (image.shape[1] / new_unpad[0], image.shape[0] / new_unpad[1])
        return image, boxes, labels


class BasicTransform:
    def __init__(self, input_size, mean=MEAN, std=STD):
        mean = np.array(mean, dtype=np.float32)
        std = np.array(std, dtype=np.float32)
        self.tfs = Compose([
            LetterBox(new_shape=input_size),
            Normalize(mean=mean, std=std)
        ])

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image, boxes, labels = self.tfs(image, boxes, labels)
        return image, boxes, labels


class AugmentTransform:
    def __init__(self, input_size, mean=MEAN, std=STD):
        mean = np.array(mean, dtype=np.float32)
        std = np.array(std, dtype=np.float32)
        self.tfs = Compose([
            #### Photometric Augment ####
            RandomBrightness(),
            RandomContrast(),
            ConvertColor(color_from="RGB", color_to="HSV"),
            RandomHue(),
            RandomSaturation(),
            ConvertColor(color_from="HSV", color_to="RGB"),
            ##### Geometric Augment #####
            ToXminYminXmaxYmax(),
            ToAbsoluteCoords(),
            Expand(mean=mean),
            RandomSampleCrop(),
            HorizontalFlip(),
            ToPercentCoords(),
            ToXcenYcenWH(),
            #############################
            LetterBox(new_shape=input_size),
            Normalize(mean=mean, std=std)
        ])

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image, boxes, labels = self.tfs(image, boxes, labels)
        return image, boxes, labels


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, boxes=None, labels=None):
        for tf in self.transforms:
            image, boxes, labels = tf(image, boxes, labels)
        return image, boxes, labels


class Normalize:
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image /= 255.
        image -= self.mean
        image /= self.std
        return image, boxes, labels


class Resize:
    def __init__(self, size=640):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image, (self.size, self.size))
        return image, boxes, labels


class RandomBrightness:
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if np.random.randint(2):
            image += np.random.uniform(-self.delta, self.delta)
        return image, boxes, labels


class RandomContrast:
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None):
        if np.random.randint(2):
            image *= np.random.uniform(self.lower, self.upper)
        return image, boxes, labels


class ConvertColor:
    def __init__(self, color_from="RGB", color_to="HSV"):
        self.color_from = color_from
        self.color_to = color_to
        
    def __call__(self, image, boxes=None, labels=None):
        if self.color_from == "RGB" and self.color_to == "HSV":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif self.color_from == "HSV" and self.color_to == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        else:
            raise NotImplementedError
        return image, boxes, labels


class RandomHue:
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if np.random.randint(2):
            image[:, :, 0] += np.random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels


class RandomSaturation:
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None):
        if np.random.randint(2):
            image[:, :, 1] *= np.random.uniform(self.lower, self.upper)
        return image, boxes, labels


class ToAbsoluteCoords:
    def __call__(self, image, boxes, labels=None):
        height, width, _ = image.shape
        boxes[:, [0, 2]] *= width
        boxes[:, [1, 3]] *= height
        return image, boxes, labels


class ToPercentCoords:
    def __call__(self, image, boxes, labels=None):
        height, width, _ = image.shape
        boxes[:, [0, 2]] /= width
        boxes[:, [1, 3]] /= height
        return image, boxes, labels


class ToXminYminXmaxYmax:
    def __call__(self, image, boxes, labels=None):
        x1y1 = boxes[:, :2] - boxes[:, 2:] / 2
        x2y2 = boxes[:, :2] + boxes[:, 2:] / 2
        boxes = np.concatenate((x1y1, x2y2), axis=1).clip(min=0, max=1)
        return image, boxes, labels


class ToXcenYcenWH:
    def __call__(self, image, boxes, labels=None):
        wh = boxes[:, 2:] - boxes[:, :2]
        xcyc = boxes[:, :2] + wh / 2
        boxes = np.concatenate((xcyc, wh), axis=1).clip(min=0, max=1)
        return image, boxes, labels


class HorizontalFlip:
    def __call__(self, image, boxes, labels=None):
        _, width, _ = image.shape
        if np.random.randint(2):
            image = image[:, ::-1, :]
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes, labels


class Expand:
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, boxes, labels=None):
        if np.random.randint(2):
            return image, boxes, labels

        height, width, channel = image.shape
        scale = np.random.uniform(1, 4)
        left = np.random.uniform(0, width*scale - width)
        top = np.random.uniform(0, height*scale - height)
        
        expand_image = np.zeros((int(height*scale), int(width*scale), channel), dtype=image.dtype)
        expand_image[...] = self.mean
        expand_image[int(top):int(top+height), int(left):int(left+width), :] = image
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))
        return expand_image, boxes, labels


class RandomSampleCrop:
    def __init__(self):
        self.sample_option = (
            # use entire image
            None,
            # sample a patch s.t. MIN IoU w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, boxes, labels):
        height, width, _ = image.shape

        while True:
            sample_id = np.random.randint(len(self.sample_option))
            sample_mode = self.sample_option[sample_id]
            if sample_mode is None:
                return image, boxes, labels

            min_iou, max_iou = sample_mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')
            
            for _ in range(50):
                current_image = image
                w = np.random.uniform(0.3 * width, width)
                h = np.random.uniform(0.3 * height, height)
                if max(h,w) / min(h,w) > 2:
                    continue

                left = np.random.uniform(width - w)
                top = np.random.uniform(height - h)
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                overlap = self.compute_IoU(boxes, rect)
                if overlap.min() < min_iou and overlap.max() > max_iou:
                    continue

                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])
                mask = m1 * m2
                if not mask.any():
                    continue

                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2], :]
                current_boxes = boxes[mask, :]
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2], rect[:2])
                current_boxes[:, :2] -= rect[:2]
                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:], rect[2:])
                current_boxes[:, 2:] -= rect[:2]
                current_labels = labels[mask]
                return current_image, current_boxes, current_labels

    def compute_IoU(self, boxA, boxB):
        inter = self.intersect(boxA, boxB)
        areaA = ((boxA[:, 2]-boxA[:, 0]) * (boxA[:, 3]-boxA[:, 1]))
        areaB = ((boxB[2]-boxB[0]) * (boxB[3]-boxB[1]))
        union = areaA + areaB - inter
        return inter / union

    def intersect(self, boxA, boxB):
        max_xy = np.minimum(boxA[:, 2:], boxB[2:])
        min_xy = np.maximum(boxA[:, :2], boxB[:2])
        inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
        return inter[:, 0] * inter[:, 1]



if __name__ == "__main__":
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))

    from dataloader import Dataset
    from utils import visualize_target, generate_random_color

    yaml_path = ROOT / 'data' / 'toy.yaml'
    input_size = 448
    index = -1
    train_dataset = Dataset(yaml_path=yaml_path, phase='train')
    train_transformer = AugmentTransform(input_size=input_size)
    # train_transformer = BasicTransform(input_size=input_size)
    class_list = train_dataset.class_list
    color_list = generate_random_color(len(class_list))

    filename, image, label = train_dataset.get_GT_item(index)
    image, boxes, labels = train_transformer(image=image, boxes=label[:, 1:5], labels=label[:, 0])
    label = np.concatenate((labels[:, np.newaxis], boxes), axis=1)
    image = denormalize(image)
    image_with_bbox = visualize_target(image=image, label=label, class_list=class_list, color_list=color_list)

    cv2.imwrite(f'./asset/augment.jpg', image_with_bbox)
