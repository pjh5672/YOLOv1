import os
import sys
import json
from pathlib import Path

import cv2
import yaml
import torch
import numpy as np
from tqdm import tqdm

from transform import *

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils import clip_box_coordinate, transform_xcycwh_to_x1y1wh



class Dataset:
    def __init__(self, yaml_path, phase):
        with open(yaml_path, mode="r") as f:
            data_item = yaml.load(f, Loader=yaml.FullLoader)
        self.phase = phase
        self.class_list = data_item["CLASS_INFO"]

        self.image_paths = []
        for sub_dir in data_item[self.phase.upper()]:
            image_dir = Path(data_item["PATH"]) / sub_dir
            self.image_paths += [str(image_dir / fn) for fn in os.listdir(image_dir) if fn.lower().endswith(("png", "jpg", "jpeg"))]
        self.label_paths = self.replace_image2label_path(self.image_paths)
        self.generate_no_label(self.label_paths)
        
        self.mAP_file_path = None
        if phase == "val":
            self.generate_mAP_source(save_dir=Path("./data/eval_src"), mAP_file_name=data_item["MAP_FILE_NAME"])


    def __len__(self): return len(self.image_paths)


    def __getitem__(self, index):
        filename, image, label = self.get_GT_item(index)
        input_tensor, label = self.transformer(image=image, label=label)
        label[:, 1:5] = clip_box_coordinate(label[:, 1:5])
        label = torch.from_numpy(label)
        ori_img_size = image.shape
        return filename, input_tensor, label, ori_img_size

    
    def get_GT_item(self, index):
        filename, image = self.get_image(index)
        label = self.get_label(index)
        label = self.check_no_label(label)
        return filename, image, label


    def get_image(self, index):
        filename = self.image_paths[index].split(os.sep)[-1]
        image = cv2.imread(self.image_paths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return filename, image


    def get_label(self, index):
        with open(self.label_paths[index], mode="r") as f:
            item = [x.split() for x in f.read().splitlines()]
        return np.array(item, dtype=np.float32)


    def replace_image2label_path(self, image_paths):
        sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
        return [sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt" for x in image_paths]


    def generate_no_label(self, label_paths):
        for label_path in label_paths:
            if not os.path.isfile(label_path):
                f = open(str(label_path), mode="w")
                f.close()


    def check_no_label(self, label):
        if len(label) == 0:
            label = np.array([[-1, 0.5, 0.5, 1., 1.]], dtype=np.float32)
        return label


    def load_transformer(self, transformer):
        self.transformer = transformer


    def generate_mAP_source(self, save_dir, mAP_file_name):
        if not save_dir.is_dir():
            os.makedirs(save_dir, exist_ok=True)
        self.mAP_file_path = save_dir / mAP_file_name
        
        if not self.mAP_file_path.is_file():
            class_id2category = self.class_list
            category2class_id = {t:l for l,t in class_id2category.items()}
    
            cocoAPI_formatter = {}
            cocoAPI_formatter["imageToid"] = {}
            cocoAPI_formatter["images"] = []
            cocoAPI_formatter["annotations"] = []
            cocoAPI_formatter["categories"] = []

            lbl_id = 0
            for i in tqdm(range(len(self))):
                filename, image, label = self.get_GT_item(i)
                img_h, img_w = image.shape[:2]
                cocoAPI_formatter["imageToid"][filename] = i
                cocoAPI_formatter["images"].append({"id": i, "width": img_w, "height": img_h})
                
                label[:, 1:5] = transform_xcycwh_to_x1y1wh(label[:, 1:5])
                label[:, [1,3]] *= img_w
                label[:, [2,4]] *= img_h
                for j in range(len(label)):
                    x = {}
                    x["id"] = lbl_id
                    x["image_id"] = i
                    x["bbox"] = [round(item, 2) for item in label[j][1:5].tolist()]
                    x["area"] = round((x["bbox"][2] * x["bbox"][3]), 2)
                    x["iscrowd"] = 0
                    x["category_id"] = int(label[j][0])
                    x["segmentation"] = []
                    cocoAPI_formatter["annotations"].append(x)
                    lbl_id += 1

            for i, cate_name in class_id2category.items():
                cocoAPI_formatter["categories"].append({"id": i, "supercategory": "", "name": cate_name})

            with open(self.mAP_file_path, "w") as outfile:
                json.dump(cocoAPI_formatter, outfile)


    @staticmethod
    def collate_fn(minibatch):
        filenames = []
        images = []
        labels = []
        ori_img_sizes = []
        
        for index, items in enumerate(minibatch):
            filenames.append(items[0])
            images.append(items[1])
            labels.append(items[2])
            ori_img_sizes.append(items[3])
        return filenames, torch.stack(images, dim=0), labels, ori_img_sizes



if __name__ == "__main__":
    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[1]
    
    yaml_path = ROOT / 'data' / 'toy.yaml'
    input_size = 224
    transformer = BasicTransform(input_size=input_size)
    # transformer = AugmentTransform(input_size=input_size)
    
    train_dataset = Dataset(yaml_path=yaml_path, phase='train')
    train_dataset.load_transformer(transformer=transformer)
    val_dataset = Dataset(yaml_path=yaml_path, phase='val')
    val_dataset.load_transformer(transformer=transformer)

    for index, minibatch in enumerate(train_dataset):
        filename, image, label, ori_img_size = train_dataset[index]
        print(filename, label, image.shape, ori_img_size)
    
    for index, minibatch in enumerate(val_dataset):
        filename, image, label, ori_img_size = val_dataset[index]
        print(filename, label, image.shape, ori_img_size)