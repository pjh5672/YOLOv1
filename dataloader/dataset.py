import os
import sys
import json
from pathlib import Path

import cv2
import yaml
import torch
import numpy as np
from tqdm import tqdm

from transform import BasicTransform, AugmentTransform, to_tensor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils import transform_xcycwh_to_x1y1wh



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
        
        self.mAP_filepath = None
        if phase == "val":
            self.generate_mAP_source(save_dir=Path("./data/eval_src"), mAP_filename=data_item["VAL_FILE"])


    def __len__(self): return len(self.image_paths)


    def __getitem__(self, index):
        filename, image, label = self.get_GT_item(index)
        shape = image.shape
        image, boxes, labels = self.transformer(image=image, boxes=label[:, 1:5], labels=label[:, 0])
        img_tensor = to_tensor(image)
        label = torch.from_numpy(np.concatenate((labels[:, np.newaxis], boxes), axis=1))
        return filename, img_tensor, label, shape

    
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
            label = np.array([[-1, 0, 0, 0, 0]], dtype=np.float32)
        return label


    def load_transformer(self, transformer):
        self.transformer = transformer


    def generate_mAP_source(self, save_dir, mAP_filename):
        if not save_dir.is_dir():
            os.makedirs(save_dir, exist_ok=True)
        self.mAP_filepath = save_dir / mAP_filename
        
        if not self.mAP_filepath.is_file():
            class_id2category = self.class_list
    
            mAP_file_formatter = {}
            mAP_file_formatter["imageToid"] = {}
            mAP_file_formatter["images"] = []
            mAP_file_formatter["annotations"] = []
            mAP_file_formatter["categories"] = []

            lbl_id = 0
            for i in tqdm(range(len(self))):
                filename, image, label = self.get_GT_item(i)
                img_h, img_w = image.shape[:2]
                mAP_file_formatter["imageToid"][filename] = i
                mAP_file_formatter["images"].append({"id": i, "width": img_w, "height": img_h})
                
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
                    mAP_file_formatter["annotations"].append(x)
                    lbl_id += 1

            for i, cate_name in class_id2category.items():
                mAP_file_formatter["categories"].append({"id": i, "supercategory": "", "name": cate_name})

            with open(self.mAP_filepath, "w") as outfile:
                json.dump(mAP_file_formatter, outfile)


    @staticmethod
    def collate_fn(minibatch):
        filenames = []
        images = []
        labels = []
        shapes = []
        
        for _, items in enumerate(minibatch):
            filenames.append(items[0])
            images.append(items[1])
            labels.append(items[2])
            shapes.append(items[3])
        return filenames, torch.stack(images, dim=0), labels, shapes


if __name__ == "__main__":
    yaml_path = ROOT / 'data' / 'toy.yaml'
    input_size = 448
    
    train_dataset = Dataset(yaml_path=yaml_path, phase='train')
    train_transformer = AugmentTransform(input_size=input_size)
    train_dataset.load_transformer(transformer=train_transformer)
    val_dataset = Dataset(yaml_path=yaml_path, phase='val')
    val_transformer = BasicTransform(input_size=input_size)
    val_dataset.load_transformer(transformer=val_transformer)

    print(len(train_dataset), len(val_dataset))
    for index, minibatch in enumerate(train_dataset):
        filename, image, label, shape = train_dataset[index]
    print(f"train dataset sanity-check done")

    for index, minibatch in enumerate(val_dataset):
        filename, image, label, shape = val_dataset[index]
    print(f"val dataset sanity-check done")
