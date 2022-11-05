import os
import glob
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET


class_list = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", 
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]


def xml_to_yolo_bbox(bbox, img_w, img_h):
    x_center = ((bbox[0] + bbox[2]) / 2) / img_w
    y_center = ((bbox[1] + bbox[3]) / 2) / img_h
    width = (bbox[2] - bbox[0]) / img_w
    height = (bbox[3] - bbox[1]) / img_h
    return [x_center, y_center, width, height]


def parse_content_from_xml(filepath, class_list):
    res = []
    tree = ET.parse(filepath)
    root = tree.getroot()
    width = int(root.find("size").find("width").text)
    height = int(root.find("size").find("height").text)

    for obj in root.findall("object"):
        label = obj.find("name").text
        if label not in class_list:
            class_list.append(label)
        index = class_list.index(label)
        bbox = [
            max(float(obj.find("bndbox").find("xmin").text), 0),
            max(float(obj.find("bndbox").find("ymin").text), 0),
            min(float(obj.find("bndbox").find("xmax").text), width),
            min(float(obj.find("bndbox").find("ymax").text), height),
        ]
        yolo_bbox = xml_to_yolo_bbox(bbox, width, height)
        res.append(f"{index} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}")
    return res


def convert_voc_to_yolo(data_dir, output_dir):
    image_dir = Path(data_dir) / "JPEGImages"
    anno_dir = Path(data_dir) / "Annotations"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir.replace("labels", "images"), exist_ok=True)

    files = glob.glob(str(anno_dir / '*.xml'))
    cnt = 0
    for file in files:
        filename = os.path.basename(file)
        imgname = filename.replace("xml", "jpg")
        txtname = filename.replace("xml", "txt")

        if (image_dir / f"{imgname}").is_file():
            shutil.copy(src=image_dir / f"{imgname}", dst=os.path.join(output_dir.replace("labels", "images"), imgname))
        else:
            print(f" {imgname} not exist ! ")
            cnt += 1
            continue

        out = parse_content_from_xml(filepath=file, class_list=class_list)
        with open(os.path.join(output_dir, f"{txtname}"), "w", encoding="utf-8") as f:
            f.write("\n".join(out))
    print(f" Complete! (Total {len(files)-cnt} images found, {cnt} not found.) ")



if __name__ == "__main__":
    data_dirs = [
        "D:/DATASET/PASCAL-VOC/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007",
        "D:/DATASET/PASCAL-VOC/VOCtrainval_11-May-2012/VOCdevkit/VOC2012",
        "D:/DATASET/PASCAL-VOC/VOCtest_06-Nov-2007/VOCdevkit/VOC2007",
        ]
    output_dirs = [
        "D:/DATASET/PASCAL-VOC/labels/trainval2007",
        "D:/DATASET/PASCAL-VOC/labels/trainval2012",
        "D:/DATASET/PASCAL-VOC/labels/test2007",
        ]

    for data_dir, output_dir in zip(data_dirs, output_dirs):
        convert_voc_to_yolo(data_dir=data_dir, output_dir=output_dir)