import os
import sys
import json
import pprint
import platform
import argparse
import subprocess
from pathlib import Path
from collections import defaultdict

import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[0]
OS_SYSTEM = platform.system()
seed_num = 2023

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except:
    if OS_SYSTEM == 'Windows':
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pycocotools-windows'])
    elif OS_SYSTEM == 'Linux':
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pycocotools'])

from dataloader import Dataset, BasicTransform, to_image
from model import YoloModel
from utils import build_basic_logger, generate_random_color, transform_xcycwh_to_x1y1x2y2, \
                  filter_confidence, run_NMS, scale_to_original, transform_x1y1x2y2_to_x1y1wh, \
                  visualize_prediction, imwrite


METRIC_FORMAT = [
    "   {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ]".format("Average Precision", "(AP)", "0.50:0.95", "all", 100),
    "   {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ]".format("Average Precision", "(AP)", "0.50", "all", 100),
    "   {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ]".format("Average Precision", "(AP)", "0.75", "all", 100),
    "   {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ]".format("Average Precision", "(AP)", "0.50:0.95", "small", 100),
    "   {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ]".format("Average Precision", "(AP)", "0.50:0.95", "medium", 100),
    "   {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ]".format("Average Precision", "(AP)", "0.50:0.95", "large", 100),
    "   {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ]".format("Average Recall", "(AR)", "0.50:0.95", "all", 1),
    "   {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ]".format("Average Recall", "(AR)", "0.50:0.95", "all", 10),
    "   {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ]".format("Average Recall", "(AR)", "0.50:0.95", "all", 100),
    "   {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ]".format("Average Recall", "(AR)", "0.50:0.95", "small", 100),
    "   {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ]".format("Average Recall", "(AR)", "0.50:0.95", "medium", 100),
    "   {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ]".format("Average Recall", "(AR)", "0.50:0.95", "large", 100),
]



@torch.no_grad()
def validate(args, dataloader, model, epoch=0):
    model.eval()
    with open(args.mAP_file_path, mode="r") as f:
        mAP_json = json.load(f)

    cocoPred = []
    check_images, check_preds, check_results = [], [], []
    mAP_stats = None
    imageToid = mAP_json["imageToid"]

    for i, minibatch in enumerate(dataloader):
        filenames, images, labels, shapes = minibatch
        predictions = model(images.cuda(args.rank, non_blocking=True))

        for j in range(len(filenames)):
            prediction = predictions[j].cpu().numpy()
            prediction[:, 1:5] = transform_xcycwh_to_x1y1x2y2(boxes=prediction[:, 1:5], clip_max=1.0)
            prediction = filter_confidence(prediction=prediction, conf_threshold=args.conf_thres)
            prediction = run_NMS(prediction=prediction, iou_threshold=args.nms_thres)

            if len(check_images) < 5:
                check_images.append(to_image(images[j]))
                check_preds.append(prediction.copy())
                
            if len(prediction) > 0:
                filename = filenames[j]
                shape = shapes[j]
                cls_id = prediction[:, [0]]
                conf = prediction[:, [-1]]
                box_x1y1x2y2 = scale_to_original(boxes=prediction[:, 1:5], scale_w=shape[1], scale_h=shape[0])
                box_x1y1wh = transform_x1y1x2y2_to_x1y1wh(boxes=box_x1y1x2y2)
                img_id = np.array((imageToid[filename],) * len(cls_id))[:, np.newaxis]
                cocoPred.append(np.concatenate((img_id, box_x1y1wh, conf, cls_id), axis=1))

    if (epoch % args.img_interval == 0) and args.img_log_dir:
        for k in range(len(check_images)):
            check_image = check_images[k]
            check_pred = check_preds[k]
            check_result = visualize_prediction(image=check_image, prediction=check_pred, class_list=args.class_list, color_list=args.color_list)
            check_results.append(check_result)
        concat_result = np.concatenate(check_results, axis=1)
        imwrite(str(args.img_log_dir / f'EP_{epoch:03d}.jpg'), concat_result)

    if len(cocoPred) > 0:
        cocoDt = args.cocoGt.loadRes(np.concatenate(cocoPred, axis=0))
        cocoEval = COCOeval(cocoGt=args.cocoGt, cocoDt=cocoDt, iouType="bbox")
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        mAP_stats = cocoEval.stats
    return mAP_stats


def parse_args(make_dirs=True):
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True, help="Name to log training")
    parser.add_argument("--data", type=str, default="toy.yaml", help="Path to data.yaml")
    parser.add_argument("--backbone", type=str, default="resnet18", help="Model architecture")
    parser.add_argument("--img_size", type=int, default=448, help="Model input size")
    parser.add_argument("--bs", type=int, default=16, help="Batch size")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="Threshold to filter confidence score")
    parser.add_argument("--nms_thres", type=float, default=0.6, help="Threshold to filter Box IoU of NMS process")
    parser.add_argument("--ckpt_name", type=str, default="best.pt", help="Path to trained model")
    parser.add_argument("--rank", type=int, default=0, help="Process id for computation")
    parser.add_argument("--img_interval", type=int, default=10, help="Interval to log train/val image")
    parser.add_argument("--workers", type=int, default=4, help="Number of workers used in dataloader")
    args = parser.parse_args()
    args.data = ROOT / "data" / args.data
    args.exp_path = ROOT / 'experiment' / args.exp_name
    args.ckpt_path = args.exp_path / 'weight' / args.ckpt_name
    args.img_log_dir = args.exp_path / 'val_image'

    if make_dirs:
        os.makedirs(args.img_log_dir, exist_ok=True)
    return args


def main():
    torch.manual_seed(seed_num)
    args = parse_args(make_dirs=True)
    logger = build_basic_logger(args.exp_path / 'val.log', set_level=1)
    logger.info(f"[Arguments]\n{pprint.pformat(vars(args))}\n")

    val_dataset = Dataset(yaml_path=args.data, phase='val')
    val_transformer = BasicTransform(input_size=args.img_size)
    val_dataset.load_transformer(transformer=val_transformer)
    val_loader = DataLoader(dataset=val_dataset, collate_fn=Dataset.collate_fn, batch_size=args.bs, shuffle=False, pin_memory=True,  num_workers=args.workers)

    args.class_list = val_dataset.class_list
    args.color_list = generate_random_color(len(args.class_list))

    ckpt = torch.load(args.ckpt_path, map_location = {"cpu":"cuda:%d" %args.rank})
    model = YoloModel(backbone=args.backbone, num_classes=len(args.class_list)).cuda(args.rank)
    model.load_state_dict(ckpt, strict=True)

    args.mAP_file_path = val_dataset.mAP_file_path
    args.cocoGt = COCO(annotation_file=args.mAP_file_path)
    mAP_stats = validate(args=args, dataloader=val_loader, model=model)

    mAP_str = "\n"
    for mAP_format, mAP_value in zip(METRIC_FORMAT, mAP_stats):
        mAP_str += f"{mAP_format} = {mAP_value:.3f}\n"
    logger.info(mAP_str)


if __name__ == "__main__":
    main()