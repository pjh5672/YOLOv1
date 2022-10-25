import os
import sys
import json
import platform
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import torch
import numpy as np
from torch import optim
from torch.backends import cudnn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[0]
OS_SYSTEM = platform.system()
TIMESTAMP = datetime.today().strftime('%Y-%m-%d_%H-%M')
cudnn.benchmark = True
seed_num = 2023

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except:
    if OS_SYSTEM == 'Windows':
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pycocotools-windows'])
    elif OS_SYSTEM == 'Linux':
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pycocotools'])

from dataloader import Dataset, BasicTransform, AugmentTransform, to_image
from model import YoloModel
from utils import YoloLoss, generate_random_color, build_basic_logger
from val import validate, METRIC_FORMAT



def train(args, dataloader, model, criterion, optimizer):
    loss_type = {'multipart', 'obj', 'noobj', 'box', 'cls'}
    losses = defaultdict(float)

    model.train()
    optimizer.zero_grad()
    for i, minibatch in enumerate(dataloader):
        images, labels = minibatch[1].cuda(args.rank, non_blocking=True), minibatch[2]
        predictions = model(images)
        loss = criterion(predictions=predictions, labels=labels)
        loss[0].backward()
        optimizer.step()
        optimizer.zero_grad()

        for loss_name, loss_value in zip(loss_type, loss):
            if not torch.isfinite(loss_value) and loss_name != 'multipart':
                print(f'############## {loss_name} Loss is Nan/Inf ! {loss_value} ##############')
                sys.exit(0)
            else:
                losses[loss_name] += loss_value.item()

    loss_str = f"[Epoch:{epoch:03d}] "
    for loss_name in loss_type:
        losses[loss_name] /= len(dataloader)
        loss_str += f"{loss_name}: {losses[loss_name]:.4f}  "
    logger.info(loss_str)


def parse_args(make_dirs=True):
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True, help="Name to log training")
    parser.add_argument("--data", type=str, default="toy.yaml", help="Path to data.yaml")
    parser.add_argument("--img_size", type=int, default=224, help="Model input size")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="Threshold to filter confidence score")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="Threshold to filter Box IoU of NMS process")
    parser.add_argument("--rank", type=int, default=0, help="Process id for computation")
    parser.add_argument("--img_interval", type=int, default=10, help="Interval to log train/val image")

    args = parser.parse_args()
    args.data = ROOT / "data" / args.data
    args.exp_path = ROOT / 'experiment' / args.exp_name
    args.weight_dir = args.exp_path / 'weight'
    args.img_log_dir = args.exp_path / 'image'

    if make_dirs:
        os.makedirs(args.weight_dir, exist_ok=True)
        os.makedirs(args.img_log_dir, exist_ok=True)
    return args


def main():
    global epoch, logger
    torch.manual_seed(seed_num)
    args = parse_args(make_dirs=True)
    logger = build_basic_logger(args.exp_path / 'train.log', set_level=1)

    train_dataset = Dataset(yaml_path=args.data, phase='train')
    train_transformer = BasicTransform(input_size=args.img_size)
    # train_transformer = AugmentTransform(input_size=args.img_size)
    train_dataset.load_transformer(transformer=train_transformer)
    train_loader = DataLoader(dataset=train_dataset, collate_fn=Dataset.collate_fn, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_dataset = Dataset(yaml_path=args.data, phase='val')
    val_transformer = BasicTransform(input_size=args.img_size)
    val_dataset.load_transformer(transformer=val_transformer)
    val_loader = DataLoader(dataset=val_dataset, collate_fn=Dataset.collate_fn, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    args.class_list = train_dataset.class_list
    args.num_classes = len(args.class_list)
    args.color_list = generate_random_color(args.num_classes)
    
    model = YoloModel(input_size=args.img_size, num_classes=args.num_classes, num_boxes=2).cuda(args.rank)
    criterion = YoloLoss(input_size=args.img_size, num_classes=args.num_classes, lambda_coord=5.0, lambda_noobj=0.5)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    args.mAP_file_path = val_dataset.mAP_file_path
    args.cocoGt = COCO(annotation_file=args.mAP_file_path)
    best_mAP = 0.0
    mAP_str = "\n"

    for epoch in range(args.num_epochs):
        train(args=args, dataloader=train_loader, model=model, criterion=criterion, optimizer=optimizer)
        mAP_stats = validate(args=args, dataloader=val_loader, model=model, epoch=epoch)

        if (mAP_stats is not None) and (mAP_stats[0] > best_mAP):
            best_mAP = mAP_stats[0]
            for mAP_format, mAP_value in zip(METRIC_FORMAT, mAP_stats):
                mAP_str += f"{mAP_format} = {mAP_value:.3f}\n"
            logger.info(mAP_str)
            torch.save(model.state_dict(), args.weight_dir / "best.pt")
    torch.save(model.state_dict(), args.weight_dir / "last.pt")
    logger.info(f"[Best mAP]{mAP_str}")

if __name__ == "__main__":
    main()
 