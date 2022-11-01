import os
import sys
import json
import pprint
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
from utils import YoloLoss, generate_random_color, build_basic_logger, set_lr
from val import validate, METRIC_FORMAT



def train(args, dataloader, model, criterion, optimizer):
    loss_type = ['multipart', 'obj', 'noobj', 'txty', 'twth', 'cls']
    losses = defaultdict(float)

    model.train()
    for i, minibatch in enumerate(dataloader):
        ni = i + len(dataloader) * epoch
        if ni <= args.nw:
            set_lr(optimizer, np.interp(ni, [0, args.nw], [args.init_lr, args.base_lr]))

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
    parser.add_argument("--img_size", type=int, default=448, help="Model input size")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--warmup_epoch", type=int, default=1, help="Epochs for warming up training")
    parser.add_argument("--init_lr", type=float, default=0.001, help="Learning rate for inital training")
    parser.add_argument("--base_lr", type=float, default=0.01, help="Base learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="Weight decay")
    parser.add_argument("--lambda_coord", type=float, default=5.0, help="Lambda for box regression loss")
    parser.add_argument("--lambda_noobj", type=float, default=0.5, help="Lambda for no-objectness loss")
    parser.add_argument("--conf_thres", type=float, default=0.01, help="Threshold to filter confidence score")
    parser.add_argument("--nms_thres", type=float, default=0.6, help="Threshold to filter Box IoU of NMS process")
    parser.add_argument("--rank", type=int, default=0, help="Process id for computation")
    parser.add_argument("--img_interval", type=int, default=5, help="Interval to log train/val image")
    args = parser.parse_args()
    
    ROOT = Path(__file__).resolve().parents[0]
    args.data = ROOT / "data" / args.data
    args.exp_path = ROOT / 'experiment' / args.exp_name
    args.weight_dir = args.exp_path / 'weight'
    args.img_log_dir = args.exp_path / 'train_image'

    if make_dirs:
        os.makedirs(args.weight_dir, exist_ok=True)
        os.makedirs(args.img_log_dir, exist_ok=True)
    return args


def main():
    global epoch, logger
    torch.manual_seed(seed_num)
    args = parse_args(make_dirs=True)
    logger = build_basic_logger(args.exp_path / 'train.log', set_level=1)
    logger.info(f"[Arguments]\n{pprint.pformat(vars(args))}\n")

    train_dataset = Dataset(yaml_path=args.data, phase='train')
    # train_transformer = BasicTransform(input_size=args.img_size)
    train_transformer = AugmentTransform(input_size=args.img_size)
    train_dataset.load_transformer(transformer=train_transformer)
    train_loader = DataLoader(dataset=train_dataset, collate_fn=Dataset.collate_fn, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_dataset = Dataset(yaml_path=args.data, phase='val')
    val_transformer = BasicTransform(input_size=args.img_size)
    val_dataset.load_transformer(transformer=val_transformer)
    val_loader = DataLoader(dataset=val_dataset, collate_fn=Dataset.collate_fn, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    args.nw = max(round(args.warmup_epoch * len(train_loader)), 100)
    args.class_list = train_dataset.class_list
    args.num_classes = len(args.class_list)
    args.color_list = generate_random_color(args.num_classes)
    
    model = YoloModel(num_classes=args.num_classes, grid_size=7, num_boxes=2).cuda(args.rank)
    criterion = YoloLoss(num_classes=args.num_classes, grid_size=model.grid_size, lambda_coord=args.lambda_coord, lambda_noobj=args.lambda_noobj)
    optimizer = optim.SGD(model.parameters(), lr=args.init_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 175], gamma=0.1)

    args.mAP_file_path = val_dataset.mAP_file_path
    args.cocoGt = COCO(annotation_file=args.mAP_file_path)
    best_epoch, best_score, best_mAP_str = 0, 0, "\n"
    
    for epoch in range(args.num_epochs):
        train(args=args, dataloader=train_loader, model=model, criterion=criterion, optimizer=optimizer)
        mAP_stats = validate(args=args, dataloader=val_loader, model=model, epoch=epoch)
        scheduler.step()
        torch.save(model.state_dict(), args.weight_dir / "last.pt")

        if mAP_stats is not None:
            ap95, ap50 = mAP_stats[:2]
            mAP_str = "\n"
            for mAP_format, mAP_value in zip(METRIC_FORMAT, mAP_stats):
                mAP_str += f"{mAP_format} = {mAP_value:.3f}\n"
            logger.info(mAP_str)

            if ap50 > best_score:
                best_epoch, best_score, best_mAP_str = epoch, ap50, mAP_str
                torch.save(model.state_dict(), args.weight_dir / "best.pt")
    
    if best_score > 0:
        logger.info(f"[Best mAP : Epoch{best_epoch}]{best_mAP_str}")


if __name__ == "__main__":
    main()