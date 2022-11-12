import os
import sys
import pprint
import platform
import argparse
from pathlib import Path
from copy import deepcopy
from datetime import datetime
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm, trange
from thop import profile
from torch import nn
from torch import optim
from torch.backends import cudnn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[0]
OS_SYSTEM = platform.system()
TIMESTAMP = datetime.today().strftime('%Y-%m-%d_%H-%M')
cudnn.benchmark = True
seed_num = 2023

from dataloader import Dataset, BasicTransform, AugmentTransform
from model import YoloModel
from utils import YoloLoss, Evaluator, generate_random_color, build_basic_logger, set_lr
from val import validate, result_analyis



def train(args, dataloader, model, criterion, optimizer):
    loss_type = ['multipart', 'obj', 'noobj', 'box', 'cls']
    losses = defaultdict(float)
    model.train()
    optimizer.zero_grad()

    for i, minibatch in enumerate(dataloader):
        ni = i + len(dataloader) * epoch
        if ni <= args.nw:
            xi = [0, args.nw]
            args.accumulate = max(1, np.interp(ni, xi, [1, args.nbs / args.bs]).round())
            set_lr(optimizer, np.interp(ni, xi, [args.init_lr, args.base_lr]))

        images, labels = minibatch[1], minibatch[2]
        predictions = model(images.cuda(args.rank, non_blocking=True))
        loss = criterion(predictions=predictions, labels=labels)
        loss[0].backward()

        if ni - args.last_opt_step >= args.accumulate:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            optimizer.zero_grad()
            args.last_opt_step = ni
    
        for loss_name, loss_value in zip(loss_type, loss):
            if not torch.isfinite(loss_value) and loss_name != 'multipart':
                print(f'############## {loss_name} Loss is Nan/Inf ! {loss_value} ##############')
                sys.exit(0)
            else:
                losses[loss_name] += loss_value.item()

    loss_str = f"[Train-Epoch:{epoch:03d}] "
    for loss_name in loss_type:
        losses[loss_name] /= len(dataloader)
        loss_str += f"{loss_name}: {losses[loss_name]:.4f}  "
    return loss_str


def parse_args(make_dirs=True):
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True, help="Name to log training")
    parser.add_argument("--resume", type=str, nargs='?', const=True ,help="Name to resume path")
    parser.add_argument("--data", type=str, default="toy.yaml", help="Path to data.yaml")
    parser.add_argument("--backbone", type=str, default="resnet18", help="Model architecture")
    parser.add_argument("--img_size", type=int, default=448, help="Model input size")
    parser.add_argument("--bs", type=int, default=64, help="Batch size")
    parser.add_argument("--nbs", type=int, default=64, help="Nominal batch size")
    parser.add_argument("--num_epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument('--lr_decay', nargs='+', default=[90, 120], type=int, help='Epoch to learning rate decay')
    parser.add_argument("--warmup", type=int, default=1, help="Epochs for warming up training")
    parser.add_argument("--init_lr", type=float, default=0.0001, help="Learning rate for inital training")
    parser.add_argument("--base_lr", type=float, default=0.001, help="Base learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="Weight decay")
    parser.add_argument("--conf_thres", type=float, default=0.01, help="Threshold to filter confidence score")
    parser.add_argument("--nms_thres", type=float, default=0.6, help="Threshold to filter Box IoU of NMS process")
    parser.add_argument("--rank", type=int, default=0, help="Process id for computation")
    parser.add_argument("--img_interval", type=int, default=10, help="Interval to log train/val image")
    parser.add_argument("--workers", type=int, default=8, help="Number of workers used in dataloader")
    args = parser.parse_args()
    args.data = ROOT / "data" / args.data
    args.exp_path = ROOT / 'experiment' / args.exp_name
    args.weight_dir = args.exp_path / 'weight'
    args.img_log_dir = args.exp_path / 'train_image'
    args.load_path = args.weight_dir / 'last.pt' if args.resume else None

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
    train_transformer = AugmentTransform(input_size=args.img_size)
    train_dataset.load_transformer(transformer=train_transformer)
    train_loader = DataLoader(dataset=train_dataset, collate_fn=Dataset.collate_fn, batch_size=args.bs, shuffle=True, pin_memory=True, num_workers=args.workers)
    val_dataset = Dataset(yaml_path=args.data, phase='val')
    val_transformer = BasicTransform(input_size=args.img_size)
    val_dataset.load_transformer(transformer=val_transformer)
    val_loader = DataLoader(dataset=val_dataset, collate_fn=Dataset.collate_fn, batch_size=args.bs, shuffle=False, pin_memory=True, num_workers=args.workers)
    
    args.class_list = train_dataset.class_list
    args.color_list = generate_random_color(len(args.class_list))
    args.nw = max(round(args.warmup * len(train_loader)), 100)
    args.accumulate = max(round(args.nbs / args.bs), 1)
    args.last_opt_step = -1
    args.mAP_file_path = val_dataset.mAP_file_path

    model = YoloModel(input_size=args.img_size, backbone=args.backbone, num_classes=len(args.class_list))
    macs, params = profile(deepcopy(model), inputs=(torch.randn(1, 3, args.img_size, args.img_size),), verbose=False)
    model = model.cuda(args.rank)
    criterion = YoloLoss(grid_size=model.grid_size)
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_decay, gamma=0.1)
    evaluator = Evaluator(annotation_file=args.mAP_file_path)

    if args.resume:
        assert args.load_path.is_file(), "Not exist trained weights in the directory path !"
        
        ckpt = torch.load(args.load_path, map_location='cpu')
        start_epoch = ckpt["running_epoch"]
        model.load_state_dict(ckpt["model_state"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda(args.rank)
    else:
        start_epoch = 1
        logger.info(f"[Arguments]\n{pprint.pformat(vars(args))}\n")
        logger.info(f"YOLOv1 Architecture Info - Params(M): {params/1e+6:.2f}, FLOPS(B): {2*macs/1E+9:.2f}")

    progress_bar = trange(start_epoch, args.num_epochs, total=args.num_epochs, initial=start_epoch, ncols=115)
    best_epoch, best_score, best_mAP_str, mAP_dict = 0, 0, "", None
    for epoch in progress_bar:
        train_loader = tqdm(train_loader, desc=f"[TRAIN:{epoch:03d}/{args.num_epochs:03d}]", ncols=115, leave=False)
        train_loss_str = train(args=args, dataloader=train_loader, model=model, criterion=criterion, optimizer=optimizer)
        logger.info(train_loss_str)

        save_opt = {"running_epoch": epoch,
                    "class_list": args.class_list,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict()}

        if epoch >= 10:
            val_loader = tqdm(val_loader, desc=f"[VAL:{epoch:03d}/{args.num_epochs:03d}]", ncols=115, leave=False)
            mAP_dict, eval_text = validate(args=args, dataloader=val_loader, model=model, evaluator=evaluator, epoch=epoch)
            ap50 = mAP_dict["all"]["mAP_50"]

            if ap50 > best_score:
                logger.info(eval_text)
                result_analyis(args=args, mAP_dict=mAP_dict["all"])
                best_epoch, best_score, best_mAP_str = epoch, ap50, eval_text
                torch.save(save_opt, args.weight_dir / "best.pt")
    
        torch.save(save_opt, args.weight_dir / "last.pt")
        scheduler.step()

    if mAP_dict:
        logger.info(f"[Best mAP at {best_epoch}]\n{best_mAP_str}")
        

if __name__ == "__main__":
    main()