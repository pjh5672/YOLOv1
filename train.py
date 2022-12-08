import os
import sys
import random
import pprint
import platform
import argparse
from pathlib import Path
from copy import deepcopy
from datetime import datetime
from collections import defaultdict

import torch
import numpy as np
from torch.cuda import amp
from tqdm import tqdm, trange
from thop import profile
from torch import optim
from torch.backends import cudnn
from torch.utils.data import DataLoader, distributed
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).resolve().parents[0]
OS_SYSTEM = platform.system()
TIMESTAMP = datetime.today().strftime('%Y-%m-%d_%H-%M')
cudnn.benchmark = True
SEED = 2023
random.seed(SEED)
torch.manual_seed(SEED)

from dataloader import Dataset, BasicTransform, AugmentTransform
from model import YoloModel
from utils import YoloLoss, Evaluator, generate_random_color, set_lr, build_basic_logger, setup_worker_logging, setup_primary_logging
from val import validate, result_analyis


def setup(rank, world_size):
    if OS_SYSTEM == 'Linux':
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12345'
        dist.init_process_group('nccl', rank=rank, world_size=world_size)


def cleanup():
    if OS_SYSTEM == 'Linux':
        dist.destroy_process_group()


def train(args, dataloader, model, criterion, optimizer, scaler):
    loss_type = ['multipart', 'obj', 'noobj', 'box', 'cls']
    losses = defaultdict(float)
    model.train()
    optimizer.zero_grad()

    for i, minibatch in enumerate(dataloader):
        ni = i + len(dataloader) * (epoch - 1)
        if ni <= args.nw:
            args.grad_accumulate = max(1, np.interp(ni, [0, args.nw], [1, args.acc_batch_size / args.batch_size]).round())
            set_lr(optimizer, args.base_lr * pow(ni / (args.nw), 2))

        images, labels = minibatch[1], minibatch[2]
        
        with amp.autocast(enabled=not args.no_amp):
            predictions = model(images.cuda(args.rank, non_blocking=True))
            loss = criterion(predictions=predictions, labels=labels)

        scaler.scale((loss[0] / args.grad_accumulate) * args.world_size).backward()
        
        if ni - args.last_opt_step >= args.grad_accumulate:
            scaler.step(optimizer)
            scaler.update()
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
    parser.add_argument("--exp", type=str, required=True, help="Name to log training")
    parser.add_argument("--resume", type=str, nargs='?', const=True ,help="Name to resume path")
    parser.add_argument("--data", type=str, default="toy.yaml", help="Path to data.yaml")
    parser.add_argument("--img_size", type=int, default=416, help="Model input size")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--acc_batch_size", type=int, default=64, help="Batch size for gradient accumulation")
    parser.add_argument("--backbone", type=str, default="resnet18", help="Model architecture")
    parser.add_argument("--num_epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument('--lr_decay', nargs='+', default=[90, 120], type=int, help='Epoch to learning rate decay')
    parser.add_argument("--warmup", type=int, default=1, help="Epochs for warming up training")
    parser.add_argument("--base_lr", type=float, default=0.001, help="Base learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="Weight decay")
    parser.add_argument("--conf_thres", type=float, default=0.001, help="Threshold to filter confidence score")
    parser.add_argument("--nms_thres", type=float, default=0.6, help="Threshold to filter Box IoU of NMS process")
    parser.add_argument("--img_interval", type=int, default=10, help="Interval to log train/val image")
    parser.add_argument("--workers", type=int, default=8, help="Number of workers used in dataloader")
    parser.add_argument("--world_size", type=int, default=1, help="Number of available GPU devices")
    parser.add_argument("--rank", type=int, default=0, help="Process id for computation")
    parser.add_argument("--no_amp", action="store_true", help="Use of FP32 training (default: AMP training)")
    parser.add_argument("--depthwise", action="store_true", help="Use of Depth-separable conv operation")

    args = parser.parse_args()
    args.data = ROOT / "data" / args.data
    args.exp_path = ROOT / "experiment" / args.exp
    args.weight_dir = args.exp_path / "weight"
    args.img_log_dir = args.exp_path / "train_image"
    args.load_path = args.weight_dir / "last.pt" if args.resume else None
    assert args.world_size > 0, "Executable GPU machine does not exist, This training supports on CUDA available environment."

    if make_dirs:
        os.makedirs(args.weight_dir, exist_ok=True)
        os.makedirs(args.img_log_dir, exist_ok=True)
    return args


def main_work(rank, world_size, args, logger):
    ################################### Init Process ####################################
    setup(rank, world_size)
    torch.manual_seed(SEED)
    torch.cuda.set_device(rank)

    if OS_SYSTEM == 'Linux':
        import logging
        setup_worker_logging(rank, logger)
    else:
        logging = logger
    ################################### Init Instance ###################################
    global epoch

    args.rank = rank
    args.last_opt_step = -1
    args.batch_size = args.batch_size // world_size
    args.grad_accumulate = max(round(args.acc_batch_size / args.batch_size), 1)
    args.workers = min([os.cpu_count() // max(world_size, 1), args.batch_size if args.batch_size > 1 else 0, args.workers])
    
    train_dataset = Dataset(yaml_path=args.data, phase="train")
    train_transformer = AugmentTransform(input_size=args.img_size)
    train_dataset.load_transformer(transformer=train_transformer)
    train_sampler = distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=args.rank, shuffle=True)
    train_loader = DataLoader(dataset=train_dataset, collate_fn=Dataset.collate_fn, batch_size=args.batch_size, 
                              shuffle=False, pin_memory=True, num_workers=args.workers, sampler=train_sampler)
    val_dataset = Dataset(yaml_path=args.data, phase="val")
    val_transformer = BasicTransform(input_size=args.img_size)
    val_dataset.load_transformer(transformer=val_transformer)
    val_loader = DataLoader(dataset=val_dataset, collate_fn=Dataset.collate_fn, batch_size=args.batch_size, 
                            shuffle=False, pin_memory=True, num_workers=args.workers)

    args.class_list = train_dataset.class_list
    args.color_list = generate_random_color(len(args.class_list))
    args.nw = max(round(args.warmup * len(train_loader)), 100)
    args.mAP_file_path = val_dataset.mAP_file_path
    
    model = YoloModel(input_size=args.img_size, backbone=args.backbone, num_classes=len(args.class_list), depthwise=args.depthwise)
    macs, params = profile(deepcopy(model), inputs=(torch.randn(1, 3, args.img_size, args.img_size),), verbose=False)
    criterion = YoloLoss(grid_size=model.grid_size)
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_decay, gamma=0.1)
    evaluator = Evaluator(annotation_file=args.mAP_file_path)
    scaler = amp.GradScaler(enabled=not args.no_amp)

    model = model.cuda(args.rank)
    if OS_SYSTEM == 'Linux':
        model = DDP(model, device_ids=[args.rank])
        dist.barrier()
    #################################### Load Model #####################################

    if args.resume:
        assert args.load_path.is_file(), "Not exist trained weights in the directory path !"
        
        ckpt = torch.load(args.load_path, map_location="cpu")
        start_epoch = ckpt["running_epoch"]
        if hasattr(model, "module"):
            model.module.load_state_dict(ckpt["model_state"], strict=True)
        else:
            model.load_state_dict(ckpt["model_state"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer_state"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda(args.rank)
        scheduler.load_state_dict(ckpt["scheduler_state"])
        scaler.load_state_dict(ckpt['scaler_state_dict'])
    else:
        start_epoch = 1
        if args.rank == 0:
            logging.warning(f"[Arguments]\n{pprint.pformat(vars(args))}\n")
            logging.warning(f"Architecture Info - Params(M): {params/1e+6:.2f}, FLOPS(B): {2*macs/1E+9:.2f}")

    #################################### Train Model ####################################
    if args.rank == 0:
        progress_bar = trange(start_epoch, args.num_epochs, total=args.num_epochs, initial=start_epoch, ncols=115)
    else:
        progress_bar = range(start_epoch, args.num_epochs)

    best_epoch, best_score, best_mAP_str, mAP_dict = 0, 0, "", None

    for epoch in progress_bar:
        if args.rank == 0:
            train_loader = tqdm(train_loader, desc=f"[TRAIN:{epoch:03d}/{args.num_epochs:03d}]", ncols=115, leave=False)
        train_sampler.set_epoch(epoch)
        train_loss_str = train(args=args, dataloader=train_loader, model=model, criterion=criterion, optimizer=optimizer, scaler=scaler)

        if args.rank == 0:
            logging.warning(train_loss_str) 
            save_opt = {"running_epoch": epoch,
                        "backbone": args.backbone,
                        "depthwise": args.depthwise,
                        "class_list": args.class_list,
                        "model_state": deepcopy(model.module).state_dict() if hasattr(model, "module") else deepcopy(model).state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        'scaler_state_dict': scaler.state_dict()}
            torch.save(save_opt, args.weight_dir / "last.pt")

            if epoch % 10 == 0:
                val_loader = tqdm(val_loader, desc=f"[VAL:{epoch:03d}/{args.num_epochs:03d}]", ncols=115, leave=False)
                mAP_dict, eval_text = validate(args=args, dataloader=val_loader, model=model, evaluator=evaluator, epoch=epoch)
                ap50 = mAP_dict["all"]["mAP_50"]
                logging.warning(eval_text)

                if ap50 > best_score:
                    result_analyis(args=args, mAP_dict=mAP_dict["all"])
                    best_epoch, best_score, best_mAP_str = epoch, ap50, eval_text
                    torch.save(save_opt, args.weight_dir / "best.pt")
    
        scheduler.step()

    if mAP_dict and args.rank == 0:
        logging.warning(f"[Best mAP at {best_epoch}]\n{best_mAP_str}")
        
    cleanup()



if __name__ == "__main__":
    args = parse_args(make_dirs=True)

    if OS_SYSTEM == 'Linux':
        torch.multiprocessing.set_start_method('spawn', force=True)
        logger = setup_primary_logging(args.exp_path / 'train.log')
        mp.spawn(main_work, args=(args.world_size, args, logger), nprocs=args.world_size, join=True)
    else:
        logger = build_basic_logger(args.exp_path / "train.log")
        main_work(rank=0, world_size=1, args=args, logger=logger)
    