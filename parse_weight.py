import argparse
from pathlib import Path
from collections import OrderedDict

import torch

ROOT = Path(__file__).resolve().parents[0]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True, help="Name to log training")
    parser.add_argument("--ckpt-name", type=str, default="best.pt", help="Path to trained model")

    args = parser.parse_args()
    args.exp_path = ROOT / "experiment" / args.exp
    args.ckpt_path = args.exp_path / "weight" / args.ckpt_name
    return args
    

if __name__ == "__main__":
    args = parse_args()

    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    trained_model_state = ckpt["model_state"]
    parsed_model_state = OrderedDict()

    for key, val in trained_model_state.items():
        if not key.startswith("head"):
            parsed_model_state[key] = val
    
    torch.save({"model_state": parsed_model_state}, f"./yolov1-{args.exp}.pt")
    