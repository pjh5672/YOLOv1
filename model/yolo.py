import os
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
from torch import nn

from backbone import build_resnet18
from head import YoloHead
from utils import set_grid



class YoloModel(nn.Module):
    def __init__(self, num_classes, grid_size=7, num_boxes=2):
        super().__init__()
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.backbone, feat_dims = build_resnet18(pretrained=True)
        self.head = YoloHead(in_channels=feat_dims, num_classes=num_classes, grid_size=grid_size, num_boxes=num_boxes)
        grid_x, grid_y = set_grid(grid_size=grid_size)
        self.grid_x = grid_x.contiguous().view((1, -1)).tile(1, self.num_boxes)
        self.grid_y = grid_y.contiguous().view((1, -1)).tile(1, self.num_boxes)


    def forward(self, x):
        self.device = x.device
        batch_size = x.shape[0]
        
        out = self.backbone(x)
        out = self.head(out)
        out = out.permute(0, 2, 3, 1).contiguous()
        pred_obj = out[..., [0, 5]].view(batch_size, -1, 1)
        pred_box = torch.cat((out[..., 1:5], out[..., 6:10]), dim=-1).view(batch_size, -1, 4)
        pred_cls = out[..., 10:].view(batch_size, -1, self.num_classes).tile(self.num_boxes, 1)

        if self.training:
            return torch.cat((pred_obj, pred_box, pred_cls), dim=-1)
        else:
            pred_box = self.transform_pred_box(pred_box)
            return torch.cat((pred_obj, pred_box, pred_cls), dim=-1)


    def transform_pred_box(self, pred_box):
        xc = (pred_box[..., 0] + self.grid_x.to(self.device)) / self.grid_size
        yc = (pred_box[..., 1] + self.grid_y.to(self.device)) / self.grid_size
        # xc = pred_box[..., 0] + (self.grid_x / self.grid_size).to(self.device)
        # yc = pred_box[..., 1] + (self.grid_y / self.grid_size).to(self.device)
        w = pred_box[..., 2]
        h = pred_box[..., 3]
        return torch.stack((xc, yc, w, h), dim=-1)



if __name__ == "__main__":
    input_size = 448
    num_classes = 1
    inp = torch.randn(1, 3, input_size, input_size)
    device = torch.device('cpu')

    model = YoloModel(num_classes=num_classes).to(device)
    model.train()
    out = model(inp.to(device))
    print(out)

    model.eval()
    out = model(inp.to(device))
    print(out.device)
    print(out)