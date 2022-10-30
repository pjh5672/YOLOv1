import math

import torch
from torch import nn

from backbone import build_resnet18
from head import YoloHead



class YoloModel(nn.Module):
    def __init__(self, input_size, num_classes, num_boxes=2):
        super().__init__()
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.backbone, feat_dims = build_resnet18(pretrained=True)
        self.head = YoloHead(in_channels=feat_dims, num_classes=num_classes, num_boxes=num_boxes)
        self.set_grid_size(input_size=input_size)
    

    def set_grid_size(self, input_size=448):
        out = self(torch.randn(1, 3, input_size, input_size))
        self.grid_size = int(math.sqrt(out.shape[1]/self.num_boxes))


    def forward(self, x):
        self.device = x.device
        batch_size = x.shape[0]
        out = self.backbone(x)
        out = self.head(out)
        out = out.permute(0, 2, 3, 1).contiguous()
        pred_obj = out[..., [0, 5]].view(batch_size, -1, 1)
        pred_box = torch.cat((out[..., 1:5], out[..., 6:10]), dim=-1).view(batch_size, -1, 4)
        pred_cls = out[..., 10:].view(batch_size, -1, self.num_classes).tile(self.num_boxes, 1)
        return torch.cat((pred_obj, pred_box, pred_cls), dim=-1)



if __name__ == "__main__":
    input_size = 448
    num_classes = 1
    inp = torch.randn(1, 3, input_size, input_size)
    device = torch.device('cpu')

    model = YoloModel(input_size=input_size, num_classes=num_classes).to(device)
    model.train()
    out = model(inp.to(device))
    print(out.shape)

    model.eval()
    out = model(inp.to(device))
    print(out.device)
    print(out.shape)