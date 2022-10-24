import torch
from torch import nn

from backbone import build_resnet18
from head import YoloHead



class YoloModel(nn.Module):
    def __init__(self, input_size, num_classes, device, num_boxes=2):
        super().__init__()
        self.stride = 32
        self.device = device
        self.num_boxes = num_boxes
        self.input_size = input_size
        self.num_classes = num_classes
        self.grid_size = input_size // self.stride
        self.backbone, feat_dims = build_resnet18(pretrained=True)
        self.head = YoloHead(in_channels=feat_dims, num_classes=num_classes, num_boxes=num_boxes)


    def forward(self, x):
        batch_size = x.shape[0]
        out = self.backbone(x)
        out = self.head(out)
        return out



if __name__ == "__main__":
    input_size = 224
    num_classes = 1
    inp = torch.randn(1, 3, input_size, input_size)
    device = torch.device('cpu')

    model = YoloModel(input_size=input_size, num_classes=num_classes, device=device).to(device)
    out = model(inp.to(device))
    print(out.shape)
