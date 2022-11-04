import torch
from torch import nn

from element import Conv, weight_init_kaiming_uniform



class YoloHead(nn.Module):
    def __init__(self, in_channels, num_classes, grid_size=7, num_boxes=2):
        super().__init__()
        self.num_attributes = (1 + 4) * num_boxes + num_classes
        self.conv1 = Conv(in_channels, in_channels//2, kernel_size=1)
        self.conv2 = Conv(in_channels//2, in_channels, kernel_size=3, padding=1)
        self.conv3 = Conv(in_channels, in_channels//2, kernel_size=1)
        self.conv4 = Conv(in_channels//2, 512, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(512, self.num_attributes, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(grid_size, grid_size))
        self.apply(weight_init_kaiming_uniform)


    def forward(self, x):
        out = self.pool(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        return out



if __name__ == "__main__":
    from backbone import build_backbone

    input_size = 448
    num_classes = 1
    backbone, feat_dims = build_backbone(model_name="resnet18", pretrained=True)
    head = YoloHead(in_channels=feat_dims, num_classes=num_classes, num_boxes=2)

    inp = torch.randn(1, 3, input_size, input_size)
    out = backbone(inp)
    print(out.shape)
    out = head(out)
    print(out.shape)

