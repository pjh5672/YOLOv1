import torch
from torch import nn

from element import weight_init_kaiming_uniform
from neck import ConvBlock


class YoloHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.num_attributes = (1 + 4) * 1 + self.num_classes
        self.convs = ConvBlock(in_channels=in_channels, out_channels=512)
        self.detect = nn.Conv2d(512, self.num_attributes, kernel_size=1)
        self.apply(weight_init_kaiming_uniform)


    def forward(self, x):
        out = self.convs(x)
        out = self.detect(out)
        return out



if __name__ == "__main__":
    from backbone import build_backbone

    input_size = 448
    num_classes = 1
    backbone, feat_dims = build_backbone(arch_name="resnet18", pretrained=True)
    head = YoloHead(in_channels=feat_dims, num_classes=num_classes)

    inp = torch.randn(1, 3, input_size, input_size)
    out = backbone(inp)
    print(out.shape)
    out = head(out)
    print(out.shape)

