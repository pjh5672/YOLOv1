import torch
from torch import nn



class YoloHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.num_attributes = (1 + 4) * 1 + self.num_classes
        self.detect = nn.Conv2d(in_channels, self.num_attributes, kernel_size=1)


    def forward(self, x):
        out = self.detect(x)
        return out



if __name__ == "__main__":
    from backbone import build_backbone
    from neck import ConvBlock

    input_size = 448
    num_classes = 1
    backbone, feat_dims = build_backbone(arch_name="resnet18", pretrained=True)
    neck = ConvBlock(in_channels=feat_dims, out_channels=512)
    head = YoloHead(in_channels=512, num_classes=num_classes)

    inp = torch.randn(1, 3, input_size, input_size)
    out = backbone(inp)
    print(out.shape)
    out = neck(out)
    print(out.shape)
    out = head(out)
    print(out.shape)
