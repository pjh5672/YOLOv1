from torch import nn
from element import Conv



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convs = nn.Sequential(
            Conv(in_channels, out_channels, kernel_size=1),
            Conv(out_channels, out_channels*2, kernel_size=3, padding=1), 
            Conv(out_channels*2, out_channels, kernel_size=1), 
            Conv(out_channels, out_channels*2, kernel_size=3, padding=1),
            Conv(out_channels*2, out_channels, kernel_size=1),
        )


    def forward(self, x):
        return self.convs(x)



if __name__ == "__main__":
    import torch
    from backbone import build_backbone

    input_size = 448
    num_classes = 1
    backbone, feat_dims = build_backbone(arch_name="resnet18", pretrained=True)
    neck = ConvBlock(in_channels=feat_dims, out_channels=512)
    inp = torch.randn(1, 3, input_size, input_size)
    out = backbone(inp)
    print(out.shape)
    out = neck(out)
    print(out.shape)