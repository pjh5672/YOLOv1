from torch import nn


class Conv(nn.Module):
    def __init__(self, c1, c2, kernel_size, stride=1, padding=0, dilation=1, act="leaky_relu", depthwise=False):
        super().__init__()

        if act == "identity":
            act_func = nn.Identity()
        elif act == "relu":
            act_func = nn.ReLU()
        elif act == "leaky_relu":
            act_func = nn.LeakyReLU(0.1)

        if depthwise:
            self.conv = nn.Sequential(
                ### Depth-wise ###
                nn.Conv2d(c1, c1, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=c1, bias=False),
                nn.BatchNorm2d(c1),
                act_func,
                ### Point-wise ###
                nn.Conv2d(c1, c2, kernel_size=1, bias=False),
                nn.BatchNorm2d(c2),
                act_func,
            )
        else:
            self.conv = nn.Sequential(
                ### General Conv ###
                nn.Conv2d(c1, c2, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=1, bias=False),
                nn.BatchNorm2d(c2),
                act_func,
            )

    def forward(self, x):
        return self.conv(x)



class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out



class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out