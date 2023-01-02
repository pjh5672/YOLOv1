from torch import nn
import torch.utils.model_zoo as model_zoo

from element import BasicBlock, BottleNeck


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg16-bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
}


class ResNet(nn.Module):
    def __init__(self, block, layers, zero_init_residual=False):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        if zero_init_residual:
            # Zero-initialize the last BN in each residual branch,
            # so that the residual branch starts with zeros, and each residual block behaves like an identity.
            # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
            for m in self.modules():
                if isinstance(m, BottleNeck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)


    def _make_layer(self, block, out_channels, num_block, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_block):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        C1 = self.conv1(x)
        C1 = self.bn1(C1)
        C1 = self.relu(C1)
        C1 = self.maxpool(C1)
        C2 = self.layer1(C1)
        C3 = self.layer2(C2)
        C4 = self.layer3(C3)
        C5 = self.layer4(C4)
        return C5


class VGG16(nn.Module):
    def __init__(self, batch_norm=False):
        super().__init__()
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        self.features = self.make_layers(cfg, batch_norm)

    def forward(self, x):
        x = self.features(x)
        return x

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)


def build_backbone(arch_name="resnet18", pretrained=True):
    feat_dims = 512
    if arch_name == "vgg16":
        model = VGG16(batch_norm=False)
    elif arch_name == "vgg16-bn":
        model = VGG16(batch_norm=True)
    elif arch_name == "resnet18":
        model = ResNet(BasicBlock, [2, 2, 2, 2])
    elif arch_name == "resnet34":
        model = ResNet(BasicBlock, [3, 4, 6, 3])
    elif arch_name == "resnet50":
        model = ResNet(BottleNeck, [3, 4, 6, 3])
        feat_dims = 2048
    elif arch_name == "resnet101":
        model = ResNet(BottleNeck, [3, 4, 23, 3])
        feat_dims = 2048
    else:
        raise RuntimeError("Only support model in [vgg16, vgg16_bn, resnet18, resnet34, resnet50, resnet101]")
    
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls[arch_name]), strict=False)
    return model, feat_dims



if __name__ == "__main__":
    import torch

    input_size = 448
    device = torch.device('cpu')
    backbone, feat_dims = build_backbone(arch_name="resnet18", pretrained=True)
    backbone.to(device)

    x = torch.randn(1, 3, input_size, input_size).to(device)
    y = backbone(x)
    print(y.shape)