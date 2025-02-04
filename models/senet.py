import torch
from torch.nn import AdaptiveAvgPool2d, Linear, Module, ReLU, Sequential, Sigmoid
from torchvision.models import resnet
from torchvision.models.resnet import ResNet50_Weights


class SEModule(Module):
    def __init__(self, channels: int, reduction=16):
        super().__init__()
        c_over_r = channels // reduction
        self.global_pooling = AdaptiveAvgPool2d(1)
        self.excitation = Sequential(
            Linear(channels, c_over_r, bias=False),
            ReLU(),
            Linear(c_over_r, channels, bias=False),
            Sigmoid()
        )

    def forward(self, x):
        out = self.global_pooling(x)  # (B, C, 1, 1)
        out = torch.squeeze(out, (2, 3))  # (B, C)
        out = self.excitation(out)  # (B, C)
        b, c, h, w = x.shape
        return x * out.view(b, c, 1, 1)  # (B, C, H, W)


class SEResNet(Module):
    def __init__(self, num_classes: int):
        super().__init__()
        backbone = resnet.resnet50(ResNet50_Weights.DEFAULT)
        self.se_conv1 = Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool)
        self.se_conv2_x = self.__make_se_layer(256, backbone.layer1)
        self.se_conv3_x = self.__make_se_layer(512, backbone.layer2)
        self.se_conv4_x = self.__make_se_layer(1024, backbone.layer3)
        self.se_conv5_x = self.__make_se_layer(2048, backbone.layer4)

        self.avgpool = backbone.avgpool
        self.fc = Linear(backbone.fc.in_features, num_classes)

    def forward(self, x):
        x = self.se_conv1(x)
        x = self.se_conv2_x(x)
        x = self.se_conv3_x(x)
        x = self.se_conv4_x(x)
        x = self.se_conv5_x(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def __make_se_layer(self, se_channels: int, residual_block: Module) -> Sequential:
        se_layer = Sequential(
            residual_block,
            SEModule(se_channels))
        return se_layer
