import torch
from torch.nn import AdaptiveAvgPool2d, Linear, Module, ReLU, Sequential, Sigmoid
from torchvision.models.resnet import ResNet


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

    @classmethod
    def make_se_layer(cls, se_channels: int, residual_block: Module) -> Sequential:
        se_layer = Sequential(
            residual_block,
            SEModule(se_channels))
        return se_layer

    def __init__(self, num_classes: int, backbone: ResNet):
        super().__init__()
        self.se_conv1 = Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool)
        expansion = backbone.layer1[0].expansion
        self.se_conv2_x = self.make_se_layer(64 * expansion, backbone.layer1)
        self.se_conv3_x = self.make_se_layer(128 * expansion, backbone.layer2)
        self.se_conv4_x = self.make_se_layer(256 * expansion, backbone.layer3)
        self.se_conv5_x = self.make_se_layer(512 * expansion, backbone.layer4)

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


if __name__ == "__main__":
    from torchvision.models import resnet
    test_batch = torch.rand(4, 3, 224, 224)
    backbone = resnet.resnet50()
    model = SEResNet(5, backbone)
    model(test_batch)
    backbone = resnet.resnet18()
    model = SEResNet(5, backbone)
    model(test_batch)
