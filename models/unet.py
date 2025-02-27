import torch
from torch import nn, Tensor
from torch.nn import Module, init
from torchvision.models import resnet
from torchvision.models.resnet import BasicBlock


def convtrans3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1, output_padding: int = 0) -> nn.ConvTranspose2d:
    """3x3 convolution with padding"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation,
                              output_padding = output_padding, groups=groups, bias=False,
                              dilation=dilation)


def convtrans1x1(in_planes: int, out_planes: int, stride: int = 1, output_padding: int = 0) -> nn.ConvTranspose2d:
    """1x1 convolution"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False,
                           output_padding=output_padding)


class DecoderBuildingBlock(Module):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        output_padding: int = 0,
        upsample: Module | None = None
    ):
        super().__init__()
        self.conv2 = convtrans3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv1 = convtrans3x3(planes, inplanes, stride, output_padding=output_padding)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv2(x)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn1(out)
        if self.upsample is not None:
            identity = self.upsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet34Decoder(Module):
    def __init__(self, num_classes: int, upsample_size: int):
        super().__init__()
        self.inplanes = 2048
        self.upsample = nn.Upsample(size=upsample_size)
        self.layer4 = self._make_layer(512, 3, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer1 = self._make_layer(64, 3, output_padding = 0,
                                       last_block_dim=64)
        self.conv1 = nn.ConvTranspose2d(self.inplanes, self.inplanes, kernel_size=7, stride=2,
                                     padding=3, output_padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.upsample_top = nn.Upsample(scale_factor=2)
        self.classifier = nn.Conv2d(self.inplanes, num_classes, kernel_size=1, bias=False)
        self.init_weights()

    def forward(self, x: Tensor) -> Tensor:
        x = self.upsample(x)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = self.upsample_top(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.classifier(x)
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def _make_layer(
        self,
        planes: int,
        blocks: int,
        stride: int = 1,
        output_padding: int = 1,
        last_block_dim: int = 0,
    ) -> nn.Sequential:
        upsample = None
        self.inplanes = planes * BasicBlock.expansion
        if last_block_dim == 0:
            last_block_dim = self.inplanes // 2
        if stride != 1 or self.inplanes != planes * BasicBlock.expansion or output_padding == 0:
            upsample = nn.Sequential(
                convtrans1x1(planes * BasicBlock.expansion, last_block_dim, stride,
                             output_padding),
                nn.BatchNorm2d(last_block_dim))
        layers = []
        last_block = DecoderBuildingBlock(last_block_dim, planes, stride, output_padding, upsample)
        for _ in range(1, blocks):
            layers.append( DecoderBuildingBlock(self.inplanes, planes) )
        layers.append(last_block)
        return nn.Sequential(*layers)


class ResNetUNet(Module):
    """Implementation of the Deep learning segmentation of hyperautofluorescent fleck lesions in Stargardt disease by Charn, J et al.
    """
    def __init__(self, num_classes: int, input_size: int):
        super().__init__()
        upsample_size = input_size // 32
        self.input_size = input_size
        self.encoder = resnet.resnet34()
        self.decoder = ResNet34Decoder(num_classes, upsample_size)

    def forward(self, x: Tensor) -> Tensor:
        # (C, H, W)
        # Encoder
        x = self.encoder.conv1(x)  # (64, H/2, W/2)
        x = self.encoder.bn1(x)  # (64, H/2, W/2)
        x = self.encoder.relu(x) # (64, H/2, W/2)
        stage1 = x
        x = self.encoder.maxpool(x)  # (64, H/4, W/4)
        x = self.encoder.layer1(x)  # (64, H/4, W/4)
        block1 = x
        x = self.encoder.layer2(x)  # (128, H/8, W/8)
        block2 = x
        x = self.encoder.layer3(x)  # (256, H/16, W/16)
        block3 = x
        x = self.encoder.layer4(x)  # (512, H/32, W/32)
        block4 = x
        x = self.encoder.avgpool(x)  # (512, 1, 1)

        # Decoder
        x = self.decoder.upsample(x)  # (512, H/32, W/32)
        x += block4
        x = self.decoder.layer4(x)  # (256, H/16, W/16)
        x = x + block3
        x = self.decoder.layer3(x)  # (128, H/8, W/8)
        x = x + block2
        x = self.decoder.layer2(x)  # (64, H/4, W/4)
        x = x + block1
        x = self.decoder.layer1(x)  # (64, H/4, W/4)
        x = self.decoder.upsample_top(x)  # (64, H/2, W/2)
        x = x + stage1
        x = self.decoder.conv1(x) # (64, H, W)
        x = self.decoder.bn1(x)
        x = self.decoder.relu(x)
        x = self.decoder.classifier(x) # (N, H, W)
        return x


def load_resnetunet(num_classes: int, input_size: int) -> ResNetUNet:
    model = ResNetUNet(num_classes, input_size)
    opt_model = torch.compile(model, dynamic=False, mode="reduce-overhead")
    return opt_model # type: ignore


if __name__ == "__main__":
    WINDOW_SIZE = 224
    unet = ResNetUNet(2, WINDOW_SIZE)
    image = torch.rand(1, 3, WINDOW_SIZE, WINDOW_SIZE)
    unet.eval()
    with torch.no_grad():
        predictions = unet(image)
