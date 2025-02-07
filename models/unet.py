import torch
from torch import nn, Tensor
from torch.nn import Module, init


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation,
                     groups=groups, bias=False, dilation=dilation)


def convtrans3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1, output_padding: int = 0) -> nn.ConvTranspose2d:
    """3x3 convolution with padding"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation,
                              output_padding = output_padding, groups=groups, bias=False,
                              dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def convtrans1x1(in_planes: int, out_planes: int, stride: int = 1, output_padding: int = 0) -> nn.ConvTranspose2d:
    """1x1 convolution"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False,
                           output_padding=output_padding)


class BuildingBlock(Module):
    EXPANSION: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Module | None = None,
    ):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity  # Shortcut connection
        out = self.relu(out)
        return out


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


class ResNet34(Module):

    BLOCK_EXPANSION = BuildingBlock.EXPANSION
    LAYERS = [3, 4, 6, 3]

    def __init__(
        self,
        num_classes: int,
        groups: int = 1,
        width_per_group: int = 64,
    ):
        super().__init__()
        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, self.LAYERS[0])
        self.layer2 = self._make_layer(128, self.LAYERS[1], stride=2)
        self.layer3 = self._make_layer(256, self.LAYERS[2], stride=2)
        self.layer4 = self._make_layer(512, self.LAYERS[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BuildingBlock.EXPANSION, num_classes)
        self.init_weights()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                init.normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def _make_layer(
        self,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        downsample = None
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * BuildingBlock.EXPANSION:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * BuildingBlock.EXPANSION, stride),
                nn.BatchNorm2d(planes * BuildingBlock.EXPANSION))
        layers = []
        layers.append( BuildingBlock(self.inplanes, planes, stride, downsample) )
        self.inplanes = planes * BuildingBlock.EXPANSION
        for _ in range(1, blocks):
            layers.append( BuildingBlock(self.inplanes, planes) )
        return nn.Sequential(*layers)


class ResNet34Decoder(Module):
    def __init__(self, num_classes: int, upsample_size: int):
        super().__init__()
        self.inplanes = 2048
        self.upsample = nn.Upsample(size=upsample_size)
        self.layer4 = self._make_layer(512, ResNet34.LAYERS[3], stride=2)
        self.layer3 = self._make_layer(256, ResNet34.LAYERS[2], stride=2)
        self.layer2 = self._make_layer(128, ResNet34.LAYERS[1], stride=2)
        self.layer1 = self._make_layer(64, ResNet34.LAYERS[0], output_padding = 0,
                                       last_block_dim=64)
        self.classifier = nn.ConvTranspose2d(64, num_classes, kernel_size=6, stride=4, padding=1,
                                             dilation=1)
        self.init_weights()

    def forward(self, x: Tensor) -> Tensor:
        x = self.upsample(x)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
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
        self.inplanes = planes * BuildingBlock.EXPANSION
        if last_block_dim == 0:
            last_block_dim = self.inplanes // 2
        if stride != 1 or self.inplanes != planes * BuildingBlock.EXPANSION or output_padding == 0:
            upsample = nn.Sequential(
                convtrans1x1(planes * BuildingBlock.EXPANSION, last_block_dim, stride,
                             output_padding),
                nn.BatchNorm2d(last_block_dim))
        layers = []
        last_block = DecoderBuildingBlock(last_block_dim, planes, stride, output_padding, upsample)
        for _ in range(1, blocks):
            layers.append( DecoderBuildingBlock(self.inplanes, planes) )
        layers.append(last_block)
        return nn.Sequential(*layers)


class UNet(Module):
    def __init__(self, num_classes: int, input_size: int):
        super().__init__()
        upsample_size = input_size // 32
        self.input_size = input_size
        self.encoder = ResNet34(10)
        self.decoder = ResNet34Decoder(num_classes, upsample_size)

    def forward(self, x: Tensor) -> Tensor:
        # (C, H, W)
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)
        stage1 = x # (64, 128, 128)
        x = self.encoder.layer1(x)
        block1 = x # (64, 128, 128)
        x = self.encoder.layer2(x)
        block2 = x # (128, 64, 64)
        x = self.encoder.layer3(x)
        block3 = x # (256, 32, 32)
        x = self.encoder.layer4(x)
        block4 = x # (512, 16, 16)
        x = self.encoder.avgpool(x) # (512, 1, 1)
        x = self.decoder.upsample(x) # (512, 16, 16)
        x += block4
        x = self.decoder.layer4(x) # (256, 32, 32)
        x1 = x + block3
        x1 = self.decoder.layer3(x1) # (128, 64, 64)
        x2 = x1 + block2
        x2 = self.decoder.layer2(x2) # (64, 128, 128)
        x3 = x2 + block1
        x3 = self.decoder.layer1(x3) # (64, 128, 128)
        x4 = x3 + stage1
        x4 = self.decoder.classifier(x4) # (N, H, W)
        return x4


def load_unet(num_classes: int, input_size: int) -> UNet:
    model = UNet(num_classes, input_size)
    opt_model = torch.compile(model, dynamic=False, mode="reduce-overhead")
    return opt_model # type: ignore


if __name__ == "__main__":
    WINDOW_SIZE = 224
    unet = UNet(2, WINDOW_SIZE)
    image = torch.rand(1, 3, WINDOW_SIZE, WINDOW_SIZE)
    unet.eval()
    with torch.no_grad():
        predictions = unet(image)
