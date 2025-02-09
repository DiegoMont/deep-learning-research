import torch
from torch.nn import AdaptiveAvgPool2d, AdaptiveMaxPool2d, ConvTranspose2d, Conv2d, Module, ReLU, Sequential, Sigmoid, Upsample
from torchvision.models import resnet
from torchvision.models.resnet import ResNet50_Weights
from torchvision.ops import MLP

from models.senet import SEResNet


class ChannelAttentionModule(Module):
    def __init__(self, channels: int, reduction_ratio: int):
        super().__init__()
        self.max_pool = AdaptiveMaxPool2d(1)
        self.avg_pool = AdaptiveAvgPool2d(1)
        c_over_q = channels // reduction_ratio
        self.mlp = MLP(channels, hidden_channels=[c_over_q, channels])
        self.encoder_layer = None

    def forward(self, x):
        b, c, _, _ = x.shape
        out = self.max_pool(x)  # (B, C, 1, 1)
        out = torch.squeeze(out)  # (B, C)
        out = self.mlp(out)  # (B, C)
        x = self.avg_pool(x)  # (B, C, 1, 1)
        x = torch.squeeze(x)  # (B, C)
        x = self.mlp(x)  # (B, C)
        out = out + x  # (B, C)
        out = Sigmoid()(out)
        return out.view(b, c, 1, 1)


class SpatialAttentionModule(Module):
    def __init__(self):
        super().__init__()
        filter_size = 3
        self.conv = Conv2d(2, 1, filter_size, padding=(filter_size - 1) // 2, bias=False)
        self.relu = ReLU()

    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True).values  # (B, 1, H, W)
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        x = torch.cat([max_pool, avg_pool], dim=1) # (B, 2, H, W)
        x = self.conv(x)  # (B, 1, H, W)
        out = self.relu(x)  # (B, 1, H, W)
        return out  # (B, 1, H, W)


class DualAttentionModule(Module):
    def __init__(self, channels: int, q: int):
        super().__init__()
        self.channel_attention = ChannelAttentionModule(channels, q)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        ch_attention = self.channel_attention(x)
        ch_attention = ch_attention * x
        space_attention = self.spatial_attention(ch_attention)
        out = space_attention * x
        return out


class TinyCrackNet(Module):
    def __init__(self, num_classes: int):
        super().__init__()
        # bottom-up encoder
        backbone = resnet.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.encoder = SEResNet(1, backbone)

        # up-sampling decoder
        self.decoder0 = SEResNet.make_se_layer(1024, resnet.conv3x3(2048, 1024))
        self.deconv0 = ConvTranspose2d(1024, 1024, kernel_size=2, stride=2)
        self.decoder1 = SEResNet.make_se_layer(1024, resnet.conv3x3(2048, 1024))
        self.deconv1 = ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder2 = SEResNet.make_se_layer(512, resnet.conv3x3(1024, 512))
        self.deconv2 = ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = SEResNet.make_se_layer(1024, resnet.conv3x3(512, 1024))
        self.decoder4 = Sequential(
            Conv2d(1024, 1024, 3, padding=1, bias=False),
            #Softmax2d()
        )

        # attention fusion architecture
        self.dual_attention = DualAttentionModule(1024, 16)

        # head
        self.segmentation_map = Sequential(
            Conv2d(1024, 512, 3, padding=1, bias=False),
            Conv2d(512, 256, 3, padding=1, bias=False),
            Upsample(scale_factor=4))
        self.classifier: Module = Conv2d(256, num_classes, 1, bias=False)

    def forward(self, x):
        _, _, H, W = x.shape
        x = self.encoder.se_conv1(x)  # (64, H/4, W/4)
        concat1 = self.encoder.se_conv2_x(x)  # (256, H/4, W/4)
        concat2 = self.encoder.se_conv3_x(concat1)  # (512, H/8, W/8)
        concat3 = self.encoder.se_conv4_x(concat2)  # (1024, H/16, W/16)
        x = self.encoder.se_conv5_x(concat3)  # (2048, H/32, W/32)

        x = self.decoder0(x)  # (1024, H/32, W/32)
        x = self.deconv0(x)  # (1024, H/16, W/16)
        x = torch.cat([concat3, x], dim=1)  # (2048, H/16, W/16)
        x = self.decoder1(x)  # (1024, H/16, W/16)
        x = self.deconv1(x)  # (512, H/8, W/8)
        x = torch.cat([concat2, x], dim=1)  # (1024, H/8, W/8)
        x = self.decoder2(x)  # (512, H/8, W/8)
        x = self.deconv2(x)  # (256, H/4, W/4)
        x = torch.cat([concat1, x], dim=1)  # (512, H/4, W/4)
        x = self.decoder3(x)  # (1024, H/4, W/4)
        soft_attention = x
        x = self.decoder4(x)  # (1024, H/4, W/4)

        soft_attention = self.dual_attention(soft_attention)  # (1024, H/4, W/4)

        x = x + soft_attention  # (1024, H/4, W/4)
        x = self.segmentation_map(x)  # (256, H, W)
        x = self.classifier(x)  # (N, H, W)

        return x


if __name__ == "__main__":
    device = torch.device("cuda")
    test_batch = torch.rand(16, 3, 480, 640).to(device)  # (B, C, H, W)
    tcn = TinyCrackNet(1).to(device)
    output = tcn(test_batch)
    pass
