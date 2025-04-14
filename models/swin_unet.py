import torch
from torch.nn import Module, Sequential
from torchvision import models


class SwinUnet(Module):
    def __init__(self):
        super().__init__()
        encoder = models.swin_v2_t()
        self.encoder_stage1 = Sequential(encoder.features[0:2])
        self.encoder_stage2 = Sequential(encoder.features[2:4])
        self.encoder_stage3 = Sequential(encoder.features[4:6])
        self.bottleneck = Sequential(encoder.features[6:])

    def forward(self, x):
        x = self.encoder_stage1(x)
        skip1 = x
        x = self.encoder_stage2(x)
        skip2 = x
        x = self.encoder_stage3(x)
        skip3 = x
        x = self.bottleneck(x)
        return x


if __name__ == "__main__":
    WINDOW_SIZE = 224
    swin_unet = SwinUnet()
    image = torch.rand(1, 3, WINDOW_SIZE, WINDOW_SIZE)
    swin_unet.eval()
    with torch.no_grad():
        predictions = swin_unet(image)
