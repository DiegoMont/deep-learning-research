import torch
from torch.nn import AdaptiveAvgPool2d, Linear, Module, ReLU, Sequential, Sigmoid


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


if __name__ == "__main__":
    test_batch = torch.rand(32, 3, 224, 224)
    model = Sequential(
        AdaptiveAvgPool2d(1)
    )
    output = model(test_batch)
    pass
