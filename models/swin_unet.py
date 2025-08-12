from einops import rearrange
import torch
from torch.nn import Conv2d, init, LayerNorm, Linear, Module, ModuleList, Sequential
from torchvision import models
from torchvision.models.swin_transformer import SwinTransformerBlockV2, Swin_V2_T_Weights


class PatchExpanding(Module):
    def __init__(self, dim, dim_scale=2, norm_layer=LayerNorm):
        super().__init__()
        self.norm = norm_layer(dim)
        self.expand = Linear(dim, dim_scale * dim, bias=False)

    def forward(self, x):
        x = self.norm(x)  # (H, W, C)
        x = self.expand(x)  # (H, W, 2C)
        C = x.shape[3]
        x = rearrange(x, 'b h w (p1 p2 c) -> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        return x


class FinalPatchExpanding(Module):
    def __init__(self, dim, norm_layer=LayerNorm):
        super().__init__()
        self.norm = norm_layer(dim)
        self.expand = Linear(dim, 16 * dim, bias=False)

    def forward(self, x):
        x = self.norm(x)  # (H, W, C)
        x = self.expand(x)  # (H, W, 16C)
        C = x.shape[3]
        dim_scale = 4
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=dim_scale, p2=dim_scale,
                      c=C // (dim_scale ** 2))
        return x


class SwinUnet(Module):
    def __init__(self, num_classes: int):
        super().__init__()
        C = 96
        # Encoder
        encoder = models.swin_v2_t(weights=Swin_V2_T_Weights.IMAGENET1K_V1)
        self.patch_embed = encoder.features[0]
        self.encoder_stage1 = Sequential(encoder.features[1:3])
        self.encoder_stage2 = Sequential(encoder.features[3:5])
        self.encoder_stage3 = Sequential(encoder.features[5:7])
        self.bottleneck = Sequential(encoder.features[7])
        self.norm = LayerNorm(8 * C)

        # Decoder
        self.decoder_expands = ModuleList()
        self.decoder_blocks = ModuleList()
        depths_decoder = [6, 2, 2]
        dim = C*8
        num_heads = 12
        for i, depth in enumerate(depths_decoder):
            self.decoder_expands.append(PatchExpanding(dim))
            self.decoder_blocks.append(self._build_decoder_block(dim, depth, num_heads))
            num_heads //= 2
            dim //= 2
        self.norm_up = LayerNorm(C)

        # Upsample and head
        self.final_upsample = FinalPatchExpanding(C)
        self.segmentation_head = Conv2d(C, num_classes, 1, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, Linear):
            init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, Linear) and m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm):
            init.constant_(m.bias, 0)
            init.constant_(m.weight, 1.0)

    def forward(self, x):
        skip1 = self.patch_embed(x)  # (H/4, W/4, C)
        skip2 = self.encoder_stage1(skip1)  # (H/8, W/8, 2C)
        skip3 = self.encoder_stage2(skip2)  # (H/16, W/16, 4C)
        x = self.encoder_stage3(skip3)  # (H/32, W/32, 8C)
        x = self.bottleneck(x)  # (H/32, W/32, 8C)
        x = self.norm(x)

        for i, skip_feature in enumerate([skip3, skip2, skip1]):
            x = self.decoder_expands[i](x)  # (Hd, Wd, Cd)
            x = torch.cat([x, skip_feature], dim=3)  # (Hd, Wd, 2Cd)
            x = self.decoder_blocks[i](x)  # (Hd, Wd, Cd)
        x = self.norm_up(x)

        x = self.final_upsample(x)  # (C, H, W)
        x = x.permute(0, 3, 1, 2)
        x = self.segmentation_head(x)  # (N, H, W)
        return x

    def _build_decoder_block(self, in_channels: int, depth: int, num_heads: int):
        dim = in_channels // 2
        window_size = [8, 8]
        layers: list[Module] = [Linear(in_channels, dim, bias=False)]
        for i_layer in range(depth):
            block = SwinTransformerBlockV2(dim, num_heads, window_size=window_size,
                        shift_size=[0 if i_layer % 2 == 0 else w // 2 for w in window_size])
            layers.append(block)
        return Sequential(*layers)


if __name__ == "__main__":
    device = torch.device("cpu")
    WINDOW_SIZE = 224
    swin_unet = SwinUnet(1).to(device)
    image = torch.rand(4, 3, WINDOW_SIZE, WINDOW_SIZE).to(device)
    swin_unet.eval()
    with torch.no_grad():
        predictions = swin_unet(image)
