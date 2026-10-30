from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int, preferred: int = 8) -> int:
    for groups in (preferred, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def _spectral(module: nn.Module, enabled: bool) -> nn.Module:
    return nn.utils.spectral_norm(module) if enabled else module


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        norm_groups: int = 8,
        use_se: bool = True,
    ) -> None:
        super().__init__()
        groups_in = _group_count(in_channels, norm_groups)
        groups_out = _group_count(out_channels, norm_groups)
        self.block = nn.Sequential(
            nn.GroupNorm(groups_in, in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups_out, out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            SqueezeExcite(out_channels) if use_se else nn.Identity(),
        )
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.skip(x) + self.block(x)


class SelfAttention2d(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4, norm_groups: int = 8) -> None:
        super().__init__()
        if channels % num_heads != 0:
            num_heads = 1
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(_group_count(channels, norm_groups), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, channels, height, width = x.shape
        qkv = self.qkv(self.norm(x)).reshape(bsz, 3, self.num_heads, channels // self.num_heads, height * width)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        scale = (channels // self.num_heads) ** -0.5
        attn = torch.softmax(torch.einsum("bhcn,bhcm->bhnm", q * scale, k), dim=-1)
        out = torch.einsum("bhnm,bhcm->bhcn", attn, v).reshape(bsz, channels, height, width)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.conv(x)


class ResUNetGenerator(nn.Module):
    """Strong deterministic pix2pix-style generator with residual blocks and low-res attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: int = 256,
        base_channels: int = 64,
        channel_mults: Sequence[int] = (1, 2, 4, 8),
        blocks_per_level: int = 2,
        dropout: float = 0.0,
        attention_resolutions: Sequence[int] = (16,),
        attention_heads: int = 4,
        norm_groups: int = 8,
        use_se: bool = True,
        final_activation: str = "tanh",
    ) -> None:
        super().__init__()
        if len(channel_mults) < 2:
            raise ValueError("channel_mults must contain at least two levels.")
        self.final_activation = final_activation
        channels = [int(base_channels * mult) for mult in channel_mults]
        self.stem = nn.Conv2d(in_channels, channels[0], 3, padding=1)

        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        resolution = int(image_size)
        in_ch = channels[0]
        for level, out_ch in enumerate(channels):
            blocks: list[nn.Module] = []
            for block_idx in range(blocks_per_level):
                block_in = in_ch if block_idx == 0 else out_ch
                blocks.append(ResidualBlock(block_in, out_ch, dropout=dropout, norm_groups=norm_groups, use_se=use_se))
            if resolution in attention_resolutions:
                blocks.append(SelfAttention2d(out_ch, num_heads=attention_heads, norm_groups=norm_groups))
            self.encoder_blocks.append(nn.Sequential(*blocks))
            in_ch = out_ch
            if level < len(channels) - 1:
                self.downsamples.append(Downsample(out_ch, channels[level + 1]))
                in_ch = channels[level + 1]
                resolution //= 2

        bottleneck_ch = channels[-1]
        self.bottleneck = nn.Sequential(
            ResidualBlock(bottleneck_ch, bottleneck_ch, dropout=dropout, norm_groups=norm_groups, use_se=use_se),
            SelfAttention2d(bottleneck_ch, num_heads=attention_heads, norm_groups=norm_groups),
            ResidualBlock(bottleneck_ch, bottleneck_ch, dropout=dropout, norm_groups=norm_groups, use_se=use_se),
        )

        self.upsamples = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        current_ch = channels[-1]
        for level in reversed(range(len(channels) - 1)):
            skip_ch = channels[level]
            self.upsamples.append(Upsample(current_ch, skip_ch))
            dec_blocks: list[nn.Module] = [
                ResidualBlock(skip_ch + skip_ch, skip_ch, dropout=dropout, norm_groups=norm_groups, use_se=use_se)
            ]
            for _ in range(max(blocks_per_level - 1, 0)):
                dec_blocks.append(ResidualBlock(skip_ch, skip_ch, dropout=dropout, norm_groups=norm_groups, use_se=use_se))
            resolution = image_size // (2**level)
            if resolution in attention_resolutions:
                dec_blocks.append(SelfAttention2d(skip_ch, num_heads=attention_heads, norm_groups=norm_groups))
            self.decoder_blocks.append(nn.Sequential(*dec_blocks))
            current_ch = skip_ch

        self.out = nn.Sequential(
            nn.GroupNorm(_group_count(current_ch, norm_groups), current_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(current_ch, out_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        skips: list[torch.Tensor] = []
        for level, blocks in enumerate(self.encoder_blocks):
            h = blocks(h)
            skips.append(h)
            if level < len(self.downsamples):
                h = self.downsamples[level](h)
        h = self.bottleneck(h)
        skips.pop()
        for upsample, blocks in zip(self.upsamples, self.decoder_blocks):
            h = upsample(h)
            skip = skips.pop()
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = blocks(torch.cat([h, skip], dim=1))
        h = self.out(h)
        if self.final_activation == "tanh":
            return torch.tanh(h)
        if self.final_activation == "sigmoid":
            return torch.sigmoid(h)
        if self.final_activation in {"identity", "none", None}:
            return h
        raise ValueError(f"Unsupported final activation: {self.final_activation}")


class PatchDiscriminator(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_channels: int = 64,
        num_layers: int = 4,
        max_channels: int = 512,
        spectral_norm: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            _spectral(nn.Conv2d(in_channels, base_channels, 4, stride=2, padding=1), spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        ch = base_channels
        for layer_idx in range(1, num_layers):
            out_ch = min(base_channels * (2**layer_idx), max_channels)
            stride = 1 if layer_idx == num_layers - 1 else 2
            layers.extend(
                [
                    _spectral(nn.Conv2d(ch, out_ch, 4, stride=stride, padding=1, bias=False), spectral_norm),
                    nn.InstanceNorm2d(out_ch, affine=True),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            ch = out_ch
        layers.append(_spectral(nn.Conv2d(ch, 1, 4, stride=1, padding=1), spectral_norm))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        features: list[torch.Tensor] = []
        h = x
        for layer in self.layers:
            h = layer(h)
            if return_features:
                features.append(h)
        return (h, features) if return_features else h


class MultiScalePatchDiscriminator(nn.Module):
    def __init__(
        self,
        condition_channels: int,
        target_channels: int,
        base_channels: int = 64,
        num_layers: int = 4,
        num_scales: int = 2,
        spectral_norm: bool = True,
    ) -> None:
        super().__init__()
        in_channels = condition_channels + target_channels
        self.discriminators = nn.ModuleList(
            [
                PatchDiscriminator(
                    in_channels=in_channels,
                    base_channels=base_channels,
                    num_layers=num_layers,
                    spectral_norm=spectral_norm,
                )
                for _ in range(num_scales)
            ]
        )

    def forward(self, condition: torch.Tensor, target: torch.Tensor, return_features: bool = False):
        x = torch.cat([condition, target], dim=1)
        outputs = []
        features = []
        for idx, discriminator in enumerate(self.discriminators):
            if idx > 0:
                x = F.avg_pool2d(x, kernel_size=3, stride=2, padding=1, count_include_pad=False)
            if return_features:
                logits, disc_features = discriminator(x, return_features=True)
                outputs.append(logits)
                features.append(disc_features)
            else:
                outputs.append(discriminator(x))
        return (outputs, features) if return_features else outputs


def build_generator(model_cfg: dict, condition_channels: int, target_channels: int) -> ResUNetGenerator:
    return ResUNetGenerator(
        in_channels=condition_channels,
        out_channels=target_channels,
        image_size=int(model_cfg.get("image_size", 256)),
        base_channels=int(model_cfg.get("base_channels", model_cfg.get("inner_channel", 64))),
        channel_mults=tuple(model_cfg.get("channel_mults", [1, 2, 4, 8])),
        blocks_per_level=int(model_cfg.get("blocks_per_level", model_cfg.get("res_blocks", 2))),
        dropout=float(model_cfg.get("dropout", 0.0)),
        attention_resolutions=tuple(model_cfg.get("attention_resolutions", model_cfg.get("attn_res", [16]))),
        attention_heads=int(model_cfg.get("attention_heads", 4)),
        norm_groups=int(model_cfg.get("norm_groups", 8)),
        use_se=bool(model_cfg.get("use_se", True)),
        final_activation=str(model_cfg.get("final_activation", "tanh")),
    )


def build_discriminator(model_cfg: dict, condition_channels: int, target_channels: int) -> MultiScalePatchDiscriminator:
    disc_cfg = model_cfg.get("discriminator", model_cfg)
    return MultiScalePatchDiscriminator(
        condition_channels=condition_channels,
        target_channels=target_channels,
        base_channels=int(disc_cfg.get("base_channels", 64)),
        num_layers=int(disc_cfg.get("num_layers", 4)),
        num_scales=int(disc_cfg.get("num_scales", 2)),
        spectral_norm=bool(disc_cfg.get("spectral_norm", True)),
    )
