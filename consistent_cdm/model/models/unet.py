
from __future__ import annotations

from abc import abstractmethod
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn import checkpoint, count_flops_attn, gamma_embedding, normalization, zero_module


class SiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class EmbedBlock(nn.Module):
    @abstractmethod
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class EmbedSequential(nn.Sequential, EmbedBlock):
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        for layer in self:
            x = layer(x, emb) if isinstance(layer, EmbedBlock) else layer(x)
        return x


class Upsample(nn.Module):
    def __init__(self, channels: int, use_conv: bool, out_channel: int | None = None):
        super().__init__()
        self.conv = nn.Conv2d(channels, out_channel or channels, 3, padding=1) if use_conv else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x) if self.conv is not None else x


class Downsample(nn.Module):
    def __init__(self, channels: int, use_conv: bool, out_channel: int | None = None):
        super().__init__()
        self.op = nn.Conv2d(channels, out_channel or channels, 3, stride=2, padding=1) if use_conv else nn.AvgPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class ResBlock(EmbedBlock):
    def __init__(
        self,
        channels: int,
        emb_channels: int,
        dropout: float,
        out_channel: int | None = None,
        use_conv: bool = False,
        use_scale_shift_norm: bool = False,
        use_checkpoint: bool = False,
        up: bool = False,
        down: bool = False,
    ):
        super().__init__()
        self.out_channel = out_channel or channels
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.in_layers = nn.Sequential(normalization(channels), SiLU(), nn.Conv2d(channels, self.out_channel, 3, padding=1))
        self.updown = up or down
        if up:
            self.h_upd = Upsample(channels, False)
            self.x_upd = Upsample(channels, False)
        elif down:
            self.h_upd = Downsample(channels, False)
            self.x_upd = Downsample(channels, False)
        else:
            self.h_upd = nn.Identity()
            self.x_upd = nn.Identity()
        self.emb_layers = nn.Sequential(
            SiLU(),
            nn.Linear(emb_channels, 2 * self.out_channel if use_scale_shift_norm else self.out_channel),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channel),
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(self.out_channel, self.out_channel, 3, padding=1)),
        )
        if self.out_channel == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(channels, self.out_channel, 3, padding=1)
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channel, 1)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        return checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)

    def _forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = self.h_upd(in_rest(x))
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = self.out_layers(h + emb_out)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int = 1,
        num_head_channels: int = -1,
        use_checkpoint: bool = False,
        use_new_attention_order: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads if num_head_channels == -1 else channels // num_head_channels
        self.norm = normalization(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.attention = QKVAttention(self.num_heads) if use_new_attention_order else QKVAttentionLegacy(self.num_heads)
        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, *spatial = x.shape
        flat = x.reshape(batch, channels, -1)
        h = self.attention(self.qkv(self.norm(flat)))
        return (flat + self.proj_out(h)).reshape(batch, channels, *spatial)


class QKVAttentionLegacy(nn.Module):
    def __init__(self, n_heads: int):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        batch, width, length = qkv.shape
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(batch * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.softmax(torch.einsum("bct,bcs->bts", q * scale, k * scale).float(), dim=-1).type(q.dtype)
        return torch.einsum("bts,bcs->bct", weight, v).reshape(batch, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    def __init__(self, n_heads: int):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        batch, width, length = qkv.shape
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.softmax(
            torch.einsum(
                "bct,bcs->bts",
                (q * scale).view(batch * self.n_heads, ch, length),
                (k * scale).view(batch * self.n_heads, ch, length),
            ).float(),
            dim=-1,
        ).type(q.dtype)
        return torch.einsum("bts,bcs->bct", weight, v.reshape(batch * self.n_heads, ch, length)).reshape(batch, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class UNet(nn.Module):
    def __init__(
        self,
        image_size: int,
        in_channel: int,
        inner_channel: int,
        out_channel: int,
        res_blocks: int,
        attn_res: list[int],
        dropout: float = 0.0,
        channel_mults: tuple[int, ...] = (1, 2, 4, 8),
        conv_resample: bool = True,
        use_checkpoint: bool = False,
        use_fp16: bool = False,
        num_heads: int = 1,
        num_head_channels: int = -1,
        num_heads_upsample: int = -1,
        use_scale_shift_norm: bool = True,
        resblock_updown: bool = True,
        use_new_attention_order: bool = False,
        num_side_classes: int | None = None,
        branch_decoder: bool = False,
        branch_split_blocks: int | None = None,
    ):
        super().__init__()
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        self.inner_channel = inner_channel
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.branch_decoder = branch_decoder
        cond_embed_dim = inner_channel * 4
        self.cond_embed = nn.Sequential(nn.Linear(inner_channel, cond_embed_dim), SiLU(), nn.Linear(cond_embed_dim, cond_embed_dim))
        self.side_embed = nn.Embedding(num_side_classes, cond_embed_dim) if num_side_classes is not None else None

        ch = input_ch = int(channel_mults[0] * inner_channel)
        self.input_blocks = nn.ModuleList([EmbedSequential(nn.Conv2d(in_channel, ch, 3, padding=1))])
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mults):
            for _ in range(res_blocks):
                layers: list[nn.Module] = [
                    ResBlock(ch, cond_embed_dim, dropout, out_channel=int(mult * inner_channel), use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)
                ]
                ch = int(mult * inner_channel)
                if ds in attn_res:
                    layers.append(AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads, num_head_channels=num_head_channels, use_new_attention_order=use_new_attention_order))
                self.input_blocks.append(EmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mults) - 1:
                out_ch = ch
                self.input_blocks.append(
                    EmbedSequential(
                        ResBlock(ch, cond_embed_dim, dropout, out_channel=out_ch, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm, down=True)
                        if resblock_updown
                        else Downsample(ch, conv_resample, out_channel=out_ch)
                    )
                )
                input_block_chans.append(out_ch)
                ds *= 2

        self.middle_block = EmbedSequential(
            ResBlock(ch, cond_embed_dim, dropout, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm),
            AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads, num_head_channels=num_head_channels, use_new_attention_order=use_new_attention_order),
            ResBlock(ch, cond_embed_dim, dropout, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm),
        )
        output_block_specs: list[tuple[int, int, bool, bool]] = []
        for level, mult in list(enumerate(channel_mults))[::-1]:
            for i in range(res_blocks + 1):
                ich = input_block_chans.pop()
                out_ch = int(inner_channel * mult)
                has_attn = ds in attn_res
                has_upsample = bool(level and i == res_blocks)
                output_block_specs.append((ch + ich, out_ch, has_attn, has_upsample))
                ch = out_ch
                if has_upsample:
                    ds //= 2

        total_output_blocks = len(output_block_specs)
        if self.branch_decoder:
            if out_channel != 2:
                raise ValueError("branch_decoder currently expects out_channel=2 for left/right prediction")
            shared_blocks = total_output_blocks // 2 if branch_split_blocks is None else int(branch_split_blocks)
            if not 0 <= shared_blocks < total_output_blocks:
                raise ValueError(f"branch_split_blocks must be in [0, {total_output_blocks - 1}], got {shared_blocks}")
        else:
            shared_blocks = total_output_blocks

        self.shared_output_blocks = nn.ModuleList(
            [
                self._build_output_block(
                    in_channels=spec[0],
                    out_channels=spec[1],
                    use_attn=spec[2],
                    upsample=spec[3],
                    cond_embed_dim=cond_embed_dim,
                    dropout=dropout,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                    num_heads=num_heads_upsample,
                    num_head_channels=num_head_channels,
                    use_new_attention_order=use_new_attention_order,
                    resblock_updown=resblock_updown,
                    conv_resample=conv_resample,
                )
                for spec in output_block_specs[:shared_blocks]
            ]
        )
        self.output_blocks = self.shared_output_blocks

        final_shared_channels = output_block_specs[shared_blocks - 1][1] if shared_blocks > 0 else ch
        if self.branch_decoder:
            branch_specs = output_block_specs[shared_blocks:]
            self.left_output_blocks = nn.ModuleList(
                [
                    self._build_output_block(
                        in_channels=spec[0],
                        out_channels=spec[1],
                        use_attn=spec[2],
                        upsample=spec[3],
                        cond_embed_dim=cond_embed_dim,
                        dropout=dropout,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        num_heads=num_heads_upsample,
                        num_head_channels=num_head_channels,
                        use_new_attention_order=use_new_attention_order,
                        resblock_updown=resblock_updown,
                        conv_resample=conv_resample,
                    )
                    for spec in branch_specs
                ]
            )
            self.right_output_blocks = nn.ModuleList(
                [
                    self._build_output_block(
                        in_channels=spec[0],
                        out_channels=spec[1],
                        use_attn=spec[2],
                        upsample=spec[3],
                        cond_embed_dim=cond_embed_dim,
                        dropout=dropout,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        num_heads=num_heads_upsample,
                        num_head_channels=num_head_channels,
                        use_new_attention_order=use_new_attention_order,
                        resblock_updown=resblock_updown,
                        conv_resample=conv_resample,
                    )
                    for spec in branch_specs
                ]
            )
            final_branch_channels = branch_specs[-1][1]
            self.out_left = nn.Sequential(normalization(final_branch_channels), SiLU(), zero_module(nn.Conv2d(final_branch_channels, 1, 3, padding=1)))
            self.out_right = nn.Sequential(normalization(final_branch_channels), SiLU(), zero_module(nn.Conv2d(final_branch_channels, 1, 3, padding=1)))
            self.out = None
        else:
            self.left_output_blocks = None
            self.right_output_blocks = None
            self.out_left = None
            self.out_right = None
            self.out = nn.Sequential(normalization(final_shared_channels), SiLU(), zero_module(nn.Conv2d(final_shared_channels, out_channel, 3, padding=1)))

    def _build_output_block(
        self,
        in_channels: int,
        out_channels: int,
        use_attn: bool,
        upsample: bool,
        cond_embed_dim: int,
        dropout: float,
        use_checkpoint: bool,
        use_scale_shift_norm: bool,
        num_heads: int,
        num_head_channels: int,
        use_new_attention_order: bool,
        resblock_updown: bool,
        conv_resample: bool,
    ) -> EmbedSequential:
        layers: list[nn.Module] = [
            ResBlock(in_channels, cond_embed_dim, dropout, out_channel=out_channels, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)
        ]
        if use_attn:
            layers.append(
                AttentionBlock(
                    out_channels,
                    use_checkpoint=use_checkpoint,
                    num_heads=num_heads,
                    num_head_channels=num_head_channels,
                    use_new_attention_order=use_new_attention_order,
                )
            )
        if upsample:
            layers.append(
                ResBlock(out_channels, cond_embed_dim, dropout, out_channel=out_channels, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm, up=True)
                if resblock_updown
                else Upsample(out_channels, conv_resample, out_channel=out_channels)
            )
        return EmbedSequential(*layers)

    def _forward_decoder_branch(
        self,
        h: torch.Tensor,
        hs: list[torch.Tensor],
        emb: torch.Tensor,
        blocks: nn.ModuleList,
    ) -> torch.Tensor:
        branch_h = h
        branch_hs = list(hs)
        for module in blocks:
            branch_h = module(torch.cat([branch_h, branch_hs.pop()], dim=1), emb)
        return branch_h

    def forward(self, x: torch.Tensor, gammas: torch.Tensor, side: torch.Tensor | None = None) -> torch.Tensor:
        hs = []
        emb = self.cond_embed(gamma_embedding(gammas.view(-1), self.inner_channel))
        if self.side_embed is not None:
            if side is None:
                raise ValueError("side labels are required when num_side_classes is enabled")
            emb = emb + self.side_embed(side.view(-1).long())
        h = x.type(torch.float32)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.shared_output_blocks:
            h = module(torch.cat([h, hs.pop()], dim=1), emb)
        if not self.branch_decoder:
            assert self.out is not None
            return self.out(h.type(x.dtype))

        assert self.left_output_blocks is not None
        assert self.right_output_blocks is not None
        assert self.out_left is not None
        assert self.out_right is not None
        left_h = self._forward_decoder_branch(h, hs, emb, self.left_output_blocks)
        right_h = self._forward_decoder_branch(h, hs, emb, self.right_output_blocks)
        left = self.out_left(left_h.type(x.dtype))
        right = self.out_right(right_h.type(x.dtype))
        return torch.cat([left, right], dim=1)
