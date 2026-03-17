import jax
import jax.numpy as jnp

import equinox as eqx

from src.model.blocks import (
    AttentionBlock,
    Downsample,
    ResBlock,
    SinusoidalEmbedding,
    Upsample,
)
from src.utils.utils import resolve_import
from typing import Callable, List, Optional


class UNet(eqx.Module):
    stem: eqx.nn.Conv2d
    time_emb: SinusoidalEmbedding
    encoder_blocks: List[List[ResBlock]]
    downsamples: List[Optional[Downsample]]
    mid_block1: ResBlock
    mid_attn: AttentionBlock
    mid_block2: ResBlock
    decoder_blocks: List[List[ResBlock]]
    upsamples: List[Optional[Upsample]]
    final_norm: eqx.nn.GroupNorm
    final_conv: eqx.nn.Conv2d
    activation: Callable = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int,
        channel_multipliers: List[int],
        num_res_blocks: int,
        num_heads: int,
        num_groups: int,
        activation: Callable,
        key: jax.Array,
    ):
        keys = jax.random.split(key, 256)
        ki = 0
        time_emb_dim = base_channels * 4
        L = len(channel_multipliers)

        if isinstance(activation, str):
            activation = resolve_import(activation)

        self.activation = activation
        self.stem = eqx.nn.Conv2d(
            in_channels, base_channels, 3, padding=1, key=keys[ki]
        )
        ki += 1
        self.time_emb = SinusoidalEmbedding(time_emb_dim, activation, keys[ki])
        ki += 1

        # Encoder
        enc_blocks, downsamples = [], []
        ch_in = base_channels
        for l in range(L):
            ch_out = base_channels * channel_multipliers[l]
            level = []
            for i in range(num_res_blocks):
                block_ch_in = ch_in if i == 0 else ch_out
                level.append(
                    ResBlock(
                        block_ch_in,
                        ch_out,
                        time_emb_dim,
                        num_groups,
                        activation,
                        keys[ki],
                    )
                )
                ki += 1
            enc_blocks.append(level)
            ch_in = ch_out
            if l < L - 1:
                downsamples.append(Downsample(ch_out, keys[ki]))
                ki += 1
            else:
                downsamples.append(None)
        self.encoder_blocks = enc_blocks
        self.downsamples = downsamples

        # Bottleneck
        ch_bot = base_channels * channel_multipliers[-1]
        self.mid_block1 = ResBlock(
            ch_bot, ch_bot, time_emb_dim, num_groups, activation, keys[ki]
        )
        ki += 1
        self.mid_attn = AttentionBlock(ch_bot, num_heads, keys[ki])
        ki += 1
        self.mid_block2 = ResBlock(
            ch_bot, ch_bot, time_emb_dim, num_groups, activation, keys[ki]
        )
        ki += 1

        # Decoder (iterate levels from L-1 down to 0)
        dec_blocks, upsample_list = [], []
        ch_current = ch_bot
        for l in reversed(range(L)):
            ch_skip = base_channels * channel_multipliers[l]
            ch_out = ch_skip
            if l < L - 1:
                upsample_list.append(Upsample(ch_current, keys[ki]))
                ki += 1
            else:
                upsample_list.append(None)
            level = []
            level.append(
                ResBlock(
                    ch_current + ch_skip,
                    ch_out,
                    time_emb_dim,
                    num_groups,
                    activation,
                    keys[ki],
                )
            )
            ki += 1
            for _ in range(num_res_blocks - 1):
                level.append(
                    ResBlock(
                        ch_out, ch_out, time_emb_dim, num_groups, activation, keys[ki]
                    )
                )
                ki += 1
            dec_blocks.append(level)
            ch_current = ch_out
        self.decoder_blocks = dec_blocks
        self.upsamples = upsample_list

        # Final head (output channels = base_channels after last decoder level)
        self.final_norm = eqx.nn.GroupNorm(num_groups, base_channels)
        self.final_conv = eqx.nn.Conv2d(
            base_channels, out_channels, 3, padding=1, key=keys[ki]
        )
        ki += 1

    def __call__(self, t: jax.Array, x_t: jax.Array, *args, **kwargs) -> jax.Array:
        time_emb = self.time_emb(t)
        h = self.stem(x_t)

        skips = []
        for blocks, downsample in zip(self.encoder_blocks, self.downsamples):
            for block in blocks:
                h = block(h, time_emb)
            skips.append(h)
            if downsample is not None:
                h = downsample(h)

        h = self.mid_block1(h, time_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, time_emb)

        for blocks, upsample, skip in zip(
            self.decoder_blocks, self.upsamples, skips[::-1]
        ):
            if upsample is not None:
                _, target_h, target_w = skip.shape
                h = upsample(h, target_h, target_w)
            h = jnp.concatenate([h, skip], axis=0)
            for block in blocks:
                h = block(h, time_emb)

        h = self.final_norm(h)
        h = self.activation(h)
        h = self.final_conv(h)
        return h
