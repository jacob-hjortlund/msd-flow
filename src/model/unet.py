"""UNet velocity-field predictor for flow matching.

Encoder–bottleneck–decoder architecture with skip connections,
sinusoidal time conditioning, and mid-level self-attention.
"""

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
    """Time-conditioned UNet that predicts the velocity field v_t = UNet(t, x_t).

    Attributes:
        stem: Initial convolution projecting input channels to base channels.
        time_emb: Sinusoidal time embedding module.
        cond_dim: Number of conditioning dimensions (0 = unconditional).
        cond_embed: Conditioning embedding module (``None`` when ``cond_dim=0``).
        null_cond_emb: Learnable null conditioning embedding (``None`` when ``cond_dim=0``).
        encoder_blocks: Per-level lists of ``ResBlock`` modules.
        downsamples: Per-level downsamplers (``None`` at the deepest level).
        mid_block1: First bottleneck ``ResBlock``.
        mid_attn: Bottleneck ``AttentionBlock``.
        mid_block2: Second bottleneck ``ResBlock``.
        decoder_blocks: Per-level lists of ``ResBlock`` modules.
        upsamples: Per-level upsamplers (``None`` at the deepest level).
        final_norm: Output ``GroupNorm``.
        final_conv: 3x3 convolution projecting to output channels.
        activation: Activation function used throughout.
    """

    stem: eqx.nn.Conv2d
    time_emb: SinusoidalEmbedding
    cond_dim: int = eqx.field(static=True)
    cond_embed: Optional[SinusoidalEmbedding]
    null_cond_emb: Optional[jax.Array]
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
        cond_dim: int = 0,
    ):
        """Initialise encoder, bottleneck, and decoder stages.

        Args:
            in_channels: Number of input image channels.
            out_channels: Number of output channels (velocity field).
            base_channels: Channel count at the first encoder level.
            channel_multipliers: Per-level channel multipliers.
            num_res_blocks: ``ResBlock`` count per encoder/decoder level.
            num_heads: Attention heads in the bottleneck.
            num_groups: Groups for all ``GroupNorm`` layers.
            activation: Activation function (or import string).
            key: JAX PRNG key.
            cond_dim: Number of conditioning dimensions. Supports 0 (unconditional)
                or 1 (scalar condition such as redshift). Values > 1 raise
                ``ValueError``.
        """
        keys = jax.random.split(key, 256)
        ki = 0
        time_emb_dim = base_channels * 4
        L = len(channel_multipliers)

        if cond_dim > 1:
            raise ValueError(
                f"cond_dim={cond_dim} is not supported; only cond_dim=0 "
                "(unconditional) or cond_dim=1 (scalar condition) are implemented."
            )

        if isinstance(activation, str):
            activation = resolve_import(activation)

        self.activation = activation
        self.stem = eqx.nn.Conv2d(
            in_channels, base_channels, 3, padding=1, key=keys[ki]
        )
        ki += 1
        self.time_emb = SinusoidalEmbedding(time_emb_dim, activation, keys[ki])
        ki += 1
        self.cond_dim = cond_dim
        if cond_dim > 0:
            self.cond_embed = SinusoidalEmbedding(time_emb_dim, activation, keys[ki])
            ki += 1
            self.null_cond_emb = jnp.zeros(time_emb_dim)
        else:
            self.cond_embed = None
            self.null_cond_emb = None

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

    def __call__(self, t: jax.Array, x_t: jax.Array, cond: jax.Array, cond_mask: jax.Array) -> jax.Array:
        """Predict the velocity field at time *t*.

        Args:
            t: Scalar time value in ``[0, 1]``.
            x_t: Noisy image of shape ``(C, H, W)``.
            cond: Conditioning vector of shape ``(cond_dim,)``. When
                ``cond_dim=1``, ``cond[0]`` is embedded as a scalar.
                When ``cond_dim=0`` the model ignores ``cond`` entirely;
                pass ``jnp.zeros(1)`` as a safe dummy value.
            cond_mask: Scalar bool. ``True`` uses the real condition
                embedding; ``False`` uses the learnable null embedding.
                When ``cond_dim=0`` this argument is ignored; pass
                ``jnp.array(False)`` by convention.

        Returns:
            Predicted velocity field of shape ``(C, H, W)``.
        """
        time_emb = self.time_emb(t)

        if self.cond_dim > 0:
            cond_emb = self.cond_embed(cond[0])
            combined_emb = time_emb + jnp.where(cond_mask, cond_emb, self.null_cond_emb)
        else:
            combined_emb = time_emb

        h = self.stem(x_t)

        skips = []
        for blocks, downsample in zip(self.encoder_blocks, self.downsamples):
            for block in blocks:
                h = block(h, combined_emb)
            skips.append(h)
            if downsample is not None:
                h = downsample(h)

        h = self.mid_block1(h, combined_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, combined_emb)

        for blocks, upsample, skip in zip(
            self.decoder_blocks, self.upsamples, skips[::-1]
        ):
            if upsample is not None:
                _, target_h, target_w = skip.shape
                h = upsample(h, target_h, target_w)
            h = jnp.concatenate([h, skip], axis=0)
            for block in blocks:
                h = block(h, combined_emb)

        h = self.final_norm(h)
        h = self.activation(h)
        h = self.final_conv(h)
        return h
