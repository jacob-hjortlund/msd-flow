"""NCSN++ (non-progressive) velocity-field predictor for flow matching.

Implements the architecture from Song et al. 2021, "Score-Based Generative
Modeling Through Stochastic Differential Equations". Uses BigGAN-style
residual blocks with integrated up/down resampling, skip rescaling, and
self-attention at configurable resolutions.
"""

from typing import Callable, List, Optional

import jax
import jax.numpy as jnp
import equinox as eqx

from src.model.blocks import (
    AttnBlockNCSN,
    GaussianFourierProjection,
    ResBlockBigGAN,
)
from src.utils.utils import resolve_import


class NCSNpp(eqx.Module):
    """NCSN++ (non-progressive) that predicts v_t = NCSNpp(t, x_t).

    Attributes:
        stem: Initial 3x3 convolution.
        time_emb: Gaussian Fourier projection for time conditioning.
        cond_dim: Number of conditioning dimensions (0 = unconditional).
        cond_embed: Conditioning embedding module (``None`` when ``cond_dim=0``).
        null_cond_emb: Null conditioning embedding (``None`` when ``cond_dim=0``).
        encoder_blocks: Flat list of encoder ResBlock/AttnBlock modules.
        encoder_is_attn: Boolean flag per encoder block (True = attention).
        downsample_blocks: ResBlockBigGAN with down=True, one per level except last.
        mid_block1: First bottleneck ResBlock.
        mid_attn: Bottleneck attention.
        mid_block2: Second bottleneck ResBlock.
        decoder_blocks: Flat list of decoder ResBlock/AttnBlock modules.
        decoder_is_attn: Boolean flag per decoder block (True = attention).
        upsample_blocks: ResBlockBigGAN with up=True, one per level except last.
        final_norm: Output GroupNorm.
        final_conv: Output 3x3 convolution.
        activation: Activation function.
    """

    stem: eqx.nn.Conv2d
    time_emb: GaussianFourierProjection
    cond_dim: int = eqx.field(static=True)
    cond_embed: Optional[GaussianFourierProjection]
    null_cond_emb: Optional[jax.Array]

    encoder_blocks: List
    encoder_is_attn: List[bool]
    downsample_blocks: List

    mid_block1: ResBlockBigGAN
    mid_attn: AttnBlockNCSN
    mid_block2: ResBlockBigGAN

    decoder_blocks: List
    decoder_is_attn: List[bool]
    upsample_blocks: List

    final_norm: eqx.nn.GroupNorm
    final_conv: eqx.nn.Conv2d

    activation: Callable = eqx.field(static=True)
    channel_multipliers: List[int] = eqx.field(static=True)
    num_res_blocks: int = eqx.field(static=True)
    attn_resolutions: List[int] = eqx.field(static=True)
    image_size: int = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int,
        channel_multipliers: List[int],
        num_res_blocks: int,
        attn_resolutions: List[int],
        dropout: float,
        num_groups: int,
        num_heads: int,
        activation: Callable,
        fourier_scale: float,
        skip_rescale: bool,
        image_size: int,
        key: jax.Array,
        cond_dim: int = 0,
    ):
        """Initialise the NCSN++ architecture.

        Args:
            in_channels: Number of input image channels.
            out_channels: Number of output channels (velocity field).
            base_channels: Channel count at the first level.
            channel_multipliers: Per-level channel multipliers.
            num_res_blocks: ResBlocks per encoder/decoder level.
            attn_resolutions: Spatial resolutions at which to apply attention.
            dropout: Dropout rate for ResBlocks.
            num_groups: Groups for all GroupNorm layers.
            num_heads: Attention heads.
            activation: Activation function (or import string).
            fourier_scale: Scale for Gaussian Fourier projection.
            skip_rescale: If True, divide residual sums by sqrt(2).
            image_size: Input spatial resolution (assumed square).
            key: JAX PRNG key.
            cond_dim: Number of conditioning dimensions. Supports 0 (unconditional)
                or 1 (scalar condition such as redshift). Values > 1 raise
                ``ValueError``.
        """
        if isinstance(activation, str):
            activation = resolve_import(activation)

        self.activation = activation
        self.channel_multipliers = list(channel_multipliers)
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = list(attn_resolutions)
        self.image_size = image_size
        self.cond_dim = cond_dim

        if cond_dim > 1:
            raise ValueError(
                f"cond_dim={cond_dim} is not supported; only cond_dim=0 "
                "(unconditional) or cond_dim=1 (scalar condition) are implemented."
            )

        keys = jax.random.split(key, 512)
        ki = 0
        L = len(channel_multipliers)
        time_emb_dim = base_channels * 4

        # -- Stem --
        self.stem = eqx.nn.Conv2d(
            in_channels, base_channels, 3, padding=1, key=keys[ki]
        )
        ki += 1

        # -- Time embedding --
        self.time_emb = GaussianFourierProjection(time_emb_dim, fourier_scale, keys[ki])
        ki += 1

        # -- Conditioning embedding --
        # Use GaussianFourierProjection for the condition embedding (matching
        # NCSNpp's time embedding type) rather than SinusoidalEmbedding, so
        # both embeddings live in the same representational space.
        if cond_dim > 0:
            self.cond_embed = GaussianFourierProjection(time_emb_dim, fourier_scale, keys[ki])
            ki += 1
            self.null_cond_emb = jnp.zeros(time_emb_dim)
        else:
            self.cond_embed = None
            self.null_cond_emb = None

        # -- Encoder --
        # Track actual channel dim of every skip connection for decoder.
        enc_blocks = []
        enc_is_attn = []
        downsample_blocks = []
        skip_channels = [base_channels]  # stem output

        ch_in = base_channels
        current_res = image_size

        for level in range(L):
            ch_out = base_channels * channel_multipliers[level]

            for block_idx in range(num_res_blocks):
                block_in = ch_in if block_idx == 0 else ch_out
                enc_blocks.append(
                    ResBlockBigGAN(
                        in_channels=block_in,
                        out_channels=ch_out,
                        time_emb_dim=time_emb_dim,
                        num_groups=num_groups,
                        activation=activation,
                        dropout=dropout,
                        skip_rescale=skip_rescale,
                        key=keys[ki],
                    )
                )
                enc_is_attn.append(False)
                ki += 1

                if current_res in attn_resolutions:
                    enc_blocks.append(
                        AttnBlockNCSN(
                            channels=ch_out,
                            num_heads=num_heads,
                            num_groups=num_groups,
                            skip_rescale=skip_rescale,
                            key=keys[ki],
                        )
                    )
                    enc_is_attn.append(True)
                    ki += 1

                skip_channels.append(ch_out)

            ch_in = ch_out

            if level < L - 1:
                downsample_blocks.append(
                    ResBlockBigGAN(
                        in_channels=ch_out,
                        out_channels=ch_out,
                        time_emb_dim=time_emb_dim,
                        num_groups=num_groups,
                        activation=activation,
                        dropout=dropout,
                        skip_rescale=skip_rescale,
                        key=keys[ki],
                        down=True,
                    )
                )
                ki += 1
                skip_channels.append(ch_out)
                current_res = current_res // 2

        self.encoder_blocks = enc_blocks
        self.encoder_is_attn = enc_is_attn
        self.downsample_blocks = downsample_blocks

        # -- Bottleneck --
        ch_bot = base_channels * channel_multipliers[-1]
        self.mid_block1 = ResBlockBigGAN(
            ch_bot,
            ch_bot,
            time_emb_dim,
            num_groups,
            activation,
            dropout,
            skip_rescale,
            keys[ki],
        )
        ki += 1
        self.mid_attn = AttnBlockNCSN(
            ch_bot,
            num_heads,
            num_groups,
            skip_rescale,
            keys[ki],
        )
        ki += 1
        self.mid_block2 = ResBlockBigGAN(
            ch_bot,
            ch_bot,
            time_emb_dim,
            num_groups,
            activation,
            dropout,
            skip_rescale,
            keys[ki],
        )
        ki += 1

        # -- Decoder --
        # Pop from skip_channels to get the correct input dimension
        # for each decoder block's concatenated [h, skip] input.
        dec_blocks = []
        dec_is_attn = []
        upsample_blocks = []

        ch_current = ch_bot

        for level in reversed(range(L)):
            ch_out = base_channels * channel_multipliers[level]

            for block_idx in range(num_res_blocks + 1):
                actual_skip_ch = skip_channels.pop()
                block_in = ch_current + actual_skip_ch
                dec_blocks.append(
                    ResBlockBigGAN(
                        block_in,
                        ch_out,
                        time_emb_dim,
                        num_groups,
                        activation,
                        dropout,
                        skip_rescale,
                        keys[ki],
                    )
                )
                dec_is_attn.append(False)
                ki += 1
                ch_current = ch_out

                if current_res in attn_resolutions:
                    dec_blocks.append(
                        AttnBlockNCSN(
                            ch_out,
                            num_heads,
                            num_groups,
                            skip_rescale,
                            keys[ki],
                        )
                    )
                    dec_is_attn.append(True)
                    ki += 1

            if level > 0:
                upsample_blocks.append(
                    ResBlockBigGAN(
                        ch_out,
                        ch_out,
                        time_emb_dim,
                        num_groups,
                        activation,
                        dropout,
                        skip_rescale,
                        keys[ki],
                        up=True,
                    )
                )
                ki += 1
                current_res = current_res * 2

        assert (
            len(skip_channels) == 0
        ), f"Skip channel mismatch: {len(skip_channels)} skips remaining"

        self.decoder_blocks = dec_blocks
        self.decoder_is_attn = dec_is_attn
        self.upsample_blocks = upsample_blocks

        # -- Output head --
        self.final_norm = eqx.nn.GroupNorm(
            num_groups, base_channels * channel_multipliers[0]
        )
        self.final_conv = eqx.nn.Conv2d(
            base_channels * channel_multipliers[0],
            out_channels,
            3,
            padding=1,
            key=keys[ki],
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
                embedding; ``False`` uses the null embedding.
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

        # -- Encoder: collect skip connections --
        skips = [h]
        L = len(self.channel_multipliers)
        block_idx = 0

        for level in range(L):
            for _ in range(self.num_res_blocks):
                h = self.encoder_blocks[block_idx](h, combined_emb)
                block_idx += 1
                if (
                    block_idx < len(self.encoder_blocks)
                    and self.encoder_is_attn[block_idx]
                ):
                    h = self.encoder_blocks[block_idx](h)
                    block_idx += 1
                skips.append(h)

            if level < L - 1:
                h = self.downsample_blocks[level](h, combined_emb)
                skips.append(h)

        # -- Bottleneck --
        h = self.mid_block1(h, combined_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, combined_emb)

        # -- Decoder: consume skip connections in reverse --
        dec_idx = 0
        for level in reversed(range(L)):
            for _ in range(self.num_res_blocks + 1):
                skip = skips.pop()
                h = jnp.concatenate([h, skip], axis=0)
                h = self.decoder_blocks[dec_idx](h, combined_emb)
                dec_idx += 1
                if dec_idx < len(self.decoder_blocks) and self.decoder_is_attn[dec_idx]:
                    h = self.decoder_blocks[dec_idx](h)
                    dec_idx += 1

            if level > 0:
                up_idx = L - 1 - level
                h = self.upsample_blocks[up_idx](h, combined_emb)

        # -- Output head --
        h = self.final_norm(h)
        h = self.activation(h)
        h = self.final_conv(h)
        return h
