"""Building blocks for the UNet velocity-field predictor.

Provides ``SinusoidalEmbedding``, ``GaussianFourierProjection``, ``ResBlock``,
``ResBlockBigGAN``, ``AttentionBlock``, ``AttnBlockNCSN``, ``Downsample``,
and ``Upsample`` modules built on Equinox.
"""

import jax

import equinox as eqx
import jax.numpy as jnp

from typing import Callable, Optional


class SinusoidalEmbedding(eqx.Module):
    lin1: eqx.nn.Linear
    lin2: eqx.nn.Linear
    dim: int = eqx.field(static=True)
    activation: Callable = eqx.field(static=True)
    """Sinusoidal positional embedding with a two-layer MLP."""

    def __init__(self, dim: int, activation: Callable, key: jax.Array):
        """Initialise the embedding layers.

        Args:
            dim (int): Embedding dimension. Must be even.
            activation (Callable): Activation function.
            key (jax.Array): RNG key.

        Raises:
            ValueError: If dim is not even.
        """

        if (dim % 2) != 0:
            raise ValueError("embedding dimension must be even.")

        k1, k2 = jax.random.split(key)
        self.dim = dim
        self.activation = activation
        self.lin1 = eqx.nn.Linear(dim, dim, key=k1)
        self.lin2 = eqx.nn.Linear(dim, dim, key=k2)

    def __call__(self, t: jax.Array) -> jax.Array:
        """
        Embed a time t.

        Args:
            t (jax.Array): Time to embed

        Returns:
            jax.Array: Sinusoidal time embedding
        """

        freqs = jnp.exp(-jnp.log(10000.0) * 2 * jnp.arange(self.dim // 2) / self.dim)
        emb = jnp.concatenate([jnp.sin(t * freqs), jnp.cos(t * freqs)])
        emb = self.lin1(emb)
        emb = self.activation(emb)
        emb = self.lin2(emb)
        return emb


class Downsample(eqx.Module):
    """Spatial 2x downsampling via strided convolution."""

    conv: eqx.nn.Conv2d

    def __init__(self, channels: int, key: jax.Array):
        """Args:
        channels: Number of input and output channels.
        key: JAX PRNG key.
        """
        self.conv = eqx.nn.Conv2d(
            channels, channels, kernel_size=3, stride=2, padding=1, key=key
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Downsample *x* by a factor of 2.

        Args:
            x: Input array of shape ``(C, H, W)``.

        Returns:
            Array of shape ``(C, H/2, W/2)``.
        """
        return self.conv(x)


class Upsample(eqx.Module):
    """Spatial upsampling via nearest-neighbor resize followed by convolution."""

    conv: eqx.nn.Conv2d

    def __init__(self, channels: int, key: jax.Array):
        self.conv = eqx.nn.Conv2d(channels, channels, kernel_size=3, padding=1, key=key)

    def __call__(self, x: jax.Array, target_h: int, target_w: int) -> jax.Array:
        """Upsample spatial dimensions to a target size.

        Args:
            x: Input array of shape ``(C, H, W)``.
            target_h: Target height after upsampling.
            target_w: Target width after upsampling.

        Returns:
            Upsampled array of shape ``(C, target_h, target_w)``.
        """
        c, h, w = x.shape
        x = jax.image.resize(x, shape=(c, target_h, target_w), method="nearest")
        return self.conv(x)


class ResBlock(eqx.Module):
    """Residual block with time-embedding conditioning and optional skip projection."""

    conv1: eqx.nn.Conv2d
    norm1: eqx.nn.GroupNorm
    time_proj: eqx.nn.Linear
    conv2: eqx.nn.Conv2d
    skip: Optional[eqx.nn.Conv2d]
    activation: Callable = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        num_groups: int,
        activation: Callable,
        key: jax.Array,
    ):
        """Args:
        in_channels: Input channel count.
        out_channels: Output channel count.
        time_emb_dim: Dimensionality of the time embedding.
        num_groups: Groups for ``GroupNorm``.
        activation: Activation function.
        key: JAX PRNG key.
        """
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.conv1 = eqx.nn.Conv2d(in_channels, out_channels, 3, padding=1, key=k1)
        self.norm1 = eqx.nn.GroupNorm(num_groups, out_channels)
        self.time_proj = eqx.nn.Linear(time_emb_dim, out_channels, key=k2)
        self.conv2 = eqx.nn.Conv2d(out_channels, out_channels, 3, padding=1, key=k3)
        self.skip = (
            None
            if in_channels == out_channels
            else eqx.nn.Conv2d(in_channels, out_channels, 1, key=k4)
        )
        self.activation = activation

    def __call__(self, x: jax.Array, time_emb: jax.Array) -> jax.Array:
        """Apply the residual block.

        Args:
            x: Input feature map of shape ``(C_in, H, W)``.
            time_emb: Time embedding vector.

        Returns:
            Output feature map of shape ``(C_out, H, W)``.
        """
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.activation(h)
        h = h + self.time_proj(time_emb).reshape(-1, 1, 1)
        h = self.conv2(h)
        # TODO: Might fail due to boolean branching in jit compile. If so follow up
        skip = x if self.skip is None else self.skip(x)
        return h + skip


class AttentionBlock(eqx.Module):
    """Self-attention over spatial tokens."""

    attn: eqx.nn.MultiheadAttention

    def __init__(self, channels: int, num_heads: int, key: jax.Array):
        """Args:
        channels: Channel dimension (used as query size).
        num_heads: Number of attention heads.
        key: JAX PRNG key.
        """
        self.attn = eqx.nn.MultiheadAttention(
            num_heads=num_heads, query_size=channels, key=key
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply multi-head self-attention over spatial positions.

        Args:
            x: Input array of shape ``(C, H, W)``.

        Returns:
            Attention output of shape ``(C, H, W)``.
        """
        c, h, w = x.shape
        tokens = x.reshape(h * w, c)
        out = self.attn(query=tokens, key_=tokens, value=tokens)
        return out.reshape(c, h, w)


class GaussianFourierProjection(eqx.Module):
    """Random Fourier feature time embedding with a two-layer MLP.

    Uses fixed (non-trainable) random frequencies to project a scalar
    timestep into a high-dimensional embedding, following Song et al. 2021.
    The frequency matrix ``W`` is frozen via ``jax.lax.stop_gradient``.
    """

    W: jax.Array
    lin1: eqx.nn.Linear
    lin2: eqx.nn.Linear
    embed_dim: int = eqx.field(static=True)

    def __init__(self, embed_dim: int, scale: float, key: jax.Array):
        """Initialise the Gaussian Fourier projection layers.

        Args:
            embed_dim: Output embedding dimension. Must be even.
            scale: Standard deviation of the random frequencies.
            key: JAX PRNG key.

        Raises:
            ValueError: If embed_dim is not even.
        """
        if embed_dim % 2 != 0:
            raise ValueError("embed_dim must be even.")

        k1, k2, k3 = jax.random.split(key, 3)
        half_dim = embed_dim // 2
        self.W = jax.random.normal(k1, (half_dim,)) * scale
        self.embed_dim = embed_dim
        self.lin1 = eqx.nn.Linear(embed_dim, embed_dim, key=k2)
        self.lin2 = eqx.nn.Linear(embed_dim, embed_dim, key=k3)

    def __call__(self, t: jax.Array) -> jax.Array:
        """Embed a scalar timestep via random Fourier features.

        Args:
            t: Scalar time value.

        Returns:
            Embedding vector of shape ``(embed_dim,)``.
        """
        W = jax.lax.stop_gradient(self.W)
        t_proj = t * W * 2 * jnp.pi
        emb = jnp.concatenate([jnp.sin(t_proj), jnp.cos(t_proj)])
        emb = self.lin1(emb)
        emb = jax.nn.swish(emb)
        emb = self.lin2(emb)
        return emb


class ResBlockBigGAN(eqx.Module):
    """BigGAN-style residual block with optional integrated up/down resampling.

    Pre-activation ordering (GroupNorm -> act -> Conv) with time-embedding
    conditioning injected between the two convolutions. When ``skip_rescale``
    is True, the residual sum is divided by sqrt(2) for training stability.
    """

    norm1: eqx.nn.GroupNorm
    conv1: eqx.nn.Conv2d
    time_proj: eqx.nn.Linear
    norm2: eqx.nn.GroupNorm
    conv2: eqx.nn.Conv2d
    skip_conv: Optional[eqx.nn.Conv2d]
    dropout: eqx.nn.Dropout
    activation: Callable = eqx.field(static=True)
    skip_rescale: bool = eqx.field(static=True)
    up: bool = eqx.field(static=True)
    down: bool = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        num_groups: int,
        activation: Callable,
        dropout: float,
        skip_rescale: bool,
        key: jax.Array,
        up: bool = False,
        down: bool = False,
    ):
        """Args:
        in_channels: Input channel count.
        out_channels: Output channel count.
        time_emb_dim: Dimensionality of the time embedding vector.
        num_groups: Groups for GroupNorm.
        activation: Activation function.
        dropout: Dropout probability.
        skip_rescale: If True, divide residual sum by sqrt(2).
        key: JAX PRNG key.
        up: If True, upsample 2x within the block.
        down: If True, downsample 2x within the block.
        """
        if up and down:
            raise ValueError("Cannot set both up=True and down=True.")

        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.activation = activation
        self.skip_rescale = skip_rescale
        self.up = up
        self.down = down

        self.norm1 = eqx.nn.GroupNorm(num_groups, in_channels)
        self.conv1 = eqx.nn.Conv2d(in_channels, out_channels, 3, padding=1, key=k1)
        self.time_proj = eqx.nn.Linear(time_emb_dim, out_channels, key=k2)
        self.norm2 = eqx.nn.GroupNorm(num_groups, out_channels)
        self.dropout = eqx.nn.Dropout(dropout)
        self.conv2 = eqx.nn.Conv2d(out_channels, out_channels, 3, padding=1, key=k3)

        self.skip_conv = (
            None
            if (in_channels == out_channels and not up and not down)
            else eqx.nn.Conv2d(in_channels, out_channels, 1, key=k4)
        )

    def _resample(self, x: jax.Array) -> jax.Array:
        """Apply up- or downsampling to a (C, H, W) tensor."""
        if self.up:
            c, h, w = x.shape
            x = jax.image.resize(x, (c, h * 2, w * 2), method="nearest")
        elif self.down:
            x = x.reshape(1, *x.shape)
            x = (
                jax.lax.reduce_window(
                    x, 0.0, jax.lax.add, (1, 1, 2, 2), (1, 1, 2, 2), "SAME"
                )
                / 4.0
            )
            x = x.squeeze(0)
        return x

    def __call__(self, x: jax.Array, time_emb: jax.Array, key: jax.Array) -> jax.Array:
        """Apply the BigGAN residual block.

        Args:
            x: Input feature map of shape ``(C_in, H, W)``.
            time_emb: Time embedding vector of shape ``(time_emb_dim,)``.
            key: JAX PRNG key.

        Returns:
            Output feature map of shape ``(C_out, H', W')`` where
            ``H', W'`` depend on up/down settings.
        """
        h = self.norm1(x)
        h = self.activation(h)
        h = self._resample(h)
        h = self.conv1(h)
        h = h + self.time_proj(time_emb).reshape(-1, 1, 1)
        h = self.norm2(h)
        h = self.activation(h)
        h = self.dropout(h, key=key)
        h = self.conv2(h)

        skip = self._resample(x)
        if self.skip_conv is not None:
            skip = self.skip_conv(skip)

        out = h + skip
        if self.skip_rescale:
            out = out / jnp.sqrt(2.0)
        return out


class AttnBlockNCSN(eqx.Module):
    """Self-attention block with GroupNorm and skip rescaling.

    Projects input to Q, K, V via 1x1 convolutions (NIN), applies
    scaled dot-product self-attention, and adds a residual connection
    optionally scaled by 1/sqrt(2).
    """

    norm: eqx.nn.GroupNorm
    qkv_proj: eqx.nn.Conv2d
    out_proj: eqx.nn.Conv2d
    channels: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    skip_rescale: bool = eqx.field(static=True)

    def __init__(
        self,
        channels: int,
        num_heads: int,
        num_groups: int,
        skip_rescale: bool,
        key: jax.Array,
    ):
        """Args:
        channels: Channel dimension.
        num_heads: Number of attention heads.
        num_groups: Groups for GroupNorm.
        skip_rescale: If True, divide residual sum by sqrt(2).
        key: JAX PRNG key.
        """
        k1, k2 = jax.random.split(key)
        self.channels = channels
        self.num_heads = num_heads
        self.skip_rescale = skip_rescale
        self.norm = eqx.nn.GroupNorm(num_groups, channels)
        self.qkv_proj = eqx.nn.Conv2d(channels, channels * 3, 1, key=k1)
        self.out_proj = eqx.nn.Conv2d(channels, channels, 1, key=k2)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply self-attention over spatial positions.

        Args:
            x: Input of shape ``(C, H, W)``.

        Returns:
            Output of shape ``(C, H, W)``.
        """
        c, h, w = x.shape
        residual = x

        x_norm = self.norm(x)
        qkv = self.qkv_proj(x_norm)  # (3*C, H, W)
        qkv = qkv.reshape(3, c, h * w)  # (3, C, N)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (C, N)

        head_dim = c // self.num_heads
        scale = head_dim ** (-0.5)

        q = q.reshape(self.num_heads, head_dim, h * w)
        k = k.reshape(self.num_heads, head_dim, h * w)
        v = v.reshape(self.num_heads, head_dim, h * w)

        attn = jnp.einsum("hdn,hdm->hnm", q, k) * scale
        attn = jax.nn.softmax(attn, axis=-1)

        out = jnp.einsum("hnm,hdm->hdn", attn, v)
        out = out.reshape(c, h, w)

        out = self.out_proj(out)
        out = out + residual
        if self.skip_rescale:
            out = out / jnp.sqrt(2.0)
        return out
