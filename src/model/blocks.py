import jax

import equinox as eqx
import jax.numpy as jnp

from typing import Callable, Optional


class SinusoidalEmbedding(eqx.Module):
    lin1: eqx.nn.Linear
    lin2: eqx.nn.Linear
    dim: int = eqx.field(static=True)
    activation: Callable = eqx.field(static=True)
    """
    _summary_
    """

    def __init__(self, dim: int, activation: Callable, key: jax.Array):
        """
        _summary_

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
    conv: eqx.nn.Conv2d

    def __init__(self, channels: int, key: jax.Array):
        self.conv = eqx.nn.Conv2d(
            channels, channels, kernel_size=3, stride=2, padding=1, key=key
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.conv(x)


class Upsample(eqx.Module):
    """Spatial upsampling via nearest-neighbor resize followed by convolution."""

    conv: eqx.nn.Conv2d

    def __init__(self, channels: int, key: jax.Array):
        self.conv = eqx.nn.Conv2d(channels, channels, kernel_size=3, padding=1, key=key)

    def __call__(
        self, x: jax.Array, target_h: int, target_w: int
    ) -> jax.Array:
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
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.activation(h)
        h = h + self.time_proj(time_emb).reshape(-1, 1, 1)
        h = self.conv2(h)
        # TODO: Might fail due to boolean branching in jit compile. If so follow up
        skip = x if self.skip is None else self.skip(x)
        return h + skip


class AttentionBlock(eqx.Module):
    attn: eqx.nn.MultiheadAttention

    def __init__(self, channels: int, num_heads: int, key: jax.Array):
        self.attn = eqx.nn.MultiheadAttention(
            num_heads=num_heads, query_size=channels, key=key
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        c, h, w = x.shape
        tokens = x.reshape(h * w, c)
        out = self.attn(query=tokens, key_=tokens, value=tokens)
        return out.reshape(c, h, w)
