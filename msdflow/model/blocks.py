"""Building blocks for the UNet velocity-field predictor.

Provides ``SinusoidalEmbedding``, ``GaussianFourierProjection``, ``ResBlock``,
``ResBlockBigGAN``, ``AttentionBlock``, ``AttnBlockNCSN``, ``Downsample``,
and ``Upsample`` modules built on Equinox.
"""

import jax

import equinox as eqx
import jax.numpy as jnp

from typing import Callable, Optional, Tuple, Sequence


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
            sArray of shape ``(C, H/2, W/2)``.
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
    dropout: float = eqx.nn.Dropout
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
    """Self-attention block with GroupNorm pre-norm and skip rescaling.

    Wraps :class:`AttentionBlock` with GroupNorm pre-normalization and a
    residual connection optionally scaled by 1/sqrt(2).
    """

    norm: eqx.nn.GroupNorm
    attn: AttentionBlock
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
        self.skip_rescale = skip_rescale
        self.norm = eqx.nn.GroupNorm(num_groups, channels)
        self.attn = AttentionBlock(channels=channels, num_heads=num_heads, key=key)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply self-attention over spatial positions.

        Args:
            x: Input of shape ``(C, H, W)``.

        Returns:
            Output of shape ``(C, H, W)``.
        """
        residual = x
        h = self.norm(x)
        h = self.attn(h)
        out = h + residual
        if self.skip_rescale:
            out = out / jnp.sqrt(2.0)
        return out


class Identity(eqx.Module):
    def __call__(self, x, *, key=None):
        return x


class LayerNorm2d(eqx.Module):
    """timm-style LayerNorm2d for inputs of shape (C, H, W).

    Normalizes across channels only, independently at each spatial location.
    """

    weight: jax.Array
    bias: jax.Array
    eps: float = eqx.field(static=True)

    def __init__(
        self,
        num_channels: int,
        eps: float = 1e-6,
        *,
        dtype=jnp.float32,
    ):
        self.weight = jnp.ones((num_channels,), dtype=dtype)
        self.bias = jnp.zeros((num_channels,), dtype=dtype)
        self.eps = eps

    def __call__(self, x: jax.Array) -> jax.Array:
        # x: (C, H, W)
        mean = jnp.mean(x, axis=0, keepdims=True)  # (1, H, W)
        var = jnp.mean((x - mean) ** 2, axis=0, keepdims=True)  # (1, H, W)
        x = (x - mean) / jnp.sqrt(var + self.eps)
        x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x


class DropPath(eqx.Module):
    """Sample-wise stochastic depth for unbatched (C,H,W) inputs.

    Under vmap, this becomes normal per-example stochastic depth.
    """

    p: float = eqx.field(static=True)
    inference: bool = eqx.field(static=True)

    def __init__(self, p: float = 0.0, inference: bool = False):
        self.p = p
        self.inference = inference

    def __call__(self, x: jax.Array, *, key: Optional[jax.Array] = None) -> jax.Array:
        if self.inference or self.p == 0.0:
            return x
        if key is None:
            raise ValueError("DropPath requires a PRNG key when p > 0.")
        keep_prob = 1.0 - self.p
        mask = jax.random.bernoulli(key, keep_prob, shape=(1, 1, 1))
        return x * mask.astype(x.dtype) / keep_prob


class ConvNeXtBlock(eqx.Module):
    """ConvNeXt block for (C, H, W) inputs.

    Matches the conv-MLP variant shown in your timm printout:
      dwconv7x7 -> LayerNorm2d -> 1x1 conv -> GELU -> 1x1 conv -> gamma -> residual
    """

    conv_dw: eqx.nn.Conv2d
    norm: LayerNorm2d
    fc1: eqx.nn.Conv2d
    fc2: eqx.nn.Conv2d
    gamma: Optional[jax.Array]
    drop_path: DropPath

    def __init__(
        self,
        dim: int,
        *,
        mlp_ratio: int = 4,
        kernel_size: int = 7,
        ls_init_value: Optional[float] = 1e-6,
        drop_path: float = 0.0,
        inference: bool = False,
        dtype=jnp.float32,
        key: jax.Array,
    ):
        k1, k2, k3 = jax.random.split(key, 3)
        hidden_dim = dim * mlp_ratio
        padding = kernel_size // 2

        self.conv_dw = eqx.nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=dim,
            use_bias=True,
            dtype=dtype,
            key=k1,
        )
        self.norm = LayerNorm2d(dim, eps=1e-6, dtype=dtype)
        self.fc1 = eqx.nn.Conv2d(
            in_channels=dim,
            out_channels=hidden_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            use_bias=True,
            dtype=dtype,
            key=k2,
        )
        self.fc2 = eqx.nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=dim,
            kernel_size=1,
            stride=1,
            padding=0,
            use_bias=True,
            dtype=dtype,
            key=k3,
        )
        self.gamma = (
            None
            if ls_init_value is None
            else jnp.full((dim,), ls_init_value, dtype=dtype)
        )
        self.drop_path = DropPath(drop_path, inference=inference)

    def __call__(self, x: jax.Array, *, key: Optional[jax.Array] = None) -> jax.Array:
        residual = x

        x = self.conv_dw(x)
        x = self.norm(x)
        x = self.fc1(x)
        x = jax.nn.gelu(x, approximate=False)
        x = self.fc2(x)

        if self.gamma is not None:
            x = x * self.gamma[:, None, None]

        x = self.drop_path(x, key=key)
        return residual + x


class ConvNeXtDownsample(eqx.Module):
    """Stage downsample: LayerNorm2d -> Conv2d(kernel=2, stride=2)."""

    norm: LayerNorm2d
    conv: eqx.nn.Conv2d

    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        *,
        stride: int = 2,
        dtype=jnp.float32,
        key: jax.Array,
    ):
        if stride not in (1, 2):
            raise ValueError(f"Expected stride 1 or 2, got {stride}.")
        self.norm = LayerNorm2d(in_chs, eps=1e-6, dtype=dtype)
        self.conv = eqx.nn.Conv2d(
            in_channels=in_chs,
            out_channels=out_chs,
            kernel_size=stride,
            stride=stride,
            padding=0,
            use_bias=True,
            dtype=dtype,
            key=key,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.norm(x)
        x = self.conv(x)
        return x


class ConvNeXtStage(eqx.Module):
    """One ConvNeXt stage: optional downsample + repeated ConvNeXt blocks."""

    downsample: eqx.Module
    blocks: Tuple[ConvNeXtBlock, ...]

    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        depth: int,
        *,
        stride: int,
        kernel_size: int = 7,
        mlp_ratio: int = 4,
        ls_init_value: Optional[float] = 1e-6,
        drop_path_rates: Sequence[float] | None = None,
        inference: bool = False,
        dtype=jnp.float32,
        key: jax.Array,
    ):
        if depth < 1:
            raise ValueError("depth must be >= 1")

        nkeys = depth + 1
        keys = jax.random.split(key, nkeys)

        if in_chs == out_chs and stride == 1:
            self.downsample = Identity()
        else:
            self.downsample = ConvNeXtDownsample(
                in_chs=in_chs,
                out_chs=out_chs,
                stride=stride,
                dtype=dtype,
                key=keys[0],
            )

        if drop_path_rates is None:
            drop_path_rates = [0.0] * depth
        if len(drop_path_rates) != depth:
            raise ValueError("drop_path_rates must have length == depth")

        self.blocks = tuple(
            ConvNeXtBlock(
                dim=out_chs,
                mlp_ratio=mlp_ratio,
                kernel_size=kernel_size,
                ls_init_value=ls_init_value,
                drop_path=drop_path_rates[i],
                inference=inference,
                dtype=dtype,
                key=keys[i + 1],
            )
            for i in range(depth)
        )

    def __call__(self, x: jax.Array, *, key: Optional[jax.Array] = None) -> jax.Array:
        x = self.downsample(x)

        if key is None:
            for block in self.blocks:
                x = block(x)
        else:
            keys = jax.random.split(key, len(self.blocks))
            for block, k in zip(self.blocks, keys):
                x = block(x, key=k)

        return x


class ConvNeXtStem(eqx.Module):
    """Patch stem: Conv2d(patch_size, stride=patch_size) -> LayerNorm2d."""

    conv: eqx.nn.Conv2d
    norm: LayerNorm2d

    def __init__(
        self,
        in_chans: int,
        out_chs: int,
        *,
        patch_size: int = 4,
        dtype=jnp.float32,
        key: jax.Array,
    ):
        self.conv = eqx.nn.Conv2d(
            in_channels=in_chans,
            out_channels=out_chs,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            use_bias=True,
            dtype=dtype,
            key=key,
        )
        self.norm = LayerNorm2d(out_chs, eps=1e-6, dtype=dtype)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.conv(x)
        x = self.norm(x)
        return x


class ConvNeXtHead(eqx.Module):
    """Matches your printed head for num_classes=0:
    global avg pool (keepdims) -> LayerNorm2d -> flatten
    """

    norm: LayerNorm2d

    def __init__(self, dim: int, *, dtype=jnp.float32):
        self.norm = LayerNorm2d(dim, eps=1e-6, dtype=dtype)

    def __call__(self, x: jax.Array) -> jax.Array:
        # x: (C, H, W)
        x = jnp.mean(x, axis=(1, 2), keepdims=True)  # (C, 1, 1)
        x = self.norm(x)  # (C, 1, 1)
        x = jnp.reshape(x, (-1,))  # (C,)
        return x
