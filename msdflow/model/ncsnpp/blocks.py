"""Building blocks for the NCSN++ velocity-field predictor."""

from typing import Callable, Optional, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp

from msdflow.model.common_blocks import AttentionBlock, _apply_conv2d, _apply_linear

__all__ = [
    "AttentionBlock",
    "AttnBlockNCSN",
    "Conv2d",
    "CoordConv",
    "GaussianFourierProjection",
    "RALAAttentionBlock",
    "ResBlockBigGAN",
]


class CoordConv(eqx.Module):

    conv: eqx.nn.Conv2d
    power: float = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = (1, 1),
        padding: str | int | Sequence[int] | Sequence[tuple[int, int]] = (0, 0),
        dilation: int | Sequence[int] = (1, 1),
        groups: int = 1,
        use_bias: bool = True,
        padding_mode: str = "ZEROS",
        dtype=None,
        power: float = 1.0,
        *,
        key: jax.Array,
    ):

        in_channels += 1
        self.power = power
        self.conv = eqx.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            use_bias=use_bias,
            padding_mode=padding_mode,
            dtype=dtype,
            key=key,
        )

    def __call__(self, x: jax.Array):

        H, W = x.shape[-2:]
        _x = jnp.linspace(-1, 1, W, dtype=x.dtype)
        _y = jnp.linspace(1, -1, H, dtype=x.dtype)

        X, Y = jnp.meshgrid(_x, _y, indexing="xy")
        _R = 1.0 - jnp.sqrt(X**2 + Y**2) / jnp.sqrt(2)
        _R = jnp.power(_R, self.power)
        R = (_R - jnp.mean(_R)) / jnp.std(_R)

        x = jnp.concatenate([x, R[None, ...]], axis=0)
        out = _apply_conv2d(self.conv, x)

        return out


class Conv2d(eqx.Module):

    conv: eqx.nn.Conv2d

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = (1, 1),
        padding: str | int | Sequence[int] | Sequence[tuple[int, int]] = (0, 0),
        dilation: int | Sequence[int] = (1, 1),
        groups: int = 1,
        use_bias: bool = True,
        padding_mode: str = "ZEROS",
        dtype=None,
        use_radial=True,
        *,
        key: jax.Array,
    ):

        self.conv = eqx.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            use_bias=use_bias,
            padding_mode=padding_mode,
            dtype=dtype,
            key=key,
        )

    def __call__(self, x: jax.Array):

        out = _apply_conv2d(self.conv, x)

        return out


def _rotate_every_two(x: jax.Array) -> jax.Array:
    """Rotate adjacent feature pairs for RALA rotary embeddings.

    Args:
        x: Input array with an even final dimension.

    Returns:
        Array with each adjacent feature pair rotated.
    """
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return jnp.stack([-x2, x1], axis=-1).reshape(x.shape)


def _rala_2d_rope_factors(
    height: int,
    width: int,
    head_dim: int,
    dtype: jnp.dtype,
) -> tuple[jax.Array, jax.Array]:
    """Build the official RALA 2D RoPE sine and cosine factors.

    Args:
        height: Spatial height.
        width: Spatial width.
        head_dim: Per-head channel dimension.
        dtype: Output dtype.

    Returns:
        Tuple ``(sin, cos)`` with shape ``(height * width, head_dim)``.
    """
    base = jnp.linspace(0.0, 1.0, head_dim // 4, dtype=jnp.float32)
    angle = 1.0 / (10000.0**base)
    angle = jnp.repeat(angle[:, None], 2, axis=1).reshape(-1)

    index_h = jnp.arange(height, dtype=jnp.float32)
    index_w = jnp.arange(width, dtype=jnp.float32)
    sin_h = jnp.sin(index_h[:, None] * angle[None, :])
    sin_w = jnp.sin(index_w[:, None] * angle[None, :])
    cos_h = jnp.cos(index_h[:, None] * angle[None, :])
    cos_w = jnp.cos(index_w[:, None] * angle[None, :])

    sin_h = jnp.repeat(sin_h[:, None, :], width, axis=1)
    sin_w = jnp.repeat(sin_w[None, :, :], height, axis=0)
    cos_h = jnp.repeat(cos_h[:, None, :], width, axis=1)
    cos_w = jnp.repeat(cos_w[None, :, :], height, axis=0)

    sin = jnp.concatenate([sin_h, sin_w], axis=-1).reshape(height * width, head_dim)
    cos = jnp.concatenate([cos_h, cos_w], axis=-1).reshape(height * width, head_dim)
    return sin.astype(dtype), cos.astype(dtype)


def _theta_shift(x: jax.Array, sin: jax.Array, cos: jax.Array) -> jax.Array:
    """Apply the official RALA rotary shift to token features.

    Args:
        x: Input array of shape ``(heads, tokens, head_dim)``.
        sin: Sine factors of shape ``(tokens, head_dim)``.
        cos: Cosine factors of shape ``(tokens, head_dim)``.

    Returns:
        Rotary-shifted array with the same shape as ``x``.
    """
    sin = sin[None, :, :]
    cos = cos[None, :, :]
    return (x * cos) + (_rotate_every_two(x) * sin)


class RALAAttentionBlock(eqx.Module):
    """Rank-Augmented Linear Attention over spatial tokens.

    This ports the official RALA ``GateLinearAttentionNoSilu`` block to
    JAX/Equinox for unbatched ``(C, H, W)`` tensors.

    Attributes:
        qkvo: Joint 1x1 convolution for query, key, value, and output gate.
        lepe: Depthwise local positional encoding convolution over values.
        proj: Output 1x1 projection.
        num_heads: Number of attention heads.
        head_dim: Per-head channel dimension.
        attention_dtype: Dtype used for projection and attention math.
        scale: Rank-augmentation attention scale.
    """

    qkvo: eqx.nn.Conv2d
    lepe: eqx.nn.Conv2d
    proj: eqx.nn.Conv2d
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    attention_dtype: jnp.dtype = eqx.field(static=True)
    scale: float = eqx.field(static=True)

    def __init__(
        self,
        channels: int,
        num_heads: int,
        key: jax.Array,
        *,
        attention_dtype: jnp.dtype = jnp.float32,
    ):
        """Initialise the RALA attention projections.

        Args:
            channels: Channel dimension.
            num_heads: Number of attention heads. Must divide ``channels``.
            key: JAX PRNG key.
            attention_dtype: Dtype used for RALA projections and attention math.

        Raises:
            ValueError: If ``channels`` is not divisible by ``num_heads`` or if
                the per-head dimension is not divisible by 4 for 2D RoPE.
        """
        if channels % num_heads != 0:
            raise ValueError(
                f"channels={channels} must be divisible by num_heads={num_heads}."
            )
        head_dim = channels // num_heads
        if head_dim % 4 != 0:
            raise ValueError(
                f"RALA head_dim={head_dim} must be divisible by 4 for 2D RoPE."
            )

        kqkvo, klepe, kproj = jax.random.split(key, 3)
        self.qkvo = eqx.nn.Conv2d(channels, channels * 4, 1, key=kqkvo)
        self.lepe = eqx.nn.Conv2d(
            channels, channels, 5, padding=2, groups=channels, key=klepe
        )
        self.proj = eqx.nn.Conv2d(channels, channels, 1, key=kproj)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.attention_dtype = attention_dtype
        self.scale = head_dim**-0.5

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply RALA over spatial positions.

        Args:
            x: Input array of shape ``(C, H, W)``.

        Returns:
            Attention output of shape ``(C, H, W)`` with the same dtype as ``x``.
        """
        channels, height, width = x.shape
        tokens = height * width
        orig_dtype = x.dtype
        x_attn = x.astype(self.attention_dtype)

        qkvo = _apply_conv2d(self.qkvo, x_attn)
        qkv = qkvo[: 3 * channels]
        gate = qkvo[3 * channels :]
        value_map = qkv[2 * channels :]
        lepe = _apply_conv2d(self.lepe, value_map)

        q, k, v = jnp.split(qkv, 3, axis=0)
        q = q.reshape(self.num_heads, self.head_dim, tokens).transpose(0, 2, 1)
        k = k.reshape(self.num_heads, self.head_dim, tokens).transpose(0, 2, 1)
        v = v.reshape(self.num_heads, self.head_dim, tokens).transpose(0, 2, 1)

        q = jax.nn.elu(q) + 1.0
        k = jax.nn.elu(k) + 1.0

        q_mean = jnp.mean(q, axis=1, keepdims=True)
        eff = self.scale * (q_mean @ jnp.swapaxes(k, -1, -2))
        eff = jax.nn.softmax(eff, axis=-1)
        k = k * jnp.swapaxes(eff, -1, -2) * tokens

        sin, cos = _rala_2d_rope_factors(
            height, width, self.head_dim, self.attention_dtype
        )
        q_rope = _theta_shift(q, sin, cos)
        k_rope = _theta_shift(k, sin, cos)

        k_mean = jnp.mean(k, axis=1, keepdims=True)
        z = 1.0 / ((q @ jnp.swapaxes(k_mean, -1, -2)) + 1e-6)
        token_scale = tokens**-0.5
        kv = (jnp.swapaxes(k_rope, -1, -2) * token_scale) @ (v * token_scale)

        out = (q_rope @ kv) * z
        out = out.transpose(0, 2, 1).reshape(channels, height, width)
        out = out + lepe
        out = _apply_conv2d(self.proj, out * gate)
        return out.astype(orig_dtype)


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
    conv1: Conv2d | CoordConv
    time_proj: eqx.nn.Linear
    norm2: eqx.nn.GroupNorm
    conv2: eqx.nn.Conv2d | CoordConv
    skip_conv: Optional[eqx.nn.Conv2d | CoordConv]
    dropout: float = eqx.nn.Dropout
    activation: Callable = eqx.field(static=True)
    skip_rescale: bool = eqx.field(static=True)
    up: bool = eqx.field(static=True)
    down: bool = eqx.field(static=True)
    compute_dtype: jnp.dtype = eqx.field(static=True)

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
        compute_dtype: jnp.dtype = jnp.float32,
        use_coord_conv: bool = False,
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
        compute_dtype: Dtype used for convolution and time-projection math.
        """
        if up and down:
            raise ValueError("Cannot set both up=True and down=True.")

        if not use_coord_conv:
            conv_layer = Conv2d
        else:
            conv_layer = CoordConv

        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.activation = activation
        self.skip_rescale = skip_rescale
        self.up = up
        self.down = down
        self.compute_dtype = compute_dtype

        self.norm1 = eqx.nn.GroupNorm(num_groups, in_channels)
        self.conv1 = conv_layer(in_channels, out_channels, 3, padding=1, key=k1)
        self.time_proj = eqx.nn.Linear(time_emb_dim, out_channels, key=k2)
        self.norm2 = eqx.nn.GroupNorm(num_groups, out_channels)
        self.dropout = eqx.nn.Dropout(dropout)
        self.conv2 = conv_layer(out_channels, out_channels, 3, padding=1, key=k3)

        self.skip_conv = (
            None
            if (in_channels == out_channels and not up and not down)
            else conv_layer(in_channels, out_channels, 1, key=k4)
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
        orig_dtype = x.dtype

        h = self.norm1(x)
        h = self.activation(h)
        h = self._resample(h)
        h = self.conv1(h.astype(self.compute_dtype)).astype(orig_dtype)

        time_h = _apply_linear(
            self.time_proj,
            time_emb.astype(self.compute_dtype),
        ).astype(orig_dtype)
        h = h + time_h.reshape(-1, 1, 1)

        h = self.norm2(h)
        h = self.activation(h)
        h = self.dropout(h, key=key)
        h = self.conv2(h.astype(self.compute_dtype)).astype(orig_dtype)

        skip = self._resample(x)
        if self.skip_conv is not None:
            skip = self.skip_conv(skip.astype(self.compute_dtype)).astype(orig_dtype)

        out = h + skip
        if self.skip_rescale:
            out = out / jnp.sqrt(2.0)
        return out


class AttnBlockNCSN(eqx.Module):
    """Self-attention block with GroupNorm pre-norm and skip rescaling.

    Wraps a selected spatial attention algorithm with GroupNorm
    pre-normalization and a residual connection optionally scaled by 1/sqrt(2).
    ``attention_type="dot_product"`` preserves the existing
    :class:`AttentionBlock` behavior, while ``attention_type="rala"`` uses
    :class:`RALAAttentionBlock`.
    """

    norm: eqx.nn.GroupNorm
    attn: AttentionBlock | RALAAttentionBlock
    skip_rescale: bool = eqx.field(static=True)

    def __init__(
        self,
        channels: int,
        num_heads: int,
        num_groups: int,
        skip_rescale: bool,
        key: jax.Array,
        *,
        attention_dtype: jnp.dtype = jnp.float32,
        implementation: Optional[str] = None,
        attention_type: str = "dot_product",
    ):
        """Args:
        channels: Channel dimension.
        num_heads: Number of attention heads.
        num_groups: Groups for GroupNorm.
        skip_rescale: If True, divide residual sum by sqrt(2).
        key: JAX PRNG key.
        attention_dtype: Dtype for Q/K/V projections and the attention call.
            Output is upcast back to the input's dtype after attention. The
            GroupNorm pre-norm and residual sum run in the input's dtype.
        implementation: Backend for the inner ``AttentionBlock``. ``None``
            auto-detects ``'cudnn'`` on GPU, ``'xla'`` otherwise.
            Only used when ``attention_type`` is ``"dot_product"``.
        attention_type: Attention algorithm selector. Choose ``"dot_product"``
            for :class:`AttentionBlock` or ``"rala"`` for
            :class:`RALAAttentionBlock`.

        Raises:
            ValueError: If ``attention_type`` is not ``"dot_product"`` or
                ``"rala"``.
        """
        self.skip_rescale = skip_rescale
        self.norm = eqx.nn.GroupNorm(num_groups, channels)
        if attention_type == "dot_product":
            self.attn = AttentionBlock(
                channels=channels,
                num_heads=num_heads,
                key=key,
                attention_dtype=attention_dtype,
                implementation=implementation,
            )
        elif attention_type == "rala":
            self.attn = RALAAttentionBlock(
                channels=channels,
                num_heads=num_heads,
                key=key,
                attention_dtype=attention_dtype,
            )
        else:
            raise ValueError(
                f"attention_type={attention_type!r} is not supported; "
                "choose 'dot_product' or 'rala'."
            )

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
