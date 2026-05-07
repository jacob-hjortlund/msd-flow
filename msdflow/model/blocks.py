"""Building blocks for the UNet velocity-field predictor.

Provides ``SinusoidalEmbedding``, ``GaussianFourierProjection``, ``ResBlock``,
``ResBlockBigGAN``, ``AttentionBlock``, ``AttnBlockNCSN``, ``Downsample``,
and ``Upsample`` modules built on Equinox.
"""

import jax
import warnings

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


class CoordConv(eqx.Module):

    conv: eqx.nn.Conv2d
    use_radial: bool = eqx.field(static=True)

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

        in_channels += 2
        if use_radial:
            in_channels += 1

        self.use_radial = use_radial
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
        _x = jnp.linspace(-1, 1, W)
        _y = jnp.linspace(1, -1, H)

        X, Y = jnp.meshgrid(_x, _y, indexing="xy")

        if self.use_radial:
            R = jnp.sqrt(X**2 + Y**2) / jnp.sqrt(2)
            coords = jnp.stack([X, Y, R], axis=0)
        else:
            coords = jnp.stack([X, Y], axis=0)

        x = jnp.concatenate([x, coords], axis=0)
        out = self.conv(x)

        pass


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


def _apply_linear(linear: eqx.nn.Linear, x: jax.Array) -> jax.Array:
    """Apply ``eqx.nn.Linear`` with weights cast to ``x.dtype``.

    Args:
        linear: Equinox Linear layer with weight shape
            ``(out_features, in_features)``.
        x: Input array with trailing dimension ``in_features``.

    Returns:
        Output array with trailing dimension ``out_features`` in ``x.dtype``.
    """
    weight = linear.weight.astype(x.dtype)
    out = x @ weight.T
    if linear.use_bias:
        bias = linear.bias.astype(x.dtype)
        out = out + bias
    return out


def _apply_conv2d(conv: eqx.nn.Conv2d, x: jax.Array) -> jax.Array:
    """Apply a convolution with parameter copies cast to the input dtype.

    Args:
        conv: Equinox convolution layer.
        x: Input array of shape ``(C, H, W)``.

    Returns:
        Convolution output in ``x.dtype``.
    """
    weight = conv.weight.astype(x.dtype)
    bias = None if conv.bias is None else conv.bias.astype(x.dtype)
    cast_conv = eqx.tree_at(lambda c: (c.weight, c.bias), conv, (weight, bias))
    return cast_conv(x)


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


class AttentionBlock(eqx.Module):
    """Self-attention over spatial tokens using ``jax.nn.dot_product_attention``.

    Q, K, V, and output projections are explicit ``eqx.nn.Linear`` layers stored
    in fp32. When ``attention_dtype`` is bfloat16, activations and weight copies
    are cast to bfloat16 for the Q/K/V projections and the attention call; the
    result is upcast back to the input's dtype immediately after the attention
    call, before the output projection.

    The attention backend is selected by ``implementation``: when ``None`` (the
    default), it auto-detects ``'cudnn'`` on GPU and ``'xla'`` on CPU. The
    backend is resolved once at construction (via ``jax.devices()[0].platform``)
    and stored as a static field, so it does not adapt if the model is later
    moved between devices. If you construct on CPU and run on GPU (or vice
    versa), pass ``implementation='cudnn'`` or ``implementation='xla'``
    explicitly to override auto-detection.

    When using ``implementation='cudnn'``, the head dimension must satisfy the
    cuDNN flash-attention constraints used here: ``head_dim <= 128`` and
    ``head_dim`` must be a multiple of 8. Since
    ``head_dim = channels // num_heads``, the requested ``num_heads`` may be
    adjusted automatically to the closest valid value that divides ``channels``.
    If this happens, a warning is emitted showing the requested and resolved
    ``num_heads``/``head_dim`` values.
    """

    q_proj: eqx.nn.Linear
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    out_proj: eqx.nn.Linear
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    attention_dtype: jnp.dtype = eqx.field(static=True)
    implementation: str = eqx.field(static=True)

    def __init__(
        self,
        channels: int,
        num_heads: int,
        key: jax.Array,
        *,
        attention_dtype: jnp.dtype = jnp.float32,
        implementation: Optional[str] = None,
    ):
        """Args:
        channels: Channel dimension, also used as query/key/value/output size.
        num_heads: Requested number of attention heads. For ``'xla'``, this
            must divide ``channels`` exactly. For ``'cudnn'``, this is treated
            as the requested value; if it does not produce a cuDNN-compatible
            ``head_dim``, it is automatically changed to the closest valid
            number of heads.
        key: JAX PRNG key.
        attention_dtype: Dtype used for Q/K/V projections and the attention
            call. Output is upcast back to the input's dtype after attention.
        implementation: Backend for ``jax.nn.dot_product_attention``. ``None``
            auto-detects ``'cudnn'`` on GPU and ``'xla'`` otherwise. Pass
            ``'xla'`` or ``'cudnn'`` to override.

        Notes:
        For ``implementation='cudnn'``, the resolved ``head_dim`` must be
        ``<= 128`` and a multiple of 8. The resolved ``num_heads`` is chosen
        as the valid divisor of ``channels`` closest to the requested
        ``num_heads``. Ties prefer the larger number of heads, giving the
        smaller ``head_dim``.
        """

        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}.")

        kq, kk, kv, ko = jax.random.split(key, 4)
        self.q_proj = eqx.nn.Linear(channels, channels, key=kq)
        self.k_proj = eqx.nn.Linear(channels, channels, key=kk)
        self.v_proj = eqx.nn.Linear(channels, channels, key=kv)
        self.out_proj = eqx.nn.Linear(channels, channels, key=ko)

        self.attention_dtype = attention_dtype

        if implementation is None:
            implementation = "cudnn" if jax.devices()[0].platform == "gpu" else "xla"
        self.implementation = implementation

        requested_num_heads = num_heads

        if implementation == "cudnn":
            num_heads, head_dim = self._resolve_cudnn_heads(
                channels=channels,
                requested_num_heads=requested_num_heads,
            )
        else:
            if channels % num_heads != 0:
                raise ValueError(
                    f"channels={channels} must be divisible by num_heads={num_heads}."
                )
            head_dim = channels // num_heads

        self.num_heads = num_heads
        self.head_dim = head_dim

    @staticmethod
    def _resolve_cudnn_heads(
        *,
        channels: int,
        requested_num_heads: int,
    ) -> tuple[int, int]:
        """Resolve ``num_heads``/``head_dim`` for cuDNN flash attention.

        cuDNN requires ``head_dim <= 128`` and ``head_dim % 8 == 0``. Since
        ``head_dim = channels // num_heads``, we search over valid head
        dimensions and choose the corresponding number of heads closest to the
        requested value.
        """

        def is_valid(num_heads: int) -> bool:
            return (
                num_heads > 0
                and channels % num_heads == 0
                and channels // num_heads <= 128
                and channels // num_heads % 8 == 0
            )

        if is_valid(requested_num_heads):
            return requested_num_heads, channels // requested_num_heads

        candidates: list[tuple[int, int]] = []

        max_head_dim = min(channels, 128)
        for head_dim in range(8, max_head_dim + 1, 8):
            if channels % head_dim == 0:
                candidate_num_heads = channels // head_dim
                candidates.append((candidate_num_heads, head_dim))

        if not candidates:
            raise ValueError(
                "Could not find a cuDNN-compatible attention head configuration "
                f"for channels={channels}. Need a head_dim that divides channels, "
                "is <= 128, and is a multiple of 8."
            )

        # Choose the closest num_heads. On ties, prefer more heads, i.e. smaller
        # head_dim, which is usually safer for cuDNN flash attention.
        resolved_num_heads, resolved_head_dim = min(
            candidates,
            key=lambda pair: (
                abs(pair[0] - requested_num_heads),
                pair[0] < requested_num_heads,
            ),
        )

        requested_head_dim = (
            channels // requested_num_heads
            if channels % requested_num_heads == 0
            else f"{channels}/{requested_num_heads} (non-integer)"
        )

        warnings.warn(
            "AttentionBlock received implementation='cudnn', but the requested "
            f"num_heads={requested_num_heads} gives head_dim={requested_head_dim}, "
            "which does not satisfy the cuDNN flash-attention constraints "
            "head_dim <= 128 and head_dim % 8 == 0. "
            f"Using num_heads={resolved_num_heads} instead, giving "
            f"head_dim={resolved_head_dim}.",
            RuntimeWarning,
            stacklevel=2,
        )

        return resolved_num_heads, resolved_head_dim

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply multi-head self-attention over spatial positions.

        Args:
            x: Input array of shape ``(C, H, W)``.

        Returns:
            Attention output of shape ``(C, H, W)`` with the same dtype as ``x``.
        """
        c, h, w = x.shape
        orig_dtype = x.dtype

        tokens = x.reshape(c, h * w).T  # (T, C)
        tokens_attn = tokens.astype(self.attention_dtype)

        q = _apply_linear(self.q_proj, tokens_attn)
        k = _apply_linear(self.k_proj, tokens_attn)
        v = _apply_linear(self.v_proj, tokens_attn)

        q = q.reshape(h * w, self.num_heads, self.head_dim)
        k = k.reshape(h * w, self.num_heads, self.head_dim)
        v = v.reshape(h * w, self.num_heads, self.head_dim)

        attn_out = jax.nn.dot_product_attention(
            q, k, v, implementation=self.implementation
        )  # (T, num_heads, head_dim)

        attn_out = attn_out.reshape(h * w, c).astype(orig_dtype)
        out = _apply_linear(self.out_proj, attn_out)  # (T, C) in orig_dtype
        return out.T.reshape(c, h, w)


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
    conv1: eqx.nn.Conv2d | CoordConv
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
            conv_layer = eqx.nn.Conv2d
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
        h = _apply_conv2d(self.conv1, h.astype(self.compute_dtype)).astype(orig_dtype)

        time_h = _apply_linear(
            self.time_proj,
            time_emb.astype(self.compute_dtype),
        ).astype(orig_dtype)
        h = h + time_h.reshape(-1, 1, 1)

        h = self.norm2(h)
        h = self.activation(h)
        h = self.dropout(h, key=key)
        h = _apply_conv2d(self.conv2, h.astype(self.compute_dtype)).astype(orig_dtype)

        skip = self._resample(x)
        if self.skip_conv is not None:
            skip = _apply_conv2d(
                self.skip_conv,
                skip.astype(self.compute_dtype),
            ).astype(orig_dtype)

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
