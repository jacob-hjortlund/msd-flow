"""Shared model blocks used across model families."""

import warnings
from typing import Callable
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp


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


class SinusoidalEmbedding(eqx.Module):
    """Sinusoidal positional embedding with a two-layer MLP."""

    lin1: eqx.nn.Linear
    lin2: eqx.nn.Linear
    dim: int = eqx.field(static=True)
    frequency_dim: int = eqx.field(static=True)
    activation: Callable = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        activation: Callable,
        key: jax.Array,
        frequency_dim: Optional[int] = None,
    ):
        """Initialise the embedding layers.

        Args:
            dim: Output embedding dimension. Must be even.
            activation: Activation function.
            key: JAX PRNG key.
            frequency_dim: Sinusoidal basis dimension before projection. Defaults
                to ``dim`` and must be even.

        Raises:
            ValueError: If ``dim`` or ``frequency_dim`` is not even.
        """

        if (dim % 2) != 0:
            raise ValueError("embedding dimension must be even.")

        if frequency_dim is None:
            frequency_dim = dim
        if (frequency_dim % 2) != 0:
            raise ValueError("frequency dimension must be even.")

        k1, k2 = jax.random.split(key)
        self.dim = dim
        self.frequency_dim = frequency_dim
        self.activation = activation
        self.lin1 = eqx.nn.Linear(frequency_dim, dim, key=k1)
        self.lin2 = eqx.nn.Linear(dim, dim, key=k2)

    def __call__(self, t: jax.Array) -> jax.Array:
        """Embed a scalar time value.

        Args:
            t: Time to embed.

        Returns:
            Sinusoidal time embedding of shape ``(dim,)``.
        """

        half = self.frequency_dim // 2
        freqs = jnp.exp(
            -jnp.log(10000.0)
            * 2
            * jnp.arange(half, dtype=jnp.float32)
            / self.frequency_dim
        )
        emb = jnp.concatenate([jnp.sin(t * freqs), jnp.cos(t * freqs)])
        emb = self.lin1(emb)
        emb = self.activation(emb)
        emb = self.lin2(emb)
        return emb


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
