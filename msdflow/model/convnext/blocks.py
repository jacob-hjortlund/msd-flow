"""ConvNeXt-specific Equinox blocks."""

from typing import Optional, Sequence, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp

__all__ = [
    "ConvNeXtBlock",
    "ConvNeXtDownsample",
    "ConvNeXtHead",
    "ConvNeXtStage",
    "ConvNeXtStem",
    "DropPath",
    "Identity",
    "LayerNorm2d",
]


class Identity(eqx.Module):
    """Identity layer that returns inputs unchanged."""

    def __call__(self, x, *, key=None):
        """Return the input unchanged.

        Args:
            x: Input value.
            key: Optional PRNG key accepted for call compatibility.

        Returns:
            The unchanged input value.
        """
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
        """Initialize the per-channel affine normalization parameters.

        Args:
            num_channels: Number of input channels.
            eps: Numerical stability epsilon.
            dtype: Parameter dtype.
        """
        self.weight = jnp.ones((num_channels,), dtype=dtype)
        self.bias = jnp.zeros((num_channels,), dtype=dtype)
        self.eps = eps

    def __call__(self, x: jax.Array) -> jax.Array:
        """Normalize a channel-first feature map.

        Args:
            x: Input array of shape ``(C, H, W)``.

        Returns:
            Normalized array of shape ``(C, H, W)``.
        """
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
        """Initialize stochastic depth.

        Args:
            p: Probability of dropping the path.
            inference: Whether to disable stochastic depth.
        """
        self.p = p
        self.inference = inference

    def __call__(self, x: jax.Array, *, key: Optional[jax.Array] = None) -> jax.Array:
        """Apply stochastic depth to an input feature map.

        Args:
            x: Input feature map.
            key: PRNG key required when ``p > 0`` and not in inference mode.

        Returns:
            Feature map after stochastic depth.

        Raises:
            ValueError: If a PRNG key is required but not provided.
        """
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
        """Initialize a ConvNeXt block.

        Args:
            dim: Number of input and output channels.
            mlp_ratio: Expansion ratio for the pointwise convolution MLP.
            kernel_size: Depthwise convolution kernel size.
            ls_init_value: Initial layer-scale value, or ``None`` to disable.
            drop_path: Stochastic depth probability.
            inference: Whether to disable stochastic depth.
            dtype: Parameter dtype.
            key: JAX PRNG key.
        """
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
        """Apply the ConvNeXt block.

        Args:
            x: Input feature map of shape ``(C, H, W)``.
            key: Optional PRNG key for stochastic depth.

        Returns:
            Output feature map of shape ``(C, H, W)``.
        """
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
        """Initialize a ConvNeXt downsampling layer.

        Args:
            in_chs: Number of input channels.
            out_chs: Number of output channels.
            stride: Downsampling stride.
            dtype: Parameter dtype.
            key: JAX PRNG key.

        Raises:
            ValueError: If ``stride`` is not 1 or 2.
        """
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
        """Downsample a feature map.

        Args:
            x: Input feature map of shape ``(C, H, W)``.

        Returns:
            Downsampled feature map.
        """
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
        """Initialize a ConvNeXt stage.

        Args:
            in_chs: Number of input channels.
            out_chs: Number of output channels.
            depth: Number of ConvNeXt blocks.
            stride: Stage stride.
            kernel_size: Depthwise convolution kernel size.
            mlp_ratio: Expansion ratio for block MLPs.
            ls_init_value: Initial layer-scale value, or ``None`` to disable.
            drop_path_rates: Per-block stochastic depth probabilities.
            inference: Whether to disable stochastic depth.
            dtype: Parameter dtype.
            key: JAX PRNG key.

        Raises:
            ValueError: If ``depth`` is less than 1, or drop-path rates have
                the wrong length.
        """
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
        """Apply the ConvNeXt stage.

        Args:
            x: Input feature map.
            key: Optional PRNG key for per-block stochastic depth.

        Returns:
            Output feature map after downsampling and stage blocks.
        """
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
        """Initialize the ConvNeXt patch stem.

        Args:
            in_chans: Number of input channels.
            out_chs: Number of output channels.
            patch_size: Patch size and convolution stride.
            dtype: Parameter dtype.
            key: JAX PRNG key.
        """
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
        """Apply the patch stem.

        Args:
            x: Input image tensor of shape ``(C, H, W)``.

        Returns:
            Stem feature map.
        """
        x = self.conv(x)
        x = self.norm(x)
        return x


class ConvNeXtHead(eqx.Module):
    """Matches your printed head for num_classes=0:
    global avg pool (keepdims) -> LayerNorm2d -> flatten
    """

    norm: LayerNorm2d

    def __init__(self, dim: int, *, dtype=jnp.float32):
        """Initialize the ConvNeXt pooling head.

        Args:
            dim: Number of channels in the input feature map.
            dtype: Parameter dtype.
        """
        self.norm = LayerNorm2d(dim, eps=1e-6, dtype=dtype)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Pool and flatten the final ConvNeXt feature map.

        Args:
            x: Input feature map of shape ``(C, H, W)``.

        Returns:
            Pooled feature vector of shape ``(C,)``.
        """
        # x: (C, H, W)
        x = jnp.mean(x, axis=(1, 2), keepdims=True)  # (C, 1, 1)
        x = self.norm(x)  # (C, 1, 1)
        x = jnp.reshape(x, (-1,))  # (C,)
        return x
