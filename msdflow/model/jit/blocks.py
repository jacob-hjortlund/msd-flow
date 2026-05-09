"""JiT geometry, patch embedding, and positional helper blocks."""

import equinox as eqx
import jax
import jax.numpy as jnp

from msdflow.model.common_blocks import _apply_conv2d


def _validate_grid_size(grid_size: int) -> None:
    """Validate that a patch grid has positive spatial extent.

    Args:
        grid_size: Number of patches along each image axis.

    Raises:
        ValueError: If ``grid_size`` is not positive.
    """

    if grid_size <= 0:
        raise ValueError("grid_size must be positive.")


def _patch_coordinates(grid_size: int) -> tuple[jax.Array, jax.Array]:
    """Return flattened row and column coordinates for a square patch grid.

    Args:
        grid_size: Number of patches along each image axis.

    Returns:
        Tuple of flattened row and column coordinates as float32 arrays.
    """

    _validate_grid_size(grid_size)
    positions = jnp.arange(grid_size, dtype=jnp.float32)
    rows, cols = jnp.meshgrid(positions, positions, indexing="ij")
    return rows.reshape(-1), cols.reshape(-1)


def normalized_patch_radius(grid_size: int) -> jax.Array:
    """Compute patch-center radii normalized to the unit interval.

    Args:
        grid_size: Number of patches along each image axis.

    Returns:
        Flattened normalized radius for every patch center.
    """

    _validate_grid_size(grid_size)
    axis = (jnp.arange(grid_size, dtype=jnp.float32) + 0.5) / grid_size * 2.0 - 1.0
    yy, xx = jnp.meshgrid(axis, axis, indexing="ij")
    radius = jnp.sqrt(xx**2 + yy**2)
    max_radius = jnp.max(radius)
    radius = jnp.where(max_radius > 0.0, radius / max_radius, radius)
    return radius.reshape(-1)


def _polar_coordinates(grid_size: int) -> tuple[jax.Array, jax.Array]:
    """Return flattened normalized polar coordinates for a patch grid.

    Args:
        grid_size: Number of patches along each image axis.

    Returns:
        Tuple containing normalized radius and angle arrays.
    """

    _validate_grid_size(grid_size)
    axis = (jnp.arange(grid_size, dtype=jnp.float32) + 0.5) / grid_size * 2.0 - 1.0
    yy, xx = jnp.meshgrid(axis, axis, indexing="ij")
    radius = normalized_patch_radius(grid_size)
    angle = (jnp.arctan2(yy, xx) + jnp.pi) / (2.0 * jnp.pi)
    return radius, angle.reshape(-1)


def _fixed_1d_sincos_pos_embed(embed_dim: int, positions: jax.Array) -> jax.Array:
    """Build a fixed one-dimensional sinusoidal position embedding.

    Args:
        embed_dim: Embedding dimension. Must be even.
        positions: One-dimensional position array.

    Returns:
        Sinusoidal embedding array with shape ``(positions.size, embed_dim)``.

    Raises:
        ValueError: If ``embed_dim`` is not even.
    """

    if (embed_dim % 2) != 0:
        raise ValueError("embed_dim must be even.")

    half_dim = embed_dim // 2
    omega = jnp.arange(half_dim, dtype=jnp.float32)
    omega = 1.0 / (10000.0 ** (omega / half_dim))
    args = positions.astype(jnp.float32)[:, None] * omega[None, :]
    return jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)


def fixed_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> jax.Array:
    """Build fixed two-dimensional sinusoidal patch embeddings.

    Args:
        embed_dim: Output embedding dimension. Must be a multiple of 4.
        grid_size: Number of patches along each image axis.

    Returns:
        Flattened patch embedding array with shape
        ``(grid_size * grid_size, embed_dim)`` and dtype float32.

    Raises:
        ValueError: If ``embed_dim`` is not a multiple of 4.
    """

    _validate_grid_size(grid_size)
    if (embed_dim % 4) != 0:
        raise ValueError("embed_dim must be a multiple of 4.")

    rows, cols = _patch_coordinates(grid_size)
    half_dim = embed_dim // 2
    row_embed = _fixed_1d_sincos_pos_embed(half_dim, rows)
    col_embed = _fixed_1d_sincos_pos_embed(half_dim, cols)
    return jnp.concatenate([row_embed, col_embed], axis=-1).astype(jnp.float32)


class BottleneckPatchEmbed(eqx.Module):
    """Patch embedding block with a low-dimensional convolutional bottleneck."""

    proj1: eqx.nn.Conv2d
    proj2: eqx.nn.Conv2d
    image_size: int = eqx.field(static=True)
    patch_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)
    compute_dtype: jnp.dtype = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        image_size: int,
        patch_size: int,
        bottleneck_dim: int,
        hidden_size: int,
        compute_dtype: jnp.dtype,
        key: jax.Array,
    ):
        """Initialize the patch embedding convolutions.

        Args:
            in_channels: Number of input image channels.
            image_size: Input image height and width.
            patch_size: Spatial size and stride for each patch.
            bottleneck_dim: Intermediate channel dimension.
            hidden_size: Output token dimension.
            compute_dtype: Activation dtype used while applying convolutions.
            key: JAX PRNG key.

        Raises:
            ValueError: If any dimension is nonpositive, or if ``image_size``
                is not divisible by ``patch_size``.
        """

        for name, value in (
            ("in_channels", in_channels),
            ("image_size", image_size),
            ("patch_size", patch_size),
            ("bottleneck_dim", bottleneck_dim),
            ("hidden_size", hidden_size),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")

        if (image_size % patch_size) != 0:
            raise ValueError("image_size must be divisible by patch_size.")

        k1, k2 = jax.random.split(key)
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.compute_dtype = compute_dtype
        self.proj1 = eqx.nn.Conv2d(
            in_channels,
            bottleneck_dim,
            kernel_size=patch_size,
            stride=patch_size,
            use_bias=False,
            key=k1,
        )
        self.proj2 = eqx.nn.Conv2d(
            bottleneck_dim,
            hidden_size,
            kernel_size=1,
            key=k2,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Embed an image into flattened patch tokens.

        Args:
            x: Input image with shape ``(channels, image_size, image_size)``.

        Returns:
            Patch token array with shape ``(num_patches, hidden_size)``.

        Raises:
            ValueError: If the input spatial size does not match ``image_size``.
        """

        if x.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(
                f"Input image size must be {(self.image_size, self.image_size)}, "
                f"got {x.shape[-2:]}."
            )

        original_dtype = x.dtype
        h = _apply_conv2d(self.proj1, x.astype(self.compute_dtype))
        h = _apply_conv2d(self.proj2, h)
        channels, height, width = h.shape
        return h.reshape(channels, height * width).T.astype(original_dtype)


class TwoDimensionalRoPE(eqx.Module):
    """Two-dimensional rotary positional embedding for image patch tokens."""

    rope_a: eqx.nn.RotaryPositionalEmbedding
    rope_b: eqx.nn.RotaryPositionalEmbedding
    coords_a: tuple[float, ...] = eqx.field(static=True)
    coords_b: tuple[float, ...] = eqx.field(static=True)
    grid_size: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    mode: str = eqx.field(static=True)
    coord_dim: int = eqx.field(static=True)
    dtype: jnp.dtype = eqx.field(static=True)

    def __init__(
        self,
        grid_size: int,
        head_dim: int,
        mode: str,
        theta: float,
        dtype: jnp.dtype,
    ):
        """Initialize two-axis RoPE coordinates and embedding modules.

        Args:
            grid_size: Number of patches along each image axis.
            head_dim: Query/key head dimension. Must be a multiple of 4.
            mode: Coordinate system, either ``"cartesian"`` or ``"polar"``.
            theta: RoPE frequency base.
            dtype: RoPE frequency dtype.

        Raises:
            ValueError: If ``mode`` or ``head_dim`` is invalid.
        """

        _validate_grid_size(grid_size)
        if mode not in ("cartesian", "polar"):
            raise ValueError("rope_mode must be 'cartesian' or 'polar'.")
        if (head_dim % 4) != 0:
            raise ValueError("head_dim must be a multiple of 4.")

        self.grid_size = grid_size
        self.head_dim = head_dim
        self.mode = mode
        self.coord_dim = head_dim // 2
        self.dtype = dtype
        if mode == "cartesian":
            coords_a, coords_b = _patch_coordinates(grid_size)
        else:
            coords_a, coords_b = _polar_coordinates(grid_size)
        self.coords_a = tuple(float(coord) for coord in coords_a)
        self.coords_b = tuple(float(coord) for coord in coords_b)
        self.rope_a = eqx.nn.RotaryPositionalEmbedding(
            self.coord_dim, theta=theta, dtype=dtype
        )
        self.rope_b = eqx.nn.RotaryPositionalEmbedding(
            self.coord_dim, theta=theta, dtype=dtype
        )

    def _apply_explicit_positions(
        self,
        rope: eqx.nn.RotaryPositionalEmbedding,
        x: jax.Array,
        coords: tuple[float, ...],
    ) -> jax.Array:
        """Apply RoPE using explicit floating-point token coordinates.

        Args:
            rope: Equinox RoPE module providing frequency configuration.
            x: Token-head features with shape ``(tokens, heads, coord_dim)``.
            coords: Static per-token coordinates.

        Returns:
            Rotated features with the same shape and dtype as ``x``.
        """

        coords_array = jnp.asarray(coords, dtype=jnp.float32)
        freqs = 1.0 / (
            rope.theta
            ** (
                jnp.arange(0.0, rope.embedding_size, 2, dtype=jnp.float32)
                / rope.embedding_size
            )
        )
        angles = coords_array[:, None] * freqs[None, :]
        cos = jnp.tile(jnp.cos(angles), (1, 2)).astype(rope.dtype).astype(x.dtype)
        sin = jnp.tile(jnp.sin(angles), (1, 2)).astype(rope.dtype).astype(x.dtype)
        cos = cos[:, None, :]
        sin = sin[:, None, :]
        rotated = eqx.nn.RotaryPositionalEmbedding.rotate_half(x)
        return x * cos + rotated * sin

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply two-dimensional RoPE to token-head features.

        Args:
            x: Query or key features with shape ``(tokens, heads, head_dim)``.

        Returns:
            Rotated features with the same shape and dtype as ``x``.

        Raises:
            ValueError: If token or feature dimensions do not match the grid.
        """

        expected_tokens = self.grid_size * self.grid_size
        if x.shape[0] != expected_tokens:
            raise ValueError(
                f"Expected {expected_tokens} tokens for grid_size={self.grid_size}, "
                f"got {x.shape[0]}."
            )
        if x.shape[-1] != self.head_dim:
            raise ValueError(
                f"Expected head_dim={self.head_dim}, got {x.shape[-1]}."
            )

        x_a, x_b = jnp.split(x, 2, axis=-1)
        x_a = self._apply_explicit_positions(self.rope_a, x_a, self.coords_a)
        x_b = self._apply_explicit_positions(self.rope_b, x_b, self.coords_b)
        return jnp.concatenate([x_a, x_b], axis=-1).astype(x.dtype)
