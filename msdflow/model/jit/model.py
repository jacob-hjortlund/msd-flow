"""Top-level Just image Transformer flow-matching model."""

from typing import Callable, List, Optional

import equinox as eqx
import jax
import jax.numpy as jnp

from msdflow.model.common_blocks import SinusoidalEmbedding
from msdflow.model.jit.blocks import (
    BottleneckPatchEmbed,
    FinalLayer,
    JiTBlock,
    TwoDimensionalRoPE,
    fixed_2d_sincos_pos_embed,
    normalized_patch_radius,
)


class JiT(eqx.Module):
    """Just image Transformer model with the NCSN++ call signature.

    The model predicts an image-shaped flow-matching target from a scalar time,
    an input image, and an optional one-dimensional condition.
    """

    x_embedder: BottleneckPatchEmbed
    time_embed: SinusoidalEmbedding
    cond_dim: int = eqx.field(static=True)
    cond_embed: Optional[SinusoidalEmbedding]
    null_cond_emb: Optional[jax.Array]
    pos_embed: tuple[tuple[float, ...], ...] = eqx.field(static=True)
    radial_values: Optional[tuple[float, ...]] = eqx.field(static=True)
    radial_embed: Optional[SinusoidalEmbedding]
    rope: TwoDimensionalRoPE
    blocks: List[JiTBlock]
    final_layer: FinalLayer
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    image_size: int = eqx.field(static=True)
    patch_size: int = eqx.field(static=True)
    grid_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    prediction_type: str = eqx.field(static=True)
    activation: Callable = eqx.field(static=True)
    compute_dtype: jnp.dtype = eqx.field(static=True)
    attention_dtype: jnp.dtype = eqx.field(static=True)
    attention_implementation: Optional[str] = eqx.field(static=True)
    rope_mode: str = eqx.field(static=True)
    use_radial_embedding: bool = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: int,
        patch_size: int,
        hidden_size: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        bottleneck_dim: int,
        key: jax.Array,
        cond_dim: int = 0,
        prediction_type: str = "velocity",
        activation: Callable = jax.nn.silu,
        frequency_embedding_dim: int = 256,
        dropout: float = 0.0,
        compute_dtype: jnp.dtype = jnp.float32,
        attention_dtype: jnp.dtype = jnp.float32,
        attention_implementation: Optional[str] = None,
        rope_mode: str = "cartesian",
        rope_theta: float = 10000.0,
        use_radial_embedding: bool = False,
        radial_power: float = 1.0,
    ):
        """Initialize the top-level JiT model.

        Args:
            in_channels: Number of input image channels.
            out_channels: Number of output image channels.
            image_size: Input and output image height and width.
            patch_size: Patch side length.
            hidden_size: Transformer token width.
            depth: Number of transformer blocks.
            num_heads: Number of attention heads.
            mlp_ratio: Feed-forward expansion ratio.
            bottleneck_dim: Patch embedding bottleneck channel width.
            key: JAX PRNG key.
            cond_dim: Optional condition dimension. Only ``0`` and ``1`` are
                currently supported.
            prediction_type: Prediction target, either ``"velocity"`` or
                ``"image"``.
            activation: Activation function for embedding and modulation MLPs.
            frequency_embedding_dim: Sinusoidal basis width.
            dropout: Dropout probability inside transformer blocks.
            compute_dtype: Dtype used for compute-heavy linear projections.
            attention_dtype: Dtype used for attention projections and RoPE.
            attention_implementation: Optional JAX attention backend override.
            rope_mode: Two-dimensional RoPE coordinate mode.
            rope_theta: RoPE frequency base.
            use_radial_embedding: Whether to add learned embeddings of fixed
                radial patch coordinates.
            radial_power: Power applied to inverted normalized patch radius.

        Raises:
            ValueError: If the configured dimensions or modes are invalid.
        """

        for name, value in (
            ("in_channels", in_channels),
            ("out_channels", out_channels),
            ("image_size", image_size),
            ("patch_size", patch_size),
            ("hidden_size", hidden_size),
            ("depth", depth),
            ("num_heads", num_heads),
            ("bottleneck_dim", bottleneck_dim),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")
        if cond_dim < 0 or cond_dim > 1:
            raise ValueError("cond_dim must be 0 or 1.")
        if prediction_type not in ("velocity", "image"):
            raise ValueError("prediction_type must be 'velocity' or 'image'.")
        if (image_size % patch_size) != 0:
            raise ValueError("image_size must be divisible by patch_size.")
        if (hidden_size % num_heads) != 0:
            raise ValueError("hidden_size must be divisible by num_heads.")
        if (hidden_size % 4) != 0:
            raise ValueError("hidden_size must be divisible by 4.")
        head_dim = hidden_size // num_heads
        if (head_dim % 4) != 0:
            raise ValueError("head_dim must be a multiple of 4.")
        if (frequency_embedding_dim % 2) != 0:
            raise ValueError("frequency_embedding_dim must be even.")
        if radial_power < 0.0:
            raise ValueError("radial_power must be nonnegative.")

        k_patch, k_time, k_cond, k_radial, k_final, key = jax.random.split(key, 6)
        grid_size = image_size // patch_size
        pos_embed = fixed_2d_sincos_pos_embed(hidden_size, grid_size)
        self.pos_embed = tuple(tuple(float(value) for value in row) for row in pos_embed)

        if use_radial_embedding:
            radial_values = (1.0 - normalized_patch_radius(grid_size)) ** radial_power
            self.radial_values = tuple(float(value) for value in radial_values)
            self.radial_embed = SinusoidalEmbedding(
                dim=hidden_size,
                activation=activation,
                key=k_radial,
                frequency_dim=frequency_embedding_dim,
            )
        else:
            self.radial_values = None
            self.radial_embed = None

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = grid_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.cond_dim = cond_dim
        self.prediction_type = prediction_type
        self.activation = activation
        self.compute_dtype = compute_dtype
        self.attention_dtype = attention_dtype
        self.attention_implementation = attention_implementation
        self.rope_mode = rope_mode
        self.use_radial_embedding = use_radial_embedding

        self.x_embedder = BottleneckPatchEmbed(
            in_channels=in_channels,
            image_size=image_size,
            patch_size=patch_size,
            bottleneck_dim=bottleneck_dim,
            hidden_size=hidden_size,
            compute_dtype=compute_dtype,
            key=k_patch,
        )
        self.time_embed = SinusoidalEmbedding(
            dim=hidden_size,
            activation=activation,
            key=k_time,
            frequency_dim=frequency_embedding_dim,
        )
        if cond_dim > 0:
            self.cond_embed = SinusoidalEmbedding(
                dim=hidden_size,
                activation=activation,
                key=k_cond,
                frequency_dim=frequency_embedding_dim,
            )
            self.null_cond_emb = jnp.zeros((hidden_size,), dtype=jnp.float32)
        else:
            self.cond_embed = None
            self.null_cond_emb = None

        self.rope = TwoDimensionalRoPE(
            grid_size=grid_size,
            head_dim=head_dim,
            mode=rope_mode,
            theta=rope_theta,
            dtype=attention_dtype,
        )
        block_keys = jax.random.split(key, depth)
        self.blocks = [
            JiTBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                activation=activation,
                compute_dtype=compute_dtype,
                attention_dtype=attention_dtype,
                attention_implementation=attention_implementation,
                key=block_key,
            )
            for block_key in block_keys
        ]
        self.final_layer = FinalLayer(
            hidden_size=hidden_size,
            patch_size=patch_size,
            out_channels=out_channels,
            activation=activation,
            compute_dtype=compute_dtype,
            key=k_final,
        )

    def _unpatchify(self, patches: jax.Array) -> jax.Array:
        """Convert flattened patch predictions back to channel-first images.

        Args:
            patches: Flattened patch predictions with shape
                ``(grid_size * grid_size, patch_size * patch_size * out_channels)``.

        Returns:
            Image tensor with shape
            ``(out_channels, image_size, image_size)``.

        Raises:
            ValueError: If ``patches`` does not have the expected shape.
        """

        expected_shape = (
            self.grid_size * self.grid_size,
            self.patch_size * self.patch_size * self.out_channels,
        )
        if patches.shape != expected_shape:
            raise ValueError(f"Expected patches shape {expected_shape}, got {patches.shape}.")

        patches = patches.reshape(
            self.grid_size,
            self.grid_size,
            self.patch_size,
            self.patch_size,
            self.out_channels,
        )
        patches = patches.transpose(4, 0, 2, 1, 3)
        return patches.reshape(self.out_channels, self.image_size, self.image_size)

    def __call__(
        self,
        t: jax.Array,
        x_t: jax.Array,
        cond: jax.Array,
        cond_mask: jax.Array,
        key: jax.Array,
    ) -> jax.Array:
        """Predict a flow-matching target using the NCSN++ call signature.

        Args:
            t: Scalar diffusion or flow time.
            x_t: Image tensor with shape
                ``(in_channels, image_size, image_size)``.
            cond: Optional condition vector. For conditional models this has
                shape ``(1,)``.
            cond_mask: Boolean selecting the provided condition when true and
                the learned null condition when false.
            key: JAX PRNG key used by stochastic transformer blocks.

        Returns:
            Predicted image tensor with shape
            ``(out_channels, image_size, image_size)``.

        Raises:
            ValueError: If ``x_t`` does not match the configured image shape.
        """

        expected_shape = (self.in_channels, self.image_size, self.image_size)
        if x_t.shape != expected_shape:
            raise ValueError(f"Input image size must be {expected_shape}, got {x_t.shape}.")

        combined_emb = self.time_embed(t)
        if self.cond_dim > 0:
            if self.cond_embed is None or self.null_cond_emb is None:
                raise ValueError("cond_dim is positive but condition embedding is missing.")
            cond_emb = self.cond_embed(cond.reshape(-1)[0])
            cond_emb = jnp.where(jnp.asarray(cond_mask), cond_emb, self.null_cond_emb)
            combined_emb = combined_emb + cond_emb

        h = self.x_embedder(x_t)
        pos_embed = jax.lax.stop_gradient(jnp.asarray(self.pos_embed, dtype=h.dtype))
        h = h + pos_embed
        if self.use_radial_embedding:
            if self.radial_values is None or self.radial_embed is None:
                raise ValueError("Radial embedding is enabled but radial values are missing.")
            radial_values = jax.lax.stop_gradient(
                jnp.asarray(self.radial_values, dtype=jnp.float32)
            )
            radial_embed = jax.vmap(self.radial_embed)(radial_values).astype(h.dtype)
            h = h + radial_embed

        block_keys = jax.random.split(key, self.depth)
        for block, block_key in zip(self.blocks, block_keys):
            h = block(h, combined_emb, self.rope, block_key)

        patches = self.final_layer(h, combined_emb)
        return self._unpatchify(patches)
