import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from msdflow.model.jit import JiT
from msdflow.model.jit.blocks import (
    BottleneckPatchEmbed,
    FinalLayer,
    JiTAttention,
    JiTBlock,
    SwiGLUFFN,
    TwoDimensionalRoPE,
    fixed_2d_sincos_pos_embed,
    normalized_patch_radius,
)

KEY = jax.random.PRNGKey(0)
SMALL_CFG = dict(
    in_channels=1,
    out_channels=1,
    image_size=8,
    patch_size=2,
    hidden_size=16,
    depth=2,
    num_heads=2,
    mlp_ratio=4.0,
    bottleneck_dim=8,
    activation=jax.nn.silu,
    frequency_embedding_dim=16,
    dropout=0.0,
    attention_implementation="xla",
)
SMALL_CFG_COND = {**SMALL_CFG, "cond_dim": 1}


def _array_leaf_dtypes(pytree):
    """Return dtypes for array leaves in a pytree."""

    return {leaf.dtype for leaf in jax.tree.leaves(eqx.filter(pytree, eqx.is_array))}


def test_fixed_2d_sincos_pos_embed_shape_and_dtype():
    pos_embed = fixed_2d_sincos_pos_embed(embed_dim=16, grid_size=2)

    assert pos_embed.shape == (4, 16)
    assert pos_embed.dtype == jnp.float32


def test_fixed_2d_sincos_pos_embed_requires_multiple_of_four():
    with pytest.raises(ValueError, match="multiple of 4"):
        fixed_2d_sincos_pos_embed(embed_dim=10, grid_size=2)


def test_normalized_patch_radius_is_unit_interval():
    radii = normalized_patch_radius(grid_size=3)

    assert radii.shape == (9,)
    assert jnp.all(radii >= 0.0)
    assert jnp.all(radii <= 1.0)
    assert float(jnp.max(radii)) == pytest.approx(1.0)
    assert float(radii[4]) == pytest.approx(0.0)


def test_bottleneck_patch_embed_output_shape():
    embed = BottleneckPatchEmbed(
        in_channels=1,
        image_size=8,
        patch_size=4,
        bottleneck_dim=6,
        hidden_size=12,
        compute_dtype=jnp.float32,
        key=KEY,
    )

    assert isinstance(embed, eqx.Module)
    tokens = embed(jnp.ones((1, 8, 8), dtype=jnp.float32))

    assert tokens.shape == (4, 12)
    assert tokens.dtype == jnp.float32


def test_bottleneck_patch_embed_validates_input_shape():
    embed = BottleneckPatchEmbed(
        in_channels=1,
        image_size=8,
        patch_size=4,
        bottleneck_dim=6,
        hidden_size=12,
        compute_dtype=jnp.float32,
        key=KEY,
    )

    with pytest.raises(ValueError, match="Input image size"):
        embed(jnp.ones((1, 4, 8), dtype=jnp.float32))


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("in_channels", 0),
        ("image_size", 0),
        ("patch_size", 0),
        ("bottleneck_dim", 0),
        ("hidden_size", 0),
    ],
)
def test_bottleneck_patch_embed_rejects_nonpositive_dimensions(parameter, value):
    kwargs = {
        "in_channels": 1,
        "image_size": 8,
        "patch_size": 4,
        "bottleneck_dim": 6,
        "hidden_size": 12,
        "compute_dtype": jnp.float32,
        "key": KEY,
    }
    kwargs[parameter] = value

    with pytest.raises(ValueError, match=parameter):
        BottleneckPatchEmbed(**kwargs)


def test_two_dimensional_rope_cartesian_changes_query_features():
    rope = TwoDimensionalRoPE(
        grid_size=2,
        head_dim=8,
        mode="cartesian",
        theta=10000.0,
        dtype=jnp.float32,
    )
    x = jnp.arange(4 * 2 * 8, dtype=jnp.float32).reshape(4, 2, 8)

    out = rope(x)

    assert out.shape == x.shape
    assert not jnp.allclose(out, x)


def test_two_dimensional_rope_polar_changes_query_features():
    rope = TwoDimensionalRoPE(
        grid_size=2,
        head_dim=8,
        mode="polar",
        theta=10000.0,
        dtype=jnp.float32,
    )
    x = jnp.arange(4 * 2 * 8, dtype=jnp.float32).reshape(4, 2, 8)

    out = rope(x)

    assert out.shape == x.shape
    assert not jnp.allclose(out, x)


def test_two_dimensional_rope_has_no_trainable_array_leaves():
    rope = TwoDimensionalRoPE(
        grid_size=2,
        head_dim=8,
        mode="cartesian",
        theta=10000.0,
        dtype=jnp.float32,
    )

    assert not jax.tree.leaves(eqx.filter(rope, eqx.is_inexact_array))


def test_two_dimensional_rope_rejects_none_mode():
    with pytest.raises(ValueError, match="rope_mode"):
        TwoDimensionalRoPE(
            grid_size=2,
            head_dim=8,
            mode="none",
            theta=10000.0,
            dtype=jnp.float32,
        )


def test_two_dimensional_rope_requires_head_dim_multiple_of_four():
    with pytest.raises(ValueError, match="head_dim"):
        TwoDimensionalRoPE(
            grid_size=2,
            head_dim=6,
            mode="cartesian",
            theta=10000.0,
            dtype=jnp.float32,
        )


def test_swiglu_ffn_preserves_token_shape():
    """SwiGLU FFN should map hidden tokens back to hidden size."""
    ffn = SwiGLUFFN(
        hidden_size=16,
        mlp_ratio=4.0,
        dropout=0.0,
        activation=jax.nn.silu,
        compute_dtype=jnp.float32,
        key=KEY,
    )
    x = jnp.ones((4, 16), dtype=jnp.float32)
    out = ffn(x, KEY)
    assert out.shape == x.shape
    assert out.dtype == jnp.float32


@pytest.mark.parametrize(
    ("hidden_size", "mlp_ratio", "match"),
    [
        (0, 4.0, "hidden_size"),
        (16, 0.0, "mlp_ratio"),
        (4, 0.1, "swiglu_hidden"),
    ],
)
def test_swiglu_ffn_rejects_invalid_dimensions(hidden_size, mlp_ratio, match):
    """SwiGLU FFN should reject nonpositive and zero-width dimensions."""
    with pytest.raises(ValueError, match=match):
        SwiGLUFFN(
            hidden_size=hidden_size,
            mlp_ratio=mlp_ratio,
            dropout=0.0,
            activation=jax.nn.silu,
            compute_dtype=jnp.float32,
            key=KEY,
        )


def test_jit_attention_preserves_token_shape_and_exposes_backend():
    """JiT attention should preserve token shape and store backend choice."""
    rope = TwoDimensionalRoPE(
        grid_size=2,
        head_dim=8,
        mode="cartesian",
        theta=10000.0,
        dtype=jnp.float32,
    )
    attn = JiTAttention(
        hidden_size=16,
        num_heads=2,
        dropout=0.0,
        attention_dtype=jnp.float32,
        implementation="xla",
        key=KEY,
    )
    x = jnp.ones((4, 16), dtype=jnp.float32)
    out = attn(x, rope, KEY)
    assert out.shape == x.shape
    assert out.dtype == jnp.float32
    assert attn.implementation == "xla"


@pytest.mark.parametrize(
    ("hidden_size", "num_heads", "match"),
    [
        (0, 2, "hidden_size"),
        (16, 0, "num_heads"),
    ],
)
def test_jit_attention_rejects_invalid_dimensions(hidden_size, num_heads, match):
    """JiT attention should reject nonpositive dimensions clearly."""
    with pytest.raises(ValueError, match=match):
        JiTAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=0.0,
            attention_dtype=jnp.float32,
            implementation="xla",
            key=KEY,
        )


def test_jit_block_preserves_token_shape():
    """One AdaLN-gated JiT block should preserve token shape."""
    rope = TwoDimensionalRoPE(
        grid_size=2,
        head_dim=8,
        mode="cartesian",
        theta=10000.0,
        dtype=jnp.float32,
    )
    block = JiTBlock(
        hidden_size=16,
        num_heads=2,
        mlp_ratio=4.0,
        dropout=0.0,
        activation=jax.nn.silu,
        compute_dtype=jnp.float32,
        attention_dtype=jnp.float32,
        attention_implementation="xla",
        key=KEY,
    )
    x = jnp.ones((4, 16), dtype=jnp.float32)
    cond = jnp.ones((16,), dtype=jnp.float32)
    out = block(x, cond, rope, KEY)
    assert out.shape == x.shape
    assert out.dtype == jnp.float32


def test_final_layer_projects_tokens_to_patch_pixels():
    """FinalLayer should produce one flattened patch per token."""
    layer = FinalLayer(
        hidden_size=16,
        patch_size=2,
        out_channels=1,
        activation=jax.nn.silu,
        compute_dtype=jnp.float32,
        key=KEY,
    )
    x = jnp.ones((4, 16), dtype=jnp.float32)
    cond = jnp.ones((16,), dtype=jnp.float32)
    out = layer(x, cond)
    assert out.shape == (4, 4)
    assert out.dtype == jnp.float32


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("hidden_size", 0),
        ("patch_size", 0),
        ("out_channels", 0),
    ],
)
def test_final_layer_rejects_invalid_dimensions(parameter, value):
    """FinalLayer should reject nonpositive dimensions clearly."""
    kwargs = {
        "hidden_size": 16,
        "patch_size": 2,
        "out_channels": 1,
        "activation": jax.nn.silu,
        "compute_dtype": jnp.float32,
        "key": KEY,
    }
    kwargs[parameter] = value

    with pytest.raises(ValueError, match=parameter):
        FinalLayer(**kwargs)


def test_jit_output_shape_matches_input():
    """JiT should predict an image-shaped tensor."""
    model = JiT(**SMALL_CFG, key=KEY)
    x = jnp.ones((1, 8, 8), dtype=jnp.float32)

    out = model(0.5, x, jnp.empty((0,), dtype=jnp.float32), False, KEY)

    assert out.shape == x.shape
    assert out.dtype == jnp.float32


def test_jit_output_is_finite():
    """JiT should produce finite outputs for finite image inputs."""
    model = JiT(**SMALL_CFG, key=KEY)
    x = jax.random.normal(KEY, (1, 8, 8), dtype=jnp.float32)

    out = model(0.5, x, jnp.empty((0,), dtype=jnp.float32), False, KEY)

    assert jnp.all(jnp.isfinite(out))


def test_jit_filter_vmap_over_batch():
    """JiT should vectorize over the NCSN++ call signature."""
    model = JiT(**SMALL_CFG, key=KEY)
    ts = jnp.array([0.1, 0.9], dtype=jnp.float32)
    xs = jnp.ones((2, 1, 8, 8), dtype=jnp.float32)
    cond = jnp.empty((2, 0), dtype=jnp.float32)
    masks = jnp.array([False, False])
    keys = jax.random.split(KEY, 2)

    out = eqx.filter_vmap(model)(ts, xs, cond, masks, keys)

    assert out.shape == xs.shape


def test_jit_cond_mask_routes_condition():
    """JiT should use the null condition when the condition mask is false."""
    model = JiT(**SMALL_CFG_COND, key=KEY)
    x = jnp.ones((1, 8, 8), dtype=jnp.float32)
    cond_a = jnp.array([0.0], dtype=jnp.float32)
    cond_b = jnp.array([1.0], dtype=jnp.float32)

    masked_a = model(0.5, x, cond_a, False, KEY)
    masked_b = model(0.5, x, cond_b, False, KEY)
    unmasked_a = model(0.5, x, cond_a, True, KEY)
    unmasked_b = model(0.5, x, cond_b, True, KEY)

    assert jnp.allclose(masked_a, masked_b)
    assert not jnp.allclose(unmasked_a, unmasked_b)


def test_jit_prediction_type_validation():
    """JiT should accept supported prediction targets and reject unsupported ones."""
    velocity_model = JiT(**SMALL_CFG, key=KEY)
    image_model = JiT(**SMALL_CFG, prediction_type="image", key=KEY)

    assert velocity_model.prediction_type == "velocity"
    assert image_model.prediction_type == "image"
    with pytest.raises(ValueError, match="prediction_type"):
        JiT(**SMALL_CFG, prediction_type="score", key=KEY)


def test_jit_cond_dim_gt1_raises():
    """JiT should reject multidimensional condition vectors for now."""
    with pytest.raises(ValueError, match="cond_dim"):
        JiT(**SMALL_CFG, cond_dim=2, key=KEY)


def test_jit_rejects_input_size_mismatch():
    """JiT should reject inputs with unexpected spatial size."""
    model = JiT(**SMALL_CFG, key=KEY)

    with pytest.raises(ValueError, match="Input image size"):
        model(0.5, jnp.ones((1, 4, 8), dtype=jnp.float32), jnp.empty((0,)), False, KEY)


def test_jit_rejects_invalid_patch_size():
    """JiT should reject patch sizes that do not divide the image size."""
    with pytest.raises(ValueError, match="divisible"):
        JiT(**{**SMALL_CFG, "patch_size": 3}, key=KEY)


def test_jit_rejects_hidden_size_not_divisible_by_heads():
    """JiT should reject hidden sizes that do not divide into heads."""
    with pytest.raises(ValueError, match="divisible"):
        JiT(**{**SMALL_CFG, "hidden_size": 18}, key=KEY)


def test_jit_rope_mode_polar_smoke():
    """JiT should run with polar two-dimensional RoPE."""
    model = JiT(**SMALL_CFG, rope_mode="polar", key=KEY)
    x = jnp.ones((1, 8, 8), dtype=jnp.float32)

    out = model(0.5, x, jnp.empty((0,), dtype=jnp.float32), False, KEY)

    assert out.shape == x.shape


def test_jit_rejects_rope_mode_none():
    """JiT should reject disabled RoPE mode."""
    with pytest.raises(ValueError, match="rope_mode"):
        JiT(**SMALL_CFG, rope_mode="none", key=KEY)


def test_jit_radial_embedding_smoke():
    """JiT should run with fixed radial patch embeddings."""
    model = JiT(**SMALL_CFG, use_radial_embedding=True, key=KEY)
    x = jnp.ones((1, 8, 8), dtype=jnp.float32)

    out = model(0.5, x, jnp.empty((0,), dtype=jnp.float32), False, KEY)

    assert out.shape == x.shape


def test_jit_rejects_negative_radial_power():
    """JiT should reject negative radial embedding powers."""
    with pytest.raises(ValueError, match="radial_power"):
        JiT(**SMALL_CFG, use_radial_embedding=True, radial_power=-1.0, key=KEY)


def test_jit_fixed_geometry_has_no_trainable_array_leaves():
    """JiT fixed geometry values should not appear as trainable array leaves."""
    model = JiT(**SMALL_CFG, use_radial_embedding=True, key=KEY)

    assert isinstance(model.pos_embed, tuple)
    assert isinstance(model.radial_values, tuple)
    assert jnp.dtype(jnp.float32) in _array_leaf_dtypes(model)


def test_jit_gradient_flows():
    """JiT should have nonzero gradients for at least some trainable arrays."""
    model = JiT(**SMALL_CFG, key=KEY)
    x = jax.random.normal(KEY, (1, 8, 8), dtype=jnp.float32)

    def loss_fn(model):
        """Return a scalar JiT output sum."""
        return jnp.sum(model(0.5, x, jnp.empty((0,), dtype=jnp.float32), False, KEY))

    grads = eqx.filter_grad(loss_fn)(model)
    grad_leaves = jax.tree.leaves(eqx.filter(grads, eqx.is_array))

    assert any(jnp.any(leaf != 0.0) for leaf in grad_leaves)
