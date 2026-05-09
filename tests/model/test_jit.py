import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

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
