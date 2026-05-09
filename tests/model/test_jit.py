import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from msdflow.model.jit.blocks import (
    BottleneckPatchEmbed,
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
