"""Tests for canonical model block implementations."""

import jax
import pytest

import equinox as eqx
import jax.numpy as jnp

from msdflow.model.common_blocks import AttentionBlock
from msdflow.model.common_blocks import SinusoidalEmbedding
from msdflow.model.ncsnpp.blocks import AttnBlockNCSN
from msdflow.model.ncsnpp.blocks import GaussianFourierProjection
from msdflow.model.ncsnpp.blocks import RALAAttentionBlock
from msdflow.model.ncsnpp.blocks import ResBlockBigGAN
from msdflow.model.unet.blocks import Downsample, ResBlock
from msdflow.model.unet.blocks import SinusoidalEmbedding as UNetSinusoidalEmbedding
from msdflow.model.unet.blocks import Upsample


KEY = jax.random.PRNGKey(0)
TIME_EMB_DIM = 16


def _array_leaf_dtypes(pytree):
    """Return dtypes for array leaves in a pytree."""
    return {
        leaf.dtype
        for leaf in jax.tree.leaves(eqx.filter(pytree, eqx.is_array))
    }


def _primitive_io_dtypes(jaxpr, primitive_name):
    """Return input and output dtypes for primitive equations in a jaxpr."""
    records = []
    for eqn in jaxpr.eqns:
        if eqn.primitive.name != primitive_name:
            continue
        in_dtypes = tuple(
            var.aval.dtype for var in eqn.invars if hasattr(var, "aval")
        )
        out_dtypes = tuple(
            var.aval.dtype for var in eqn.outvars if hasattr(var, "aval")
        )
        records.append((in_dtypes, out_dtypes))
    return records


def test_sinusoidal_embedding_unet_reexport_matches_common():
    """UNet block imports must re-export the common sinusoidal embedding."""
    assert UNetSinusoidalEmbedding is SinusoidalEmbedding


def test_sinusoidal_embedding_frequency_dim_projects_to_output_dim():
    """A smaller sinusoidal basis should project to the requested output dim."""
    emb = SinusoidalEmbedding(
        dim=32,
        activation=jax.nn.silu,
        key=KEY,
        frequency_dim=16,
    )
    out = emb(jnp.array(0.5))
    assert out.shape == (32,)


def test_sinusoidal_embedding_odd_frequency_dim_raises():
    """The optional frequency basis must have an even dimension."""
    with pytest.raises(ValueError, match="frequency dimension"):
        SinusoidalEmbedding(
            dim=32,
            activation=jax.nn.silu,
            key=KEY,
            frequency_dim=15,
        )


def test_sinusoidal_embedding_output_shape():
    """Verify output shape matches the requested embedding dimension."""
    dim = 32
    emb = SinusoidalEmbedding(dim=dim, activation=jax.nn.silu, key=KEY)
    t = jnp.array(0.5)
    out = emb(t)
    assert out.shape == (dim,), f"Expected ({dim},), got {out.shape}"


def test_sinusoidal_embedding_different_t_values():
    """Verify distinct timesteps produce distinct embeddings."""
    emb = SinusoidalEmbedding(dim=16, activation=jax.nn.silu, key=KEY)
    out0 = emb(jnp.array(0.0))
    out1 = emb(jnp.array(1.0))
    assert not jnp.allclose(out0, out1), "Embeddings for t=0 and t=1 should differ"


def test_sinusoidal_embedding_is_finite():
    """Verify embedding output contains only finite values."""
    emb = SinusoidalEmbedding(dim=16, activation=jax.nn.silu, key=KEY)
    out = emb(jnp.array(0.3))
    assert jnp.all(jnp.isfinite(out)), "Embedding output contains non-finite values"


def test_downsample_halves_spatial_dims():
    """Verify spatial dimensions are halved after downsampling."""
    ds = Downsample(channels=4, key=KEY)
    x = jnp.ones((4, 16, 16))
    out = ds(x)
    assert out.shape == (4, 8, 8), f"Expected (4, 8, 8), got {out.shape}"


def test_downsample_preserves_channels():
    """Verify channel count is unchanged after downsampling."""
    ds = Downsample(channels=8, key=KEY)
    x = jnp.ones((8, 16, 16))
    out = ds(x)
    assert out.shape[0] == 8


def test_upsample_doubles_spatial_dims():
    """Verify spatial dimensions are doubled after upsampling."""
    us = Upsample(channels=4, key=KEY)
    x = jnp.ones((4, 8, 8))
    out = us(x, target_h=16, target_w=16)
    assert out.shape == (4, 16, 16), f"Expected (4, 16, 16), got {out.shape}"


def test_upsample_preserves_channels():
    """Verify channel count is unchanged after upsampling."""
    us = Upsample(channels=6, key=KEY)
    x = jnp.ones((6, 8, 8))
    out = us(x, target_h=16, target_w=16)
    assert out.shape[0] == 6


def test_resblock_output_shape_same_channels():
    """Verify output shape is preserved when in_channels equals out_channels."""
    block = ResBlock(
        in_channels=4,
        out_channels=4,
        time_emb_dim=TIME_EMB_DIM,
        num_groups=2,
        activation=jax.nn.silu,
        key=KEY,
    )
    x = jnp.ones((4, 8, 8))
    t_emb = jnp.ones(TIME_EMB_DIM)
    out = block(x, t_emb)
    assert out.shape == (4, 8, 8)


def test_resblock_output_shape_different_channels():
    """Verify output channels change when in_channels differs from out_channels."""
    block = ResBlock(
        in_channels=4,
        out_channels=8,
        time_emb_dim=TIME_EMB_DIM,
        num_groups=2,
        activation=jax.nn.silu,
        key=KEY,
    )
    x = jnp.ones((4, 8, 8))
    t_emb = jnp.ones(TIME_EMB_DIM)
    out = block(x, t_emb)
    assert out.shape == (8, 8, 8)


def test_resblock_time_emb_affects_output():
    """Verify different time embeddings produce different outputs."""
    block = ResBlock(
        in_channels=4,
        out_channels=4,
        time_emb_dim=TIME_EMB_DIM,
        num_groups=2,
        activation=jax.nn.silu,
        key=KEY,
    )
    x = jnp.ones((4, 8, 8))
    out0 = block(x, jnp.zeros(TIME_EMB_DIM))
    out1 = block(x, jnp.ones(TIME_EMB_DIM))
    assert not jnp.allclose(out0, out1)


def test_attention_block_preserves_shape():
    """Verify attention block output shape matches input shape."""
    block = AttentionBlock(channels=8, num_heads=2, key=KEY)
    x = jnp.ones((8, 4, 4))
    out = block(x)
    assert out.shape == (8, 4, 4)


def test_attention_block_output_finite():
    """Verify attention block output contains only finite values."""
    block = AttentionBlock(channels=8, num_heads=2, key=KEY)
    k, _ = jax.random.split(KEY)
    x = jax.random.normal(k, (8, 4, 4))
    out = block(x)
    assert jnp.all(jnp.isfinite(out))


def test_attention_block_invalid_num_heads_raises():
    """channels not divisible by num_heads raises ValueError."""
    with pytest.raises(ValueError, match="num_heads"):
        AttentionBlock(channels=8, num_heads=3, key=KEY)


def test_attention_block_default_implementation_is_xla_on_cpu():
    """On a CPU-only test runner, auto-detection resolves to 'xla'."""
    block = AttentionBlock(channels=8, num_heads=2, key=KEY)
    assert block.implementation == "xla"


def test_attention_block_explicit_implementation_xla():
    """Explicit implementation='xla' is stored as-is."""
    block = AttentionBlock(channels=8, num_heads=2, key=KEY, implementation="xla")
    assert block.implementation == "xla"


def test_attention_block_bfloat16_preserves_input_dtype():
    """attention_dtype=bfloat16 still returns output in the input dtype."""
    block = AttentionBlock(
        channels=8, num_heads=2, key=KEY, attention_dtype=jnp.bfloat16
    )
    k, _ = jax.random.split(KEY)
    x = jax.random.normal(k, (8, 4, 4)).astype(jnp.float32)
    out = block(x)
    assert out.shape == x.shape
    assert out.dtype == jnp.float32
    assert jnp.all(jnp.isfinite(out))


def test_attention_block_bfloat16_close_to_fp32():
    """bf16 attention output is close to fp32 attention output."""
    block_fp32 = AttentionBlock(
        channels=8, num_heads=2, key=KEY, attention_dtype=jnp.float32
    )
    block_bf16 = AttentionBlock(
        channels=8, num_heads=2, key=KEY, attention_dtype=jnp.bfloat16
    )
    k, _ = jax.random.split(KEY)
    x = jax.random.normal(k, (8, 4, 4)).astype(jnp.float32)
    out_fp32 = block_fp32(x)
    out_bf16 = block_bf16(x)
    assert jnp.allclose(out_fp32, out_bf16, atol=5e-2)


def test_rala_attention_block_preserves_shape():
    """Verify RALA attention output shape matches input shape."""
    block = RALAAttentionBlock(channels=8, num_heads=2, key=KEY)
    x = jnp.ones((8, 4, 4))
    out = block(x)
    assert out.shape == (8, 4, 4)


def test_rala_attention_block_output_finite():
    """Verify RALA attention output contains only finite values."""
    block = RALAAttentionBlock(channels=8, num_heads=2, key=KEY)
    k, _ = jax.random.split(KEY)
    x = jax.random.normal(k, (8, 4, 4))
    out = block(x)
    assert jnp.all(jnp.isfinite(out))


def test_rala_attention_block_invalid_num_heads_raises():
    """channels not divisible by num_heads raises ValueError."""
    with pytest.raises(ValueError, match="num_heads"):
        RALAAttentionBlock(channels=8, num_heads=3, key=KEY)


def test_rala_attention_block_invalid_rope_head_dim_raises():
    """RALA requires head_dim divisible by 4 for 2D RoPE."""
    with pytest.raises(ValueError, match="divisible by 4"):
        RALAAttentionBlock(channels=12, num_heads=2, key=KEY)


def test_rala_attention_block_bfloat16_preserves_input_dtype():
    """attention_dtype=bfloat16 still returns output in the input dtype."""
    block = RALAAttentionBlock(
        channels=8, num_heads=2, key=KEY, attention_dtype=jnp.bfloat16
    )
    k, _ = jax.random.split(KEY)
    x = jax.random.normal(k, (8, 4, 4)).astype(jnp.float32)
    out = block(x)
    assert out.shape == x.shape
    assert out.dtype == jnp.float32
    assert jnp.all(jnp.isfinite(out))


def test_gaussian_fourier_projection_output_shape():
    """Verify output shape matches embed_dim."""
    embed_dim = 32
    gfp = GaussianFourierProjection(embed_dim=embed_dim, scale=16.0, key=KEY)
    t = jnp.array(0.5)
    out = gfp(t)
    assert out.shape == (embed_dim,), f"Expected ({embed_dim},), got {out.shape}"


def test_gaussian_fourier_projection_different_t_values():
    """Verify distinct timesteps produce distinct embeddings."""
    gfp = GaussianFourierProjection(embed_dim=32, scale=16.0, key=KEY)
    out0 = gfp(jnp.array(0.0))
    out1 = gfp(jnp.array(1.0))
    assert not jnp.allclose(out0, out1), "Embeddings for t=0 and t=1 should differ"


def test_gaussian_fourier_projection_is_finite():
    """Verify embedding output contains only finite values."""
    gfp = GaussianFourierProjection(embed_dim=32, scale=16.0, key=KEY)
    out = gfp(jnp.array(0.3))
    assert jnp.all(jnp.isfinite(out)), "Embedding output contains non-finite values"


def test_gaussian_fourier_projection_W_frozen():
    """Verify W receives zero gradients via stop_gradient."""
    gfp = GaussianFourierProjection(embed_dim=32, scale=16.0, key=KEY)
    import equinox as eqx

    def loss_fn(model, t):
        return jnp.sum(model(t))

    grads = eqx.filter_grad(loss_fn)(gfp, jnp.array(0.5))
    assert jnp.all(grads.W == 0.0), "W should have zero gradients"


def test_resblock_biggan_same_channels():
    """Verify output shape when in_channels == out_channels, no resampling."""
    block = ResBlockBigGAN(
        in_channels=8, out_channels=8, time_emb_dim=TIME_EMB_DIM,
        num_groups=2, activation=jax.nn.swish, dropout=0.0,
        skip_rescale=True, key=KEY,
    )
    x = jnp.ones((8, 16, 16))
    t_emb = jnp.ones(TIME_EMB_DIM)
    out = block(x, t_emb, jax.random.PRNGKey(0))
    assert out.shape == (8, 16, 16)


def test_resblock_biggan_different_channels():
    """Verify output channels change correctly."""
    block = ResBlockBigGAN(
        in_channels=4, out_channels=8, time_emb_dim=TIME_EMB_DIM,
        num_groups=2, activation=jax.nn.swish, dropout=0.0,
        skip_rescale=True, key=KEY,
    )
    x = jnp.ones((4, 16, 16))
    t_emb = jnp.ones(TIME_EMB_DIM)
    out = block(x, t_emb, jax.random.PRNGKey(0))
    assert out.shape == (8, 16, 16)


def test_resblock_biggan_downsample():
    """Verify spatial dims halved when down=True."""
    block = ResBlockBigGAN(
        in_channels=8, out_channels=8, time_emb_dim=TIME_EMB_DIM,
        num_groups=2, activation=jax.nn.swish, dropout=0.0,
        skip_rescale=True, down=True, key=KEY,
    )
    x = jnp.ones((8, 16, 16))
    t_emb = jnp.ones(TIME_EMB_DIM)
    out = block(x, t_emb, jax.random.PRNGKey(0))
    assert out.shape == (8, 8, 8)


def test_resblock_biggan_upsample():
    """Verify spatial dims doubled when up=True."""
    block = ResBlockBigGAN(
        in_channels=8, out_channels=8, time_emb_dim=TIME_EMB_DIM,
        num_groups=2, activation=jax.nn.swish, dropout=0.0,
        skip_rescale=True, up=True, key=KEY,
    )
    x = jnp.ones((8, 8, 8))
    t_emb = jnp.ones(TIME_EMB_DIM)
    out = block(x, t_emb, jax.random.PRNGKey(0))
    assert out.shape == (8, 16, 16)


def test_resblock_biggan_time_conditioning():
    """Verify different time embeddings produce different outputs."""
    block = ResBlockBigGAN(
        in_channels=4, out_channels=4, time_emb_dim=TIME_EMB_DIM,
        num_groups=2, activation=jax.nn.swish, dropout=0.0,
        skip_rescale=True, key=KEY,
    )
    x = jnp.ones((4, 8, 8))
    out0 = block(x, jnp.zeros(TIME_EMB_DIM), jax.random.PRNGKey(0))
    out1 = block(x, jnp.ones(TIME_EMB_DIM), jax.random.PRNGKey(0))
    assert not jnp.allclose(out0, out1)


def test_resblock_biggan_skip_rescale():
    """Verify skip rescaling divides output by sqrt(2)."""
    block_rescale = ResBlockBigGAN(
        in_channels=4, out_channels=4, time_emb_dim=TIME_EMB_DIM,
        num_groups=2, activation=jax.nn.swish, dropout=0.0,
        skip_rescale=True, key=KEY,
    )
    block_no_rescale = ResBlockBigGAN(
        in_channels=4, out_channels=4, time_emb_dim=TIME_EMB_DIM,
        num_groups=2, activation=jax.nn.swish, dropout=0.0,
        skip_rescale=False, key=KEY,
    )
    x = jnp.ones((4, 8, 8))
    t_emb = jnp.ones(TIME_EMB_DIM)
    out_rescale = block_rescale(x, t_emb, jax.random.PRNGKey(0))
    out_no_rescale = block_no_rescale(x, t_emb, jax.random.PRNGKey(0))
    assert jnp.allclose(out_rescale * jnp.sqrt(2.0), out_no_rescale, atol=1e-5)


def test_resblock_biggan_output_finite():
    """Verify output is finite for random input."""
    block = ResBlockBigGAN(
        in_channels=4, out_channels=8, time_emb_dim=TIME_EMB_DIM,
        num_groups=2, activation=jax.nn.swish, dropout=0.0,
        skip_rescale=True, key=KEY,
    )
    k, _ = jax.random.split(KEY)
    x = jax.random.normal(k, (4, 8, 8))
    t_emb = jax.random.normal(k, (TIME_EMB_DIM,))
    out = block(x, t_emb, jax.random.PRNGKey(0))
    assert jnp.all(jnp.isfinite(out))


def test_resblock_biggan_up_and_down_raises():
    """Verify setting both up=True and down=True raises ValueError."""
    with pytest.raises(ValueError, match="Cannot set both"):
        ResBlockBigGAN(
            in_channels=4, out_channels=4, time_emb_dim=TIME_EMB_DIM,
            num_groups=2, activation=jax.nn.swish, dropout=0.0,
            skip_rescale=True, up=True, down=True, key=KEY,
        )


def test_resblock_biggan_compute_dtype_bfloat16_smoke():
    """bf16 compute returns finite fp32 output for fp32 inputs."""
    block = ResBlockBigGAN(
        in_channels=4,
        out_channels=8,
        time_emb_dim=TIME_EMB_DIM,
        num_groups=2,
        activation=jax.nn.swish,
        dropout=0.0,
        skip_rescale=True,
        key=KEY,
        compute_dtype=jnp.bfloat16,
    )
    x_key, emb_key = jax.random.split(KEY)
    x = jax.random.normal(x_key, (4, 8, 8)).astype(jnp.float32)
    t_emb = jax.random.normal(emb_key, (TIME_EMB_DIM,)).astype(jnp.float32)

    out = block(x, t_emb, jax.random.PRNGKey(0))

    assert block.compute_dtype == jnp.bfloat16
    assert out.shape == (8, 8, 8)
    assert out.dtype == jnp.float32
    assert jnp.all(jnp.isfinite(out))


def test_resblock_biggan_compute_dtype_keeps_stored_arrays_float32():
    """bf16 compute does not convert stored trainable arrays."""
    block = ResBlockBigGAN(
        in_channels=4,
        out_channels=8,
        time_emb_dim=TIME_EMB_DIM,
        num_groups=2,
        activation=jax.nn.swish,
        dropout=0.0,
        skip_rescale=True,
        key=KEY,
        compute_dtype=jnp.bfloat16,
    )

    assert _array_leaf_dtypes(block) == {jnp.dtype(jnp.float32)}


def test_resblock_biggan_bfloat16_compute_close_to_float32_compute():
    """bf16 compute stays close to the default fp32 compute path."""
    block_fp32 = ResBlockBigGAN(
        in_channels=4,
        out_channels=8,
        time_emb_dim=TIME_EMB_DIM,
        num_groups=2,
        activation=jax.nn.swish,
        dropout=0.0,
        skip_rescale=True,
        key=KEY,
    )
    block_bf16 = ResBlockBigGAN(
        in_channels=4,
        out_channels=8,
        time_emb_dim=TIME_EMB_DIM,
        num_groups=2,
        activation=jax.nn.swish,
        dropout=0.0,
        skip_rescale=True,
        key=KEY,
        compute_dtype=jnp.bfloat16,
    )
    x_key, emb_key = jax.random.split(KEY)
    x = jax.random.normal(x_key, (4, 8, 8)).astype(jnp.float32)
    t_emb = jax.random.normal(emb_key, (TIME_EMB_DIM,)).astype(jnp.float32)
    dropout_key = jax.random.PRNGKey(0)

    out_fp32 = block_fp32(x, t_emb, dropout_key)
    out_bf16 = block_bf16(x, t_emb, dropout_key)

    assert jnp.allclose(out_fp32, out_bf16, atol=2e-1, rtol=2e-1)


def test_resblock_biggan_compute_dtype_reaches_conv_and_time_projection_primitives():
    """bf16 compute uses bf16 primitive IO for convs and time projection."""
    block_fp32 = ResBlockBigGAN(
        in_channels=4,
        out_channels=8,
        time_emb_dim=TIME_EMB_DIM,
        num_groups=2,
        activation=jax.nn.swish,
        dropout=0.0,
        skip_rescale=True,
        key=KEY,
    )
    block_bf16 = ResBlockBigGAN(
        in_channels=4,
        out_channels=8,
        time_emb_dim=TIME_EMB_DIM,
        num_groups=2,
        activation=jax.nn.swish,
        dropout=0.0,
        skip_rescale=True,
        key=KEY,
        compute_dtype=jnp.bfloat16,
    )
    x_key, emb_key = jax.random.split(KEY)
    x = jax.random.normal(x_key, (4, 8, 8)).astype(jnp.float32)
    t_emb = jax.random.normal(emb_key, (TIME_EMB_DIM,)).astype(jnp.float32)
    dropout_key = jax.random.PRNGKey(0)

    def apply_block(block, x, t_emb, key):
        return block(x, t_emb, key)

    bf16_jaxpr = eqx.filter_make_jaxpr(apply_block)(
        block_bf16, x, t_emb, dropout_key
    )[0].jaxpr
    fp32_jaxpr = eqx.filter_make_jaxpr(apply_block)(
        block_fp32, x, t_emb, dropout_key
    )[0].jaxpr

    bf16_conv_dtypes = _primitive_io_dtypes(bf16_jaxpr, "conv_general_dilated")
    bf16_dot_dtypes = _primitive_io_dtypes(bf16_jaxpr, "dot_general")
    fp32_conv_dtypes = _primitive_io_dtypes(fp32_jaxpr, "conv_general_dilated")
    fp32_dot_dtypes = _primitive_io_dtypes(fp32_jaxpr, "dot_general")

    assert len(bf16_conv_dtypes) == 3
    assert all(
        jnp.dtype(jnp.bfloat16) in in_dtypes
        and jnp.dtype(jnp.bfloat16) in out_dtypes
        for in_dtypes, out_dtypes in bf16_conv_dtypes[:3]
    )
    assert any(
        jnp.dtype(jnp.bfloat16) in in_dtypes
        and jnp.dtype(jnp.bfloat16) in out_dtypes
        for in_dtypes, out_dtypes in bf16_dot_dtypes
    )

    assert not any(
        jnp.dtype(jnp.bfloat16) in in_dtypes
        or jnp.dtype(jnp.bfloat16) in out_dtypes
        for in_dtypes, out_dtypes in fp32_conv_dtypes + fp32_dot_dtypes
    )


def test_attn_block_ncsn_preserves_shape():
    """Verify attention block output shape matches input shape."""
    block = AttnBlockNCSN(channels=8, num_heads=1, num_groups=2, skip_rescale=True, key=KEY)
    x = jnp.ones((8, 4, 4))
    out = block(x)
    assert out.shape == (8, 4, 4)


def test_attn_block_ncsn_output_finite():
    """Verify attention block output contains only finite values."""
    block = AttnBlockNCSN(channels=8, num_heads=1, num_groups=2, skip_rescale=True, key=KEY)
    k, _ = jax.random.split(KEY)
    x = jax.random.normal(k, (8, 4, 4))
    out = block(x)
    assert jnp.all(jnp.isfinite(out))


def test_attn_block_ncsn_skip_rescale():
    """Verify skip rescaling is applied."""
    block_rescale = AttnBlockNCSN(
        channels=8, num_heads=1, num_groups=2, skip_rescale=True, key=KEY
    )
    block_no_rescale = AttnBlockNCSN(
        channels=8, num_heads=1, num_groups=2, skip_rescale=False, key=KEY
    )
    k, _ = jax.random.split(KEY)
    x = jax.random.normal(k, (8, 4, 4))
    out_rescale = block_rescale(x)
    out_no_rescale = block_no_rescale(x)
    assert not jnp.allclose(out_rescale, out_no_rescale)


def test_attn_block_ncsn_bfloat16_preserves_input_dtype():
    """AttnBlockNCSN with attention_dtype=bfloat16 returns output in input dtype."""
    block = AttnBlockNCSN(
        channels=8,
        num_heads=2,
        num_groups=2,
        skip_rescale=True,
        key=KEY,
        attention_dtype=jnp.bfloat16,
    )
    k, _ = jax.random.split(KEY)
    x = jax.random.normal(k, (8, 4, 4)).astype(jnp.float32)
    out = block(x)
    assert out.shape == x.shape
    assert out.dtype == jnp.float32
    assert jnp.all(jnp.isfinite(out))


def test_attn_block_ncsn_implementation_passthrough():
    """AttnBlockNCSN forwards implementation kwarg to inner AttentionBlock."""
    block = AttnBlockNCSN(
        channels=8,
        num_heads=2,
        num_groups=2,
        skip_rescale=True,
        key=KEY,
        implementation="xla",
    )
    assert block.attn.implementation == "xla"


def test_attn_block_ncsn_attention_type_rala():
    """AttnBlockNCSN can wrap RALAAttentionBlock."""
    block = AttnBlockNCSN(
        channels=8,
        num_heads=2,
        num_groups=2,
        skip_rescale=True,
        key=KEY,
        attention_type="rala",
    )
    assert isinstance(block.attn, RALAAttentionBlock)
    x = jnp.ones((8, 4, 4))
    out = block(x)
    assert out.shape == x.shape


def test_attn_block_ncsn_invalid_attention_type_raises():
    """Unknown attention_type raises ValueError."""
    with pytest.raises(ValueError, match="attention_type"):
        AttnBlockNCSN(
            channels=8,
            num_heads=2,
            num_groups=2,
            skip_rescale=True,
            key=KEY,
            attention_type="unknown",
        )
