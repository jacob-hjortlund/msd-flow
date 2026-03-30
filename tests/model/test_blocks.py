"""Tests for msdflow.model.blocks."""

import jax
import pytest

import jax.numpy as jnp

from msdflow.model.blocks import ResBlock
from msdflow.model.blocks import AttentionBlock
from msdflow.model.blocks import SinusoidalEmbedding
from msdflow.model.blocks import Downsample, Upsample


KEY = jax.random.PRNGKey(0)
TIME_EMB_DIM = 16


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


from msdflow.model.blocks import GaussianFourierProjection


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


from msdflow.model.blocks import ResBlockBigGAN


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


from msdflow.model.blocks import AttnBlockNCSN


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
