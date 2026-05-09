"""Tests for msdflow.model.convnext."""

import jax
import jax.numpy as jnp

from msdflow.model.convnext import ConvNeXtEncoder


def test_convnext_encoder_small_forward_no_pretrained_download():
    """A tiny ConvNeXtEncoder runs without invoking the pretrained builder."""
    model = ConvNeXtEncoder(
        in_chans=1,
        depths=(1,),
        dims=(4,),
        patch_size=2,
        kernel_sizes=(3,),
        key=jax.random.PRNGKey(0),
    )
    x = jnp.ones((1, 8, 8))

    out = model(x)

    assert out.shape == (4,)
    assert jnp.all(jnp.isfinite(out))
