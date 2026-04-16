"""ConvNeXt encoder for encoding images.

Implements the architecture from [INSERT ConvNeXt REFERENCE]. Made to match
the specific implementation used in the HuggingFace timm package.
"""

import jax
import timm
import torch

import numpy as np
import equinox as eqx
import jax.numpy as jnp

from typing import Callable, Optional, Tuple, Sequence
from msdflow.model.blocks import ConvNeXtStem, ConvNeXtStage, ConvNeXtHead, Identity


def copy_timm_convnext_encoder_to_eqx(pt_model, eq_model, *, verbose: bool = True):
    """Copy weights from a timm ConvNeXt-like PyTorch model into an Equinox ConvNeXtEncoder.

    Assumptions
    -----------
    - `pt_model` is a timm ConvNeXt model with the structure shown in your printout.
    - `eq_model` is an instance of the Equinox `ConvNeXtEncoder` defined earlier.
    - The Equinox model returns pooled features (i.e. corresponds to timm model with num_classes=0).

    Returns
    -------
    A new Equinox model with copied weights.
    """

    def _torch_to_jax(pt_tensor, like=None):
        arr = np.asarray(pt_tensor.detach().cpu())
        if like is not None:
            return jnp.asarray(arr, dtype=like.dtype)
        return jnp.asarray(arr)

    def _shape_check(name, src, dst):
        if tuple(src.shape) != tuple(dst.shape):
            raise ValueError(
                f"Shape mismatch for {name}: "
                f"PyTorch {tuple(src.shape)} vs Equinox {tuple(dst.shape)}"
            )

    def _match_bias_shape(src_bias, dst_bias, name):
        if tuple(src_bias.shape) == tuple(dst_bias.shape):
            return src_bias

        if src_bias.size == dst_bias.size:
            return jnp.reshape(src_bias, dst_bias.shape)

        raise ValueError(
            f"Shape mismatch for {name}.bias: "
            f"PyTorch {tuple(src_bias.shape)} vs Equinox {tuple(dst_bias.shape)}"
        )

    def _copy_conv2d(eq_conv, pt_conv, name):
        # weight
        new_weight = _torch_to_jax(pt_conv.weight, like=eq_conv.weight)
        _shape_check(f"{name}.weight", new_weight, eq_conv.weight)
        eq_conv = eqx.tree_at(lambda m: m.weight, eq_conv, new_weight)

        # bias
        pt_bias = pt_conv.bias
        eq_bias = eq_conv.bias
        if (pt_bias is None) != (eq_bias is None):
            raise ValueError(
                f"Bias mismatch for {name}: "
                f"PyTorch bias is {'present' if pt_bias is not None else 'absent'}, "
                f"Equinox bias is {'present' if eq_bias is not None else 'absent'}."
            )
        if pt_bias is not None:
            new_bias = _torch_to_jax(pt_bias, like=eq_bias)
            new_bias = _match_bias_shape(new_bias, eq_bias, name)
            eq_conv = eqx.tree_at(
                lambda m: m.bias,
                eq_conv,
                new_bias,
                is_leaf=lambda x: x is None,
            )

        return eq_conv

    def _copy_layernorm2d(eq_ln, pt_ln, name):
        new_weight = _torch_to_jax(pt_ln.weight, like=eq_ln.weight)
        new_bias = _torch_to_jax(pt_ln.bias, like=eq_ln.bias)

        _shape_check(f"{name}.weight", new_weight, eq_ln.weight)
        _shape_check(f"{name}.bias", new_bias, eq_ln.bias)

        eq_ln = eqx.tree_at(lambda m: m.weight, eq_ln, new_weight)
        eq_ln = eqx.tree_at(lambda m: m.bias, eq_ln, new_bias)
        return eq_ln

    def _copy_block(eq_block, pt_block, name):
        eq_block = eqx.tree_at(
            lambda b: b.conv_dw,
            eq_block,
            _copy_conv2d(eq_block.conv_dw, pt_block.conv_dw, f"{name}.conv_dw"),
        )
        eq_block = eqx.tree_at(
            lambda b: b.norm,
            eq_block,
            _copy_layernorm2d(eq_block.norm, pt_block.norm, f"{name}.norm"),
        )
        eq_block = eqx.tree_at(
            lambda b: b.fc1,
            eq_block,
            _copy_conv2d(eq_block.fc1, pt_block.mlp.fc1, f"{name}.mlp.fc1"),
        )
        eq_block = eqx.tree_at(
            lambda b: b.fc2,
            eq_block,
            _copy_conv2d(eq_block.fc2, pt_block.mlp.fc2, f"{name}.mlp.fc2"),
        )

        # timm ConvNeXt usually has a learnable gamma/layer-scale parameter
        pt_gamma = getattr(pt_block, "gamma", None)
        eq_gamma = eq_block.gamma
        if (pt_gamma is None) != (eq_gamma is None):
            raise ValueError(
                f"Gamma mismatch for {name}: "
                f"PyTorch gamma is {'present' if pt_gamma is not None else 'absent'}, "
                f"Equinox gamma is {'present' if eq_gamma is not None else 'absent'}."
            )
        if pt_gamma is not None:
            new_gamma = _torch_to_jax(pt_gamma, like=eq_gamma)
            _shape_check(f"{name}.gamma", new_gamma, eq_gamma)
            eq_block = eqx.tree_at(
                lambda b: b.gamma,
                eq_block,
                new_gamma,
                is_leaf=lambda x: x is None,
            )

        return eq_block

    def _copy_stem(eq_model, pt_model):
        eq_stem = eq_model.stem
        pt_stem = pt_model.stem

        eq_stem = eqx.tree_at(
            lambda s: s.conv,
            eq_stem,
            _copy_conv2d(eq_stem.conv, pt_stem[0], "stem.0"),
        )
        eq_stem = eqx.tree_at(
            lambda s: s.norm,
            eq_stem,
            _copy_layernorm2d(eq_stem.norm, pt_stem[1], "stem.1"),
        )
        return eqx.tree_at(lambda m: m.stem, eq_model, eq_stem)

    def _copy_stage(eq_stage, pt_stage, stage_idx):
        # downsample
        eq_down = eq_stage.downsample
        pt_down = pt_stage.downsample

        eq_is_identity = eq_down.__class__.__name__ == "Identity"
        pt_is_identity = pt_down.__class__.__name__ == "Identity"

        if eq_is_identity != pt_is_identity:
            raise ValueError(
                f"Downsample mismatch in stage {stage_idx}: "
                f"PyTorch is {'Identity' if pt_is_identity else 'non-Identity'}, "
                f"Equinox is {'Identity' if eq_is_identity else 'non-Identity'}."
            )

        if not eq_is_identity:
            eq_down = eqx.tree_at(
                lambda d: d.norm,
                eq_down,
                _copy_layernorm2d(
                    eq_down.norm, pt_down[0], f"stages.{stage_idx}.downsample.0"
                ),
            )
            eq_down = eqx.tree_at(
                lambda d: d.conv,
                eq_down,
                _copy_conv2d(
                    eq_down.conv, pt_down[1], f"stages.{stage_idx}.downsample.1"
                ),
            )
            eq_stage = eqx.tree_at(lambda s: s.downsample, eq_stage, eq_down)

        # blocks
        if len(eq_stage.blocks) != len(pt_stage.blocks):
            raise ValueError(
                f"Block count mismatch in stage {stage_idx}: "
                f"PyTorch has {len(pt_stage.blocks)}, Equinox has {len(eq_stage.blocks)}."
            )

        new_blocks = []
        for block_idx, (eq_block, pt_block) in enumerate(
            zip(eq_stage.blocks, pt_stage.blocks)
        ):
            new_blocks.append(
                _copy_block(
                    eq_block, pt_block, f"stages.{stage_idx}.blocks.{block_idx}"
                )
            )

        eq_stage = eqx.tree_at(lambda s: s.blocks, eq_stage, tuple(new_blocks))
        return eq_stage

    def _copy_head(eq_model, pt_model):
        # Your printed head is:
        # global_pool -> norm -> flatten -> pre_logits -> drop -> fc(identity)
        # so only head.norm has parameters to port.
        eq_head = eq_model.head
        pt_head = pt_model.head

        eq_head = eqx.tree_at(
            lambda h: h.norm,
            eq_head,
            _copy_layernorm2d(eq_head.norm, pt_head.norm, "head.norm"),
        )
        return eqx.tree_at(lambda m: m.head, eq_model, eq_head)

    # ---------- top-level copy ----------
    eq_model = _copy_stem(eq_model, pt_model)

    if len(eq_model.stages) != len(pt_model.stages):
        raise ValueError(
            f"Stage count mismatch: PyTorch has {len(pt_model.stages)}, "
            f"Equinox has {len(eq_model.stages)}."
        )

    new_stages = []
    for stage_idx, (eq_stage, pt_stage) in enumerate(
        zip(eq_model.stages, pt_model.stages)
    ):
        new_stages.append(_copy_stage(eq_stage, pt_stage, stage_idx))
    eq_model = eqx.tree_at(lambda m: m.stages, eq_model, tuple(new_stages))

    eq_model = _copy_head(eq_model, pt_model)

    if verbose:
        print("Successfully copied timm ConvNeXt encoder weights to Equinox model.")

    return eq_model


def build_zoobot_nano():

    model_name = "hf_hub:mwalmsley/zoobot-encoder-greyscale-convnext_nano"
    torch_model = timm.create_model(
        model_name, pretrained=True, num_classes=0, in_chans=1
    ).eval()

    key = jax.random.PRNGKey(0)
    eq_model = ConvNeXtEncoder(
        in_chans=1,
        depths=(2, 2, 8, 2),
        dims=(80, 160, 320, 640),
        patch_size=4,
        kernel_sizes=(7, 7, 7, 7),
        mlp_ratio=4,
        ls_init_value=1e-6,
        drop_path_rate=0.0,
        inference=True,
        key=key,
    )

    model = copy_timm_convnext_encoder_to_eqx(pt_model=torch_model, eq_model=eq_model)

    def apply_model(x):
        x = (x + 1.0) / 2.0
        x = jax.image.resize(
            x,
            shape=(1, 224, 224),
            method="bilinear",
            antialias=True,
        )
        out = model(x)
        return out

    return apply_model


class ConvNeXtEncoder(eqx.Module):
    """Full ConvNeXt encoder ending in the pooled feature vector."""

    stem: ConvNeXtStem
    stages: Tuple[ConvNeXtStage, ...]
    norm_pre: eqx.Module
    head: ConvNeXtHead

    def __init__(
        self,
        *,
        in_chans: int = 1,
        depths: Tuple[int, int, int, int] = (2, 2, 8, 2),
        dims: Tuple[int, int, int, int] = (80, 160, 320, 640),
        patch_size: int = 4,
        kernel_sizes: Tuple[int, int, int, int] = (7, 7, 7, 7),
        mlp_ratio: int = 4,
        ls_init_value: Optional[float] = 1e-6,
        drop_path_rate: float = 0.0,
        inference: bool = False,
        dtype=jnp.float32,
        key: jax.Array,
    ):

        n_depths = len(depths)
        n_dims = len(dims)
        n_kernels = len(kernel_sizes)
        if (n_depths != n_dims) or (n_depths != n_kernels) or (n_dims != n_kernels):
            raise ValueError(
                "depths, dims, and kernel_sizes must all have same length."
            )

        total_blocks = sum(depths)
        dpr = jnp.linspace(0.0, drop_path_rate, total_blocks).tolist()

        keys = jax.random.split(key, n_depths + 1)
        self.stem = ConvNeXtStem(
            in_chans=in_chans,
            out_chs=dims[0],
            patch_size=patch_size,
            dtype=dtype,
            key=keys[0],
        )

        stages = []
        block_idx = 0
        in_dim = dims[0]

        for i in range(n_depths):
            out_dim = dims[i]
            depth = depths[i]
            stride = 1 if i == 0 else 2
            stage = ConvNeXtStage(
                in_chs=in_dim,
                out_chs=out_dim,
                depth=depth,
                stride=stride,
                kernel_size=kernel_sizes[i],
                mlp_ratio=mlp_ratio,
                ls_init_value=ls_init_value,
                drop_path_rates=dpr[block_idx : block_idx + depth],
                inference=inference,
                dtype=dtype,
                key=keys[i + 1],
            )
            stages.append(stage)
            in_dim = out_dim
            block_idx += depth

        self.stages = tuple(stages)
        self.norm_pre = Identity()
        self.head = ConvNeXtHead(dims[-1], dtype=dtype)

    def __call__(
        self,
        x: jax.Array,
        *,
        key: Optional[jax.Array] = None,
    ):
        x = self.stem(x)

        feats = []
        if key is None:
            stage_keys = [None] * len(self.stages)
        else:
            stage_keys = jax.random.split(key, len(self.stages))

        for stage, k in zip(self.stages, stage_keys):
            x = stage(x, key=k)
            feats.append(x)

        x = self.norm_pre(x)
        x = self.head(x)

        return x
