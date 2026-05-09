"""Compatibility re-exports for model building blocks.

The block implementations live in model-family packages:
`msdflow.model.unet.blocks`, `msdflow.model.ncsnpp.blocks`,
`msdflow.model.convnext.blocks`, and `msdflow.model.common_blocks`.
"""

from msdflow.model.common_blocks import AttentionBlock
from msdflow.model.common_blocks import SinusoidalEmbedding
from msdflow.model.convnext.blocks import (
    ConvNeXtBlock,
    ConvNeXtDownsample,
    ConvNeXtHead,
    ConvNeXtStage,
    ConvNeXtStem,
    DropPath,
    Identity,
    LayerNorm2d,
)
from msdflow.model.ncsnpp.blocks import (
    AttnBlockNCSN,
    Conv2d,
    CoordConv,
    GaussianFourierProjection,
    RALAAttentionBlock,
    ResBlockBigGAN,
)
from msdflow.model.unet.blocks import (
    Downsample,
    ResBlock,
    Upsample,
)

__all__ = [
    "AttentionBlock",
    "AttnBlockNCSN",
    "Conv2d",
    "ConvNeXtBlock",
    "ConvNeXtDownsample",
    "ConvNeXtHead",
    "ConvNeXtStage",
    "ConvNeXtStem",
    "CoordConv",
    "Downsample",
    "DropPath",
    "GaussianFourierProjection",
    "Identity",
    "LayerNorm2d",
    "RALAAttentionBlock",
    "ResBlock",
    "ResBlockBigGAN",
    "SinusoidalEmbedding",
    "Upsample",
]
