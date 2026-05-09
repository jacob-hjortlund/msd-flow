"""Import compatibility tests for model package refactors."""


def test_unet_package_exports_match_top_level_export():
    """UNet is available from package and top-level public imports."""
    from msdflow.model import UNet as TopLevelUNet
    from msdflow.model.unet import UNet as PackageUNet
    from msdflow.model.unet.model import UNet as ModuleUNet

    assert TopLevelUNet is PackageUNet
    assert PackageUNet is ModuleUNet


def test_unet_block_exports_are_available_from_new_package():
    """UNet blocks are importable from the new package path."""
    from msdflow.model.common_blocks import AttentionBlock
    from msdflow.model.unet.blocks import AttentionBlock as UNetAttentionBlock
    from msdflow.model.unet.blocks import Downsample, ResBlock, SinusoidalEmbedding
    from msdflow.model.unet.blocks import Upsample

    assert UNetAttentionBlock is AttentionBlock
    assert Downsample.__name__ == "Downsample"
    assert ResBlock.__name__ == "ResBlock"
    assert SinusoidalEmbedding.__name__ == "SinusoidalEmbedding"
    assert Upsample.__name__ == "Upsample"


def test_ncsnpp_package_exports_match_top_level_export():
    """NCSNpp is available from package and top-level public imports."""
    from msdflow.model import NCSNpp as TopLevelNCSNpp
    from msdflow.model.ncsnpp import NCSNpp as PackageNCSNpp
    from msdflow.model.ncsnpp.model import NCSNpp as ModuleNCSNpp

    assert TopLevelNCSNpp is PackageNCSNpp
    assert PackageNCSNpp is ModuleNCSNpp


def test_ncsnpp_block_exports_are_available_from_new_package():
    """NCSN++ blocks are importable from the new package path."""
    from msdflow.model.common_blocks import AttentionBlock
    from msdflow.model.ncsnpp import RALAAttentionBlock as PackageRALAAttentionBlock
    from msdflow.model.ncsnpp import ResBlockBigGAN as PackageResBlockBigGAN
    from msdflow.model.ncsnpp.blocks import AttnBlockNCSN, Conv2d, CoordConv
    from msdflow.model.ncsnpp.blocks import GaussianFourierProjection
    from msdflow.model.ncsnpp.blocks import RALAAttentionBlock, ResBlockBigGAN
    from msdflow.model.ncsnpp.blocks import AttentionBlock as NCSNAttentionBlock

    assert NCSNAttentionBlock is AttentionBlock
    assert PackageRALAAttentionBlock is RALAAttentionBlock
    assert PackageResBlockBigGAN is ResBlockBigGAN
    assert AttnBlockNCSN.__name__ == "AttnBlockNCSN"
    assert Conv2d.__name__ == "Conv2d"
    assert CoordConv.__name__ == "CoordConv"
    assert GaussianFourierProjection.__name__ == "GaussianFourierProjection"
    assert RALAAttentionBlock.__name__ == "RALAAttentionBlock"
    assert ResBlockBigGAN.__name__ == "ResBlockBigGAN"


def test_convnext_package_exports_builder_and_encoder():
    """ConvNeXt public symbols are available from the package path."""
    from msdflow.model.convnext import ConvNeXtEncoder, build_zoobot_nano
    from msdflow.model.convnext.blocks import ConvNeXtBlock, ConvNeXtHead
    from msdflow.model.convnext.model import ConvNeXtEncoder as ModuleEncoder

    assert ConvNeXtEncoder is ModuleEncoder
    assert build_zoobot_nano.__name__ == "build_zoobot_nano"
    assert ConvNeXtBlock.__name__ == "ConvNeXtBlock"
    assert ConvNeXtHead.__name__ == "ConvNeXtHead"


def test_inceptionv3_package_exports_builder_and_model():
    """InceptionV3 public symbols are available from the package path."""
    from msdflow.model.inceptionv3 import (
        Array,
        Dtype,
        InceptionV3,
        PRNGKey,
        Shape,
        build_headless_inceptionv3,
        reshape,
    )
    from msdflow.model.inceptionv3.blocks import Array as BlockArray
    from msdflow.model.inceptionv3.blocks import BasicConv2d, InceptionA
    from msdflow.model.inceptionv3.blocks import Dtype as BlockDtype
    from msdflow.model.inceptionv3.blocks import PRNGKey as BlockPRNGKey
    from msdflow.model.inceptionv3.blocks import Shape as BlockShape
    from msdflow.model.inceptionv3.model import InceptionV3 as ModuleInceptionV3
    from msdflow.model.inceptionv3.model import reshape as ModuleReshape

    assert Array is BlockArray
    assert Dtype is BlockDtype
    assert InceptionV3 is ModuleInceptionV3
    assert PRNGKey is BlockPRNGKey
    assert Shape is BlockShape
    assert reshape is ModuleReshape
    assert build_headless_inceptionv3.__name__ == "build_headless_inceptionv3"
    assert BasicConv2d.__name__ == "BasicConv2d"
    assert InceptionA.__name__ == "InceptionA"


def test_legacy_hydra_model_target_paths_resolve():
    """Hydra can resolve legacy model target strings after package moves."""
    from hydra.utils import get_class, get_method

    from msdflow.model.convnext import build_zoobot_nano
    from msdflow.model.inceptionv3 import InceptionV3
    from msdflow.model.ncsnpp import NCSNpp
    from msdflow.model.unet import UNet

    assert get_class("msdflow.model.unet.UNet") is UNet
    assert get_class("msdflow.model.ncsnpp.NCSNpp") is NCSNpp
    assert get_class("msdflow.model.inceptionv3.InceptionV3") is InceptionV3
    assert get_method("msdflow.model.convnext.build_zoobot_nano") is build_zoobot_nano


def test_root_blocks_module_reexports_moved_blocks():
    """The old msdflow.model.blocks import surface remains compatible."""
    from msdflow.model import blocks as root_blocks
    from msdflow.model import common_blocks
    from msdflow.model.convnext import blocks as convnext_blocks
    from msdflow.model.ncsnpp import blocks as ncsnpp_blocks
    from msdflow.model.unet import blocks as unet_blocks

    expected_exports = {
        "AttentionBlock": common_blocks.AttentionBlock,
        "AttnBlockNCSN": ncsnpp_blocks.AttnBlockNCSN,
        "Conv2d": ncsnpp_blocks.Conv2d,
        "ConvNeXtBlock": convnext_blocks.ConvNeXtBlock,
        "ConvNeXtDownsample": convnext_blocks.ConvNeXtDownsample,
        "ConvNeXtHead": convnext_blocks.ConvNeXtHead,
        "ConvNeXtStage": convnext_blocks.ConvNeXtStage,
        "ConvNeXtStem": convnext_blocks.ConvNeXtStem,
        "CoordConv": ncsnpp_blocks.CoordConv,
        "Downsample": unet_blocks.Downsample,
        "DropPath": convnext_blocks.DropPath,
        "GaussianFourierProjection": ncsnpp_blocks.GaussianFourierProjection,
        "Identity": convnext_blocks.Identity,
        "LayerNorm2d": convnext_blocks.LayerNorm2d,
        "RALAAttentionBlock": ncsnpp_blocks.RALAAttentionBlock,
        "ResBlock": unet_blocks.ResBlock,
        "ResBlockBigGAN": ncsnpp_blocks.ResBlockBigGAN,
        "SinusoidalEmbedding": unet_blocks.SinusoidalEmbedding,
        "Upsample": unet_blocks.Upsample,
    }

    assert sorted(root_blocks.__all__) == sorted(expected_exports)
    for name, expected_obj in expected_exports.items():
        assert getattr(root_blocks, name) is expected_obj
