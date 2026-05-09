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
