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
