"""InceptionV3 model definition and feature extraction helpers."""

import pickle
from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp

from msdflow.model.inceptionv3.blocks import (
    BasicConv2d,
    Dense,
    InceptionA,
    InceptionAux,
    InceptionB,
    InceptionC,
    InceptionD,
    InceptionE,
    avg_pool,
)
from msdflow.model.inceptionv3.weights import download, get


def reshape(x):
    """Convert a channel-first grayscale image to resized RGB.

    Args:
        x: Input image with shape ``(1, height, width)``.

    Returns:
        A resized RGB image with shape ``(299, 299, 3)``.
    """
    x = jnp.moveaxis(x, 0, -1)
    x = jnp.broadcast_to(x, (*x.shape[:2], 3))
    x = jax.image.resize(
        x,
        shape=(299, 299, 3),
        method="bilinear",
        antialias=True,
    )
    return x


def build_headless_inceptionv3():
    """Build a pretrained headless InceptionV3 feature extractor.

    Returns:
        Callable that maps one channel-first grayscale image to InceptionV3
        features.
    """
    model = InceptionV3(pretrained=True)
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, jnp.ones((1, 299, 299, 3)))
    model_call_fn = partial(model.apply, params, train=False)

    def apply_fn(x):
        """Apply the headless InceptionV3 feature extractor.

        Args:
            x: Channel-first grayscale input image.

        Returns:
            InceptionV3 feature vector.
        """

        x = reshape(x)
        out = model_call_fn(x)
        return out

    return apply_fn


class InceptionV3(nn.Module):
    """
    InceptionV3 network.
    Reference: https://arxiv.org/abs/1512.00567
    Ported mostly from: https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py

    Attributes:
        include_head (bool): If True, include classifier head.
        num_classes (int): Number of classes.
        pretrained (bool): If True, use pretrained weights.
        transform_input (bool): If True, preprocesses the input according to the method with which it
                                was trained on ImageNet.
        aux_logits (bool): If True, add an auxiliary branch that can improve training.
        dtype (str): Data type.
    """

    include_head: bool = False
    num_classes: int = 1000
    pretrained: bool = False
    transform_input: bool = False
    aux_logits: bool = False
    ckpt_path: str = (
        "https://www.dropbox.com/s/xt6zvlvt22dcwck/inception_v3_weights_fid.pickle?dl=1"
    )
    dtype: str = "float32"

    def setup(self):
        """Load pretrained parameters when requested."""
        if self.pretrained:
            ckpt_file = download(self.ckpt_path)
            self.params_dict = pickle.load(open(ckpt_file, "rb"))
            self.num_classes_ = 1000
        else:
            self.params_dict = None
            self.num_classes_ = self.num_classes

    @nn.compact
    def __call__(self, x, train=True, rng=jax.random.PRNGKey(0)):
        """Run the InceptionV3 network.

        Args:
            x (tensor): Input image, shape [B, H, W, C].
            train (bool): If True, training mode.
            rng (jax.random.PRNGKey): Random seed.

        Returns:
            Model output features or logits, with auxiliary logits when enabled.
        """
        single_image = x.ndim == 3
        if single_image:
            x = x[None, ...]

        x = self._transform_input(x)
        x = BasicConv2d(
            out_channels=32,
            kernel_size=(3, 3),
            strides=(2, 2),
            params_dict=get(self.params_dict, "Conv2d_1a_3x3"),
            dtype=self.dtype,
        )(x, train)
        x = BasicConv2d(
            out_channels=32,
            kernel_size=(3, 3),
            params_dict=get(self.params_dict, "Conv2d_2a_3x3"),
            dtype=self.dtype,
        )(x, train)
        x = BasicConv2d(
            out_channels=64,
            kernel_size=(3, 3),
            padding=((1, 1), (1, 1)),
            params_dict=get(self.params_dict, "Conv2d_2b_3x3"),
            dtype=self.dtype,
        )(x, train)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))
        x = BasicConv2d(
            out_channels=80,
            kernel_size=(1, 1),
            params_dict=get(self.params_dict, "Conv2d_3b_1x1"),
            dtype=self.dtype,
        )(x, train)
        x = BasicConv2d(
            out_channels=192,
            kernel_size=(3, 3),
            params_dict=get(self.params_dict, "Conv2d_4a_3x3"),
            dtype=self.dtype,
        )(x, train)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))
        x = InceptionA(
            pool_features=32,
            params_dict=get(self.params_dict, "Mixed_5b"),
            dtype=self.dtype,
        )(x, train)
        x = InceptionA(
            pool_features=64,
            params_dict=get(self.params_dict, "Mixed_5c"),
            dtype=self.dtype,
        )(x, train)
        x = InceptionA(
            pool_features=64,
            params_dict=get(self.params_dict, "Mixed_5d"),
            dtype=self.dtype,
        )(x, train)
        x = InceptionB(params_dict=get(self.params_dict, "Mixed_6a"), dtype=self.dtype)(
            x, train
        )
        x = InceptionC(
            channels_7x7=128,
            params_dict=get(self.params_dict, "Mixed_6b"),
            dtype=self.dtype,
        )(x, train)
        x = InceptionC(
            channels_7x7=160,
            params_dict=get(self.params_dict, "Mixed_6c"),
            dtype=self.dtype,
        )(x, train)
        x = InceptionC(
            channels_7x7=160,
            params_dict=get(self.params_dict, "Mixed_6d"),
            dtype=self.dtype,
        )(x, train)
        x = InceptionC(
            channels_7x7=192,
            params_dict=get(self.params_dict, "Mixed_6e"),
            dtype=self.dtype,
        )(x, train)
        aux = None
        if self.aux_logits and train:
            aux = InceptionAux(
                num_classes=self.num_classes_,
                params_dict=get(self.params_dict, "AuxLogits"),
                dtype=self.dtype,
            )(x, train)
        x = InceptionD(params_dict=get(self.params_dict, "Mixed_7a"), dtype=self.dtype)(
            x, train
        )
        x = InceptionE(
            avg_pool, params_dict=get(self.params_dict, "Mixed_7b"), dtype=self.dtype
        )(x, train)
        # Following the implementation by @mseitzer, we use max pooling instead
        # of average pooling here.
        # See: https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/inception.py#L320
        x = InceptionE(
            nn.max_pool, params_dict=get(self.params_dict, "Mixed_7c"), dtype=self.dtype
        )(x, train)
        x = jnp.mean(x, axis=(1, 2), keepdims=False)
        if not self.include_head:
            return x[0] if single_image else x
        x = nn.Dropout(rate=0.5)(x, deterministic=not train, rng=rng)
        x = Dense(
            features=self.num_classes_,
            params_dict=get(self.params_dict, "fc"),
            dtype=self.dtype,
        )(x)
        if self.aux_logits:
            if single_image:
                return x[0], aux[0]
            return x, aux

        return x[0] if single_image else x

    def _transform_input(self, x):
        """Apply ImageNet channel normalization when requested.

        Args:
            x: Input image batch.

        Returns:
            Transformed image batch.
        """
        if self.transform_input:
            x_ch0 = (
                jnp.expand_dims(x[..., 0], axis=-1) * (0.229 / 0.5)
                + (0.485 - 0.5) / 0.5
            )
            x_ch1 = (
                jnp.expand_dims(x[..., 1], axis=-1) * (0.224 / 0.5)
                + (0.456 - 0.5) / 0.5
            )
            x_ch2 = (
                jnp.expand_dims(x[..., 2], axis=-1) * (0.225 / 0.5)
                + (0.406 - 0.5) / 0.5
            )
            x = jnp.concatenate((x_ch0, x_ch1, x_ch2), axis=-1)
        return x
