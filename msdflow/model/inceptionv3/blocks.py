"""InceptionV3 block definitions."""

from functools import partial
from typing import Any, Callable, Iterable, Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.module import merge_param
from jax import lax
from jax.nn import initializers

from msdflow.model.inceptionv3.weights import get

PRNGKey = Any
Array = Any
Shape = Tuple[int]
Dtype = Any

__all__ = [
    "Array",
    "BasicConv2d",
    "BatchNorm",
    "Dense",
    "Dtype",
    "InceptionA",
    "InceptionAux",
    "InceptionB",
    "InceptionC",
    "InceptionD",
    "InceptionE",
    "PRNGKey",
    "Shape",
    "avg_pool",
    "pool",
]


class Dense(nn.Module):
    """Dense layer with optional pretrained parameter initialization.

    Attributes:
        features: Number of output features.
        kernel_init: Initializer for the kernel when no parameters are loaded.
        bias_init: Initializer for the bias when no parameters are loaded.
        params_dict: Optional pretrained parameter dictionary.
        dtype: Computation dtype.
    """

    features: int
    kernel_init: partial = nn.initializers.lecun_normal()
    bias_init: partial = nn.initializers.zeros
    params_dict: dict = None
    dtype: str = "float32"

    @nn.compact
    def __call__(self, x):
        """Apply the dense layer.

        Args:
            x: Input activations.

        Returns:
            Output activations.
        """
        x = nn.Dense(
            features=self.features,
            kernel_init=(
                self.kernel_init
                if self.params_dict is None
                else lambda *_: jnp.array(self.params_dict["kernel"])
            ),
            bias_init=(
                self.bias_init
                if self.params_dict is None
                else lambda *_: jnp.array(self.params_dict["bias"])
            ),
        )(x)
        return x


class BasicConv2d(nn.Module):
    """Convolution, batch normalization, and ReLU block.

    Attributes:
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        strides: Convolution strides.
        padding: Convolution padding.
        use_bias: Whether the convolution includes a bias.
        kernel_init: Initializer for the kernel when no parameters are loaded.
        bias_init: Initializer for the bias when no parameters are loaded.
        params_dict: Optional pretrained parameter dictionary.
        dtype: Computation dtype.
    """

    out_channels: int
    kernel_size: Union[int, Iterable[int]] = (3, 3)
    strides: Optional[Iterable[int]] = (1, 1)
    padding: Union[str, Iterable[Tuple[int, int]]] = "valid"
    use_bias: bool = False
    kernel_init: partial = nn.initializers.lecun_normal()
    bias_init: partial = nn.initializers.zeros
    params_dict: dict = None
    dtype: str = "float32"

    @nn.compact
    def __call__(self, x, train=True):
        """Apply the convolutional block.

        Args:
            x: Input activations.
            train: Whether to use training-mode batch normalization.

        Returns:
            Output activations.
        """
        x = nn.Conv(
            features=self.out_channels,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            use_bias=self.use_bias,
            kernel_init=(
                self.kernel_init
                if self.params_dict is None
                else lambda *_: jnp.array(self.params_dict["conv"]["kernel"])
            ),
            bias_init=(
                self.bias_init
                if self.params_dict is None
                else lambda *_: jnp.array(self.params_dict["conv"]["bias"])
            ),
            dtype=self.dtype,
        )(x)
        if self.params_dict is None:
            x = BatchNorm(
                epsilon=0.001,
                momentum=0.1,
                use_running_average=not train,
                dtype=self.dtype,
            )(x)
        else:
            x = BatchNorm(
                epsilon=0.001,
                momentum=0.1,
                bias_init=lambda *_: jnp.array(self.params_dict["bn"]["bias"]),
                scale_init=lambda *_: jnp.array(self.params_dict["bn"]["scale"]),
                mean_init=lambda *_: jnp.array(self.params_dict["bn"]["mean"]),
                var_init=lambda *_: jnp.array(self.params_dict["bn"]["var"]),
                use_running_average=not train,
                dtype=self.dtype,
            )(x)
        x = jax.nn.relu(x)
        return x


class InceptionA(nn.Module):
    """InceptionV3 mixed block with parallel 1x1, 5x5, 3x3, and pool paths.

    Attributes:
        pool_features: Number of output channels for the pooling branch.
        params_dict: Optional pretrained parameter dictionary.
        dtype: Computation dtype.
    """

    pool_features: int
    params_dict: dict = None
    dtype: str = "float32"

    @nn.compact
    def __call__(self, x, train=True):
        """Apply the InceptionA block.

        Args:
            x: Input activations.
            train: Whether to use training-mode batch normalization.

        Returns:
            Concatenated branch activations.
        """
        branch1x1 = BasicConv2d(
            out_channels=64,
            kernel_size=(1, 1),
            params_dict=get(self.params_dict, "branch1x1"),
            dtype=self.dtype,
        )(x, train)
        branch5x5 = BasicConv2d(
            out_channels=48,
            kernel_size=(1, 1),
            params_dict=get(self.params_dict, "branch5x5_1"),
            dtype=self.dtype,
        )(x, train)
        branch5x5 = BasicConv2d(
            out_channels=64,
            kernel_size=(5, 5),
            padding=((2, 2), (2, 2)),
            params_dict=get(self.params_dict, "branch5x5_2"),
            dtype=self.dtype,
        )(branch5x5, train)

        branch3x3dbl = BasicConv2d(
            out_channels=64,
            kernel_size=(1, 1),
            params_dict=get(self.params_dict, "branch3x3dbl_1"),
            dtype=self.dtype,
        )(x, train)
        branch3x3dbl = BasicConv2d(
            out_channels=96,
            kernel_size=(3, 3),
            padding=((1, 1), (1, 1)),
            params_dict=get(self.params_dict, "branch3x3dbl_2"),
            dtype=self.dtype,
        )(branch3x3dbl, train)
        branch3x3dbl = BasicConv2d(
            out_channels=96,
            kernel_size=(3, 3),
            padding=((1, 1), (1, 1)),
            params_dict=get(self.params_dict, "branch3x3dbl_3"),
            dtype=self.dtype,
        )(branch3x3dbl, train)

        branch_pool = avg_pool(
            x, window_shape=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1))
        )
        branch_pool = BasicConv2d(
            out_channels=self.pool_features,
            kernel_size=(1, 1),
            params_dict=get(self.params_dict, "branch_pool"),
            dtype=self.dtype,
        )(branch_pool, train)

        output = jnp.concatenate(
            (branch1x1, branch5x5, branch3x3dbl, branch_pool), axis=-1
        )
        return output


class InceptionB(nn.Module):
    """InceptionV3 downsampling mixed block.

    Attributes:
        params_dict: Optional pretrained parameter dictionary.
        dtype: Computation dtype.
    """

    params_dict: dict = None
    dtype: str = "float32"

    @nn.compact
    def __call__(self, x, train=True):
        """Apply the InceptionB block.

        Args:
            x: Input activations.
            train: Whether to use training-mode batch normalization.

        Returns:
            Concatenated branch activations.
        """
        branch3x3 = BasicConv2d(
            out_channels=384,
            kernel_size=(3, 3),
            strides=(2, 2),
            params_dict=get(self.params_dict, "branch3x3"),
            dtype=self.dtype,
        )(x, train)

        branch3x3dbl = BasicConv2d(
            out_channels=64,
            kernel_size=(1, 1),
            params_dict=get(self.params_dict, "branch3x3dbl_1"),
            dtype=self.dtype,
        )(x, train)
        branch3x3dbl = BasicConv2d(
            out_channels=96,
            kernel_size=(3, 3),
            padding=((1, 1), (1, 1)),
            params_dict=get(self.params_dict, "branch3x3dbl_2"),
            dtype=self.dtype,
        )(branch3x3dbl, train)
        branch3x3dbl = BasicConv2d(
            out_channels=96,
            kernel_size=(3, 3),
            strides=(2, 2),
            params_dict=get(self.params_dict, "branch3x3dbl_3"),
            dtype=self.dtype,
        )(branch3x3dbl, train)

        branch_pool = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))

        output = jnp.concatenate((branch3x3, branch3x3dbl, branch_pool), axis=-1)
        return output


class InceptionC(nn.Module):
    """InceptionV3 mixed block with factorized 7x7 branches.

    Attributes:
        channels_7x7: Intermediate channel count for 7x7 branches.
        params_dict: Optional pretrained parameter dictionary.
        dtype: Computation dtype.
    """

    channels_7x7: int
    params_dict: dict = None
    dtype: str = "float32"

    @nn.compact
    def __call__(self, x, train=True):
        """Apply the InceptionC block.

        Args:
            x: Input activations.
            train: Whether to use training-mode batch normalization.

        Returns:
            Concatenated branch activations.
        """
        branch1x1 = BasicConv2d(
            out_channels=192,
            kernel_size=(1, 1),
            params_dict=get(self.params_dict, "branch1x1"),
            dtype=self.dtype,
        )(x, train)

        branch7x7 = BasicConv2d(
            out_channels=self.channels_7x7,
            kernel_size=(1, 1),
            params_dict=get(self.params_dict, "branch7x7_1"),
            dtype=self.dtype,
        )(x, train)
        branch7x7 = BasicConv2d(
            out_channels=self.channels_7x7,
            kernel_size=(1, 7),
            padding=((0, 0), (3, 3)),
            params_dict=get(self.params_dict, "branch7x7_2"),
            dtype=self.dtype,
        )(branch7x7, train)
        branch7x7 = BasicConv2d(
            out_channels=192,
            kernel_size=(7, 1),
            padding=((3, 3), (0, 0)),
            params_dict=get(self.params_dict, "branch7x7_3"),
            dtype=self.dtype,
        )(branch7x7, train)

        branch7x7dbl = BasicConv2d(
            out_channels=self.channels_7x7,
            kernel_size=(1, 1),
            params_dict=get(self.params_dict, "branch7x7dbl_1"),
            dtype=self.dtype,
        )(x, train)
        branch7x7dbl = BasicConv2d(
            out_channels=self.channels_7x7,
            kernel_size=(7, 1),
            padding=((3, 3), (0, 0)),
            params_dict=get(self.params_dict, "branch7x7dbl_2"),
            dtype=self.dtype,
        )(branch7x7dbl, train)
        branch7x7dbl = BasicConv2d(
            out_channels=self.channels_7x7,
            kernel_size=(1, 7),
            padding=((0, 0), (3, 3)),
            params_dict=get(self.params_dict, "branch7x7dbl_3"),
            dtype=self.dtype,
        )(branch7x7dbl, train)
        branch7x7dbl = BasicConv2d(
            out_channels=self.channels_7x7,
            kernel_size=(7, 1),
            padding=((3, 3), (0, 0)),
            params_dict=get(self.params_dict, "branch7x7dbl_4"),
            dtype=self.dtype,
        )(branch7x7dbl, train)
        branch7x7dbl = BasicConv2d(
            out_channels=self.channels_7x7,
            kernel_size=(1, 7),
            padding=((0, 0), (3, 3)),
            params_dict=get(self.params_dict, "branch7x7dbl_5"),
            dtype=self.dtype,
        )(branch7x7dbl, train)

        branch_pool = avg_pool(
            x, window_shape=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1))
        )
        branch_pool = BasicConv2d(
            out_channels=192,
            kernel_size=(1, 1),
            params_dict=get(self.params_dict, "branch_pool"),
            dtype=self.dtype,
        )(branch_pool, train)

        output = jnp.concatenate(
            (branch1x1, branch7x7, branch7x7dbl, branch_pool), axis=-1
        )
        return output


class InceptionD(nn.Module):
    """InceptionV3 downsampling mixed block with factorized branches.

    Attributes:
        params_dict: Optional pretrained parameter dictionary.
        dtype: Computation dtype.
    """

    params_dict: dict = None
    dtype: str = "float32"

    @nn.compact
    def __call__(self, x, train=True):
        """Apply the InceptionD block.

        Args:
            x: Input activations.
            train: Whether to use training-mode batch normalization.

        Returns:
            Concatenated branch activations.
        """
        branch3x3 = BasicConv2d(
            out_channels=192,
            kernel_size=(1, 1),
            params_dict=get(self.params_dict, "branch3x3_1"),
            dtype=self.dtype,
        )(x, train)
        branch3x3 = BasicConv2d(
            out_channels=320,
            kernel_size=(3, 3),
            strides=(2, 2),
            params_dict=get(self.params_dict, "branch3x3_2"),
            dtype=self.dtype,
        )(branch3x3, train)

        branch7x7x3 = BasicConv2d(
            out_channels=192,
            kernel_size=(1, 1),
            params_dict=get(self.params_dict, "branch7x7x3_1"),
            dtype=self.dtype,
        )(x, train)
        branch7x7x3 = BasicConv2d(
            out_channels=192,
            kernel_size=(1, 7),
            padding=((0, 0), (3, 3)),
            params_dict=get(self.params_dict, "branch7x7x3_2"),
            dtype=self.dtype,
        )(branch7x7x3, train)
        branch7x7x3 = BasicConv2d(
            out_channels=192,
            kernel_size=(7, 1),
            padding=((3, 3), (0, 0)),
            params_dict=get(self.params_dict, "branch7x7x3_3"),
            dtype=self.dtype,
        )(branch7x7x3, train)
        branch7x7x3 = BasicConv2d(
            out_channels=192,
            kernel_size=(3, 3),
            strides=(2, 2),
            params_dict=get(self.params_dict, "branch7x7x3_4"),
            dtype=self.dtype,
        )(branch7x7x3, train)

        branch_pool = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))

        output = jnp.concatenate((branch3x3, branch7x7x3, branch_pool), axis=-1)
        return output


class InceptionE(nn.Module):
    """InceptionV3 mixed block with split 3x3 branches.

    Attributes:
        pooling: Pooling function for the pooling branch.
        params_dict: Optional pretrained parameter dictionary.
        dtype: Computation dtype.
    """

    pooling: Callable
    params_dict: dict = None
    dtype: str = "float32"

    @nn.compact
    def __call__(self, x, train=True):
        """Apply the InceptionE block.

        Args:
            x: Input activations.
            train: Whether to use training-mode batch normalization.

        Returns:
            Concatenated branch activations.
        """
        branch1x1 = BasicConv2d(
            out_channels=320,
            kernel_size=(1, 1),
            params_dict=get(self.params_dict, "branch1x1"),
            dtype=self.dtype,
        )(x, train)

        branch3x3 = BasicConv2d(
            out_channels=384,
            kernel_size=(1, 1),
            params_dict=get(self.params_dict, "branch3x3_1"),
            dtype=self.dtype,
        )(x, train)
        branch3x3_a = BasicConv2d(
            out_channels=384,
            kernel_size=(1, 3),
            padding=((0, 0), (1, 1)),
            params_dict=get(self.params_dict, "branch3x3_2a"),
            dtype=self.dtype,
        )(branch3x3, train)
        branch3x3_b = BasicConv2d(
            out_channels=384,
            kernel_size=(3, 1),
            padding=((1, 1), (0, 0)),
            params_dict=get(self.params_dict, "branch3x3_2b"),
            dtype=self.dtype,
        )(branch3x3, train)
        branch3x3 = jnp.concatenate((branch3x3_a, branch3x3_b), axis=-1)

        branch3x3dbl = BasicConv2d(
            out_channels=448,
            kernel_size=(1, 1),
            params_dict=get(self.params_dict, "branch3x3dbl_1"),
            dtype=self.dtype,
        )(x, train)
        branch3x3dbl = BasicConv2d(
            out_channels=384,
            kernel_size=(3, 3),
            padding=((1, 1), (1, 1)),
            params_dict=get(self.params_dict, "branch3x3dbl_2"),
            dtype=self.dtype,
        )(branch3x3dbl, train)
        branch3x3dbl_a = BasicConv2d(
            out_channels=384,
            kernel_size=(1, 3),
            padding=((0, 0), (1, 1)),
            params_dict=get(self.params_dict, "branch3x3dbl_3a"),
            dtype=self.dtype,
        )(branch3x3dbl, train)
        branch3x3dbl_b = BasicConv2d(
            out_channels=384,
            kernel_size=(3, 1),
            padding=((1, 1), (0, 0)),
            params_dict=get(self.params_dict, "branch3x3dbl_3b"),
            dtype=self.dtype,
        )(branch3x3dbl, train)
        branch3x3dbl = jnp.concatenate((branch3x3dbl_a, branch3x3dbl_b), axis=-1)

        branch_pool = self.pooling(
            x, window_shape=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1))
        )
        branch_pool = BasicConv2d(
            out_channels=192,
            kernel_size=(1, 1),
            params_dict=get(self.params_dict, "branch_pool"),
            dtype=self.dtype,
        )(branch_pool, train)

        output = jnp.concatenate(
            (branch1x1, branch3x3, branch3x3dbl, branch_pool), axis=-1
        )
        return output


class InceptionAux(nn.Module):
    """Auxiliary classifier branch for InceptionV3.

    Attributes:
        num_classes: Number of output classes.
        kernel_init: Initializer for the kernel when no parameters are loaded.
        bias_init: Initializer for the bias when no parameters are loaded.
        params_dict: Optional pretrained parameter dictionary.
        dtype: Computation dtype.
    """

    num_classes: int
    kernel_init: partial = nn.initializers.lecun_normal()
    bias_init: partial = nn.initializers.zeros
    params_dict: dict = None
    dtype: str = "float32"

    @nn.compact
    def __call__(self, x, train=True):
        """Apply the auxiliary classifier branch.

        Args:
            x: Input activations.
            train: Whether to use training-mode batch normalization.

        Returns:
            Auxiliary logits.
        """
        x = avg_pool(x, window_shape=(5, 5), strides=(3, 3))
        x = BasicConv2d(
            out_channels=128,
            kernel_size=(1, 1),
            params_dict=get(self.params_dict, "conv0"),
            dtype=self.dtype,
        )(x, train)
        x = BasicConv2d(
            out_channels=768,
            kernel_size=(5, 5),
            params_dict=get(self.params_dict, "conv1"),
            dtype=self.dtype,
        )(x, train)
        x = jnp.mean(x, axis=(1, 2))
        x = jnp.reshape(x, newshape=(x.shape[0], -1))
        x = Dense(
            features=self.num_classes,
            params_dict=get(self.params_dict, "fc"),
            dtype=self.dtype,
        )(x)
        return x


def _absolute_dims(rank, dims):
    """Convert possibly negative dimensions to absolute dimensions.

    Args:
        rank: Rank of the input array.
        dims: Dimension indices.

    Returns:
        Absolute dimension indices.
    """
    return tuple([rank + dim if dim < 0 else dim for dim in dims])


class BatchNorm(nn.Module):
    """Batch normalization layer compatible with pretrained Inception weights.

    Attributes:
        use_running_average: Whether to use stored running averages.
        axis: Feature axis.
        momentum: Running-average momentum.
        epsilon: Numerical stability epsilon.
        dtype: Output dtype.
        use_bias: Whether to include bias.
        use_scale: Whether to include scale.
        bias_init: Bias initializer.
        scale_init: Scale initializer.
        mean_init: Running mean initializer.
        var_init: Running variance initializer.
        axis_name: Optional distributed axis name.
        axis_index_groups: Optional distributed axis groups.
    """

    use_running_average: Optional[bool] = None
    axis: int = -1
    momentum: float = 0.99
    epsilon: float = 1e-5
    dtype: Dtype = jnp.float32
    use_bias: bool = True
    use_scale: bool = True
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros
    scale_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.ones
    mean_init: Callable[[Shape], Array] = lambda s: jnp.zeros(s, jnp.float32)
    var_init: Callable[[Shape], Array] = lambda s: jnp.ones(s, jnp.float32)
    axis_name: Optional[str] = None
    axis_index_groups: Any = None

    @nn.compact
    def __call__(self, x, use_running_average: Optional[bool] = None):
        """Normalize input activations.

        Args:
            x: Input activations.
            use_running_average: Optional override for running-average mode.

        Returns:
            Normalized activations.
        """
        use_running_average = merge_param(
            "use_running_average", self.use_running_average, use_running_average
        )
        x = jnp.asarray(x, jnp.float32)
        axis = self.axis if isinstance(self.axis, tuple) else (self.axis,)
        axis = _absolute_dims(x.ndim, axis)
        feature_shape = tuple(d if i in axis else 1 for i, d in enumerate(x.shape))
        reduced_feature_shape = tuple(d for i, d in enumerate(x.shape) if i in axis)
        reduction_axis = tuple(i for i in range(x.ndim) if i not in axis)

        # see NOTE above on initialization behavior
        initializing = self.is_mutable_collection("params")

        ra_mean = self.variable(
            "batch_stats", "mean", self.mean_init, reduced_feature_shape
        )
        ra_var = self.variable(
            "batch_stats", "var", self.var_init, reduced_feature_shape
        )

        if use_running_average:
            mean, var = ra_mean.value, ra_var.value
        else:
            mean = jnp.mean(x, axis=reduction_axis, keepdims=False)
            mean2 = jnp.mean(lax.square(x), axis=reduction_axis, keepdims=False)
            if self.axis_name is not None and not initializing:
                concatenated_mean = jnp.concatenate([mean, mean2])
                mean, mean2 = jnp.split(
                    lax.pmean(
                        concatenated_mean,
                        axis_name=self.axis_name,
                        axis_index_groups=self.axis_index_groups,
                    ),
                    2,
                )
            var = mean2 - lax.square(mean)

            if not initializing:
                ra_mean.value = (
                    self.momentum * ra_mean.value + (1 - self.momentum) * mean
                )
                ra_var.value = self.momentum * ra_var.value + (1 - self.momentum) * var

        y = x - mean.reshape(feature_shape)
        mul = lax.rsqrt(var + self.epsilon)
        if self.use_scale:
            scale = self.param("scale", self.scale_init, reduced_feature_shape).reshape(
                feature_shape
            )
            mul = mul * scale
        y = y * mul
        if self.use_bias:
            bias = self.param("bias", self.bias_init, reduced_feature_shape).reshape(
                feature_shape
            )
            y = y + bias
        return jnp.asarray(y, self.dtype)


def pool(inputs, init, reduce_fn, window_shape, strides, padding):
    """Apply a generic reduce-window pooling operation.

    Args:
        inputs: Input activations.
        init: Initial value for the reduction.
        reduce_fn: Reduction function.
        window_shape: Spatial pooling window.
        strides: Spatial pooling strides.
        padding: Pooling padding.

    Returns:
        Pooled activations.
    """
    strides = strides or (1,) * len(window_shape)
    assert len(window_shape) == len(strides), f"len({window_shape}) == len({strides})"
    strides = (1,) + strides + (1,)
    dims = (1,) + window_shape + (1,)

    is_single_input = False
    if inputs.ndim == len(dims) - 1:
        # add singleton batch dimension because lax.reduce_window always
        # needs a batch dimension.
        inputs = inputs[None]
        is_single_input = True

    assert inputs.ndim == len(dims), f"len({inputs.shape}) != len({dims})"
    if not isinstance(padding, str):
        padding = tuple(map(tuple, padding))
        assert len(padding) == len(window_shape), (
            f"padding {padding} must specify pads for same number of dims as "
            f"window_shape {window_shape}"
        )
        assert all(
            [len(x) == 2 for x in padding]
        ), f"each entry in padding {padding} must be length 2"
        padding = ((0, 0),) + padding + ((0, 0),)
    y = jax.lax.reduce_window(inputs, init, reduce_fn, dims, strides, padding)
    if is_single_input:
        y = jnp.squeeze(y, axis=0)
    return y


def avg_pool(inputs, window_shape, strides=None, padding="VALID"):
    """Apply average pooling with explicit padding counts.

    Args:
        inputs: Input activations with shape ``(batch, height, width, channels)``.
        window_shape: Spatial pooling window.
        strides: Spatial pooling strides.
        padding: Pooling padding.

    Returns:
        Average-pooled activations.
    """
    assert inputs.ndim == 4
    assert len(window_shape) == 2

    y = pool(inputs, 0.0, jax.lax.add, window_shape, strides, padding)
    ones = jnp.ones(shape=(1, inputs.shape[1], inputs.shape[2], 1)).astype(inputs.dtype)
    counts = jax.lax.conv_general_dilated(
        ones,
        jnp.expand_dims(jnp.ones(window_shape).astype(inputs.dtype), axis=(-2, -1)),
        window_strides=(1, 1),
        padding=((1, 1), (1, 1)),
        dimension_numbers=nn.linear._conv_dimension_numbers(ones.shape),
        feature_group_count=1,
    )
    y = y / counts
    return y
