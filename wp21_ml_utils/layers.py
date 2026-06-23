from typing import Callable, Tuple, Optional, Union, Any

import tensorflow as tf
from tensorflow.keras import layers, activations
from tensorflow.keras.backend import set_image_data_format
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.types.experimental import TensorLike
import numpy as np
import itertools

from wp21_ml_utils.utils import (
    unpack_momenta,
    polar_to_cartesian,
    init_dense_layer,
    cartesian_to_polar,
)

set_image_data_format("channels_last")


@register_keras_serializable("wp21_ml_utils")
class SymmetricPooling(layers.Layer):
    """
    Symmetry-aware pooling layer.

    Groups pixels that are equivalent under η–φ reflection symmetries within
    a square kernel and sums them into a reduced feature representation.

    This produces a fixed set of symmetry-invariant features which can be
    processed by subsequent dense layers.

    Input:
        Tensor of shape (B, H, W, C)

    Output:
        Tensor of shape
        (B, H-k+1, W-k+1, C * ((k//2)+1)^2)

    Parameters
    ----------
    size : int
        Odd kernel size defining the symmetry neighbourhood.
    """

    def __init__(self, size: int, **kwargs):
        super().__init__(**kwargs)

        if size % 2 != 1:
            raise ValueError("size must be odd integer")

        self.size = size

    def build(self, input_shape):
        input_channels = input_shape[-1]

        centre = self.size // 2
        n_features = (centre + 1) ** 2

        k = np.zeros(
            (self.size, self.size, 1, n_features),
            dtype=np.float32,
        )

        feature_idx = 0
        for eta_idx in range(centre + 1):
            for phi_idx in range(centre + 1):
                for i, j in itertools.product(
                    [eta_idx, self.size - 1 - eta_idx],
                    [phi_idx, self.size - 1 - phi_idx],
                ):
                    k[i, j, :, feature_idx] = 1.0
                feature_idx += 1

        assert feature_idx == n_features

        kernel = np.repeat(k, input_channels, axis=2)

        self.kernel = self.add_weight(
            name="kernel",
            shape=kernel.shape,
            initializer=tf.constant_initializer(kernel),
            trainable=False,
        )

        super().build(input_shape)

    def call(self, inputs: TensorLike) -> tf.Tensor:
        return tf.nn.depthwise_conv2d(
            inputs,
            self.kernel,
            strides=[1, 1, 1, 1],
            padding="VALID",
        )

    def get_config(self):
        config = super().get_config()
        config.update({"size": self.size})
        return config


@register_keras_serializable("wp21_ml_utils")
class SymmetricDepthwiseConv2D(layers.Layer):
    """
    Symmetry-constrained depthwise convolution.

    Performs SymmetricPooling independently for each input channel and then
    applies a channel-specific dense transformation. This provides a learned
    convolution-like operation while preserving reflection symmetries of the
    η–φ neighbourhood.

    Input:
        (B, H, W, C)

    Output:
        (B, H-k+1, W-k+1, C * depth_multiplier)

    Parameters
    ----------
    kernel_size : int
        Size of the symmetric neighbourhood.

    depth_multiplier : int
        Number of output features produced per input channel.

    activation : str, optional
        Activation applied inside the per-channel dense projection.

    use_hgq : bool, default=False
        Enables HGQ-compatible dense layers when available.
    """

    def __init__(
        self,
        kernel_size: int,
        depth_multiplier: int,
        activation: str = None,
        use_hgq: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.kernel_size = kernel_size
        self.depth_multiplier = depth_multiplier
        self.activation = activation
        self.use_hgq = use_hgq

    def build(self, input_shape):
        self.input_channels = int(input_shape[-1])

        self.pooling = SymmetricPooling(size=self.kernel_size)

        self.dense_layers = [
            init_dense_layer(
                self.depth_multiplier,
                activation=self.activation,
                use_hgq=self.use_hgq,
            )
            for _ in range(self.input_channels)
        ]
        super().build(input_shape)

    def call(self, image: TensorLike) -> tf.Tensor:
        pooled_inputs = self.pooling(image)
        pooled_inputs_by_layer = tf.split(pooled_inputs, self.input_channels, axis=-1)
        pooled_inputs_by_layer = [
            dense_layer(x)
            for dense_layer, x in zip(self.dense_layers, pooled_inputs_by_layer)
        ]
        outputs = tf.concat(pooled_inputs_by_layer, axis=-1)
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "kernel_size": self.kernel_size,
                "depth_multiplier": self.depth_multiplier,
                "activation": self.activation,
                "use_hgq": self.use_hgq,
            }
        )
        return config


@register_keras_serializable("wp21_ml_utils")
class EtaPhiPadding(layers.Layer):
    """
    Detector-aware η–φ padding.

    Applies:
      - cyclic padding in φ (periodic detector coordinate)
      - zero padding in η (non-periodic detector coordinate)

    This reproduces the topology commonly used in calorimeter image
    representations.

    Input:
        (B, E, P, C)

    Output:
        (B, E+2p, P+2p, C)

    Parameters
    ----------
    pad_size : int
        Number of cells added on each side.
    """

    def __init__(self, pad_size, **kwargs):
        super().__init__(**kwargs)
        self.pad_size = pad_size

    def cyclic_padding_at_axis(self, x: TensorLike, axis: int = 2) -> TensorLike:
        length = tf.shape(x)[axis]
        pad_before = tf.gather(x, tf.range(length - self.pad_size, length), axis=axis)
        pad_after = tf.gather(x, tf.range(0, self.pad_size), axis=axis)
        return tf.concat([pad_before, x, pad_after], axis=axis)

    def zero_padding_at_axis(self, x: TensorLike, axis: int = 1):
        rank = len(x.shape)
        paddings = [[0, 0]] * rank
        paddings[axis] = [self.pad_size, self.pad_size]
        return tf.pad(x, paddings, mode="CONSTANT", constant_values=0)

    def call(self, x: TensorLike) -> tf.Tensor:
        return self.zero_padding_at_axis(self.cyclic_padding_at_axis(x))

    def get_config(self):
        return {**super().get_config(), "pad_size": self.pad_size}


@register_keras_serializable("wp21_ml_utils")
class TowerEtaPhiLayer(layers.Layer):
    """
    Generates η and φ coordinate maps for calorimeter images.

    Converts image indices into physical η and φ coordinates and returns
    coordinate tensors aligned with the input image grid.

    Input:
        Image tensor of shape (B, E, P, C)

    Output:
        Tuple[eta, phi]
        Each of shape (B, E, P, 1)

    Parameters
    ----------
    deta : float
        η bin width.

    dphi : float
        φ bin width.
    """

    def __init__(self, deta: float = 0.1, dphi: float = np.pi / 32, **kwargs):
        super().__init__(**kwargs)
        self.deta = deta
        self.dphi = dphi

    def call(self, image: TensorLike) -> tf.Tensor:
        B, E, P, _ = tf.unstack(tf.shape(image))

        eta_idxs = tf.tile(tf.reshape(tf.range(E), (1, E, 1, 1)), (B, 1, P, 1))
        eta = tf.cast(2 * eta_idxs - E + 1, dtype=tf.float32) * self.deta / 2.0

        phi_idxs = tf.tile(tf.reshape(tf.range(P), (1, 1, P, 1)), (B, E, 1, 1))
        phi = tf.cast(2 * phi_idxs - P + 1, dtype=tf.float32) * self.dphi / 2.0

        return eta, phi

    def get_config(self):
        return {**super().get_config(), "deta": self.deta, "dphi": self.dphi}


@register_keras_serializable("wp21_ml_utils")
class SlidingConeSum(layers.Layer):
    """
    Sliding-window energy sum.

    Computes the sum of values within a circular or square neighbourhood
    centred on every pixel. Frequently used to emulate cone-based jet or
    cluster energy accumulation.

    Input:
        (B, H, W, C)

    Output:
        (B, H, W, 1)

    Parameters
    ----------
    kernel_size : int
        Size of the neighbourhood.

    shape : {"circle", "square"}
        Geometry of the accumulation region.

    radius : int, optional
        Radius used for circular masks.
    """

    def __init__(
        self,
        kernel_size: int = 9,
        shape: str = "circle",
        radius: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.kernel_size = int(kernel_size)
        self.shape = shape
        self.radius = radius

        self.kernel = self.init_kernel()
        self.pad = EtaPhiPadding(pad_size=kernel_size // 2)

    def call(self, image: TensorLike) -> tf.Tensor:
        return tf.nn.conv2d(self.pad(image), self.kernel, strides=1, padding="VALID")

    def get_config(self):
        return {
            **super().get_config(),
            "kernel_size": self.kernel_size,
            "shape": self.shape,
            "radius": self.radius,
        }

    def init_kernel(self) -> tf.Tensor:
        if self.radius is None:
            radius = self.kernel_size // 2 if self.radius is None else self.radius
        else:
            radius = self.radius

        kernel = np.ones((self.kernel_size, self.kernel_size), dtype=np.float32)

        if self.shape == "square":
            return tf.convert_to_tensor(kernel[:, :, None, None], dtype=self.dtype)

        elif self.shape == "circle":
            eta_idxs, phi_idxs = np.indices((self.kernel_size, self.kernel_size))
            deta = eta_idxs - self.kernel_size // 2
            dphi = phi_idxs - self.kernel_size // 2
            mask = (deta**2 + dphi**2) <= radius**2
            kernel = np.where(mask, kernel, 0)
            return tf.convert_to_tensor(kernel[:, :, None, None], dtype=self.dtype)
        else:
            raise ValueError("Shape must be 'square' or 'circle'")

    def build(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError(f"Expected 4D input (B, H, W, C), got {input_shape}")

        self.input_channels = input_shape[-1]
        self.pad.build(input_shape)
        super().build(input_shape)


@register_keras_serializable("wp21_ml_utils")
class CircularMaxPool(tf.keras.layers.Layer):
    """
    Circular max-pooling operator.

    Computes the maximum value inside a circular neighbourhood while ignoring
    pixels outside the circle.

    Unlike standard MaxPool2D, the receptive field follows a disk-shaped
    geometry.

    Input:
        (B, H, W, C)

    Output:
        (B, H-k+1, W-k+1, C)

    Parameters
    ----------
    kernel_size : int
        Diameter of the circular pooling region.
    """

    def __init__(self, kernel_size: int, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size

        radius = kernel_size // 2

        yy, xx = np.indices((kernel_size, kernel_size))
        yy -= radius
        xx -= radius
        circle = (xx**2 + yy**2) <= radius**2

        filt = np.where(circle, 0.0, -1e9).astype(np.float32)
        self.filter = tf.constant(filt[None, :, :, None])  # [1, k, k, 1]

    def call(self, image: TensorLike) -> tf.Tensor:
        k = self.kernel_size

        patches = tf.image.extract_patches(
            images=image,
            sizes=[1, k, k, 1],
            strides=[1, 1, 1, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )  # [B, H, W, k*k*C]

        b, h, w, c = tf.unstack(tf.shape(image))

        patches = tf.reshape(
            patches,
            [b, h - k + 1, w - k + 1, k, k, c],
        )

        patches = patches + self.filter  # broadcasts over batch/spatial

        return tf.reduce_max(patches, axis=[3, 4])


@register_keras_serializable("wp21_ml_utils")
class LocalMaxMask(layers.Layer):
    """
    Local-maximum detector.

    Produces a boolean mask indicating whether each pixel is the maximum
    within a local neighbourhood.

    Supports both square and circular neighbourhood definitions and uses
    a small random perturbation to break ties deterministically.

    Input:
        (B, H, W, C)

    Output:
        Boolean tensor of shape (B, H, W, C)

    Parameters
    ----------
    kernel_size : int
        Local neighbourhood size.

    shape : {"square", "circle"}
        Neighbourhood geometry.
    """

    def __init__(self, kernel_size: int = 9, shape: str = "square", **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.shape = shape
        self.radius = kernel_size // 2
        self.pad = EtaPhiPadding(pad_size=self.radius)

        if shape == "square":
            # Original square max-pool
            self.pool = layers.MaxPool2D(
                pool_size=(kernel_size, kernel_size),
                strides=(1, 1),
                padding="VALID",
            )

        elif shape == "circle":
            self.pool = CircularMaxPool(kernel_size)

        else:
            raise ValueError(f"shape must be 'square' or 'circle', got '{shape}'")

    def build(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError(f"Expected 4D input (B, H, W, C), got shape {input_shape}")

        self.height = input_shape[1]
        self.width = input_shape[2]
        self.channels = input_shape[3]

        self.pad.build(input_shape)
        self.pool.build(input_shape)

        super().build(input_shape)

    def call(self, image: TensorLike) -> tf.Tensor:
        # Add tiny epsilon for deterministic tie-breaking
        eps = tf.random.uniform(
            tf.shape(image), minval=-1, maxval=1, seed=42, dtype=image.dtype
        )
        image_w_eps = image + 1e-3 * eps
        pooled = self.pool(self.pad(image_w_eps))
        return tf.equal(image_w_eps, pooled)

    def get_config(self):
        return {
            **super().get_config(),
            "kernel_size": self.kernel_size,
            "shape": self.shape,
        }


@register_keras_serializable("wp21_ml_utils")
class NthLeadingPt(layers.Layer):
    """
    Extracts the N-th highest transverse momentum.

    Sorts jet constituents by pT and returns the specified rank.

    Input:
        Tensor containing momentum features where the first component
        corresponds to pT.

    Output:
        (B, 1)

    Parameters
    ----------
    index : int
        Zero-based rank after sorting in descending pT order.
    """

    def __init__(self, index: int, **kwargs):
        super().__init__(**kwargs)
        self.index = index

    def call(self, vectors: TensorLike) -> tf.Tensor:
        pt = vectors[..., 0]
        sorted_pt = tf.sort(pt, axis=-1, direction="DESCENDING")
        return sorted_pt[:, self.index, None]

    def get_config(self):
        return {
            **super().get_config(),
            "index": self.index,
        }


@register_keras_serializable("wp21_ml_utils")
class VectorSum(layers.Layer):
    """
    Vectorial momentum summation.

    Sums constituent momentum vectors and returns the total momentum in
    either Cartesian or polar coordinates.

    Supports automatic conversion between coordinate systems before and
    after summation.

    Input:
        (B, N, 3)

    Output:
        (B, 3)

    Parameters
    ----------
    input : {"polar", "cartesian"}
        Coordinate system of the inputs.

    output : {"polar", "cartesian"}
        Coordinate system of the returned vector sum.
    """

    def __init__(self, input: str = "polar", output: str = "polar", **kwargs):
        super().__init__(**kwargs)
        self.input = input
        self.output = output

        if self.input not in ["polar", "cartesian"]:
            raise ValueError(
                f"'input' must be 'polar' or 'cartesian', got '{self.input}'"
            )

        if self.output not in ["polar", "cartesian"]:
            raise ValueError(
                f"'output' must be 'polar' or 'cartesian', got '{self.output}'"
            )

    def call(self, vectors: TensorLike) -> tf.Tensor:
        components = unpack_momenta(vectors)
        if self.input == "polar":
            px, py, pz = polar_to_cartesian(components)
        else:
            px, py, pz = components

        sum_px = tf.reduce_sum(px, axis=1)
        sum_py = tf.reduce_sum(py, axis=1)
        sum_pz = tf.reduce_sum(pz, axis=1)

        if self.output == "polar":
            sum_components = cartesian_to_polar(sum_px, sum_py, sum_pz)
        else:
            sum_components = sum_px, sum_py, sum_pz

        return tf.concat([sum_components], axis=1)

    def get_config(self):
        return {
            **super().get_config(),
            "input": self.input,
            "output": self.output,
        }


@register_keras_serializable("wp21_ml_utils")
class MonoDense(layers.Dense):
    """
    Monotonic fully-connected layer.

    Extends ``tf.keras.layers.Dense`` by enforcing sign constraints on selected
    weights and by optionally partitioning the output units into convex,
    concave, and saturated activation groups. The layer is designed for
    constructing neural networks with provable monotonicity properties while
    retaining expressive nonlinear behaviour.

    Monotonicity
    ------------
    The ``monotonicity_indicator`` specifies how the output should depend on
    each input feature:

    * ``+1`` : monotonically increasing
    * ``-1`` : monotonically decreasing
    * ``0``  : unconstrained

    During the forward pass, constrained weights are projected onto the
    appropriate sign domain:

    * increasing features use ``|w|``
    * decreasing features use ``-|w|``
    * unconstrained features use ``w``

    This guarantees that the layer preserves the specified monotonic
    relationship with respect to each constrained input dimension.

    Activation Partitioning
    -----------------------
    When an activation function is specified, the output units are divided
    into three groups:

    1. Convex units
       Apply the activation directly.

    2. Concave units
       Apply the reflected activation

           f_concave(x) = -f(-x)

    3. Saturated units
       Combine convex and concave behaviour to produce a bounded,
       saturation-like response.

    The relative number of units assigned to each group is determined by
    ``activation_weights``.

    Convexity and Concavity
    -----------------------
    The layer supports architectures used in monotonic neural networks,
    partially input-convex neural networks (PICNNs), and related constrained
    function approximators.

    By combining monotonic weight constraints with convex, concave, and
    saturated activation components, the layer can represent rich nonlinear
    monotone functions while maintaining interpretable shape constraints.

    Parameters
    ----------
    units : int
        Number of output units.

    activation : str or callable, optional
        Base activation function used to construct the convex, concave,
        and saturated activation branches.

    monotonicity_indicator : int or sequence of int, default=0
        Monotonicity constraint for each input feature.

        * +1 = increasing
        * -1 = decreasing
        *  0 = unconstrained

        A scalar applies the same constraint to all inputs.

    is_convex : bool, default=False
        Marks the layer as belonging to a convex branch of a monotonic
        architecture. Stored for configuration and model construction.

    is_concave : bool, default=False
        Marks the layer as belonging to a concave branch of a monotonic
        architecture. Stored for configuration and model construction.

    activation_weights : tuple(float, float, float),
        default=(7.0, 7.0, 2.0)

        Relative allocation of output units to

        (convex, concave, saturated)

        activation groups. The values are normalised internally before
        determining the number of units assigned to each group.

    Notes
    -----
    The layer applies monotonicity constraints only to the kernel weights.
    Bias terms remain unconstrained.

    The exact number of units assigned to each activation group is obtained
    by rounding the normalised activation weights and assigning any remaining
    units to the saturated group.
    """

    def __init__(
        self,
        units: int,
        activation: Optional[Union[str, Callable]] = None,
        monotonicity_indicator: Union[int, list] = 0,
        is_convex: bool = False,
        is_concave: bool = False,
        activation_weights: Tuple[float, float, float] = (7.0, 7.0, 2.0),
        **kwargs: Any,
    ):
        super().__init__(units=units, activation=None, **kwargs)

        if hasattr(monotonicity_indicator, "__len__"):
            self.monotonicity_indicator = list(monotonicity_indicator)
        else:
            self.monotonicity_indicator = float(monotonicity_indicator)

        self.org_activation = activation
        self.is_convex = is_convex
        self.is_concave = is_concave
        self.activation_weights = activation_weights

        if is_convex and is_concave:
            raise ValueError("Layer cannot be both convex and concave.")

        if len(activation_weights) != 3:
            raise ValueError("activation_weights must have 3 components.")

        if any(w < 0 for w in activation_weights):
            raise ValueError("activation_weights must be non-negative.")

        # Will be set in build()
        self.convex_activation = None
        self.concave_activation = None
        self.saturated_activation = None

        self._s_convex = None
        self._s_concave = None
        self._s_saturated = None

    def build(self, input_shape):
        super().build(input_shape)

        if isinstance(self.monotonicity_indicator, float):
            mi = np.full(input_shape[-1], self.monotonicity_indicator, dtype=np.float32)
        else:
            mi = np.asarray(self.monotonicity_indicator, dtype=np.float32)

        self.mi = tf.constant(mi.reshape(-1, 1), dtype=tf.float32)

        # Activations
        self.convex_activation = activations.get(self.org_activation)

        def concave(x):
            return -self.convex_activation(-x)

        self.concave_activation = concave

        def saturated(x):
            c = 1.0
            cc = self.convex_activation(tf.ones_like(x) * c)
            return tf.where(
                x <= 0,
                self.convex_activation(x + c) - cc,
                concave(x - c) + cc,
            )

        self.saturated_activation = saturated

        w = np.array(self.activation_weights, dtype=np.float32)
        w = w / np.sum(w)

        s_convex = int(np.round(w[0] * self.units))
        s_concave = int(np.round(w[1] * self.units))
        s_saturated = self.units - s_convex - s_concave

        self._s_convex = s_convex
        self._s_concave = s_concave
        self._s_saturated = s_saturated

    def _apply_monotonicity(self, kernel: tf.Tensor) -> tf.Tensor:
        abs_kernel = tf.abs(kernel)

        mi = self.monotonicity_indicator

        kernel = tf.where(mi == 1.0, abs_kernel, kernel)
        kernel = tf.where(mi == -1.0, -abs_kernel, kernel)

        return kernel

    def _apply_activations(self, x: tf.Tensor) -> tf.Tensor:
        x_c, x_n, x_s = tf.split(
            x,
            [self._s_convex, self._s_concave, self._s_saturated],
            axis=-1,
        )

        y_c = self.convex_activation(x_c)
        y_n = self.concave_activation(x_n)
        y_s = self.saturated_activation(x_s)

        return tf.concat([y_c, y_n, y_s], axis=-1)

    def call(self, inputs: TensorLike) -> tf.Tensor:
        kernel = self._apply_monotonicity(self.kernel)

        outputs = tf.linalg.matmul(inputs, kernel)

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)

        if self.org_activation is not None:
            outputs = self._apply_activations(outputs)

        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "activation": self.org_activation,
                "monotonicity_indicator": self.monotonicity_indicator,
                "is_convex": self.is_convex,
                "is_concave": self.is_concave,
                "activation_weights": self.activation_weights,
            }
        )
        return config
