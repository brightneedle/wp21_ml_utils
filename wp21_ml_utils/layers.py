from typing import Callable, Tuple, Optional, Union, Any

import tensorflow as tf
from tensorflow.keras import layers, activations
from tensorflow.keras.backend import set_image_data_format
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


class SymmetricPooling(layers.Layer):
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

        # IMPORTANT: register as Keras weight (not tf.constant)
        self.kernel = self.add_weight(
            name="kernel",
            shape=kernel.shape,
            initializer=tf.constant_initializer(kernel),
            trainable=False,
        )

        super().build(input_shape)

    def call(self, inputs):
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


class SymmetricDepthwiseConv2D(layers.Layer):
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

    def call(self, inputs):
        pooled_inputs = self.pooling(inputs)
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


class EtaPhiPadding(layers.Layer):
    def __init__(self, pad_size, **kwargs):
        super().__init__(**kwargs)
        self.pad_size = pad_size

    def cyclic_padding_at_axis(self, x, axis=2):
        length = tf.shape(x)[axis]
        pad_before = tf.gather(x, tf.range(length - self.pad_size, length), axis=axis)
        pad_after = tf.gather(x, tf.range(0, self.pad_size), axis=axis)
        return tf.concat([pad_before, x, pad_after], axis=axis)

    def zero_padding_at_axis(self, x, axis=1):
        rank = len(x.shape)
        paddings = [[0, 0]] * rank
        paddings[axis] = [self.pad_size, self.pad_size]
        return tf.pad(x, paddings, mode="CONSTANT", constant_values=0)

    def call(self, x):
        return self.zero_padding_at_axis(self.cyclic_padding_at_axis(x))

    def get_config(self):
        return {**super().get_config(), "pad_size": self.pad_size}


class TowerEtaPhiLayer(layers.Layer):
    def __init__(self, deta: float = 0.1, dphi: float = np.pi / 32, **kwargs):
        super().__init__(**kwargs)
        self.deta = deta
        self.dphi = dphi

    def call(self, image):
        B, E, P, _ = tf.unstack(tf.shape(image))

        eta_idxs = tf.tile(tf.reshape(tf.range(E), (1, E, 1, 1)), (B, 1, P, 1))
        eta = tf.cast(2 * eta_idxs - E + 1, dtype=tf.float32) * self.deta / 2.0

        phi_idxs = tf.tile(tf.reshape(tf.range(P), (1, 1, P, 1)), (B, E, 1, 1))
        phi = tf.cast(2 * phi_idxs - P + 1, dtype=tf.float32) * self.dphi / 2.0

        return eta, phi

    def get_config(self):
        return {**super().get_config(), "deta": self.deta, "dphi": self.dphi}


class SlidingConeSum(layers.Layer):
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

    def call(self, x):
        return tf.nn.conv2d(self.pad(x), self.kernel, strides=1, padding="VALID")

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


class CircularMaxPool(tf.keras.layers.Layer):
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

    def call(self, x):
        k = self.kernel_size

        patches = tf.image.extract_patches(
            images=x,
            sizes=[1, k, k, 1],
            strides=[1, 1, 1, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )  # [B, H, W, k*k*C]

        b, h, w, c = tf.unstack(tf.shape(x))
        c = x.shape[-1]

        patches = tf.reshape(
            patches,
            [b, h - k + 1, w - k + 1, k, k, c],
        )

        patches = patches + self.filter  # broadcasts over batch/spatial

        return tf.reduce_max(patches, axis=[3, 4])


class LocalMaxMask(layers.Layer):
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

    def call(self, image):
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


class NthLeadingPt(layers.Layer):
    def __init__(self, index: int, **kwargs):
        super().__init__(**kwargs)
        self.index = index

    def call(self, jets):
        pt = jets[..., 0]
        sorted_pt = tf.sort(pt, axis=-1, direction="DESCENDING")
        return sorted_pt[:, self.index, None]

    def get_config(self):
        return {
            **super().get_config(),
            "index": self.index,
        }


class VectorSum(layers.Layer):
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

    def call(self, x):
        components = unpack_momenta(x)
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
            "jet_idx": self.jet_idx,
        }


class MonoDense(layers.Dense):
    """
    Monotonic Dense Layer with:
      - monotonicity constraints on weights
      - optional convex/concave/saturated activation partitioning
    """

    def __init__(
        self,
        units: int,
        *,
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

    def call(self, inputs: TensorLike) -> TensorLike:
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
