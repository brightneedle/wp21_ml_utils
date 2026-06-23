from typing import Callable

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import Constant
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.types.experimental import TensorLike

from wp21_ml_utils.utils import scaled_softplus, unpack_momenta


@register_keras_serializable("wp21_ml_utils")
class BaseQuantiser(Layer):
    """
    Abstract base class for differentiable quantisation layers.

    Maps continuous inputs onto a discrete set of quantisation levels using
    nearest-neighbour assignment. During training, a soft approximation based
    on temperature-scaled distances is used together with a straight-through
    estimator (STE) to preserve gradient flow.

    Subclasses define the quantisation levels by implementing
    ``_compute_bin_edges()``.

    Input:
        Arbitrary tensor shape.

    Output:
        Tensor with the same shape as the input, where each value is mapped
        to the nearest quantisation level.

    Parameters
    ----------
    T : float
        Temperature controlling the sharpness of the soft assignment used
        during training. Larger values approach hard quantisation.
    """

    def __init__(self, T: float, **kwargs):
        super().__init__(**kwargs)
        self.T = float(T)

    def _compute_bin_edges(self) -> tf.Tensor:
        raise NotImplementedError()

    def call(self, x: TensorLike, training: bool = False) -> tf.Tensor:
        x_shape = tf.shape(x)
        x_flat = tf.reshape(x, [-1])

        bin_edges = self._compute_bin_edges()

        d = x_flat[:, None] - bin_edges[None, :]

        if training:
            weights = tf.nn.softmax(-self.T * tf.abs(d), axis=1)
            y_soft = tf.reduce_sum(weights * bin_edges[None, :], axis=1)

            nearest_idx = tf.argmin(tf.abs(d), axis=-1)
            y_hard = tf.gather(bin_edges, nearest_idx)
            y_flat = y_soft + tf.stop_gradient(y_hard - y_soft)

        else:
            nearest_idx = tf.argmin(tf.abs(d), axis=-1)
            y_flat = tf.gather(bin_edges, nearest_idx)

        return tf.reshape(y_flat, x_shape)

    def get_config(self):
        return {**super().get_config(), "T": self.T}


@register_keras_serializable("wp21_ml_utils")
class QuadLinearQuantiser(BaseQuantiser):
    """
    Piecewise-geometric quantiser with four dynamic ranges.

    Constructs quantisation levels from four consecutive regions with
    increasing bin widths:

        lsb, lsb, G·lsb, G²·lsb, G³·lsb

    where each region contains an equal number of quantisation steps.
    This provides fine resolution near zero and progressively coarser
    resolution at larger values.

    Input:
        Arbitrary tensor shape.

    Output:
        Tensor of identical shape with values mapped to the nearest
        quantisation level.

    Parameters
    ----------
    bits : int, default=6
        Total quantiser bit width.

    lsb : float, default=0.04
        Width of the smallest quantisation step.

    G : float, default=4.0
        Geometric scaling factor between neighbouring dynamic ranges.

    trainable : bool, default=False
        If True, allows ``lsb`` and ``G`` to be optimised during training.

    T : float, default=50
        Temperature used for differentiable quantisation.
    """

    def __init__(
        self,
        bits: int = 6,
        lsb: float = 0.04,
        G: float = 4.0,
        trainable: bool = False,
        T: float = 50,
        **kwargs,
    ):
        super().__init__(T=T, **kwargs)
        self.bits = int(bits)
        self.lsb_init = float(lsb)
        self.G_init = float(G)
        self.trainable = trainable

        self.steps_per_range = int(2 ** (self.bits - 2))

    def build(self, input_shape):
        self.lsb = self.add_weight(
            shape=[1],
            initializer=Constant(self.lsb_init),
            trainable=self.trainable,
            name=f"{self.name}_lsb",
        )
        self.G = self.add_weight(
            shape=[1],
            initializer=Constant(self.G_init),
            trainable=self.trainable,
            name=f"{self.name}_G",
        )
        super().build(input_shape)

    def _compute_bin_edges(self):
        n = self.steps_per_range

        zero = tf.zeros([n], dtype=self.lsb.dtype)
        one = tf.ones([n], dtype=self.lsb.dtype)

        g = tf.fill([n], tf.reshape(self.G, []))
        g2 = tf.fill([n], tf.pow(tf.reshape(self.G, []), 2.0))
        g3 = tf.fill([n], tf.pow(tf.reshape(self.G, []), 3.0))

        bin_widths = tf.concat([zero, one, g, g2, g3], axis=0)

        return self.lsb * tf.cumsum(bin_widths)

    def get_config(self):
        return {
            **super().get_config(),
            "bits": self.bits,
            "lsb": self.lsb_init,
            "G": self.G_init,
        }


@register_keras_serializable("wp21_ml_utils")
class FlexibleQuantiser(BaseQuantiser):
    """
    Learnable non-uniform quantiser.

    Defines a quantisation grid through trainable bin widths and optionally
    trainable lower and upper range limits. Bin widths are normalised with
    a softmax operation to ensure a valid ordered set of quantisation levels.

    This layer can learn task-specific quantisation schemes while maintaining
    differentiability through the straight-through estimator inherited from
    ``BaseQuantiser``.

    Input:
        Arbitrary tensor shape.

    Output:
        Tensor of identical shape with values mapped to the nearest learned
        quantisation level.

    Parameters
    ----------
    bits : int
        Quantiser bit width.

    min_range : float, optional
        Initial lower edge of the quantisation range.

    max_range : float, optional
        Initial upper edge of the quantisation range.

    T : float, default=50
        Temperature used for differentiable quantisation.

    train_min_range : bool, default=False
        Allows the lower range limit to be learned.

    train_max_range : bool, default=False
        Allows the overall quantisation range to be learned.

    train_widths : bool, default=False
        Allows the relative bin widths to be learned.

    bin_smoothing : float, default=1e-2
        Smoothing parameter reserved for regularisation of learned bin
        widths.
    """

    def __init__(
        self,
        bits: int,
        min_range: float = None,
        max_range: float = None,
        T: float = 50,
        train_min_range: bool = False,
        train_max_range: bool = False,
        train_widths: bool = False,
        bin_smoothing: float = 1e-3,
        **kwargs,
    ):
        super().__init__(T=T, **kwargs)
        if min_range is None:
            min_range = -(2 ** (bits / 2))
        if max_range is None:
            max_range = 2 ** (bits / 2)

        self.min_range = float(min_range)
        self.max_range = float(max_range)
        self.bits = int(bits)
        self.bin_smoothing = float(bin_smoothing)
        self.train_min_range = train_min_range
        self.train_max_range = train_max_range
        self.train_widths = train_widths

        self.num_bins = 2**self.bits - 1

    def build(self, input_shape):
        self.lower = self.add_weight(
            shape=[1],
            initializer=Constant(self.min_range),
            trainable=self.train_min_range,
            name=f"{self.name}_min_range",
        )
        self.range_scale = self.add_weight(
            shape=[1],
            initializer="zeros",
            trainable=self.train_max_range,
            name=f"{self.name}_range",
        )
        self.bin_widths = self.add_weight(
            shape=[self.num_bins],
            initializer="zeros",
            trainable=self.train_widths,
            name=f"{self.name}_bin_widths",
        )
        super().build(input_shape)

    def _compute_bin_edges(self):
        range_scale = scaled_softplus(self.range_scale)
        bin_widths = tf.nn.softmax(self.bin_widths)

        edges = tf.concat(
            [tf.zeros([1], dtype=bin_widths.dtype), tf.math.cumsum(bin_widths)],
            axis=0,
        )

        bin_edges = self.lower + (self.max_range - self.min_range) * range_scale * edges
        return bin_edges

    def get_config(self):
        return {
            **super().get_config(),
            "bits": self.bits,
            "min_range": self.min_range,
            "max_range": self.max_range,
            "train_min_range": self.train_min_range,
            "train_max_range": self.train_max_range,
            "train_widths": self.train_widths,
            "bin_smoothing": self.bin_smoothing,
            "T": self.T,
        }


@register_keras_serializable("wp21_ml_utils")
class EncodeCellEt(Layer):
    def __init__(self, encoding: Callable, **kwargs):
        super().__init__(**kwargs)
        self.encoding = encoding

    def call(self, x):
        components = unpack_momenta(x)
        et = components[0]
        encoded_et = self.encoding(et)
        return tf.concat([encoded_et, *components[1:]], axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update(
            {"encoding": tf.keras.utils.serialize_keras_object(self.encoding)}
        )
        return config

    @classmethod
    def from_config(cls, config):
        custom_objects = tf.keras.utils.get_custom_objects()
        encoding_config = config["encoding"]
        config["encoding"] = tf.keras.utils.deserialize_keras_object(
            encoding_config, custom_objects=custom_objects
        )
        return cls(**config)
