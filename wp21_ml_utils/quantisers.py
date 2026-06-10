import tensorflow as tf
from tensorflow.keras import initializers, layers

from wp21_ml_utils.utils import softplus


class TrainableQuantiser(layers.Layer):
    def __init__(self, T, **kwargs):
        super().__init__(**kwargs)
        self.T = T

    def _compute_bin_edges(self):
        raise NotImplementedError()

    def call(self, x, training=False):
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

    def get_config(
        self,
    ):
        return {**super().get_config(), "T": self.T}


class QuadLinearQuantiser(TrainableQuantiser):
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
            initializer=initializers.Constant(self.lsb_init),
            trainable=self.trainable,
            name=f"{self.name}_lsb",
        )
        self.G = self.add_weight(
            shape=[1],
            initializer=initializers.Constant(self.G_init),
            trainable=self.trainable,
            name=f"{self.name}_G",
        )

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


class FlexibleQuantiser(TrainableQuantiser):
    def __init__(
        self,
        bits: int,
        min_range: float = None,
        max_range: float = None,
        T: float = 50,
        train_min_range: bool = False,
        train_max_range: bool = False,
        train_widths: bool = False,
        bin_regularisation: float = 1e-2,
        **kwargs,
    ):
        super().__init__(T=T, **kwargs)
        if min_range is None:
            min_range = -(2 ** (bits / 2))
        if max_range is None:
            max_range = 2 ** (bits / 2)

        self.min_range = min_range
        self.max_range = max_range
        self.bits = bits
        self.bin_regularisation = bin_regularisation

        self.num_bins = 2**bits - 1

        self.lower = self.add_weight(
            shape=[1],
            initializer=initializers.Constant(self.min_range),
            trainable=train_min_range,
            name=f"{self.name}_min_range",
        )
        self.range_scale = self.add_weight(
            shape=[1],
            initializer="zeros",
            trainable=train_max_range,
            name=f"{self.name}_range",
        )
        self.bin_widths = self.add_weight(
            shape=[self.num_bins],
            initializer="zeros",
            trainable=train_widths,
            name=f"{self.name}_bin_widths",
        )

    def get_bin_edges(self):
        range_scale = softplus(self.range_scale)
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
            "train_min_range": self.lower.trainable,
            "train_max_range": self.range_scale.trainable,
            "train_widths": self.bin_widths.trainable,
            "bin_regularisation": self.bin_regularisation,
            "T": self.T,
        }
