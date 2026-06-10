import tensorflow as tf
from tensorflow.keras import initializers, layers

from wp21_ml_utils.utils import softplus


class QuadLinearQuantiser(layers.Layer):
    def __init__(self, bits: int = 6, lsb: float = 0.04, G: float = 4.0, **kwargs):
        super().__init__(**kwargs)
        self.bits = int(bits)
        self.lsb = float(lsb)
        self.G = float(G)

        steps_per_range = int(2 ** (self.bits - 2))

        # Precompute values
        bin_widths = tf.constant(
            [0] * steps_per_range
            + [1] * steps_per_range
            + [self.G] * steps_per_range
            + [self.G**2] * steps_per_range
            + [self.G**3] * steps_per_range
        )

        self.bin_edges = self.lsb * tf.cumsum(bin_widths)

    def call(self, x):
        x_shape = tf.shape(x)
        x_flat = tf.reshape(x, [-1])

        bin_edges = self.get_bin_edges()

        d = x_flat[:, None] - bin_edges[None, :]

        nearest_idx = tf.argmin(tf.abs(d), axis=-1)
        y_flat = tf.gather(bin_edges, nearest_idx)

        return tf.reshape(y_flat, x_shape)

    def get_bin_edges(self):
        return self.bin_edges

    def get_config(self):
        return {
            **super().get_config(),
            "bits": self.bits,
            "lsb": self.lsb,
            "G": self.G,
        }


class TrainableQuantiser(layers.Layer):
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
        smooth_in_forward: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if min_range is None:
            min_range = -(2 ** (bits / 2))
        if max_range is None:
            max_range = 2 ** (bits / 2)

        self.min_range = min_range
        self.max_range = max_range
        self.bits = bits
        self.bin_regularisation = bin_regularisation
        self.smooth_in_forward = smooth_in_forward

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
        self.T = T

    def get_bin_edges(self):
        range_scale = softplus(self.range_scale)
        bin_widths = tf.nn.softmax(self.bin_widths)

        edges = tf.concat(
            [tf.zeros([1], dtype=bin_widths.dtype), tf.math.cumsum(bin_widths)],
            axis=0,
        )

        bin_edges = self.lower + (self.max_range - self.min_range) * range_scale * edges
        return bin_edges

    def call(self, x, training=False):
        x_shape = tf.shape(x)
        x_flat = tf.reshape(x, [-1])

        bin_edges = self.get_bin_edges()

        d = x_flat[:, None] - bin_edges[None, :]

        if self.bin_widths.trainable:
            diff = self.bin_widths[1:] - self.bin_widths[:-1]
            self.add_loss(self.bin_regularisation * tf.reduce_sum(tf.abs(diff)))

        if training:
            weights = tf.nn.softmax(-self.T * tf.abs(d), axis=1)
            y_soft = tf.reduce_sum(weights * bin_edges[None, :], axis=1)
            if self.smooth_in_forward:
                y_flat = y_soft

            else:
                nearest_idx = tf.argmin(tf.abs(d), axis=-1)
                y_hard = tf.gather(bin_edges, nearest_idx)
                y_flat = y_soft + tf.stop_gradient(y_hard - y_soft)

        else:
            nearest_idx = tf.argmin(tf.abs(d), axis=-1)
            y_flat = tf.gather(bin_edges, nearest_idx)

        return tf.reshape(y_flat, x_shape)

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
            "smooth_in_forward": self.smooth_in_forward,
        }
