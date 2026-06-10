import tensorflow as tf
from tensorflow.keras import (
    layers,
    backend,
    Model,
    Input,
)
import tensorflow_probability as tfp


from wp21_ml_utils.utils import (
    softplus,
    smooth_piecewise_linear,
    smooth_piecewise_linear_deriv,
    diff,
)

tfb = tfp.bijectors


class SmoothPieceWiseLinear(tfb.Bijector):
    def __init__(
        self,
        left_slope: float,
        right_slope: float,
        k: float = 1,
        x_shift: float = 0,
        y_shift: float = 0,
        eps=1e-6,
        validate_args: bool = False,
        **kwargs,
    ):
        super().__init__(
            validate_args=validate_args, forward_min_event_ndims=0, **kwargs
        )
        if validate_args:
            tf.debugging.assert_positive(left_slope)
            tf.debugging.assert_positive(right_slope)
            tf.debugging.assert_positive(k)

        self.left_slope = tf.maximum(left_slope, eps)
        self.right_slope = tf.maximum(right_slope, eps)
        self.x_shift = x_shift
        self.y_shift = y_shift
        self.k = tf.maximum(k, eps)
        self.eps = eps

    def _forward(self, x):
        return (
            smooth_piecewise_linear(
                x - self.x_shift, self.left_slope, self.right_slope, self.k
            )
            + self.y_shift
        )

    def _forward_log_det_jacobian(self, x):
        deriv = smooth_piecewise_linear_deriv(
            x - self.x_shift, self.left_slope, self.right_slope, self.k
        )
        deriv = backend.clip(deriv, self.eps, 1 / self.eps)
        return tf.math.log(deriv)

    def _inverse(self, y):
        y0 = y - self.y_shift

        mean_slope = 0.5 * (self.left_slope + self.right_slope)
        x = y0 * tf.math.log(2.0) / mean_slope + self.x_shift

        for _ in range(5):
            f = self._forward(x) - y
            fp = smooth_piecewise_linear_deriv(
                x - self.x_shift,
                self.left_slope,
                self.right_slope,
                self.k,
            )
            x = x - f / fp

        return x


class RQSLayer(layers.Layer):
    def __init__(
        self,
        num_bins: int,
        monotonic: bool = True,
        range_min: float = -1.0,
        range_max: float = 1.0,
        lamda: float = 1e-4,
        log_input: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_bins = num_bins
        self.monotonic = monotonic
        self.lamda = lamda
        self.log_input = log_input
        scale_init = "zeros" if self.monotonic else "ones"
        self.slopes = self.add_weight(
            shape=[4],
            initializer=scale_init,
            trainable=True,
            name=f"{self.name}_scales",
        )
        self.shifts = self.add_weight(
            shape=[4], initializer="zeros", trainable=True, name=f"{self.name}_shifts"
        )
        self.smoothing = self.add_weight(
            shape=[2],
            initializer="zeros",
            trainable=True,
            name=f"{self.name}_smoothness",
        )
        self.bin_widths = self.add_weight(
            shape=[num_bins],
            trainable=True,
            name=f"{self.name}_w",
        )
        self.bin_heights = self.add_weight(
            shape=[num_bins],
            trainable=True,
            name=f"{self.name}_h",
        )
        self.knot_slopes = self.add_weight(
            shape=[num_bins - 1],
            initializer="zeros",
            trainable=True,
            name=f"{self.name}_s",
        )
        self.range_min = range_min
        self.range_max = range_max

    def build_bijector(self):
        slopes = softplus(self.slopes) if self.monotonic else self.slopes
        shifts = self.shifts
        smoothing = softplus(self.smoothing)

        range = self.range_max - self.range_min
        bin_widths = range * tf.nn.softmax(self.bin_widths)
        bin_heights = range * tf.nn.softmax(self.bin_heights)
        knot_slopes = softplus(self.knot_slopes)

        penalty = tf.reduce_sum(
            diff(tf.math.log(knot_slopes)) ** 2 / bin_widths[..., 1:-1]
        )

        bijector_list = [
            SmoothPieceWiseLinear(
                left_slope=slopes[2],
                right_slope=slopes[3],
                x_shift=shifts[2],
                y_shift=shifts[3],
                k=smoothing[1],
            ),
            tfb.RationalQuadraticSpline(
                bin_widths=bin_widths,
                bin_heights=bin_heights,
                knot_slopes=knot_slopes,
                range_min=self.range_min,
            ),
            SmoothPieceWiseLinear(
                left_slope=slopes[0],
                right_slope=slopes[1],
                x_shift=shifts[0],
                y_shift=shifts[1],
                k=smoothing[0],
            ),
        ]

        if self.log_input:
            bijector_list += [tfb.Log(), tfb.Shift(1e-3)]

        bijector = tfb.Chain(bijector_list)
        return bijector, penalty

    def call(self, x, inverse=False):
        bijector, penalty = self.build_bijector()
        if inverse:
            return bijector.inverse(x)
        else:
            self.add_loss(self.lamda * penalty)
            return bijector.forward(x)

    def get_config(self):
        return {
            **super().get_config(),
            "num_bins": self.num_bins,
            "monotonic": self.monotonic,
            "range_min": self.range_min,
            "range_max": self.range_max,
            "lamda": self.lamda,
            "log_input": self.log_input,
        }


class ConditionalRQSLayer(layers.Layer):
    def __init__(
        self,
        num_bins: int,
        cond_dim: int,
        hidden_layer_sizes: list[int] = [32, 32],
        activation: str = "elu",
        monotonic: bool = True,
        lamda: float = 1e-4,
        range_min: float = -1.0,
        range_max: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_bins = num_bins
        self.cond_dim = cond_dim
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.monotonic = monotonic
        self.lamda = lamda
        self.range_min = range_min
        self.range_max = range_max
        self.mlp = self.init_mlp()

    def init_mlp(self):
        cond_inputs = Input((self.cond_dim,))
        x = cond_inputs
        for hls in self.hidden_layer_sizes:
            x = layers.Dense(
                hls,
                activation=self.activation,
            )(x)

        x = layers.Dense(
            3 * self.num_bins + 9,
            kernel_initializer="zeros",
            bias_initializer="zeros",
        )(x)

        slopes, shifts, smoothing, widths, heights, knot_slopes = tf.split(
            x,
            [4, 4, 2, self.num_bins, self.num_bins, self.num_bins - 1],
            axis=-1,
        )

        return Model(
            inputs=cond_inputs,
            outputs=[slopes, shifts, smoothing, widths, heights, knot_slopes],
            name=f"{self.name}_mlp",
        )

    def call(self, inputs):
        x, c = inputs
        B = tf.shape(x)[0]
        N = tf.shape(x)[1]
        x_ = tf.reshape(x, [B * N])
        c_ = tf.reshape(c, [B * N, self.cond_dim])

        slopes, shifts, smoothing, bin_widths, bin_heights, knot_slopes = self.mlp(c_)

        if self.monotonic:
            slopes = softplus(slopes)

        smoothing = softplus(smoothing)

        range = self.range_max - self.range_min
        bin_widths = range * tf.nn.softmax(bin_widths)
        bin_heights = range * tf.nn.softmax(bin_heights)
        knot_slopes = softplus(knot_slopes)

        penalty = tf.reduce_sum(
            diff(tf.math.log(knot_slopes)) ** 2 / bin_widths[..., 1:-1], axis=-1
        )
        self.add_loss(self.lamda * tf.reduce_mean(penalty))

        rqs = tfb.RationalQuadraticSpline(
            bin_widths=bin_widths,
            bin_heights=bin_heights,
            knot_slopes=knot_slopes,
            range_min=self.range_min,
        )

        y_ = (
            smooth_piecewise_linear(
                x_ - shifts[:, 0],
                left_slope=slopes[:, 0],
                right_slope=slopes[:, 1],
                k=smoothing[:, 0],
            )
            + shifts[:, 1]
        )
        y_ = rqs.forward(y_)
        y_ = (
            smooth_piecewise_linear(
                y_ - shifts[:, 2],
                left_slope=slopes[:, 2],
                right_slope=slopes[:, 3],
                k=smoothing[:, 1],
            )
            + shifts[:, 3]
        )

        y = tf.reshape(y_, tf.shape(x))
        return y

    def get_config(self):
        return {
            **super().get_config(),
            "num_bins": self.num_bins,
            "cond_dim": self.cond_dim,
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "activation": self.activation,
            "monotonic": self.monotonic,
            "lamda": self.lamda,
            "range_min": self.range_min,
            "range_max": self.range_max,
        }
