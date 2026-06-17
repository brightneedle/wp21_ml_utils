from typing import Union

import tensorflow as tf
from tensorflow.keras.regularizers import Regularizer
from tensorflow.types.experimental import TensorLike


class PushMaxWeightToUnity(Regularizer):
    """
    Regulariser that encourages the maximum weight value to equal one.

    Computes a penalty proportional to the deviation of the maximum weight
    along the specified axes from unity:

        penalty = |max(weight) - 1|

    This can be useful when learning normalised weighting maps, attention
    masks, calibration factors, or detector response templates where the
    largest response is expected to be one.

    Parameters
    ----------
    strength : float
        Multiplicative regularisation coefficient.

    axis : int or tuple[int], default=(1, 2, 3)
        Axes over which the maximum value is computed before averaging the
        penalty across the remaining dimensions.
    """

    def __init__(
        self, strength: float, axis: Union[int, tuple[int]] = (1, 2, 3), **kwargs
    ):
        super().__init__(**kwargs)
        self.strength = float(strength)
        self.axis = axis

    def __call__(self, weight: TensorLike) -> tf.Tensor:
        penalty = tf.math.abs(tf.reduce_max(weight, axis=self.axis) - 1.0)
        return self.strength * tf.reduce_mean(penalty)

    def get_config(self):
        return {"strength": self.strength, "axis": self.axis}


class SparsityPenalty(Regularizer):
    """
    Sparsity-promoting regulariser.

    Encourages tensors to contain no more than a specified fraction of active
    elements. Activity is estimated using a smooth tanh-based approximation to
    a binary activation indicator:

        active ≈ tanh(T · |x|)

    No penalty is applied while the active fraction remains below the target
    threshold. Above the threshold, a scaled ReLU penalty is applied and may
    optionally be raised to a higher power.

    This regulariser is useful for encouraging sparse feature maps, detector
    images, latent representations, or object occupancy predictions.

    Parameters
    ----------
    strength : float
        Multiplicative regularisation coefficient.

    max_active_fraction : float, default=0
        Maximum allowed fraction of active elements before penalties are
        applied.

    k : int, default=1
        Exponent applied to the sparsity penalty. Larger values increasingly
        penalise large violations.

    axis : int or tuple[int], default=(1, 2, 3)
        Axes over which the active fraction is computed.

    T : float, default=50.0
        Temperature controlling the sharpness of the activity indicator.
        Larger values more closely approximate a binary activation mask.
    """

    def __init__(
        self,
        strength: float,
        max_active_fraction: float = 0,
        k: int = 1,
        axis: Union[int, tuple[int]] = (1, 2, 3),
        T: float = 50.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.strength = float(strength)
        self.max_active_fraction = float(max_active_fraction)
        self.k = int(k)
        self.axis = axis
        self.T = float(T)

    def __call__(self, inputs: TensorLike) -> tf.Tensor:
        is_active = tf.nn.tanh(self.T * tf.abs(inputs))
        active_fraction = tf.reduce_mean(is_active, axis=self.axis)
        penalty = tf.nn.relu(active_fraction - self.max_active_fraction) / (
            1 - self.max_active_fraction
        )
        mean_penalty = tf.reduce_mean(penalty**self.k)
        return self.strength * mean_penalty

    def get_config(self):
        return {
            "max_active_fraction": self.max_active_fraction,
            "strength": self.strength,
            "k": self.k,
            "axis": self.axis,
            "T": self.T,
        }
