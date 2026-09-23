import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.backend import set_image_data_format

set_image_data_format("channels_last")


@register_keras_serializable("wp21_ml_utils")
class ReflectionSymmetry(tf.keras.constraints.Constraint):
    """Constrain a 2D convolution kernel to be reflection symmetric.

    The first two axes of a Keras convolution kernel are its spatial axes.
    Averaging over both spatial reflections projects the kernel onto the
    subspace invariant under independent eta and phi reflections.
    """

    def __call__(self, kernel: tf.Tensor) -> tf.Tensor:
        return (
            kernel
            + tf.reverse(kernel, axis=[0])
            + tf.reverse(kernel, axis=[1])
            + tf.reverse(kernel, axis=[0, 1])
        ) / 4.0
