import tensorflow as tf
from tensorflow.keras import backend, regularizers


class PushMaxWeightToUnity(regularizers.Regularizer):
    def __init__(self, strength: float, axis: int = (1, 2, 3), **kwargs):
        super().__init__(**kwargs)
        self.strength = strength
        self.axis = axis

    def __call__(self, w):
        penalty = backend.abs(backend.max(w, axis=self.axis) - 1.0)
        return self.strength * backend.mean(penalty)

    def get_config(self):
        return {"strength": self.strength, "axis": self.axis}


class SparsityPenalty(regularizers.Regularizer):
    def __init__(
        self,
        max_active_fraction: float,
        strength: float = 1.0,
        k: float = 1.0,
        axis: int = (1, 2, 3),
        scale: float = 50,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.strength = strength
        self.max_active_fraction = max_active_fraction
        self.k = k
        self.axis = axis
        self.scale = scale

    def __call__(self, inputs):
        is_active = tf.nn.tanh(self.scale * tf.abs(inputs))
        active_fraction = tf.reduce_mean(is_active, axis=self.axis)
        penalty = tf.nn.relu(active_fraction - self.max_active_fraction) / (
            1 - self.max_active_fraction
        )
        mean_penalty = tf.reduce_mean(penalty**self.k)
        return self.strength * mean_penalty

    def get_config(self):
        return {
            **super().get_config(),
            "max_active_fraction": self.max_active_fraction,
            "strength": self.strength,
            "k": self.k,
            "axis": self.axis,
            "scale": self.scale,
        }
