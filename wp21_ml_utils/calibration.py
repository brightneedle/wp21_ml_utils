import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Concatenate
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.initializers import RandomNormal

from wp21_ml_utils.layers import MonoDense
from wp21_ml_utils.utils import unpack_momenta


@register_keras_serializable("wp21_ml_utils")
class CalibrationMLP(Layer):
    """
    Learnable particle-level transverse momentum calibration network.

    This layer applies a small monotonic or standard MLP to particle kinematics
    in order to predict a calibrated transverse momentum and an auxiliary
    gating factor that suppresses unreliable low-momentum inputs.

    The network operates on (pT, η, φ) inputs and produces a corrected
    transverse momentum according to:

        pT^calib = σ(g(x)) · exp(clipped(f(x)))

    where:
        - f(x) is the regression head predicting log(pT)
        - g(x) is a gating head predicting per-particle reliability
        - σ is a sigmoid ensuring gating values in [0, 1]

    The final output preserves η and φ while replacing pT with the calibrated
    value.

    When `monotonic=True`, the model uses monotonic dense layers
    (MonoDense) to enforce physically motivated constraints, ensuring that
    the output increases monotonically with respect to selected inputs.

    Architecture:
        1. Input unpacking into (pT, η, φ)
        2. Feature construction: [pT, |η|]
        3. Log transform with numerical stabilization
        4. Fully connected hidden layers
        5. Two output heads:
           - log pT regression head
           - gating head
        6. Gated exponential reconstruction of pT
        7. Concatenation of (pT_calib, η, φ)

    Input:
        Tensor of shape (B, N, 3), where the last dimension corresponds to
        (pT, η, φ) for each particle.

    Output:
        Tensor of shape (B, N, 3) with calibrated transverse momentum and
        unchanged angular coordinates.

    Parameters
    ----------
    hidden_layer_sizes : tuple[int], default=(64, 64)
        Number of units in each hidden layer of the MLP.

    hidden_activation : str, default="softplus"
        Activation function used in hidden layers.

    eps : float, default=1e-3
        Small constant added for numerical stability in log transforms and
        masking of low-pT inputs.

    monotonic : bool, default=True
        If True, uses monotonic dense layers (MonoDense) to enforce physically
        constrained behavior in selected inputs; otherwise uses standard Dense
        layers.
    """

    def __init__(
        self,
        hidden_layer_sizes=(64, 64),
        hidden_activation="softplus",
        eps=1e-3,
        monotonic=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_layer_sizes = hidden_layer_sizes
        self.hidden_activation = hidden_activation
        self.eps = eps
        self.monotonic = monotonic

    def _get_layer(
        self,
        nodes,
        activation=None,
        monotonicity_indicator=None,
    ):
        if self.monotonic:
            kwargs = dict(
                activation=activation,
                kernel_initializer=RandomNormal(stddev=1e-2),
            )

            if monotonicity_indicator is not None:
                kwargs["monotonicity_indicator"] = monotonicity_indicator

            return MonoDense(nodes, **kwargs)
        else:
            return Dense(
                nodes,
                activation=activation,
            )

    def build(self, input_shape):
        self.hidden_layers = []

        for i, hls in enumerate(self.hidden_layer_sizes):
            if i == 0:
                layer = self._get_layer(
                    hls,
                    activation=self.hidden_activation,
                    monotonicity_indicator=[1, 0],
                )
            else:
                layer = self._get_layer(
                    hls,
                    activation=self.hidden_activation,
                )

            self.hidden_layers.append(layer)

        self.log_pt_head = self._get_layer(1)
        self.gate_head = self._get_layer(1)

        self.concat = Concatenate(axis=-1)

        super().build(input_shape)

    def call(self, momenta):
        pt, eta, phi = unpack_momenta(momenta)

        x = self.concat([pt, tf.math.abs(eta)])
        x = tf.math.log(x + self.eps)

        for layer in self.hidden_layers:
            x = layer(x)

        log_pt = self.log_pt_head(x)
        gate_logits = self.gate_head(x)

        calib_pt = tf.exp(tf.clip_by_value(log_pt, -10, 10))

        gate = tf.nn.sigmoid(gate_logits)

        gated_calib_pt = tf.where(
            pt > self.eps,
            gate * calib_pt,
            0,
        )

        return self.concat([gated_calib_pt, eta, phi])
