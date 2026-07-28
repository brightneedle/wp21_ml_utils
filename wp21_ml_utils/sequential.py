from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Layer,
    Dense,
    BatchNormalization,
    Activation,
    Dropout,
)
from keras.saving import register_keras_serializable
from hgq.layers import QDense, QBatchNormDense


@register_keras_serializable(package="custom")
class DenseNeuralNetwork(Layer):
    """
    Configurable stack of dense layers with optional HGQ and regularisation.

    Builds a feed-forward network from ``hidden_layer_sizes``. Standard Keras
    ``Dense`` layers are used by default; when ``use_hgq`` is enabled, the
    corresponding HGQ ``QDense`` or ``QBatchNormDense`` layers are used
    instead. Batch normalisation and dropout are applied after every hidden
    layer when requested.

    The layer is registered for Keras serialisation under the ``custom``
    package, so it can be saved as part of a model configuration.

    Parameters
    ----------
    hidden_layer_sizes : list[int]
        Number of units in each hidden layer. An empty list produces an
        identity sequential network.
    activation : str
        Activation applied to each hidden layer.
    use_hgq : bool
        Whether to use HGQ quantised dense layers.
    dropout : float, default=0.0
        Dropout rate applied after each hidden layer. Set to zero to disable
        dropout.
    batch_norm : bool, default=False
        Whether to apply batch normalisation to each hidden layer. With
        standard Keras layers, normalisation is followed by ``activation``.
    """

    def __init__(
        self,
        hidden_layer_sizes: list[int],
        activation: str,
        use_hgq: bool,
        dropout: float = 0.0,
        batch_norm: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_layer_sizes = list(hidden_layer_sizes)
        self.activation = activation
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.use_hgq = use_hgq

        self.nn = Sequential()
        for units in self.hidden_layer_sizes:
            if self.use_hgq:
                if self.batch_norm:
                    self.nn.add(QBatchNormDense(units, activation=self.activation))
                else:
                    self.nn.add(QDense(units, activation=self.activation))

            else:
                if self.batch_norm:
                    self.nn.add(Dense(units))
                    self.nn.add(BatchNormalization())
                    self.nn.add(Activation(self.activation))
                else:
                    self.nn.add(Dense(units, activation=self.activation))

            if self.dropout > 0:
                self.nn.add(Dropout(self.dropout))

    def call(self, inputs, training=None):
        return self.nn(inputs, training=training)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_layer_sizes": self.hidden_layer_sizes,
                "activation": self.activation,
                "dropout": self.dropout,
                "batch_norm": self.batch_norm,
                "use_hgq": self.use_hgq,
            }
        )
        return config
