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
