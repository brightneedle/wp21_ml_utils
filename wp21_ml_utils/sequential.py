from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Layer,
    Dense,
    BatchNormalization,
    Activation,
    Dropout,
    Conv2D,
    MaxPooling2D,
    AveragePooling2D,
)
from tensorflow.keras.utils import register_keras_serializable
from hgq.layers import QDense, QBatchNormDense, QConv2D


@register_keras_serializable("wp21_ml_utils")
class DenseLayers(Layer):
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


@register_keras_serializable("wp21_ml_utils")
class Conv2DPoolingLayers(Layer):
    """
    Configurable stack of two-dimensional convolution and pooling layers.

    Builds a convolutional network from ``filter_sizes``, ``kernel_sizes``,
    and ``pooling_sizes``. Standard Keras ``Conv2D`` layers are used by
    default; when ``use_hgq`` is enabled, the corresponding HGQ ``QConv2D``
    layers are used instead. A max- or average-pooling layer is added after
    each convolution when its matching pooling size is greater than one, and
    dropout is applied after each block when requested.

    The layer is registered for Keras serialisation under the
    ``wp21_ml_utils`` package, so it can be saved as part of a model
    configuration.

    Parameters
    ----------
    filter_sizes : list[int]
        Number of filters in each convolutional layer.
    kernel_sizes : list[int]
        Kernel size for each convolutional layer. Must have the same length
        as ``filter_sizes``.
    pooling_sizes : list[int]
        Pooling size after each convolutional layer. A value of one disables
        pooling for that layer. Must have the same length as ``filter_sizes``.
    activation : str
        Activation applied to each convolutional layer.
    pooling : str
        Pooling operation to use: ``max`` or ``average``.
    use_hgq : bool
        Whether to use HGQ quantised convolutional layers.
    padding : str, default="valid"
        Padding mode passed to each convolutional layer.
    dropout : float, default=0.0
        Dropout rate applied after each convolution/pooling block. Set to
        zero to disable dropout.
    """

    def __init__(
        self,
        filter_sizes: list[int],
        kernel_sizes: list[int],
        pooling_sizes: list[int],
        activation: str,
        pooling: str,
        use_hgq: bool,
        padding: str = "valid",
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.filter_sizes = list(filter_sizes)
        self.kernel_sizes = list(kernel_sizes)
        self.pooling_sizes = list(pooling_sizes)
        self.pooling = pooling
        self.padding = padding
        self.activation = activation
        self.dropout = dropout
        self.use_hgq = use_hgq

        if not (
            len(self.filter_sizes) == len(self.kernel_sizes) == len(self.pooling_sizes)
        ):
            raise ValueError(
                "filter_sizes, kernel_sizes, and pooling_sizes must have the same length."
            )

        self.nn = Sequential()

        for filters, kernel_size, pool_size in zip(
            self.filter_sizes, self.kernel_sizes, self.pooling_sizes
        ):
            if self.use_hgq:
                self.nn.add(
                    QConv2D(
                        filters=filters,
                        kernel_size=kernel_size,
                        padding=self.padding,
                        activation=self.activation,
                    )
                )
            else:
                self.nn.add(
                    Conv2D(
                        filters=filters,
                        kernel_size=kernel_size,
                        padding=self.padding,
                        activation=self.activation,
                    )
                )

            if pool_size > 1:
                if self.pooling == "max":
                    self.nn.add(MaxPooling2D(pool_size=pool_size))
                elif self.pooling == "average":
                    self.nn.add(AveragePooling2D(pool_size=pool_size))
                else:
                    raise ValueError(
                        f"Invalid pooling type: {self.pooling}. Must be 'max' or 'average'."
                    )

            if self.dropout > 0:
                self.nn.add(Dropout(self.dropout))

    def call(self, inputs, training=None):
        return self.nn(inputs, training=training)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filter_sizes": self.filter_sizes,
                "kernel_sizes": self.kernel_sizes,
                "pooling_sizes": self.pooling_sizes,
                "pooling": self.pooling,
                "padding": self.padding,
                "activation": self.activation,
                "dropout": self.dropout,
                "use_hgq": self.use_hgq,
            }
        )
        return config
