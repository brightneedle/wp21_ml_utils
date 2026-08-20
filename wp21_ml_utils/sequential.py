from tensorflow.keras.layers import (
    Dense,
    BatchNormalization,
    Activation,
    Dropout,
    Conv2D,
    MaxPooling2D,
    AveragePooling2D,
)
from tensorflow.keras.regularizers import L1L2, L1, L2
from hgq.layers import QDense, QBatchNormDense, QConv2D


class DenseLayers:
    """
    Configurable stack of dense layers with optional HGQ and regularisation.

    Builds a feed-forward network from ``hidden_layer_sizes``. Standard Keras
    ``Dense`` layers are used by default; when ``use_hgq`` is enabled, the
    corresponding HGQ ``QDense`` or ``QBatchNormDense`` layers are used
    instead. Batch normalisation and dropout are applied after every hidden
    layer when requested.

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
    l2_penalty : float, default=0.0
        L2 regularisation penalty applied to the kernel and bias of each
        dense layer. Set to zero to disable L2 regularisation.
    l1_penalty : float, default=0.0
        L1 regularisation penalty applied to the kernel and bias of each
        dense layer. Set to zero to disable L1 regularisation.
    name : str or None, default=None
        Optional prefix added to the generated layer names.
    """

    def __init__(
        self,
        hidden_layer_sizes: list[int],
        activation: str,
        use_hgq: bool,
        dropout: float = 0.0,
        batch_norm: bool = False,
        l2_penalty: float = 0.0,
        l1_penalty: float = 0.0,
        name: str = None,
    ):
        self.hidden_layer_sizes = list(hidden_layer_sizes)
        self.activation = activation
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.use_hgq = use_hgq
        self.l2_penalty = l2_penalty
        self.l1_penalty = l1_penalty

        if (self.l1_penalty > 0) & (self.l2_penalty > 0):
            self.regularizer = L1L2(l1=self.l1_penalty, l2=self.l2_penalty)
        elif self.l1_penalty > 0:
            self.regularizer = L1(l1=self.l1_penalty)
        elif self.l2_penalty > 0:
            self.regularizer = L2(l2=self.l2_penalty)
        else:
            self.regularizer = None

        self.layer_list = []
        for units in self.hidden_layer_sizes:
            if self.use_hgq:
                if self.batch_norm:
                    self.layer_list.append(
                        QBatchNormDense(
                            units,
                            activation=self.activation,
                            kernel_regularizer=self.regularizer,
                            bias_regularizer=self.regularizer,
                        )
                    )
                else:
                    self.layer_list.append(
                        QDense(
                            units,
                            activation=self.activation,
                            kernel_regularizer=self.regularizer,
                            bias_regularizer=self.regularizer,
                        )
                    )

            else:
                if self.batch_norm:
                    self.layer_list.append(
                        Dense(
                            units,
                            kernel_regularizer=self.regularizer,
                            bias_regularizer=self.regularizer,
                        )
                    )
                    self.layer_list.append(BatchNormalization())
                    self.layer_list.append(Activation(self.activation))
                else:
                    self.layer_list.append(
                        Dense(
                            units,
                            activation=self.activation,
                            kernel_regularizer=self.regularizer,
                            bias_regularizer=self.regularizer,
                        )
                    )

            if self.dropout > 0:
                self.layer_list.append(Dropout(self.dropout))

        if name:
            for layer in self.layer_list:
                layer.name = f"{name}_{layer.name}"

    def __call__(self, inputs):
        x = inputs
        for layer in self.layer_list:
            x = layer(x)
        return x


class Conv2DPoolingLayers:
    """
    Configurable stack of two-dimensional convolution and pooling layers.

    Builds a convolutional network from ``filter_sizes``, ``kernel_sizes``,
    and ``pooling_sizes``. Standard Keras ``Conv2D`` layers are used by
    default; when ``use_hgq`` is enabled, the corresponding HGQ ``QConv2D``
    layers are used instead. A max- or average-pooling layer is added after
    each convolution when its matching pooling size is greater than one, and
    dropout is applied after each block when requested.

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
    l2_penalty : float, default=0.0
        L2 regularisation penalty applied to the kernel and bias of each
        convolutional layer. Set to zero to disable L2 regularisation.
    l1_penalty : float, default=0.0
        L1 regularisation penalty applied to the kernel and bias of each
        convolutional layer. Set to zero to disable L1 regularisation.
    name : str or None, default=None
        Optional prefix added to the generated layer names.
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
        l2_penalty: float = 0.0,
        l1_penalty: float = 0.0,
        name=None,
    ):
        self.filter_sizes = list(filter_sizes)
        self.kernel_sizes = list(kernel_sizes)
        self.pooling_sizes = list(pooling_sizes)
        self.pooling = pooling
        self.padding = padding
        self.activation = activation
        self.use_hgq = use_hgq
        self.dropout = dropout
        self.l2_penalty = l2_penalty
        self.l1_penalty = l1_penalty

        if (self.l1_penalty > 0) & (self.l2_penalty > 0):
            self.regularizer = L1L2(l1=self.l1_penalty, l2=self.l2_penalty)
        elif self.l1_penalty > 0:
            self.regularizer = L1(l1=self.l1_penalty)
        elif self.l2_penalty > 0:
            self.regularizer = L2(l2=self.l2_penalty)
        else:
            self.regularizer = None

        if not (
            len(self.filter_sizes) == len(self.kernel_sizes) == len(self.pooling_sizes)
        ):
            raise ValueError(
                "filter_sizes, kernel_sizes, and pooling_sizes must have the same length."
            )

        self.layer_list = []
        for filters, kernel_size, pool_size in zip(
            self.filter_sizes, self.kernel_sizes, self.pooling_sizes
        ):
            if self.use_hgq:
                self.layer_list.append(
                    QConv2D(
                        filters=filters,
                        kernel_size=kernel_size,
                        padding=self.padding,
                        activation=self.activation,
                        kernel_regularizer=self.regularizer,
                        bias_regularizer=self.regularizer,
                    )
                )
            else:
                self.layer_list.append(
                    Conv2D(
                        filters=filters,
                        kernel_size=kernel_size,
                        padding=self.padding,
                        activation=self.activation,
                        kernel_regularizer=self.regularizer,
                        bias_regularizer=self.regularizer,
                    )
                )

            if pool_size > 1:
                if self.pooling == "max":
                    self.layer_list.append(MaxPooling2D(pool_size=pool_size))
                elif self.pooling == "average":
                    self.layer_list.append(AveragePooling2D(pool_size=pool_size))
                else:
                    raise ValueError(
                        f"Invalid pooling type: {self.pooling}. Must be 'max' or 'average'."
                    )

            if self.dropout > 0:
                self.layer_list.append(Dropout(self.dropout))

        if name:
            for layer in self.layer_list:
                layer.name = f"{name}_{layer.name}"

    def __call__(self, inputs):
        x = inputs
        for layer in self.layer_list:
            x = layer(x)
        return x
