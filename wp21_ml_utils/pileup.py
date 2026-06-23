import tensorflow as tf
from tensorflow.keras.layers import Layer, MaxPooling2D, AveragePooling2D, Concatenate
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.regularizers import Regularizer
from tensorflow.keras.initializers import Constant
from tensorflow.types.experimental import TensorLike
import numpy as np

from wp21_ml_utils.utils import take_median, init_dense_layer
from wp21_ml_utils.layers import (
    TowerEtaPhiLayer,
    EtaPhiPadding,
    SymmetricDepthwiseConv2D,
)


@register_keras_serializable("wp21_ml_utils")
class TowerSoftKiller(Layer):
    """
    SoftKiller pileup-suppression layer for calorimeter tower images.

    Implements the SoftKiller algorithm by dividing the detector image into
    patches, computing the maximum tower transverse energy in each patch, and
    determining the median of these local maxima. Towers with energy below
    the median threshold are removed.

    This procedure suppresses diffuse low-energy background activity while
    preserving localised high-energy deposits.

    Input:
        Tensor of shape (B, E, P, C), typically representing calorimeter
        tower transverse energies.

    Output:
        Tensor of identical shape with low-energy towers set to zero.

    Parameters
    ----------
    patch_size : tuple[int, int], default=(5, 8)
        Size of the η–φ patches used to determine the SoftKiller threshold.
    """

    def __init__(self, patch_size=(5, 8), **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def build(self, input_shape):
        self.max_pooling_layer = MaxPooling2D(
            pool_size=self.patch_size, padding="valid"
        )
        super().build(input_shape)

    def call(self, image: TensorLike) -> tf.Tensor:
        local_max = self.max_pooling_layer(image)
        median_max = take_median(local_max)
        sk_towers = tf.where(image > median_max, image, 0)
        return sk_towers

    def get_config(self):
        return {
            **super().get_config(),
            "patch_size": self.patch_size,
        }


@register_keras_serializable("wp21_ml_utils")
class TowerSoftKillerWithAreaCorrection(Layer):
    """
    Area-corrected SoftKiller pileup-suppression layer.

    Performs an initial event-by-event background-density subtraction before
    applying the standard SoftKiller procedure.

    The background density ρ is estimated from the median transverse energy
    density of detector patches:

        ρ = median(E_T^patch / A_patch)

    and subtracted from each tower according to its geometric area. The
    resulting image is then processed using the SoftKiller algorithm to remove
    residual low-energy background activity.

    This approach combines area-based pileup subtraction with SoftKiller
    thresholding and is particularly useful in high-occupancy environments.

    Input:
        Tensor of shape (B, E, P, C), typically representing calorimeter
        tower transverse energies.

    Output:
        Tensor of identical shape with background-subtracted and SoftKiller-
        filtered tower energies.

    Parameters
    ----------
    patch_size : tuple[int, int], default=(5, 8)
        Size of the η–φ patches used for both background-density estimation
        and SoftKiller threshold determination.

    pixel_area : float, default=0.1 * π / 32
        Geometric area associated with a single calorimeter tower in η–φ
        space.
    """

    def __init__(self, patch_size=(5, 8), pixel_area=0.1 * np.pi / 32, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.pixel_area = pixel_area

    def build(self, input_shape):
        self.pixels_per_patch = tf.cast(
            self.patch_size[0] * self.patch_size[1], dtype=tf.float32
        )
        self.max_pooling_layer = MaxPooling2D(
            pool_size=self.patch_size, padding="valid"
        )
        self.avg_pooling_layer = AveragePooling2D(pool_size=self.patch_size)

        super().build(input_shape)

    def call(self, image: TensorLike) -> tf.Tensor:
        # rho subtraction
        sum_patch_Et = self.pixels_per_patch * self.avg_pooling_layer(image)
        rho = take_median(sum_patch_Et / (self.pixel_area * self.pixels_per_patch))
        image_rho_sub = tf.maximum(image - rho * self.pixel_area, 0)

        # normal softkiller
        local_max = self.max_pooling_layer(image_rho_sub)
        median_max = take_median(local_max)
        sk_towers = tf.where(image_rho_sub > median_max, image_rho_sub, 0)
        return sk_towers

    def get_config(self):
        return {
            **super().get_config(),
            "patch_size": self.patch_size,
            "pixel_area": self.pixel_area,
        }


@register_keras_serializable("wp21_ml_utils")
class PileupCNN(Layer):
    """
    Learnable pileup-suppression layer based on local calorimeter topology.

    Applies a depthwise convolution over a local η–φ neighbourhood to extract
    tower-level features, optionally augments these with the tower pseudorapidity,
    and predicts a multiplicative weight for each input tower. The output is
    obtained by scaling the original tower energies with the predicted weights:

        E_T^out = w · E_T^in

    where the weights are constrained to the range [0, 1] through a
    hard-sigmoid activation. Towers identified as likely pileup receive
    weights close to zero, while towers associated with hard-scatter activity
    retain weights close to one.

    The architecture consists of:

    1. η–φ-aware padding.
    2. Symmetric depthwise convolution for local feature extraction.
    3. Optional inclusion of log(|η|).
    4. A stack of dense hidden layers.
    5. A per-channel weight-prediction head.

    Input:
        Tensor of shape (B, E, P, C), typically representing calorimeter
        tower transverse energies.

    Output:
        Tensor of identical shape containing pileup-suppressed tower energies.

    Parameters
    ----------
    size : int, default=3
        Size of the local convolutional neighbourhood.

    depth_multiplier : int, default=4
        Number of depthwise convolution filters produced per input channel.

    hidden_layer_sizes : list[int], default=[32, 32]
        Widths of the fully connected hidden layers used to predict tower
        weights.

    use_hgq : bool, default=False
        Whether to use HGQ-compatible layers and quantisation-aware
        implementations.

    init_as_layer_sum : bool, default=True
        Initialise the output weight head such that the layer initially
        behaves approximately like the identity transformation.

    with_abseta : bool, default=True
        If True, append log(|η|) information to the learned local features.

    weight_regulariser : keras.regularizers.Regularizer, optional
        Regulariser applied to the predicted tower weights.
    """

    def __init__(
        self,
        size: int = 3,
        depth_multiplier: int = 4,
        hidden_layer_sizes: list[int] = [32, 32],
        use_hgq: bool = False,
        init_as_layer_sum: bool = True,
        with_abseta: bool = True,
        weight_regulariser: Regularizer = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.size = size
        self.depth_multiplier = depth_multiplier
        self.hidden_layer_sizes = hidden_layer_sizes
        self.use_hgq = use_hgq
        self.init_as_layer_sum = init_as_layer_sum
        self.with_abseta = with_abseta
        self.weight_regulariser = weight_regulariser

    def build(self, input_shape):
        channels = input_shape[-1]

        self.padding = EtaPhiPadding(self.size // 2)

        self.depthwise_conv = SymmetricDepthwiseConv2D(
            kernel_size=self.size,
            depth_multiplier=self.depth_multiplier,
            use_hgq=self.use_hgq,
            activation="relu",
        )

        if self.with_abseta:
            self.eta_phi = TowerEtaPhiLayer()
            self.concat = Concatenate(axis=-1)

        self.hidden_layers = [
            init_dense_layer(
                hls,
                activation="relu",
                use_hgq=self.use_hgq,
            )
            for hls in self.hidden_layer_sizes
        ]

        self.weight_head = init_dense_layer(
            channels,
            activation="hard_sigmoid",
            kernel_initializer=(
                "zeros" if self.init_as_layer_sum else "glorot_uniform"
            ),
            bias_initializer=(Constant(3.0) if self.init_as_layer_sum else "zeros"),
            activity_regularizer=self.weight_regulariser,
            use_hgq=self.use_hgq,
        )

        super().build(input_shape)

    def call(self, inputs):
        x = self.padding(inputs)
        x = self.depthwise_conv(x)

        if self.with_abseta:
            eta, _ = self.eta_phi(inputs)
            log_abseta = tf.math.log(tf.math.abs(eta) + 1e-3)
            x = self.concat([x, log_abseta])

        for layer in self.hidden_layers:
            x = layer(x)

        w = self.weight_head(x)

        return w * inputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "size": self.size,
                "depth_multiplier": self.depth_multiplier,
                "hidden_layer_sizes": self.hidden_layer_sizes,
                "weight_regulariser": self.weight_regulariser,
                "init_as_layer_sum": self.init_as_layer_sum,
                "use_hgq": self.use_hgq,
                "with_abseta": self.with_abseta,
            }
        )
        return config
