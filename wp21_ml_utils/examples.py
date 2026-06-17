from tensorflow.keras import Input, Model, ops, initializers
from tensorflow.keras.regularizers import Regularizer
from tensorflow.keras.layers import Multiply, Concatenate

from wp21_ml_utils.converters import ImageToVectors
from wp21_ml_utils.layers import (
    EtaPhiPadding,
    SymmetricDepthwiseConv2D,
    TowerEtaPhiLayer,
    SlidingConeSum,
    LocalMaxMask,
)
from wp21_ml_utils.utils import init_dense_layer


def ConeJets(
    input_shape: tuple[int],
    kernel_size: int = 9,
    min_pt: float = 0,
    max_jets: int = 20,
    shape: str = "circle",
    radius: float = None,
    **kwargs,
) -> Model:
    """
    Keras model that performs cone-based jet reconstruction from calorimeter images.

    This model implements a simplified jet-finding pipeline inspired by cone clustering
    algorithms used in high-energy physics. It identifies localized energy deposits,
    selects stable seed regions, and converts them into a fixed-size set of jet vectors.

    Pipeline:
        1. Compute local energy sums using a sliding cone window.
        2. Detect local maxima in the original image as jet seeds.
        3. Mask cone-summed responses using the seed locations.
        4. Convert the resulting sparse image into ranked (pt, eta, phi) jet vectors.

    Args:
        input_shape (tuple[int]):
            Shape of the input calorimeter image excluding batch dimension,
            typically (eta_bins, phi_bins, channels).
        kernel_size (int):
            Size of the sliding cone window used to aggregate local energy.
        min_pt (float):
            Minimum transverse momentum threshold below which jets are discarded.
        max_jets (int):
            Maximum number of jets to return in the output representation.
        shape (str):
            Geometry of the cone window (e.g. "circle" or "square").
        radius (float, optional):
            Optional physical radius parameter for the cone shape (if supported).
        **kwargs:
            Additional keyword arguments passed to `tf.keras.Model`.

    Returns:
        tf.keras.Model:
            A Keras model mapping calorimeter images to a fixed-length tensor of
            jet vectors with shape (batch, max_jets, 3), where each vector is
            (pt, eta, phi).
    """

    image = Input(shape=input_shape)

    cone_sums = SlidingConeSum(
        kernel_size=kernel_size,
        shape=shape,
        radius=radius,
        name="cone_sum",
    )(image)

    # get good seed mask
    local_max_seed_mask = LocalMaxMask(
        kernel_size=kernel_size,
        shape=shape,
        name="local_max",
    )(image)

    # mask cone sums
    masked_cone_sums = ops.where(local_max_seed_mask, cone_sums, 0)

    # convert to 3-vectors
    jet_vectors = ImageToVectors(max_vectors=max_jets, min_pt=min_pt)(masked_cone_sums)

    return Model(inputs=image, outputs=jet_vectors, **kwargs)


def PileupCNN(
    input_shape: tuple[int],
    size: int = 3,
    depth_multiplier: int = 4,
    hidden_layer_sizes: list[int] = [32, 32],
    use_hgq: bool = False,
    init_as_layer_sum: bool = True,
    with_abseta: bool = True,
    weight_regulariser: Regularizer = None,
    **kwargs,
) -> Model:
    """
    Constructs a convolutional neural network for pileup mitigation in calorimeter images.

    The model learns per-pixel weights that down-weight pileup contamination
    while preserving physically meaningful energy deposits.

    Architecture:
    1. Optional eta–phi padding for convolution stability.
    2. Symmetric depthwise convolution over the calorimeter image.
    3. Optional augmentation with log(|eta|) features.
    4. Fully connected per-pixel refinement network.
    5. Learned multiplicative weighting of input image.

    Final output is a reweighted version of the input image.

    Args:
        input_shape (tuple[int]):
            Shape of the input calorimeter image (E, P, C).
        size (int):
            Kernel size for depthwise convolution.
        depth_multiplier (int):
            Number of filters per input channel in depthwise convolution.
        hidden_layer_sizes (list[int]):
            Sizes of intermediate dense layers applied per pixel.
        use_hgq (bool):
            Whether to use HGQ-aware layers (hardware-aware quantisation support).
        init_as_layer_sum (bool):
            If True, initializes final weighting layer to approximate identity
            (bias=3, zero kernel), encouraging initial pass-through behavior.
        with_abseta (bool):
            If True, includes log(|eta|) as an additional feature channel.
        weight_regulariser (Regularizer, optional):
            Optional regularizer applied to learned weights.
        **kwargs:
            Additional arguments passed to the Keras Model constructor.

    Returns:
        tf.keras.Model:
            A model that outputs a pileup-suppressed version of the input image,
            with the same shape as the input.
    """
    inputs = Input(shape=input_shape)

    x = EtaPhiPadding(size // 2)(inputs)
    x = SymmetricDepthwiseConv2D(
        kernel_size=size,
        depth_multiplier=depth_multiplier,
        use_hgq=use_hgq,
        activation="relu",
    )(x)

    if with_abseta:
        eta, _ = TowerEtaPhiLayer()(inputs)
        log_abseta = ops.log(ops.abs(eta) + 1e-3)
        x = Concatenate(axis=-1)([x, log_abseta])

    for layer_size in hidden_layer_sizes:
        x = init_dense_layer(layer_size, activation="relu", use_hgq=use_hgq)(x)

    w = init_dense_layer(
        input_shape[-1],
        activation="hard_sigmoid",
        kernel_initializer="zeros" if init_as_layer_sum else "glorot_uniform",
        bias_initializer=initializers.Constant(3.0) if init_as_layer_sum else "zeros",
        activity_regularizer=weight_regulariser,
        use_hgq=use_hgq,
    )(x)

    outputs = Multiply()([w, inputs])

    model = Model(
        inputs=inputs,
        outputs=outputs,
        **kwargs,
    )

    return model
