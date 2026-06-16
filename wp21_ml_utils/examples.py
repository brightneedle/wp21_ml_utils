from tensorflow.keras import Input, Model, ops, initializers
from tensorflow.keras.layers import Multiply, Concatenate

from wp21_ml_utils.converters import ImageToVectors
from wp21_ml_utils.layers import (
    EtaPhiPadding,
    SymmetricDepthwiseConv2D,
    TowerEtaPhiLayer,
    SlidingConeSum,
    LocalMaxMask,
)
from wp21_ml_utils.regularisers import PushMaxWeightToUnity
from wp21_ml_utils.utils import init_dense_layer


def ConeJets(
    input_shape: tuple[int],
    kernel_size: int = 9,
    min_pt: float = 0,
    max_jets: int = 20,
    shape: str = "circle",
    radius: float = None,
    layer_name: str = "cone-layer",
):
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

    return Model(inputs=image, outputs=jet_vectors, name=layer_name)


def PileupCNN(
    input_shape: tuple[int],
    size: int = 3,
    depth_multiplier: int = 4,
    hidden_layer_sizes: list[int] = [32, 32],
    use_hgq: bool = False,
    init_as_layer_sum: bool = True,
    with_abseta: bool = True,
    push_max_to_unity: bool = False,
    **kwargs,
):
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
        activity_regularizer=PushMaxWeightToUnity(1e-3) if push_max_to_unity else None,
        use_hgq=use_hgq,
    )(x)

    outputs = Multiply()([w, inputs])

    model = Model(
        inputs=inputs,
        outputs=outputs,
        **kwargs,
    )

    return model
