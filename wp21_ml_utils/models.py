import tensorflow as tf
from tensorflow.keras import layers, Input, ops, Model, initializers
from tensorflow.keras.models import load_model

# import hgq

from wp21_ml_utils.utils import unpack
from wp21_ml_utils.converters import ImageToVectors
from wp21_ml_utils.layers import (
    EtaPhiPadding,
    SymmetricDepthwiseConv2D,
    SlidingConeSum,
    LocalMaxMask,
    TowerEtaPhiLayer,
    MonoDense,
)
from wp21_ml_utils.regularisers import PushMaxWeightToUnity
from wp21_ml_utils.quantisers import TrainableQuantiser
from wp21_ml_utils.utils import init_dense_layer


def load_wp21_model(path, custom_objects={}):
    def collect_custom_objects():
        import inspect
        import pkgutil
        import importlib

        import wp21_ml_utils

        objects = {}
        for _, module_name, _ in pkgutil.walk_packages(
            wp21_ml_utils.__path__,
            wp21_ml_utils.__name__ + ".",
        ):
            module = importlib.import_module(module_name)

            for name, obj in inspect.getmembers(module):
                if hasattr(obj, "get_config") and isinstance(obj, type):
                    objects[name] = obj

        return objects

    custom_objects.update(collect_custom_objects())
    # custom_objects["QDense"] = hgq.layers.QDense

    return load_model(path, custom_objects=custom_objects)


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
        input_channels=input_shape[-1],
        depth_multiplier=depth_multiplier,
        use_hgq=use_hgq,
        activation="relu",
    )(x)

    if with_abseta:
        eta, _ = TowerEtaPhiLayer()(inputs)
        log_abseta = ops.log(ops.abs(eta) + 1e-3)
        x = layers.Concatenate(axis=-1)([x, log_abseta])

    for layer_size in hidden_layer_sizes:
        x = init_dense_layer(layer_size, activation="relu", use_hgq=use_hgq)(x)

    w = init_dense_layer(
        input_shape[-1],
        activation="hard_sigmoid",
        kernel_initializer="zeros" if init_as_layer_sum else "glorot_uniform",
        bias_initializer="ones" if init_as_layer_sum else "zeros",
        activity_regularizer=PushMaxWeightToUnity(1e-3) if push_max_to_unity else None,
        use_hgq=use_hgq,
    )(x)

    outputs = layers.Multiply()([w, inputs])

    model = Model(
        inputs=inputs,
        outputs=outputs,
        **kwargs,
    )

    return model


def ConeJetAlgo(
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


def JetEnergyResponseMLP(
    hidden_layer_sizes: list[int] = [64, 64],
    hidden_activation: str = "softplus",
    eps: float = 1e-3,
    name: str = "jet-calib",
    monotonic: bool = True,
    max_jets: int = 10,
):
    def get_layer(nodes, activation=None, monotonicity_indicator=None):
        if monotonic:
            if monotonicity_indicator is None:
                return MonoDense(
                    nodes,
                    activation=activation,
                    kernel_initializer=initializers.RandomNormal(stddev=1e-2),
                )

            else:
                return MonoDense(
                    nodes,
                    activation=activation,
                    kernel_initializer=initializers.RandomNormal(stddev=1e-2),
                    monotonicity_indicator=monotonicity_indicator,
                )
        else:
            return layers.Dense(
                nodes,
                activation=activation,
            )

    momenta = Input(shape=(max_jets, 3))

    pt, eta, phi = unpack(momenta)

    x = layers.Concatenate(axis=-1)([pt, ops.abs(eta)])
    x = ops.log(x + eps)
    for i, hls in enumerate(hidden_layer_sizes):
        if i == 0:
            layer = get_layer(
                hls,
                activation=hidden_activation,
                monotonicity_indicator=[1, 0],
            )
        else:
            layer = get_layer(hls, activation=hidden_activation)

        x = layer(x)

    log_pt = get_layer(1)(x)
    gate_head = get_layer(1)(x)

    calib_pt = ops.exp(ops.clip(log_pt, -10, 10))
    gate = ops.sigmoid(gate_head)

    gated_calib_pt = ops.where(pt > eps, gate * calib_pt, 0)

    calib_momenta = layers.Concatenate(axis=-1)([gated_calib_pt, eta, phi])

    return Model(inputs=momenta, outputs=calib_momenta, name=name)


def MissingHT(name: str = "htmiss", n_jets: int = 10, **kwargs):
    calib_layer = JetEnergyResponseMLP(name="htmiss_jet_calib", **kwargs)

    jets = Input(shape=(n_jets, 3))
    pt, _, phi = unpack(calib_layer(jets))

    px = pt * tf.math.cos(phi)
    py = pt * tf.math.sin(phi)

    sum_px = tf.reduce_sum(px, axis=1)
    sum_py = tf.reduce_sum(py, axis=1)

    mht_xy = layers.Concatenate()([sum_px, sum_py])

    return Model(inputs=jets, outputs=mht_xy, name=name)


def HeterogeneousTrainableQuantizer(
    input_shape: tuple[int],
    bits: int,
    min_range: float = 0,
    max_range: float = 102,
    T: float = 50,
    train_min_range: bool = False,
    train_max_range: bool = True,
    train_widths: bool = True,
    bin_regularisation: float = 1e-3,
    smooth_in_forward: bool = True,
    axis: int = -1,
    **kwargs,
):
    quantizers = [
        TrainableQuantiser(
            bits=bits,
            min_range=min_range,
            max_range=max_range,
            T=T,
            train_max_range=train_max_range,
            train_min_range=train_min_range,
            train_widths=train_widths,
            bin_regularisation=bin_regularisation,
            smooth_in_forward=smooth_in_forward,
            name=f"quantiser_{i}",
        )
        for i in range(input_shape[axis])
    ]

    x = Input(input_shape)
    y = [quant(x_) for quant, x_ in zip(quantizers, tf.unstack(x, axis=axis))]
    y = tf.stack(y, axis=axis)
    return Model(x, y, **kwargs)
