import os
import matplotlib.pyplot as plt

from tensorflow.keras import layers, Input, ops, Model

plt.rcParams["figure.dpi"] = 150
plt.rcParams["figure.constrained_layout.use"] = True

output_dir = os.path.join("./test_outputs")
os.makedirs(output_dir, exist_ok=True)


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
    from wp21_ml_utils.layers import (
        EtaPhiPadding,
        SymmetricDepthwiseConv2D,
        TowerEtaPhiLayer,
    )
    from wp21_ml_utils.regularisers import PushMaxWeightToUnity
    from wp21_ml_utils.utils import init_dense_layer

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
    from wp21_ml_utils.converters import ImageToVectors
    from wp21_ml_utils.layers import SlidingConeSum, LocalMaxMask

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


def test_imports():
    import pkgutil
    import importlib
    import wp21_ml_utils

    package = wp21_ml_utils.__name__
    for _, module_name, _ in pkgutil.walk_packages(
        wp21_ml_utils.__path__, package + "."
    ):
        importlib.import_module(module_name)


def test_model_load():
    from wp21_ml_utils.models import load_wp21_model

    model = PileupCNN((50, 64, 6), use_hgq=True)

    output_path = os.path.join(output_dir, "test_qcnn.keras")
    model.save(output_path)
    load_wp21_model(output_path)


def test_quantisers():
    import numpy as np
    from tensorflow.keras import Sequential
    from wp21_ml_utils.models import load_wp21_model
    from wp21_ml_utils.quantisers import QuadLinearQuantiser, FlexibleQuantiser

    x = np.exp(np.random.normal(size=(10000, 1)))

    for qlayer in [
        QuadLinearQuantiser(bits=4, trainable=True),
        FlexibleQuantiser(bits=4, train_max_range=True, train_widths=True),
    ]:
        model = Sequential([qlayer])
        model.compile(loss="mse", optimizer="adam")
        model.fit(x, x, batch_size=32, epochs=10)

        fig, ax = plt.subplots(ncols=2, figsize=(8, 4))
        ax[0].scatter(x, x, s=0.2, label="x")
        ax[0].scatter(x, model.predict(x), label="Q(x)", s=0.2)
        ax[0].set_yscale("log")
        ax[0].set_xscale("log")
        ax[0].legend()

        ax[1].plot(model.layers[0]._compute_bin_edges().numpy())

        plt.savefig(os.path.join(output_dir, f"quantiser_test_{qlayer.name}.png"))
        plt.close()

        output_path = os.path.join(output_dir, f"test_quantiser_{qlayer.name}.keras")
        model.save(output_path)
        load_wp21_model(output_path)


def test_mono_dense():
    import numpy as np
    from tensorflow.keras import Sequential
    from wp21_ml_utils.models import load_wp21_model
    from wp21_ml_utils.layers import MonoDense
    from scipy.stats import spearmanr

    x = np.random.uniform(0, 2 * np.pi, size=(10000, 1))
    y = x + 1.5 * np.sin(x)

    model = Sequential(
        [
            MonoDense(64, activation="softplus", monotonicity_indicator=1),
            MonoDense(64, activation="softplus", monotonicity_indicator=1),
            MonoDense(1, monotonicity_indicator=1),
        ]
    )
    model.compile(loss="mse", optimizer="adam")
    model.fit(x, y, batch_size=32, epochs=20)

    y_pred = model.predict(x)
    rank_coeff = spearmanr(x, y_pred, axis=None).statistic
    assert np.isclose(rank_coeff, 1)

    plt.figure(figsize=(4, 4))
    plt.scatter(x, y, s=0.1, label="True")
    plt.scatter(x, y_pred, s=0.1, label="MLP")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "monodense_test.png"))
    plt.close()

    output_path = os.path.join(output_dir, "monodense_mlp.keras")
    model.save(output_path)
    load_wp21_model(output_path)
