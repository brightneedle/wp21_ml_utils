import os
from pathlib import Path
import matplotlib.pyplot as plt


plt.rcParams["figure.dpi"] = 150
plt.rcParams["figure.constrained_layout.use"] = True

TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / "data"
OUTPUT_DIR = TEST_DIR / "test_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def test_imports():
    import pkgutil
    import importlib
    import wp21_ml_utils

    package = wp21_ml_utils.__name__
    for _, module_name, _ in pkgutil.walk_packages(
        wp21_ml_utils.__path__, package + "."
    ):
        importlib.import_module(module_name)


def test_pucnn():
    import numpy as np
    from tensorflow.keras import Sequential
    from wp21_ml_utils.model import load_model
    from wp21_ml_utils.pileup import PileupCNN
    from wp21_ml_utils.regularisers import PushMaxWeightToUnity, SparsityPenalty

    strength = 1e-3

    x = np.random.normal(size=(128, 50, 64, 6))

    for regulariser in [PushMaxWeightToUnity(strength), SparsityPenalty(strength)]:
        model = Sequential(
            [
                PileupCNN(
                    use_hgq=True,
                    init_as_layer_sum=True,
                    weight_regulariser=regulariser,
                ),
            ]
        )

        y = model(x)

        np.testing.assert_equal(x.shape, y.shape)
        np.testing.assert_allclose(x, y)

        output_path = os.path.join(OUTPUT_DIR, "test_qcnn.keras")
        model.save(output_path)
        load_model(output_path)

        model.compile(loss="mse", optimizer="adam")
        model.evaluate(x, x, verbose=0)


def test_quantisers():
    import numpy as np
    from tensorflow.keras import Sequential
    from wp21_ml_utils.model import load_model
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

        plt.savefig(os.path.join(OUTPUT_DIR, f"quantiser_test_{qlayer.name}.png"))
        plt.close()

        output_path = os.path.join(OUTPUT_DIR, f"test_quantiser_{qlayer.name}.keras")
        model.save(output_path)
        load_model(output_path)


def test_mono_dense():
    import numpy as np
    from tensorflow.keras import Sequential
    from wp21_ml_utils.model import load_model
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

    np.testing.assert_allclose(rank_coeff, 1)

    plt.figure(figsize=(4, 4))
    plt.scatter(x, y, s=0.1, label="True")
    plt.scatter(x, y_pred, s=0.1, label="MLP")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "monodense_test.png"))
    plt.close()

    output_path = os.path.join(OUTPUT_DIR, "monodense_mlp.keras")
    model.save(output_path)
    load_model(output_path)


def test_cone_jets():
    import numpy as np
    from tensorflow.keras import Sequential
    from wp21_ml_utils.model import load_model
    from wp21_ml_utils.clustering import ConeJet

    model = Sequential(
        [
            ConeJet(
                kernel_size=3,
                max_jets=2,
                min_pt=0,
            )
        ]
    )

    x = np.zeros((1, 9, 9, 1), dtype=np.float32)

    x[0, 2, 2, 0] = 5.0
    x[0, 6, 6, 0] = 10.0

    jets = model(x)

    assert jets[0, 0, 0] >= jets[0, 1, 0]

    np.testing.assert_allclose(jets[0, :, 0], [10.0, 5.0])

    output_path = os.path.join(OUTPUT_DIR, "test_cone.keras")
    model.save(output_path)
    load_model(output_path)


def test_towers():
    import numpy as np
    from wp21_ml_utils.converters import VectorsToImage

    layer = VectorsToImage(
        eta_edges=np.array([-1.0, 0.0, 1.0], dtype=np.float32),
        phi_edges=np.array([-np.pi, 0.0, np.pi], dtype=np.float32),
        filter_layers=range(6),
        return_layers=True,
    )

    x = np.asarray(
        [
            [
                [1.0, -0.5, -1.0, 0],
                [2.0, 0.5, 1.0, 3],
                [4.0, 0.5, 1.0, 3],
            ]
        ]
    )

    out = layer(x).numpy()

    # Expected shape: (1, 2, 2, 6)
    expected = np.zeros((1, 2, 2, 6), dtype=np.float32)
    expected[0, 0, 0, 0] = 1.0
    expected[0, 1, 1, 3] = 6.0

    np.testing.assert_allclose(out, expected)

    fig, ax = plt.subplots(ncols=2, figsize=(8, 4))
    ax[0].matshow(expected[0].sum(axis=-1))
    ax[1].matshow(out[0].sum(axis=-1))

    plt.savefig(os.path.join(OUTPUT_DIR, "tower_test.png"))
    plt.close()


def test_converters():
    import numpy as np
    from wp21_ml_utils.converters import VectorsToImage, ImageToVectors

    x = np.asarray(
        [
            [
                [4.0, -0.5, -np.pi / 2],
                [2.0, 0.5, np.pi / 2],
                [1.0, 0.5, -np.pi / 2],
            ]
        ]
    )

    vector2image = VectorsToImage(
        eta_edges=np.array([-1.0, 0.0, 1.0], dtype=np.float32),
        phi_edges=np.array([-np.pi, 0.0, np.pi], dtype=np.float32),
    )
    image2vector = ImageToVectors(deta=1.0, dphi=np.pi, max_vectors=x.shape[1])

    y = image2vector(vector2image(x))

    np.testing.assert_equal(x.shape, y.shape)
    np.testing.assert_allclose(x, y)


def test_coordinates():
    import numpy as np
    from wp21_ml_utils.utils import (
        unpack_momenta,
        polar_to_cartesian,
        cartesian_to_polar,
    )

    np.random.seed(42)

    px, py, pz = unpack_momenta(np.random.normal(size=(128, 10, 3)))

    pt, eta, phi = cartesian_to_polar(px, py, pz)

    px_, py_, pz_ = polar_to_cartesian(pt, eta, phi)

    np.testing.assert_allclose(
        (px, py, pz),
        (px_, py_, pz_),
        rtol=1e-5,
        atol=1e-6,
    )


def test_build_from_config():
    import numpy as np
    from pprint import pprint
    from wp21_ml_utils.model import (
        load_config,
        build_from_config,
        compile_from_config,
        load_model,
    )

    config = load_config(DATA_DIR / "test_model_config.yaml")
    pprint(config)

    model, _, _ = build_from_config(config)

    compile_from_config(model, config)

    train_data = np.load(DATA_DIR / "train_data.npz")

    model.evaluate(
        x={k: train_data[k] for k in config["inputs"]},
        y={k: train_data[k] for k in config["outputs"]},
        verbose=0,
    )

    save_to = OUTPUT_DIR / "test_pipeline.keras"
    model.save(save_to)
    load_model(save_to)


def test_build_from_custom_object():
    from tensorflow.keras.layers import Layer
    from wp21_ml_utils.model import build_from_config, update_custom_objects

    class CustomLayer(Layer):
        def call(self, inputs):
            return inputs * 2

    config = {
        "inputs": {
            "input": {
                "shape": [1],
            },
        },
        "layers": {
            "custom_layer": {
                "class": "CustomLayer",
                "inputs": ["input"],
            },
        },
        "outputs": {
            "custom_layer": {},
        },
    }

    update_custom_objects({"CustomLayer": CustomLayer})

    model, _, _ = build_from_config(config)
