import os
import matplotlib.pyplot as plt


plt.rcParams["figure.dpi"] = 150
plt.rcParams["figure.constrained_layout.use"] = True

output_dir = os.path.join("./test_outputs")
os.makedirs(output_dir, exist_ok=True)


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
    from wp21_ml_utils.models import load_wp21_model
    from wp21_ml_utils.examples import PileupCNN
    from wp21_ml_utils.regularisers import PushMaxWeightToUnity, SparsityPenalty

    strength = 1e-3

    x = np.random.normal(size=(128, 50, 64, 6))

    for regulariser in [PushMaxWeightToUnity(strength), SparsityPenalty(strength)]:
        model = PileupCNN(
            (50, 64, 6),
            use_hgq=True,
            init_as_layer_sum=True,
            push_max_to_unity=True,
            weight_regulariser=regulariser,
        )
        y = model(x)

        np.testing.assert_equal(x.shape, y.shape)
        np.testing.assert_allclose(x, y)

        output_path = os.path.join(output_dir, "test_qcnn.keras")
        model.save(output_path)
        load_wp21_model(output_path)

        model.compile(loss="mse", optimizer="adam")
        model.evaluate(x, x, verbose=0)


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

    np.testing.assert_allclose(rank_coeff, 1)

    plt.figure(figsize=(4, 4))
    plt.scatter(x, y, s=0.1, label="True")
    plt.scatter(x, y_pred, s=0.1, label="MLP")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "monodense_test.png"))
    plt.close()

    output_path = os.path.join(output_dir, "monodense_mlp.keras")
    model.save(output_path)
    load_wp21_model(output_path)


def test_cone_jets():
    import numpy as np
    from wp21_ml_utils.models import load_wp21_model
    from wp21_ml_utils.examples import ConeJets

    model = ConeJets(
        input_shape=(9, 9, 1),
        kernel_size=3,
        max_jets=2,
        min_pt=0,
    )

    x = np.zeros((1, 9, 9, 1), dtype=np.float32)

    x[0, 2, 2, 0] = 5.0
    x[0, 6, 6, 0] = 10.0

    jets = model.predict(x, verbose=0)

    assert jets[0, 0, 0] >= jets[0, 1, 0]

    np.testing.assert_allclose(jets[0, :, 0], [10.0, 5.0])

    output_path = os.path.join(output_dir, "test_cone.keras")
    model.save(output_path)
    load_wp21_model(output_path)


def test_towers():
    import numpy as np
    from wp21_ml_utils.converters import VectorsToImage

    layer = VectorsToImage(
        eta_edges=np.array([-1.0, 0.0, 1.0], dtype=np.float32),
        phi_edges=np.array([-np.pi, 0.0, np.pi], dtype=np.float32),
        n_layers=6,
        use_layers=True,
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

    plt.savefig(os.path.join(output_dir, "tower_test.png"))
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
