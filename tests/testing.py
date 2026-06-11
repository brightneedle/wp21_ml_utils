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


def test_model_load():
    import os
    from wp21_ml_utils.models import PileUpCNN, load_wp21_model

    model = PileUpCNN((50, 64, 6), use_hgq=True)

    output_path = os.path.join(output_dir, "test_qcnn.keras")
    model.save(output_path)
    load_wp21_model(output_path)


def test_quantiser():
    import numpy as np
    from tensorflow.keras import Sequential
    from wp21_ml_utils.models import load_wp21_model
    from wp21_ml_utils.quantisers import QuadLinearQuantiser

    model = Sequential([QuadLinearQuantiser(bits=4, trainable=True)])
    model.compile(loss="mse", optimizer="adam")

    x = np.exp(np.random.normal(size=(10000, 1)))

    model.fit(x, x, batch_size=32, epochs=10)

    G = model.layers[0].G.numpy().item()
    lsb = model.layers[0].lsb.numpy().item()

    fig, ax = plt.subplots(ncols=2, figsize=(8, 4))
    ax[0].scatter(x, x, s=0.2, label="x")
    ax[0].scatter(x, model.predict(x), label=f"Q(x) G={G:.3g}, LSB={lsb:.3g}", s=0.2)
    ax[0].set_yscale("log")
    ax[0].set_xscale("log")
    ax[0].legend()

    ax[1].plot(model.layers[0]._compute_bin_edges().numpy())

    plt.savefig(os.path.join(output_dir, "quantiser_test.png"))
    plt.close()

    output_path = os.path.join(output_dir, "test_quantiser.keras")
    model.save(output_path)
    load_wp21_model(output_path)


def test_jet_calib():
    import numpy as np
    from wp21_ml_utils.models import JetEnergyResponseMLP, load_wp21_model
    from wp21_ml_utils.losses import CalibrationLoss

    model = JetEnergyResponseMLP(n_jets=1)
    model.compile(loss=CalibrationLoss(), optimizer="adam")

    pt = np.random.uniform(0, 2 * np.pi, size=(10000, 1))
    eta = np.random.uniform(-2.5, 2.5, size=pt.shape)
    phi = np.random.uniform(-np.pi, np.pi, size=pt.shape)

    inputs = np.stack([pt, eta, phi], axis=-1)
    targets = np.stack([pt + np.sin(pt), eta, phi], axis=-1)

    model.fit(inputs, targets, batch_size=32, epochs=10)

    plt.figure(figsize=(4, 4))
    plt.scatter(inputs[..., 0], targets[..., 0], s=0.1, label="True")
    plt.scatter(pt, model.predict(inputs)[..., 0], s=0.1, label="MLP")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "jet_calib_test.png"))
    plt.close()

    output_path = os.path.join(output_dir, "test_jes_mlp.keras")
    model.save(output_path)
    load_wp21_model(output_path)
