import os

output_dir = os.path.join("./test_outputs")


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

    model = PileUpCNN((50, 64, 6))

    output_path = os.path.join(output_dir, "test_model.keras")
    os.makedirs(output_dir, exist_ok=True)

    model.save(output_path)
    load_wp21_model(output_path)


def test_quantiser():
    import numpy as np
    from tensorflow.keras import Sequential
    from wp21_ml_utils.quantisers import QuadLinearQuantiser
    import matplotlib.pyplot as plt

    model = Sequential([QuadLinearQuantiser(bits=4, trainable=True)])
    model.compile(loss="mse", optimizer="adam")

    x = np.exp(np.random.normal(size=(10000, 1)))

    model.fit(x, x, batch_size=32, epochs=10)

    G = model.layers[0].G.numpy().item()
    lsb = model.layers[0].lsb.numpy().item()

    fig, ax = plt.subplots(ncols=2, figsize=(10, 5), dpi=150, layout="constrained")
    ax[0].scatter(x, x, s=0.2)
    ax[0].scatter(x, model.predict(x), label=f"G={G:.3g}, LSB={lsb:.3g}", s=0.2)
    ax[0].set_yscale("log")
    ax[0].set_xscale("log")
    ax[1].legend()

    ax[1].plot(model.layers[0]._compute_bin_edges().numpy())

    plt.savefig(os.path.join(output_dir, "quantiser_test.png"))
    plt.close()
