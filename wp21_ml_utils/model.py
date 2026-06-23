import tensorflow as tf
import yaml


def update_custom_objects(custom_objects: dict = {}) -> None:
    from wp21_ml_utils import (
        calibration,
        clustering,
        converters,
        layers,
        losses,
        quantisers,
        pileup,
        regularisers,
    )

    for module in [
        calibration,
        clustering,
        converters,
        layers,
        losses,
        quantisers,
        pileup,
        regularisers,
    ]:
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, type):
                custom_objects[name] = obj

    tf.keras.utils.get_custom_objects().update(custom_objects)


def load_config(path) -> dict:
    return yaml.safe_load(open(path))


def build_from_config(config: dict) -> (tf.keras.Model, dict, dict):
    random_state = int(config.get("random_state", 42))

    tf.random.set_seed(random_state)
    tf.config.experimental.enable_op_determinism()

    update_custom_objects()

    input_spec = config.get("inputs", {})
    output_spec = config.get("outputs", {})

    tensor_dict = {}
    for name, spec in input_spec.items():
        tensor_dict[name] = tf.keras.Input(shape=spec["shape"], name=name)

    layers_dict = {}
    for node_name, node in config["algorithms"].items():
        op_name = node["op"]

        inputs = node["inputs"]
        if isinstance(inputs, str):
            inputs = [inputs]

        x = (
            tensor_dict[inputs[0]]
            if len(inputs) == 1
            else [tensor_dict[i] for i in inputs]
        )

        params = node.get("params", {}) or {}

        layer = tf.keras.utils.deserialize_keras_object(
            {"class_name": op_name, "config": params},
        )
        layers_dict[node_name] = layer
        tensor_dict[node_name] = layer(x)

    model = tf.keras.Model(
        inputs={name: tensor_dict[name] for name in input_spec},
        outputs={name: tensor_dict[name] for name in output_spec},
    )

    return model, layers_dict, tensor_dict


def compile_from_config(model: tf.keras.Model, config: dict) -> tf.keras.Model:
    output_spec = config.get("outputs", {})

    losses = {}
    loss_weights = {}
    for name, spec in output_spec.items():
        loss_name = spec.get("loss", None)
        obj = tf.keras.utils.get_custom_objects().get(loss_name)
        if obj is not None:
            losses[name] = obj() if isinstance(obj, type) else obj
        else:
            losses[name] = tf.keras.losses.get(loss_name)

        loss_weights[name] = spec.get("loss_weight", 1.0)

    optimiser_config = config.get("optimiser", {"op": "adam", "params": {}})
    optimiser = tf.keras.optimizers.deserialize(
        {"class_name": optimiser_config["op"], "config": optimiser_config["params"]}
    )

    model.compile(
        optimizer=optimiser,
        loss=losses if losses else None,
        loss_weights=loss_weights if loss_weights else None,
        jit_compile=False,
    )
    return model


def load_model(path, custom_objects: dict = {}, compile: bool = True) -> tf.keras.Model:
    update_custom_objects()
    return tf.keras.models.load_model(path, compile=compile)
