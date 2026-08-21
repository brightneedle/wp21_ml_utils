import tensorflow as tf
import yaml
import inspect
from contextlib import ExitStack


def update_custom_objects(custom_objects: dict = {}) -> None:
    """
    Register custom classes from the wp21_ml_utils modules with Keras.

    Parameters
    ----------
    custom_objects:
        Optional dictionary of custom objects to register. If provided, the
        discovered custom classes are added to this dictionary before it is
        registered with Keras.

    Notes
    -----
    All classes defined in the configured wp21_ml_utils modules are added to
    Keras' global custom object registry.
    """
    from wp21_ml_utils import (
        calibration,
        clustering,
        converters,
        layers,
        losses,
        quantisers,
        pileup,
        regularisers,
        sequential,
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
        sequential,
    ]:
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, type):
                custom_objects[name] = obj

    tf.keras.utils.get_custom_objects().update(custom_objects)


def load_config(path) -> dict:
    """
    Load a YAML configuration file.

    Parameters
    ----------
    path:
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Configuration loaded from the YAML file.
    """
    return yaml.safe_load(open(path))


def build_layer(class_name, params):
    """
    Construct a Keras layer from its class name and parameters.

    The class is first looked up in Keras' registered custom objects and then
    in ``tf.keras.layers``.

    Parameters
    ----------
    class_name:
        Name of the layer class to construct.
    params:
        Dictionary of keyword arguments passed to the layer constructor.

    Returns
    -------
    tf.keras.layers.Layer
        Instantiated Keras layer.

    Raises
    ------
    ValueError
        If ``class_name`` cannot be found among the registered custom objects
        or Keras layers.
    TypeError
        If ``name`` is not an expected argument for the class.
    """
    custom = tf.keras.utils.get_custom_objects()

    if class_name in custom:
        cls = custom[class_name]
    elif hasattr(tf.keras.layers, class_name):
        cls = getattr(tf.keras.layers, class_name)
    else:
        raise ValueError(f"Unknown layer '{class_name}'")

    if not issubclass(cls, tf.keras.layers.Layer):
        signature = inspect.signature(cls.__init__)
        if "name" not in signature.parameters:
            raise TypeError(
                f"Callable classes should expect 'name' argument, '{class_name}' does not."
            )

    return cls(**params)


def build_from_config(config: dict) -> tuple[tf.keras.Model, dict, dict]:
    """
    Build a Keras model from a configuration dictionary.

    The configuration specifies the model inputs, layers, and outputs. Layers
    are constructed sequentially according to their entries in the
    ``layers`` configuration, with tensors referenced by their configured
    names.

    Parameters
    ----------
    config:
        Model configuration dictionary. Expected to contain ``inputs``,
        ``layers``, and ``outputs`` entries, with optional ``random_state`` and
        ``hgq_config`` entries. ``hgq_config`` may contain a list of
        ``quantizer_scopes`` and a ``layer`` configuration. These are applied
        while the configured layers are constructed.

    Returns
    -------
    model:
        Constructed Keras model.
    layers_dict:
        Dictionary mapping configured layer names to their instantiated Keras
        layer objects.
    tensor_dict:
        Dictionary mapping configured tensor names to their corresponding
        Keras tensors.
    """
    random_state = int(config.get("random_state", 42))

    tf.random.set_seed(random_state)
    tf.config.experimental.enable_op_determinism()

    update_custom_objects()

    input_spec = config.get("inputs", {})
    output_spec = config.get("outputs", {})

    layer_spec = config["layers"]
    configured_nodes = set(input_spec) | set(layer_spec)
    connected_nodes = set(output_spec)
    pending_nodes = list(output_spec)

    while pending_nodes:
        node_name = pending_nodes.pop()
        if node_name not in layer_spec:
            continue

        inputs = layer_spec[node_name]["inputs"]
        if isinstance(inputs, str):
            inputs = [inputs]
        for input_name in inputs:
            if input_name not in connected_nodes:
                connected_nodes.add(input_name)
                pending_nodes.append(input_name)

    disconnected_nodes = configured_nodes - connected_nodes
    if disconnected_nodes:
        names = ", ".join(sorted(disconnected_nodes))
        raise ValueError(f"Hanging or unconnected nodes in model config: {names}")

    tensor_dict = {}
    for name, spec in input_spec.items():
        tensor_dict[name] = tf.keras.Input(shape=spec["shape"], name=name)

    layers_dict = {}
    with ExitStack() as stack:
        hgq_config = config.get("hgq_config")
        if hgq_config is not None:
            from hgq.config import LayerConfigScope, QuantizerConfigScope

            for scope_config in hgq_config.get("quantizer_scopes", []):
                stack.enter_context(QuantizerConfigScope(**scope_config))

            layer_config = hgq_config.get("layer")
            if layer_config is not None:
                stack.enter_context(LayerConfigScope(**layer_config))

        for node_name, node in layer_spec.items():
            class_name = node["class"]

            inputs = node["inputs"]
            if isinstance(inputs, str):
                inputs = [inputs]

            x = (
                tensor_dict[inputs[0]]
                if len(inputs) == 1
                else [tensor_dict[i] for i in inputs]
            )

            params = node.get("params", {}) or {}
            params.update({"name": node_name})

            layer = build_layer(class_name, params)
            layers_dict[node_name] = layer
            tensor_dict[node_name] = layer(x)

    model = tf.keras.Model(
        inputs={name: tensor_dict[name] for name in input_spec},
        outputs={name: tensor_dict[name] for name in output_spec},
    )

    return model, layers_dict, tensor_dict


def compile_from_config(model: tf.keras.Model, config: dict):
    """
    Compile a Keras model using optimizer, loss, metric, and loss-weight
    settings from a configuration dictionary.

    Parameters
    ----------
    model:
        Keras model to compile.
    config:
        Model configuration dictionary. Output-specific losses, metrics, and
        loss weights are read from the ``outputs`` entry, while the optimizer
        is read from the ``optimiser`` entry.

    Raises
    ------
    TypeError
        If a configured loss does not accept the supplied constructor
        parameters.
    """
    output_spec = config.get("outputs", {})

    losses = {}
    loss_weights = {}
    metrics = {}
    for name, spec in output_spec.items():
        loss_name = spec.get("loss", None)
        loss_params = spec.get("params", {})

        obj = tf.keras.utils.get_custom_objects().get(loss_name)

        if obj is None:
            obj = getattr(tf.keras.losses, loss_name, None)

        if obj is None:
            obj = tf.keras.losses.get(loss_name)

        if isinstance(obj, type):
            losses[name] = obj(**loss_params)
        elif loss_params:
            try:
                losses[name] = obj.__class__(**loss_params)
            except TypeError as e:
                raise TypeError(
                    f"Loss '{loss_name}' does not accept constructor parameters"
                ) from e
        else:
            losses[name] = obj

        loss_weights[name] = spec.get("loss_weight", 1.0)

        metric_list = spec.get("metrics", [])
        if metric_list:
            metrics[name] = []
            for metric_name in metric_list:
                obj = tf.keras.utils.get_custom_objects().get(metric_name)
                if obj is not None:
                    metrics[name].append(obj() if isinstance(obj, type) else obj)
                else:
                    metrics[name].append(tf.keras.metrics.get(metric_name))

    optimiser_config = config.get("optimiser", {"class": "adam", "params": {}})
    optimiser = tf.keras.optimizers.deserialize(
        {"class_name": optimiser_config["class"], "config": optimiser_config["params"]}
    )

    model.compile(
        optimizer=optimiser,
        loss=losses if losses else None,
        loss_weights=loss_weights if loss_weights else None,
        metrics=metrics,
        jit_compile=False,
    )


def load_model(path: str, compile: bool = True) -> tf.keras.Model:
    """
    Load a Keras model from disk with custom objects registered.

    Parameters
    ----------
    path:
        Path to the saved Keras model.
    compile:
        Whether to compile the model using its saved compilation
        configuration.

    Returns
    -------
    tf.keras.Model
        Loaded Keras model.
    """
    update_custom_objects()
    return tf.keras.models.load_model(path, compile=compile)


def extract_submodel(
    model: tf.keras.Model,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    submodel_name: str | None = None,
    append_outputs: bool = False,
) -> tf.keras.Model:
    """
    Extract a Keras sub-model from named layers of an existing model.

    Parameters
    ----------
    model:
        The original Keras model.
    input_names:
        Names of the layers to use as the sub-model inputs. If None, the
        original model inputs are reused.
    output_names:
        Names of the layers to use as the sub-model outputs. If None, the
        original model outputs are reused.
    submodel_name:
        Optional name for the extracted sub-model. If None, the name is
        generated as ``"{model.name}_submodel"``.
    append_outputs:
        If True, append the original model outputs to the selected outputs.
        Has no effect when ``output_names`` is None.

    Returns
    -------
    tf.keras.Model
        A sub-model sharing the original model's layers and weights.
    """
    if input_names is None:
        inputs = list(model.inputs)
        tensor_map = None
    else:
        selected_inputs = [model.get_layer(name).output for name in input_names]
        inputs = [
            tf.keras.Input(
                shape=tensor.shape[1:],
                dtype=tensor.dtype,
                name=name,
            )
            for name, tensor in zip(input_names, selected_inputs)
        ]
        tensor_map = {
            id(original): new for original, new in zip(selected_inputs, inputs)
        }

        for layer in model.layers:
            for node in list(layer._inbound_nodes):
                node_outputs = tf.nest.flatten(node.output_tensors)
                if all(id(tensor) in tensor_map for tensor in node_outputs):
                    continue

                node_inputs = tf.nest.flatten(node.input_tensors)
                if not node_inputs or not all(
                    id(tensor) in tensor_map for tensor in node_inputs
                ):
                    continue

                args, kwargs = node.arguments.fill_in(tensor_map)
                new_outputs = layer(*args, **kwargs)
                for original, new in zip(node_outputs, tf.nest.flatten(new_outputs)):
                    tensor_map[id(original)] = new

    if output_names is None:
        selected_output_names = list(model.output_names)
        selected_outputs = list(model.outputs)
    else:
        selected_output_names = list(output_names)
        selected_outputs = [model.get_layer(name).output for name in output_names]
        if append_outputs:
            for name, tensor in zip(model.output_names, model.outputs):
                if name not in selected_output_names:
                    selected_output_names.append(name)
                    selected_outputs.append(tensor)

    if tensor_map is None:
        outputs = selected_outputs
    else:
        missing_outputs = [
            tensor for tensor in selected_outputs if id(tensor) not in tensor_map
        ]
        if missing_outputs:
            raise ValueError(
                "Requested outputs are not reachable from the selected inputs."
            )
        outputs = [tensor_map[id(tensor)] for tensor in selected_outputs]

    return tf.keras.Model(
        inputs=inputs,
        outputs=dict(zip(selected_output_names, outputs)),
        name=(f"{model.name}_submodel" if submodel_name is None else submodel_name),
    )
