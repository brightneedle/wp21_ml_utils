from tensorflow.keras.models import load_model


def collect_custom_objects():
    import inspect
    import importlib

    objects = {}

    # only import known safe modules instead of walk_packages
    modules = [
        "wp21_ml_utils.converters",
        "wp21_ml_utils.quantisers",
        "wp21_ml_utils.regularisers",
        "wp21_ml_utils.softkiller",
        "wp21_ml_utils.layers",
        "wp21_ml_utils.losses",
    ]

    for module_name in modules:
        module = importlib.import_module(module_name)

        for name, obj in inspect.getmembers(module):
            if isinstance(obj, type) and hasattr(obj, "get_config"):
                objects[name] = obj

    return objects


def load_wp21_model(path, custom_objects={}):
    custom_objects.update(collect_custom_objects())
    return load_model(path, custom_objects=custom_objects)
