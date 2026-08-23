import copy

def update_config(config:dict, params:dict) -> dict:
    config_copy = copy.deepcopy(config)

    return _update_node(config_copy, params)

def _update_node(node, params, prefix=""):
    if isinstance(node, dict):
        
        if "type" in node and ("values" in node or "range" in node):
            param_name = prefix.rstrip("_")
            if node["type"] == "list":
                n_elements = params[f"{param_name}_length"]
                return [params[f"{param_name}_{i+1}"] for i in range(n_elements)]
            return params[param_name]

        return {key: _update_node(value, params, f"{prefix}{key}_") for key, value in node.items()}

    if isinstance(node, list):
        return [_update_node(value, params, prefix) for value in node]
    return node