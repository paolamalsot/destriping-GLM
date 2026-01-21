import inspect


def match_method_signature(method, dict):
    method_params = inspect.signature(method).parameters
    filtered_params = {
        k: v for k, v in dict.items() if k in method_params and k != "self"
    }
    return filtered_params
