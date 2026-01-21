from .parallel_executor import setup_executor
import yaml


class ExecutorDict:
    def __init__(self, input_dict):
        if "default" not in input_dict:
            raise ValueError("The input dictionary must have a 'default' key.")
        self._dict = input_dict

    def __getitem__(self, key):
        if key in self._dict.keys():
            return self._dict[key]
        else:
            return self._dict.get(key, self._dict["default"])

    def __setitem__(self, key, value):
        self._dict[key] = value

    def __delitem__(self, key):
        del self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def __repr__(self):
        return f"executor_dict({self._dict})"


def define_lambda_function(kwargs):
    return lambda output_dir: setup_executor(output_dir, **kwargs)


def configuration_dict_to_executor_dict(configuration_dict):
    executor_dict = {
        key: define_lambda_function(kwargs)
        for key, kwargs in configuration_dict.items()
    }
    return executor_dict


def load_submitit_executor_dict(yaml_file):
    with open(yaml_file) as file:
        dict_ = yaml.safe_load(file)
    executor_dict = configuration_dict_to_executor_dict(dict_)
    return ExecutorDict(executor_dict)


path_default_submitit_executor_configuration_dict = "experiments/src/custom_destriping_benchmark/analysis/submitit_executors_configs/default.yaml"


def load_default_submitit_executor_dict():
    return load_submitit_executor_dict(
        path_default_submitit_executor_configuration_dict
    )
