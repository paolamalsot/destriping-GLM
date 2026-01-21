# Import Hydra and other required modules
import hydra
from omegaconf import DictConfig
import os
import sys
import traceback
from shutil import copytree, copyfile
from hydra.utils import get_original_cwd, to_absolute_path, instantiate
import yaml
import warnings


# Define the main function for your experiment
@hydra.main(
    config_path="../../experiments/hydra_config/",
    config_name="default",
    version_base=None,
)
def run_experiment(cfg: DictConfig):
    print(f"directory: {os.getcwd()}")

    # copy code
    with open(
        to_absolute_path("./experiments/utilities/experiment_folders_to_copy.yaml"), "r"
    ) as file:
        folders_to_copy = yaml.safe_load(file)
    for folder in folders_to_copy:
        folder_code = to_absolute_path(folder)
        copytree(folder_code, folder)

    # copy parsing function
    parsing_fun_str = cfg.parsing_fun
    parsing_fun_path = to_absolute_path(parsing_fun_str.replace(".", "/") + ".py")
    copyfile(parsing_fun_path, "./parsing.py")

    original_root = get_original_cwd()

    # Instantiate and call the function specified in the config
    try:
        parsing_fun = instantiate(
            {
                "_target_": cfg.parsing_fun + ".parse_config",
                "original_root": original_root,
                "cfg": cfg,
            },
            _recursive_=False,
            _partial_=True,
        )

        parsing_fun()
    except BaseException as e:
        print(f"Error: {e}")
        traceback.print_exc(file=sys.stderr)
        raise
    finally:
        # fflush everything
        sys.stdout.flush()
        sys.stderr.flush()


if __name__ == "__main__":
    run_experiment()
