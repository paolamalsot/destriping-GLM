from utilities.are_we_in_cluster import run_script_with_runpy
from src.utilities.submitit.parallel_executor import (
    setup_executor,
    on_cluster
)
import datetime
from functools import partial
import yaml
import argparse
import os

# Either runs locally (if not on cluster), or submits with submitit a Python script whose config YAML path can be given.

# Parsing params
parser = argparse.ArgumentParser(description="Run a script locally or on a cluster.")
parser.add_argument("--script_path", type=str, required=True, help="Path to the script to be executed.")
parser.add_argument("--cluster_config", type=str, default="experiments/utilities/default_submitit_executor.yaml", 
                    help="Path to the YAML config for cluster execution.")
args = parser.parse_args()

script_path = args.script_path
cluster_config = args.cluster_config
script_name = os.path.basename(script_path)

fun = partial(run_script_with_runpy, script_path)

if on_cluster():
    # Load cluster configuration
    with open(cluster_config, "r") as config_file:
        config_args = yaml.safe_load(config_file)

    # Generate output directory name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"outfiles/{script_name}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Setup and submit to the cluster
    executor = setup_executor(output_dir, **config_args)
    executor.submit(fun)
    # Does not wait for results!
else:
    # Redirect output and stderr locally
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = f"outfiles/{script_name}_{timestamp}.out"
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    with open(log_file_path, "w") as log_file:
        try:
            # Redirect stdout and stderr
            import sys
            sys.stdout = log_file
            sys.stderr = log_file

            fun()
        finally:
            # Restore stdout and stderr
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
