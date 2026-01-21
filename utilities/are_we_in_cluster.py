import socket
import runpy


def run_script_with_runpy(script_path):
    """
    Executes a Python script using the runpy module.

    :param script_path: Path to the Python script to run.
    """
    runpy.run_path(script_path)
