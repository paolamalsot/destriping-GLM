import os


def find_config_dir(dir, substring):
    # useful to find directories created by hydra multirun, which include the run_number, like 0_destriped=True
    for item in os.listdir(dir):
        item_path = os.path.join(dir, item)
        if os.path.isdir(item_path) and substring in item:
            return item_path
    return None
