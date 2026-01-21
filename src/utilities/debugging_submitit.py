import hydra
from omegaconf import OmegaConf
from hydra import initialize_config_dir, compose
from pathlib import Path as P

def load_conf(config_dir, config_name = "config.yaml"):
    config_dir_path = P(config_dir).absolute().__str__()
    hydra.core.global_hydra.GlobalHydra.get_state().clear()
    initialize_config_dir(
        config_dir=config_dir_path, job_name="relaunch_job", version_base=None
    )

    # Compose the config
    cfg = compose(config_name=config_name)
    return cfg
