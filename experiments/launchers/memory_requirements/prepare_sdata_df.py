from src.spatialAdata.loading import load_spatialAdata
from src.destriping.GLUM.sdata_to_df import data_to_df
from pathlib import Path as P
import yaml

dataset_config_paths = {
    "mouse_brain": "experiments/hydra_config/destriping_model/mouse_brain/dataset/mouse_brain.yaml",
    "human_lymph_node": "experiments/hydra_config/destriping_model/human_lymph_node/dataset/human_lymph_node.yaml",
    "zebrafish_head": "experiments/hydra_config/destriping_model/zebrafish_head/dataset/zebrafish_head.yaml",
    "mouse_embryo": "experiments/hydra_config/destriping_model/mouse_embryo/dataset/mouse_embryo.yaml",
}

output_dir = P("results/memory_requirements/datasets_df")

for dataset_name, config_path in dataset_config_paths.items():
    cfg = yaml.safe_load(open(config_path))
    dataset_path = cfg["path_data"]
    cell_id_label = cfg["cell_id_label"]
    sdata = load_spatialAdata(dataset_path)
    df = data_to_df(sdata, cell_id_label)
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / f"df_{dataset_name}.pkl"
    df.to_pickle(output_path)
