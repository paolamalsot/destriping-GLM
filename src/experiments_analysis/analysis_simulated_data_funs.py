from __future__ import annotations
from pathlib import Path
import pandas as pd
import pandas as pd
from my_notebooks.GLUM.funs.experiment_plots import visualize_difference_solutions
from src.experiments_analysis.summary_structure_preservation import operation_dict

from src.experiments_analysis.analysis_utils import (
    load_gt_sol,
    load_poisson_sol,
)


def make_stripe_difference_plots(df_runs: pd.DataFrame, output_dir: Path) -> None:
    """
    Generate Plotly stripe-factor difference plots for each dataset.

    Uses ``visualize_difference_solutions`` from
    ``my_notebooks.GLUM.funs.experiment_plots`` to compare each fitted solution
    to the ground truth Poisson solution, when available.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = df_runs["dataset_path"].iloc[0]
    assert len(df_runs["dataset_path"].unique()) == 1
    dataset_root = Path(dataset_path).parent
    gt_sol = load_gt_sol(dataset_root)
    if gt_sol is None:
        return None
    experiments_df = df_runs.copy()
    experiments_df["sol"] = experiments_df.apply(lambda x: load_poisson_sol(x), axis=1)
    experiments_df.index = df_runs.index

    fig = visualize_difference_solutions(
        experiments_df.loc[~pd.isna(experiments_df["sol"])], gt_sol
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_dir / "stripe_factors_difference.html")
