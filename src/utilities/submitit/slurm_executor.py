# override of submitit SlurmExecutor to add account and exclude parameters


from submitit.auto.auto import AutoExecutor
import typing as tp


class MySlurmExecutor(AutoExecutor):
    def __init__(self, folder):
        super().__init__(folder)

    @classmethod
    def _valid_parameters(cls) -> tp.Set[str]:
        # Add 'account' to the valid parameter list
        return super()._valid_parameters().union({"slurm_account", "slurm_exclude"})

    def _internal_update_parameters(self, **kwargs: tp.Any) -> None:
        # Just delegate to the parent which already validates using _make_sbatch_string
        super()._internal_update_parameters(**kwargs)


## EXAMPLE USAGE
# executor = MySlurmExecutor(output_dir)
# executor.update_parameters(
#     time=10,  # min
#     mem=4000, # MB
#     cpus_per_task=2,
#     account = "tumorp",
#     exclude = "compute-biomed-17"
# )
# job = executor.submit(fun)
