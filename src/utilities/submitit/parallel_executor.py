import socket
import submitit
import os
import logging
from typing import List, Tuple, Callable
import numpy as np
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import List, Callable
import pandas as pd
from src.utilities.submitit.slurm_executor import MySlurmExecutor
import time
from submitit.core.utils import FailedJobError


def remove_slurm_keys_from_environ():
    slurm_keys = [key for key in os.environ if key.startswith("SLURM_")]
    for key in slurm_keys:
        del os.environ[key]


def on_cluster():
    """Check if running on a compute cluster."""
    on_cluster = "biomed" in socket.gethostname()
    logging.debug(f"{on_cluster=}")
    return on_cluster


def setup_executor(output_folder, mem, n_cpus, time_min, account=None, exclude=None):
    """Setup Submitit executor for parallel tasks."""
    executor = MySlurmExecutor(folder=os.path.join(output_folder, "submitit_logs"))
    executor.update_parameters(
        timeout_min=time_min, mem_gb=mem, cpus_per_task=n_cpus  # 1 hour  # GB per task
    )

    if account is not None:
        executor.update_parameters(account=account)
    if exclude is not None:
        executor.update_parameters(exclude=exclude)
    return executor


def distance_matrix_executor(output_folder):
    mem = 10
    time_min = 60
    n_cpus = 30
    return setup_executor(output_folder, mem, n_cpus, time_min)


def map_array_from_submitit_executor(executor):
    # WARNING IN CONCURRENT.FUTURES STOPS WHEN THE SHORTEST ITERABLE IS EXHAUSTED
    def map_fun(*args, **kwargs):
        jobs = executor.map_array(*args, **kwargs)
        results = []
        failed = []
        for job in jobs:
            try:
                result = job.result()
                results.append(result)
                failed.append(False)
            except Exception as e:
                print(f"Job {job.job_id} failed with exception: {e}")
                results.append(None)
                failed.append(True)

        if np.any(np.array(failed)):
            raise ValueError("one job failed...")

        return results

    return map_fun


def map_array_from_threadpool_executor(executor_):
    def map_fun(*args, **kwargs):
        with executor_ as executor:
            results = executor.map(*args, **kwargs)
            return list(results)

    return map_fun


def get_map_array_fun(submitit_executor, threadpool_executor=None, use_submitit=True):
    if threadpool_executor is None:
        threadpool_executor = ThreadPoolExecutor(max_workers=1)

    if on_cluster() and use_submitit:
        executor = submitit_executor
        map_fun = map_array_from_submitit_executor(executor)

    else:
        executor_ = threadpool_executor
        map_fun = map_array_from_threadpool_executor(executor_)
    return map_fun


def submit_with_throttle(submitit_executor, fun, args, kwargs, retries=3, wait_time=60):
    # wait time is in seconds
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            job = submitit_executor.submit(fun, *args, **kwargs)
            break  # Success
        except FailedJobError as e:
            last_error = e
            print(f"[Submitit Retry] Attempt {attempt} failed: {e}")
            if attempt < retries:
                time.sleep(wait_time)
            else:
                print(f"[Submitit Retry] All {retries} attempts failed.")
                raise RuntimeError(
                    f"Submitit job failed after {retries} retries."
                ) from last_error
    return job


def get_execute_fun(submitit_executor, use_submitit, dir_):
    """
    Returns a function to execute a given callable (function) with arguments,
    either locally or on a cluster using a submitit executor.

    Parameters:
    submitit_executor: submitit.Executor
        The executor to use for submitting jobs to a cluster.
    use_submitit: bool
        If True, jobs will be submitted to the cluster using the executor.
        If False, the function will execute locally.

    Returns:
    execute_fun: Callable
        A function that takes the following arguments:
        - fun: Callable
            The function to execute.
        - args: Any
            Positional arguments to pass to the function.
        - kwargs: dict
            Keyword arguments to pass to the function.

        When `use_submitit` is True and `on_cluster()` returns True, the function
        will submit the job to the cluster and wait for the result.
        Otherwise, the function will be executed locally.
    """
    if on_cluster() and use_submitit:
        submitit_executor = submitit_executor(dir_)
        executor = submitit_executor

        def execute_fun(fun, *args, **kwargs):
            # Submit the job to the cluster and wait for the result
            # job = executor.submit(fun, *args, **kwargs)
            job = submit_with_throttle(executor, fun, args, kwargs)
            res = job.result()
            return res

    else:

        def execute_fun(fun, *args, **kwargs):
            # Execute the function locally
            return fun(*args, **kwargs)

    return execute_fun


# from experiments.scripts.destriping_poisson.supplementary_analyses.analysis import load_default_submitit_executor_dict
# from experiments.scripts.destriping_poisson.supplementary_analyses.utils.parallel_executor import get_execute_fun
# import os
# import time

# def test_fun(time_sec, statement):
#     time.sleep(time_sec)
#     print(statement)
#     return statement

# dir_ = "test_submitit"
# use_submitit = False
# time_sec = 30
# statement = "I made it"
# os.makedirs(dir_, exist_ok = True)
# default_executor_dict = load_default_submitit_executor_dict()
# submitit_executor = default_executor_dict["default"](dir_)

# executor = get_execute_fun(submitit_executor, use_submitit)
# statement_out = executor(test_fun, time_sec, statement)


def run_tasks_with_executor(executor, tasks: List[Callable[[], None]]):
    """
    Submit tasks to the executor and wait for their completion.

    Parameters:
        executor: The executor instance (e.g., ThreadPoolExecutor).
        tasks: A list of callables (functions with pre-specified arguments).
    """
    futures = [executor.submit(task) for task in tasks]
    results = []
    errors = []

    for i, future in enumerate(futures):
        try:
            result = future.result()  # Wait for the task to complete
            results.append((i, result))  # Store the result
        except Exception as e:
            # Collect error information with traceback
            error_traceback = traceback.format_exc()
            errors.append((i, error_traceback))

    # Raise a combined error if any occurred
    if errors:
        error_messages = "\n".join(
            f"Task {index} failed with error:\n{error}" for index, error in errors
        )
        raise RuntimeError(f"Errors occurred in the following tasks:\n{error_messages}")

    return results, errors


def df_apply_in_chunks(
    df,
    fun_by_row,
    submitit_executor,
    args=(),
    load_supplementary_args_fun=None,
    n_chunks=None,
    chunksize=None,
):
    """
    Apply a function to a DataFrame by row in parallel using submitit, splitting the DataFrame into chunks.

    Parameters:
        df: pd.DataFrame
            The DataFrame to process.
        fun_by_row: Callable
            Function to apply to each row. Should accept a row and keyword arguments.
        submitit_executor: submitit.Executor
            The submitit executor to use.
        args: dict or tuple
            Additional arguments to pass to fun_by_row.
        load_supplementary_args_fun: Callable
            Function to load supplementary arguments for each chunk (called inside the chunk).
        n_chunks: int
            Number of chunks to split the DataFrame into.
        chunksize: int
            Number of rows per chunk (alternative to n_chunks).

    Returns:
        pd.Series or pd.DataFrame: Concatenated results from all chunks.
    """
    if not isinstance(args, dict):
        args = dict(args)

    if n_chunks is not None and chunksize is not None:
        raise ValueError("Specify only one of n_chunks or chunksize, not both.")

    if chunksize is not None:
        if chunksize <= 0:
            raise ValueError("chunksize must be a positive integer.")
        n_chunks = int(np.ceil(len(df) / chunksize))
    elif n_chunks is None or n_chunks <= 0:
        n_chunks = 1

    map_fun = map_array_from_submitit_executor(submitit_executor)
    chunks = np.array_split(df, n_chunks)

    def chunk_apply(chunk):
        supplementary_args = (
            load_supplementary_args_fun(chunk) if load_supplementary_args_fun else {}
        )
        merged_args = {**supplementary_args, **args}
        return chunk.apply(lambda row: fun_by_row(row, **merged_args), axis=1)

    result_chunks = map_fun(chunk_apply, chunks)
    result = pd.concat(result_chunks).sort_index()
    return result
