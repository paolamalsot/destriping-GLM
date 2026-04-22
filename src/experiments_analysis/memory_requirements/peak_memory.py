"""
Utilities to extract peak memory usage (MaxRSS) from SLURM sacct for Hydra runs.

Each Hydra run directory contains a `.submitit/` sub-directory whose immediate
children are the SLURM job IDs submitted for that run (one per array task for
sweep jobs, a single directory for single-task jobs).  `sacct` is queried with
those IDs and the MaxRSS of the python-process step (`JobID` ending in `.0`) is
returned.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def find_job_ids(run_dir: Path) -> list[str]:
    """Return SLURM job IDs found in ``<run_dir>/.submitit/``.

    The `.submitit` directory contains one sub-directory per submitted job,
    named after the numeric SLURM job ID.  Array-job task indices (e.g.
    ``52875586_0``) are *not* returned here; sacct expands them automatically
    when given the base ID.
    """
    submitit_dir = Path(run_dir) / ".submitit"
    if not submitit_dir.exists():
        raise FileNotFoundError(f"No .submitit directory found in {run_dir}")
    return [
        p.name
        for p in sorted(submitit_dir.iterdir())
        if p.is_dir() and re.match(r"^\d+$", p.name)
    ]


def query_sacct(job_ids: list[str]) -> pd.DataFrame:
    """Run ``sacct`` for *job_ids* and return a parsed DataFrame.

    Columns: ``JobID``, ``MaxRSS``, ``Elapsed``, ``State``.
    """
    result = subprocess.run(
        [
            "sacct",
            "-j", ",".join(job_ids),
            "--format=JobID,MaxRSS,Elapsed,State",
            "--noheader",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    rows = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        rows.append(
            {"JobID": parts[0], "MaxRSS": parts[1], "Elapsed": parts[2], "State": parts[3]}
        )
    return pd.DataFrame(rows, columns=["JobID", "MaxRSS", "Elapsed", "State"])


def _parse_rss_to_gb(rss_str: str) -> float | None:
    """Parse a sacct MaxRSS string (e.g. ``'9526900K'``) to GB.

    Returns ``None`` for empty or zero values.
    """
    if not rss_str or rss_str.strip() in ("", "0"):
        return None
    m = re.match(r"^(\d+)([KMG]?)$", rss_str.strip())
    if not m:
        return None
    value, unit = float(m.group(1)), m.group(2)
    kb = {"K": value, "M": value * 1024, "G": value * 1024 ** 2, "": value / 1024}.get(unit)
    return kb / (1024 ** 2) if kb is not None else None  # KB → GB


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_peak_memory_gb(run_dir: Path) -> dict:
    """Return peak memory in GB for a Hydra run directory, via sacct.

    Locates ``.submitit/``, collects SLURM job IDs, queries sacct, and returns
    the maximum MaxRSS across all python-process steps (those whose ``JobID``
    ends with ``.0``).

    Returns
    -------
    dict with keys:
        ``peak_memory_GB`` : float
        ``job_ids``        : list[str]
        ``df_sacct``       : pd.DataFrame  (full sacct output, all steps)
    """
    job_ids = find_job_ids(Path(run_dir))
    df = query_sacct(job_ids)
    df_python = df[df["JobID"].str.endswith(".0")].copy()
    df_python["MaxRSS_GB"] = df_python["MaxRSS"].apply(_parse_rss_to_gb)
    peak_gb = df_python["MaxRSS_GB"].max()
    return {"peak_memory_GB": peak_gb, "job_ids": job_ids, "df_sacct": df}


def build_memory_table_from_job_ids(runs: dict[str, str | list[str]]) -> pd.DataFrame:
    """Build a peak memory table from explicit SLURM job IDs.

    Use this when ``.submitit/`` directories are not available (e.g. the results
    directory is a symlink that does not include hidden subdirectories, or the
    job IDs are already known from a prior ``sacct`` query).

    Parameters
    ----------
    runs:
        Mapping from a human-readable label to a single job ID string or a list
        of job ID strings (for array jobs).

    Returns
    -------
    pd.DataFrame with columns ``run`` and ``peak_memory_GB``.
    """
    records = []
    for label, job_ids in runs.items():
        if isinstance(job_ids, str):
            job_ids = [job_ids]
        df = query_sacct(job_ids)
        df_python = df[df["JobID"].str.endswith(".0")].copy()
        df_python["MaxRSS_GB"] = df_python["MaxRSS"].apply(_parse_rss_to_gb)
        peak_gb = df_python["MaxRSS_GB"].max()
        records.append({"run": label, "peak_memory_GB": round(peak_gb, 2)})
    return pd.DataFrame(records)


def build_memory_table(runs: dict[str, Path | str]) -> pd.DataFrame:
    """Build a peak memory table for a mapping of ``{label: run_dir}``.

    Parameters
    ----------
    runs:
        Mapping from a human-readable label to a Hydra run directory
        (the timestamped leaf directory that contains ``.submitit/``).

    Returns
    -------
    pd.DataFrame with columns ``run`` and ``peak_memory_GB``.
    """
    records = []
    for label, run_dir in runs.items():
        try:
            result = get_peak_memory_gb(Path(run_dir))
            records.append(
                {"run": label, "peak_memory_GB": round(result["peak_memory_GB"], 2)}
            )
        except Exception as e:
            records.append(
                {"run": label, "peak_memory_GB":"not found"}
            )
    return pd.DataFrame(records)
