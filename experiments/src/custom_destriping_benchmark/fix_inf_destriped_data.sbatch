#!/bin/bash
#SBATCH --job-name=fix_inf_destriped
#SBATCH --output=fix_inf_destriped_%j.out
#SBATCH --error=fix_inf_destriped_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G

set -euo pipefail

cd "$HOME/destriping-GLM-rebuttals"
pixi run python -m experiments.src.custom_destriping_benchmark.fix_inf_destriped_data
