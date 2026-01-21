# destriping-GLM

Code and experiments for a GLM-based approach to reduce striping artifacts in 10x Genomics VisiumHD total-count images using nuclei segmentation. 

The repository includes reproducible pipelines for real-data benchmarks and synthetic simulations, plus analysis notebooks to generate figures and summaries.


---

## Motivation

10x Genomics VisiumHD enables spatial transcriptomics at ~2 μm² resolution but often exhibits **slide-specific, non-periodic striping artifacts** due to lane-width variability. These multiplicative row/column effects distort bin total counts and can bias downstream analyses.

---

## Method

We assume that the *true* size of the bin in row $i$ and column $j$ equals $h_i w_j$, where $h_i$ and $w_j$ are the corresponding horizontal and vertical lane widths, respectively. We further assume the total transcript concentration is homogeneous within nuclei, such that the expected value of total counts in bin $[i,j]$ is:

$$\mu_{ij} = c_{p(i,j)} \, h_i \, w_j$$


where $c_p$ is the total transcript concentration in nucleus $p$ and $p(i,j)$ denotes the nucleus to which bin $[i,j]$ belongs.

Finally, we model observed total counts $y_{ij}$ with a Negative Binomial distribution:

$$y_{ij} \sim \mathrm{NB}(\mu_{ij}, \phi) \,,$$

with mean $\mu_{ij}$ and dispersion parameter $\phi$.

### GLM parameterization

We fit nucleus concentrations and stripe factors jointly in a generalized linear modeling framework, with:

- **cross-validated regularization** on stripe factors $(\log(h_i), \log(w_j))$,
- **iterative dispersion estimation** for $\phi$,
- a correction step that converts the fitted parameters into a **destriped counts image**.

Implementation note: GLM fitting and regularized optimization are performed using **glum** (QuantCo):  
https://github.com/Quantco/glum

---

## Results (high level)

### Synthetic data (known ground truth)
- Improved stripe-factor estimation accuracy.
- Lower error in corrected counts compared to `bin2cell` and `bin2cell`-derived baselines.

### Public VisiumHD slides

On 4 datasets (mouse brain, mouse embryo, zebrafish head and human lymph node):
- Consistently lowers striping intensity.
- Better preserves biological structure present in global/large-scale count patterns.
- Avoids artifacts (e.g., macro-stripes/edge effects) observed with sequential quantile normalization.

---

## Installation

This repo uses `pixi`.

```bash
pixi install
```

### glum patch
The environment currently includes a patched `glum` build distributed via prefix.dev. We plan to upstream/merge the patch to the public repository soon.

---

## Experiments

### I. Real data experiments

#### 1) Download datasets
Download from the 10x Genomics website using:
- scripts in `experiments/download_datasets`, **or**
- the direct links listed in `experiments/download_datasets/links`

#### 2) Prepare data (preprocessing / segmentation)
Run the relevant scripts:
- `experiments/src/custom_destriping_benchmark/prepare_human_lymph_node.py`
- `experiments/src/custom_destriping_benchmark/prepare_mouse_embryo.py`
- `experiments/src/custom_destriping_benchmark/prepare_zebrafish_head.py`
- `experiments/src/custom_destriping_benchmark/segment_mouse_brain.py`

#### 3) Run the benchmark
Configs live in:
- `experiments/hydra_config/destriping_model`

Benchmarks are launched via:
- `experiments/launchers/custom_destriping_benchmark/benchmark`

**Note:** Hydra outputs are time-stamped. You may need to adjust paths in configs to match the latest output directories produced in step (2).

#### 4) Benchmark analysis & plots
All results and figures are reproducible with notebooks in:
- `notebooks/`

Each notebook reads the benchmark output directory from a `.yaml` file in:
- `experiments/benchmark_output_files`

After running benchmarks, update the appropriate YAML(s) with the output directory path(s) from step (3).

---

### II. Synthetic data generation

#### 1) Build a counts-structure template from the real mouse brain slide
```bash
python experiments/src/custom_destriping_benchmark/generate_n_counts_structure_for_synthetic_data.py
```

You may need to adjust the mouse brain path to match the output of:
- `segment_mouse_brain.py` (from Real Data step I.2)

#### 2) Generate synthetic datasets via Hydra configs
Hydra configs:
- `experiments/hydra_config/destriping_model/simulation_generate_data/run_weibul.yaml` (single seed)
- `experiments/hydra_config/destriping_model/simulation_generate_data/sweep_big_with_n_counts_structure_weibul_nb.yaml` (multi-seed sweep)

Launchers:
- `experiments/launchers/custom_destriping_benchmark/generate_synthetic_data_1_seed.sh`
- `experiments/launchers/custom_destriping_benchmark/generate_synthetic_data_other_seeds.sh`

Before running, update:
- `segmentation_mask_path` in  
  `experiments/hydra_config/destriping_model/simulation_generate_data/simulation_params/big_with_n_counts_structure_weibul_nb.yaml`
  to point to the output of step I.2  
- `expected_concentration_gene_profile_path` to point to the output produced by the synthetic-data setup pipeline

Remaining steps (benchmarking + analysis) mirror the real-data workflow.

---

## Repository structure

### Source code
- `src/destriping/GLUM/glum_wrapper.py`  
  `GlumWrapper`: main class fitting the Negative Binomial model to spatial transcriptomics data
- `src/destriping/GLUM/`  
  iterative fitting + cross-validation logic
- `src/spatialAdata/spatialAdata.py`  
  container class for spatial transcriptomics data
- `src/experiments_analysis/`  
  result processing + plotting utilities
- `notebooks/`  
  analyses and figures for the publication

### Experiments
- `experiments/hydra_config/`  
  Hydra configs to launch experiments
- `experiments/launchers/`  
  bash scripts to launch preprocessing / benchmark runs
- Hydra outputs are automatically time-stamped and must be reported in:
  - `experiments/benchmark_output_files/`
  so the analysis notebooks can find them.

---

## Hydra + cluster/local note

If runs are launched locally (instead of on a compute cluster), the line below in the primary Hydra config needs to be commented out:

- `- override hydra/launcher: ../../../../hydra/launcher/cluster_4G`

Also comment out cluster-specific options such as `mem_per_cpu: ...` or `timeout_min`.
