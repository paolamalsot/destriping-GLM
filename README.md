To install the environment, install pixi and then run

`pixi install`

Note that our environment contains a patch for the glum-library, downloadable from prefix.dev, that we plan to merge to the public repository soon.

I. Real data experiments

1. Downloading datasets:
To download the datasets from the 10x Genomics website, run the scripts in `experiments/download_datasets`, or directly download the files from the links described in `experiments/download_datasets/links`

2. Prepare the data for the experiments:
experiments/src/custom_destriping_benchmark/prepare_human_lymph_node.py
experiments/src/custom_destriping_benchmark/prepare_mouse_embryo.py
experiments/src/custom_destriping_benchmark/prepare_zebrafish_head.py
experiments/src/custom_destriping_benchmark/segment_mouse_brain.py

3. Run the benchmark
All configs for the benchmark are located in experiments/hydra_config/destriping_model. Note that since the outputs of our files in step 2. is time-stamped, you might need to adjust some paths relative to this.
The benchmarks can be launched through the scripts in experiments/launchers/custom_destriping_benchmark/benchmark

4. Benchmark analysis & Plots
All the results and figures are reproducible through the notebooks in notebooks/
Each notebook fetches the results directory in a .yaml file located in experiments/benchmark_output_files. So first, report the results directories of step 3 in those config files.

--- 

Synthetic data generation

1. Generate the counts structure template from the real mouse brain slide:
python experiments/src/custom_destriping_benchmark/generate_n_counts_structure_for_synthetic_data.py
(Here again, adjust the data-path from step I.2 in segment_mouse_brain)

2. The hydra-config for the synthetic data generation is in experiments/hydra_config/destriping_model/simulation_generate_data/run_weibul.yaml (for 1 seed), or  experiments/hydra_config/destriping_model/simulation_generate_data/sweep_big_with_n_counts_structure_weibul_nb.yaml for all the seeds. The corresponding launchers are:
- experiments/launchers/custom_destriping_benchmark/generate_synthetic_data_1_seed.sh
- experiments/launchers/custom_destriping_benchmark/generate_synthetic_data_other_seeds.sh
Here again, one must first adjust the segmentation_mask_path in experiments/hydra_config/destriping_model/simulation_generate_data/simulation_params/big_with_n_counts_structure_weibul_nb.yaml from the output of step I.2. and the expected_concentration_gene_profile_path from the output of step II.2

Remaining steps for results generation are similar to the real data.

Repository structure

Source-code:
- src.destriping.GLUM.glum_wrapper.GlumWrapper: main class for our model for fitting the negative binomial model to spatial transcriptomics data using glum. 
- src/destriping/GLUM: code for the iterative fitting and cross-validation.
- src.spatialAdata.spatialAdata: class for storing spatial transcriptomics class
- src/experiments_analysis: processing results and plots
- notebooks: analysis and plots for the publication

Experiments:
- experiments/hydra_config: all the config-files to launch the experiments
- The bash scripts to launch the preprocessing and benchmark experiments are located in experiments/hydra_config/launchers
- The output files are automatically time-stamped by hydra and must be reported in the appropriate config file in experiments/benchmark_output_files in order that the analysis notebooks finds them.

General note: depending on whether the runs are launched locally or on a compute-cluster the line " - override hydra/launcher: ../../../../hydra/launcher/cluster_4G" in the primary config file of hydra must be commented out, along with cluster-specific configs such as "mem_per_cpu: ..."