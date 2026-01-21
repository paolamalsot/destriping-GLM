#!/bin/bash
source ../../utilities/launching_hydra_sweep.sh

multirun destriping_model/simulation_generate_data/sweep_big_with_n_counts_structure_weibul_nb.yaml
sleep 10