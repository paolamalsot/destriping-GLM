#!/bin/bash
source ../../utilities/launching_hydra_sweep.sh

multirun destriping_model/simulation_big_dataset/big_with_struct_nb_weibul_other_seeds/glum_benchmark_v6/v6_sensitivity_dataset &
sleep 5
