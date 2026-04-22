#!/bin/bash
source ../../../utilities/launching_hydra_sweep.sh

multirun destriping_model/human_lymph_node/baselines/b2c_destriping.yaml &
sleep 5
multirun destriping_model/human_lymph_node/baselines/dividing_by_factors_ratio &
sleep 5
multirun destriping_model/human_lymph_node/baselines/dividing_by_factors_median.yaml &
sleep 5
multirun destriping_model/human_lymph_node/baselines/dividing_by_factors &
sleep 5
multirun destriping_model/human_lymph_node/baselines/bin_level_normalization.yaml &
sleep 5
multirun destriping_model/human_lymph_node/glum_benchmark_v6/None_init &
sleep 5
multirun destriping_model/human_lymph_node/glum_benchmark_v6/v6 &
sleep 5
multirun destriping_model/human_lymph_node/glum_benchmark_v6/P2_identity &
sleep 20