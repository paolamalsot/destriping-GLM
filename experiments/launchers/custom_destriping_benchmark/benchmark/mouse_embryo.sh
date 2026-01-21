#!/bin/bash
source ../../../utilities/launching_hydra_sweep.sh

multirun destriping_model/mouse_embryo/baselines/b2c_destriping.yaml &
sleep 5
multirun destriping_model/mouse_embryo/baselines/dividing_by_factors_ratio &
sleep 5
multirun destriping_model/mouse_embryo/baselines/dividing_by_factors_median.yaml &
sleep 5
multirun destriping_model/mouse_embryo/baselines/dividing_by_factors &
sleep 5
multirun destriping_model/mouse_embryo/baselines/bin_level_normalization.yaml &
sleep 5
multirun destriping_model/mouse_embryo/glum_benchmark_v3/None_init &
sleep 5
multirun destriping_model/mouse_embryo/glum_benchmark_v3/v3 &
sleep 5
multirun destriping_model/mouse_embryo/glum_benchmark_v3/P2_identity &
sleep 20