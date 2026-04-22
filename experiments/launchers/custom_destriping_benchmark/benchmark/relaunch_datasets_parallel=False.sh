#!/bin/bash
source ../../../utilities/launching_hydra_sweep.sh

multirun destriping_model/mouse_brain/glum_benchmark_v6/v6 &
sleep 5
multirun destriping_model/mouse_embryo/glum_benchmark_v6/v6 &
sleep 5
multirun destriping_model/human_lymph_node/glum_benchmark_v6/v6 &
sleep 5
multirun destriping_model/zebrafish_head/glum_benchmark_v6/v6 &
sleep 20