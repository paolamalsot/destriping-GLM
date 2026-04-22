#!/bin/bash
source ../../utilities/launching_hydra_sweep.sh

multirun segmentation_sensitivity/destriping/simulation_seed_546/collapse_label &
sleep 5
multirun segmentation_sensitivity/destriping/human_lymph_node/collapse_label &
sleep 5
