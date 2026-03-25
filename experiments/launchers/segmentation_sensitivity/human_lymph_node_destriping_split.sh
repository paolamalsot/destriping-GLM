#!/bin/bash
source ../../utilities/launching_hydra_sweep.sh

multirun segmentation_sensitivity/destriping/splitting/human_lymph_node_seed_42 &
sleep 5
multirun segmentation_sensitivity/destriping/splitting/human_lymph_node_seed_64 &
sleep 5
multirun segmentation_sensitivity/destriping/splitting/human_lymph_node_seed_754 &
sleep 5
