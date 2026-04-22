#!/bin/bash
source ../../utilities/launching_hydra_sweep.sh

multirun segmentation_sensitivity/destriping/human_lymph_node/splitting/seed_42 &
sleep 5
multirun segmentation_sensitivity/destriping/human_lymph_node/splitting/seed_64 &
sleep 5
multirun segmentation_sensitivity/destriping/human_lymph_node/splitting/seed_754 &
sleep 5
