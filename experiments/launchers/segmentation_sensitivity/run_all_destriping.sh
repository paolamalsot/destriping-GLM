#!/bin/bash
source ../../utilities/launching_hydra_sweep.sh

# Human lymph node
multirun segmentation_sensitivity/destriping/human_lymph_node/merging/seed_42 &
sleep 5
multirun segmentation_sensitivity/destriping/human_lymph_node/merging/seed_64 &
sleep 5
multirun segmentation_sensitivity/destriping/human_lymph_node/merging/seed_754 &
sleep 5

multirun segmentation_sensitivity/destriping/human_lymph_node/splitting/seed_42 &
sleep 5
multirun segmentation_sensitivity/destriping/human_lymph_node/splitting/seed_64 &
sleep 5
multirun segmentation_sensitivity/destriping/human_lymph_node/splitting/seed_754 &
sleep 5

multirun segmentation_sensitivity/destriping/human_lymph_node/subsampling/seed_42 &
sleep 5
multirun segmentation_sensitivity/destriping/human_lymph_node/subsampling/seed_64 &
sleep 5
multirun segmentation_sensitivity/destriping/human_lymph_node/subsampling/seed_754 &
sleep 5

# Simulation seed 546
multirun segmentation_sensitivity/destriping/simulation_seed_546/merging/seed_42 &
sleep 5
multirun segmentation_sensitivity/destriping/simulation_seed_546/merging/seed_64 &
sleep 5
multirun segmentation_sensitivity/destriping/simulation_seed_546/merging/seed_754 &
sleep 5

multirun segmentation_sensitivity/destriping/simulation_seed_546/splitting/seed_42 &
sleep 5
multirun segmentation_sensitivity/destriping/simulation_seed_546/splitting/seed_64 &
sleep 5
multirun segmentation_sensitivity/destriping/simulation_seed_546/splitting/seed_754 &
sleep 5

multirun segmentation_sensitivity/destriping/simulation_seed_546/subsampling/seed_42 &
sleep 5
multirun segmentation_sensitivity/destriping/simulation_seed_546/subsampling/seed_64 &
sleep 5
multirun segmentation_sensitivity/destriping/simulation_seed_546/subsampling/seed_754 &
sleep 5
