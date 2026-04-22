#!/bin/bash
source ../../utilities/launching_hydra_sweep.sh

multirun segmentation_sensitivity/destriping/simulation_seed_546/subsampling/seed_42 &
sleep 5
multirun segmentation_sensitivity/destriping/simulation_seed_546/subsampling/seed_64 &
sleep 5
multirun segmentation_sensitivity/destriping/simulation_seed_546/subsampling/seed_754 &
sleep 5
