#!/bin/bash
source ../../utilities/launching_hydra_sweep.sh

multirun segmentation_sensitivity/destriping/splitting/simulation_seed_546_seed_42 &
sleep 5
multirun segmentation_sensitivity/destriping/splitting/simulation_seed_546_seed_64 &
sleep 5
multirun segmentation_sensitivity/destriping/splitting/simulation_seed_546_seed_754 &
sleep 5
