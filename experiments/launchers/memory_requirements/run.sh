#!/bin/bash
source ../../utilities/launching_hydra_sweep.sh

# Fitting only (max_iter=1) — 4 datasets
multirun memory_requirements/fitting/mouse_brain &
sleep 5
multirun memory_requirements/fitting/mouse_embryo &
sleep 5
multirun memory_requirements/fitting/human_lymph_node &
sleep 5
multirun memory_requirements/fitting/zebrafish_head &
sleep 5

# Destriping only (canonical sol, qm_tot_counts) — 4 datasets
# multirun memory_requirements/destriping/mouse_brain &
# sleep 5
# multirun memory_requirements/destriping/mouse_embryo &
# sleep 5
# multirun memory_requirements/destriping/human_lymph_node &
# sleep 5
# multirun memory_requirements/destriping/zebrafish_head &
# sleep 5
