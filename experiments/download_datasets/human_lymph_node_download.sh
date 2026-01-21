#!/usr/bin/env bash
source ./download_datasets.sh
LINKS_FILE="${PROJECT_ROOT}/experiments/download_datasets/links/human_lymph_node.txt"
FOLDER_PATH_REL="data/Visium_HD_Human_Lymph_Node"
download_datasets ${LINKS_FILE} ${FOLDER_PATH_REL} --decompress