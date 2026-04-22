#!/usr/bin/env bash
source ./download_datasets.sh
LINKS_FILE="${PROJECT_ROOT}/experiments/download_datasets/links/mouse_brain.txt"
FOLDER_PATH_REL="data/Visium_HD_Mouse_Brain"
download_datasets ${LINKS_FILE} ${FOLDER_PATH_REL} --decompress