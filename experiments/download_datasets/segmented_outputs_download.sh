#!/usr/bin/env bash
source ./download_datasets.sh

cd ../..

# Download only nucleus_segmentations.geojson from a Space Ranger segmented_outputs archive.
# Streams curl directly into tar — the .tar.gz is never written to disk.
download_nucleus_segmentations_geojson() {
  local url="$1"
  local dest_dir="${PROJECT_ROOT}/$2"
  local target_in_archive="segmented_outputs/nucleus_segmentations.geojson"
  local dest_file="${dest_dir}/nucleus_segmentations.geojson"

  mkdir -p "${dest_dir}"

  echo "Downloading: ${url}"
  echo "Extracting : ${target_in_archive}"
  curl -L -f --retry 5 --retry-delay 2 --progress-bar "${url}" \
    | tar -xzf - -C "${dest_dir}" --strip-components=1 "${target_in_archive}"
  echo "Saved      : ${dest_file}"
  echo
}

LINKS_DIR="${PROJECT_ROOT}/experiments/download_datasets/segmented_outputs_links"

download_nucleus_segmentations_geojson "$(cat "${LINKS_DIR}/human_lymph_node.txt")" "data/Visium_HD_Human_Lymph_Node/segmented_outputs"
download_nucleus_segmentations_geojson "$(cat "${LINKS_DIR}/mouse_brain.txt")"      "data/Visium_HD_Mouse_Brain/segmented_outputs"
download_nucleus_segmentations_geojson "$(cat "${LINKS_DIR}/mouse_embryo.txt")"     "data/Visium_HD_Mouse_Embryo/segmented_outputs"
download_nucleus_segmentations_geojson "$(cat "${LINKS_DIR}/zebrafish_head.txt")"   "data/Visium_HD_Zebrafish_Head/segmented_outputs"
