#!/usr/bin/env bash
# download_datasets.sh
#
# Provides:
#   - PROJECT_ROOT (exported)
#   - download_datasets <links_file> <folder_path_rel> [--decompress]
#   - decompress_in_folder <abs_or_rel_folder>  (rel is relative to PROJECT_ROOT)

set -euo pipefail

_DOWNLOAD_DATASETS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="$(cd -- "${_DOWNLOAD_DATASETS_DIR}/../.." && pwd)"

decompress_in_folder() {
  if [[ $# -ne 1 ]]; then
    echo "Usage: decompress_in_folder <folder>" >&2
    return 2
  fi

  local folder="$1"
  # Allow relative paths (relative to PROJECT_ROOT)
  if [[ "${folder}" != /* ]]; then
    folder="${PROJECT_ROOT}/${folder}"
  fi

  if [[ ! -d "${folder}" ]]; then
    echo "ERROR: Folder not found: ${folder}" >&2
    return 1
  fi

  shopt -s nullglob

  local f
  local did_any=0

  # .tar.gz / .tgz
  for f in "${folder}"/*.tar.gz "${folder}"/*.tgz; do
    [[ -e "$f" ]] || continue
    echo "Extracting tar archive: $(basename "$f")"
    tar -xzf "$f" -C "${folder}"
    did_any=1
  done

  # .zip
  for f in "${folder}"/*.zip; do
    [[ -e "$f" ]] || continue
    if ! command -v unzip >/dev/null 2>&1; then
      echo "WARNING: unzip not installed; skipping $(basename "$f")" >&2
      continue
    fi
    echo "Extracting zip archive: $(basename "$f")"
    unzip -o "$f" -d "${folder}" >/dev/null
    did_any=1
  done

  # .gz (single-file gzip). Skip .tar.gz which we already handled.
  for f in "${folder}"/*.gz; do
    [[ -e "$f" ]] || continue
    [[ "$f" == *.tar.gz ]] && continue
    echo "Decompressing gzip: $(basename "$f")"
    gunzip -kf "$f"   # -k keep original, -f overwrite output if needed
    did_any=1
  done

  shopt -u nullglob

  if [[ $did_any -eq 0 ]]; then
    echo "No compressed files found in: ${folder}"
  else
    echo "Decompression complete in: ${folder}"
  fi
}

download_datasets() {
  if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: download_datasets <links_file> <folder_path_rel> [--decompress]" >&2
    return 2
  fi

  local links_file="$1"
  local folder_path_rel="$2"
  local do_decompress="${3:-}"

  local dest_dir="${PROJECT_ROOT}/${folder_path_rel}"

  if [[ ! -f "${links_file}" ]]; then
    echo "ERROR: Links file not found: ${links_file}" >&2
    return 1
  fi

  mkdir -p "${dest_dir}"

  local downloader=""
  if command -v curl >/dev/null 2>&1; then
    downloader="curl"
  elif command -v wget >/dev/null 2>&1; then
    downloader="wget"
  else
    echo "ERROR: Neither curl nor wget is installed." >&2
    return 1
  fi

  echo "Project root : ${PROJECT_ROOT}"
  echo "Links file   : ${links_file}"
  echo "Destination  : ${dest_dir}"
  echo "Downloader   : ${downloader}"
  echo

  local url filename out_path
  while IFS= read -r url || [[ -n "${url}" ]]; do
    # Trim whitespace
    url="${url#"${url%%[![:space:]]*}"}"
    url="${url%"${url##*[![:space:]]}"}"

    [[ -z "${url}" ]] && continue
    [[ "${url}" =~ ^# ]] && continue

    filename="$(basename "${url}")"
    out_path="${dest_dir}/${filename}"

    echo "Downloading: ${url}"
    if [[ "${downloader}" == "curl" ]]; then
      curl -L -f -C - \
        --retry 5 --retry-delay 2 \
        --progress-bar \
        -o "${out_path}.partial" "${url}"
      mv -f "${out_path}.partial" "${out_path}"
    else
      wget -c \
        --tries=5 --waitretry=2 \
        --progress=bar:force:noscroll \
        -O "${out_path}.partial" "${url}"
      mv -f "${out_path}.partial" "${out_path}"
    fi

    echo "Saved to   : ${out_path}"
    echo
  done < "${links_file}"

  echo "All downloads complete."

  if [[ "${do_decompress}" == "--decompress" ]]; then
    echo
    decompress_in_folder "${dest_dir}"
  elif [[ -n "${do_decompress}" ]]; then
    echo "ERROR: Unknown option: ${do_decompress}" >&2
    return 2
  fi
}
