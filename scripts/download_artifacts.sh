#!/usr/bin/env bash
# Download trained model weights and curated datasets from Zenodo.
#
# Usage:
#   bash scripts/download_artifacts.sh [weights|dataset|all]
#
# Default: all
#
# Requires: curl, tar, sha256sum (or shasum -a 256 on macOS).

set -euo pipefail

# Zenodo record IDs
ZENODO_WEIGHTS_RECORD="TBD-WEIGHTS"  # TODO: weights deposit pending
ZENODO_DATASET_RECORD="20350181"     # DT-Kinase dataset v1.0.0
# --------------------------------------------------------------------

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-all}"

zenodo_url() {
  echo "https://zenodo.org/records/$1/files-archive"
}

sha256_cmd() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum -c "$1"
  else
    shasum -a 256 -c "$1"
  fi
}

download_weights() {
  echo ">>> Downloading model weights from Zenodo (record $ZENODO_WEIGHTS_RECORD)"
  mkdir -p "$ROOT_DIR/results" "$ROOT_DIR/ConPLex/best_models" \
           "$ROOT_DIR/DrugBAN/results" "$ROOT_DIR/GraphBAN/results"
  local archive="$ROOT_DIR/zenodo_weights.tar.gz"
  curl -L -o "$archive" "$(zenodo_url "$ZENODO_WEIGHTS_RECORD")"
  tar -xzf "$archive" -C "$ROOT_DIR"
  rm "$archive"
  echo "Weights extracted into:"
  echo "  - results/{benchmark_*,all/benchmark_all_*}/test/level4_cnn_8M/*/seed_*/level4_cnn_model.pt"
  echo "  - ConPLex/best_models/trained_*/...best_model.pt"
  echo "  - DrugBAN/results/<corpus>/seed_*/best_model_epoch_*.pth"
  echo "  - GraphBAN/results/<corpus>/seed_*/best_model_epoch_*.pth"
}

download_dataset() {
  echo ">>> Downloading curated dataset from Zenodo (record $ZENODO_DATASET_RECORD)"
  mkdir -p "$ROOT_DIR/ConPLex/dataset" "$ROOT_DIR/scaffolds_splits/output"
  local archive="$ROOT_DIR/zenodo_dataset.tar.gz"
  curl -L -o "$archive" "$(zenodo_url "$ZENODO_DATASET_RECORD")"
  tar -xzf "$archive" -C "$ROOT_DIR"
  rm "$archive"
  echo "Dataset extracted into:"
  echo "  - ConPLex/dataset/kinase_{all,human,non_human}/{train,val,test}.csv"
  echo "  - scaffolds_splits/output/{human,non_human,universal}_{train,val,test}.tsv"
  echo "  - scaffolds_splits/output/scenarios/Sc/*.tsv"
}

case "$TARGET" in
  weights) download_weights ;;
  dataset) download_dataset ;;
  all)
    download_weights
    download_dataset
    ;;
  *)
    echo "Unknown target: $TARGET (use weights|dataset|all)" >&2
    exit 1
    ;;
esac

echo "Done."
