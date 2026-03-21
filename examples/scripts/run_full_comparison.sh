#!/bin/bash
#
# Full Comparison Script: Encoder Architectures + Baseline
# =========================================================
#
# This script runs two sets of experiments:
#   1. Encoder Architecture Comparison (Linear vs CNN vs CNN+Attention)
#   2. Baseline CrossAttention Analysis (CNN encoder, 500 epochs)
#
# Embeddings: 8M, 150M, 3B
# Dataset: non_human (can be changed below)
#
# Usage:
#   ./run_full_comparison.sh
#   ./run_full_comparison.sh human    # for human dataset
#
# Author: DockTKinase Team
# Date: January 2025

set -e  # Exit on error

# Configuration
DATASET="${1:-non_human}"
EPOCHS_COMPARISON=200
EPOCHS_BASELINE=500
EMBEDDINGS=("8M" "150M" "650M")

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Output directories
COMPARISON_OUTPUT="encoder_comparison_results"
BASELINE_OUTPUT="leakage_analysis_results"

# Log file
LOG_FILE="full_comparison_${DATASET}_$(date +%Y%m%d_%H%M%S).log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to log messages
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

log_section() {
    echo "" | tee -a "$LOG_FILE"
    echo -e "${BLUE}======================================================================${NC}" | tee -a "$LOG_FILE"
    echo -e "${BLUE}$1${NC}" | tee -a "$LOG_FILE"
    echo -e "${BLUE}======================================================================${NC}" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

# Print header
echo ""
echo "=========================================================================="
echo "         FULL ENCODER COMPARISON + BASELINE ANALYSIS"
echo "=========================================================================="
echo ""
echo "Configuration:"
echo "  Dataset:            $DATASET"
echo "  Embeddings:         ${EMBEDDINGS[*]}"
echo "  Comparison epochs:  $EPOCHS_COMPARISON"
echo "  Baseline epochs:    $EPOCHS_BASELINE"
echo "  Log file:           $LOG_FILE"
echo ""
echo "=========================================================================="
echo ""

# Check if GPU is available
if command -v nvidia-smi &> /dev/null; then
    log "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader | head -1
else
    log_warning "No GPU detected. Training will be slow on CPU."
fi

# Create output directories
mkdir -p "$COMPARISON_OUTPUT"
mkdir -p "$BASELINE_OUTPUT"

# Record start time
START_TIME=$(date +%s)

# ============================================================================
# PART 1: ENCODER ARCHITECTURE COMPARISON
# ============================================================================

log_section "PART 1: ENCODER ARCHITECTURE COMPARISON"
log "Comparing: Linear vs CNN vs CNN+Attention"
log "Epochs: $EPOCHS_COMPARISON"

for EMB in "${EMBEDDINGS[@]}"; do
    log_section "Encoder Comparison: $EMB"

    # Check if results already exist
    SHORT_NAME=$(echo "$EMB" | tr '[:upper:]' '[:lower:]')
    RESULT_FILE="${COMPARISON_OUTPUT}/${DATASET}_t*${SHORT_NAME}*_encoder_comparison.json"

    if ls $RESULT_FILE 1> /dev/null 2>&1; then
        log_warning "Results for $EMB already exist. Skipping. Use --force to recalculate."
        continue
    fi

    log "Starting encoder comparison for $EMB..."

    python compare_encoder_architectures.py \
        --embedding "$EMB" \
        --dataset "$DATASET" \
        --epochs "$EPOCHS_COMPARISON" \
        --output_dir "$COMPARISON_OUTPUT" \
        2>&1 | tee -a "$LOG_FILE"

    if [ $? -eq 0 ]; then
        log "Encoder comparison for $EMB completed successfully!"
    else
        log_error "Encoder comparison for $EMB failed!"
    fi
done

# ============================================================================
# PART 2: BASELINE CROSSATTENTION ANALYSIS
# ============================================================================

log_section "PART 2: BASELINE CROSSATTENTION ANALYSIS"
log "Using CNN encoder (original implementation)"
log "Epochs: $EPOCHS_BASELINE"

for EMB in "${EMBEDDINGS[@]}"; do
    log_section "Baseline Analysis: $EMB"

    log "Starting baseline analysis for $EMB..."

    python crossattention_split_analysis.py \
        --embedding "$EMB" \
        --dataset "$DATASET" \
        --output_dir "$BASELINE_OUTPUT" \
        2>&1 | tee -a "$LOG_FILE"

    if [ $? -eq 0 ]; then
        log "Baseline analysis for $EMB completed successfully!"
    else
        log_error "Baseline analysis for $EMB failed!"
    fi
done

# ============================================================================
# SUMMARY
# ============================================================================

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

log_section "SUMMARY"

echo "" | tee -a "$LOG_FILE"
echo "Total execution time: ${HOURS}h ${MINUTES}m ${SECONDS}s" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

echo "Results saved in:" | tee -a "$LOG_FILE"
echo "  Encoder comparison: $COMPARISON_OUTPUT/" | tee -a "$LOG_FILE"
echo "  Baseline analysis:  $BASELINE_OUTPUT/" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# List generated files
echo "Generated files:" | tee -a "$LOG_FILE"
echo "  Encoder comparison:" | tee -a "$LOG_FILE"
ls -la "$COMPARISON_OUTPUT"/*.json 2>/dev/null | tee -a "$LOG_FILE" || echo "    (none)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "  Baseline analysis:" | tee -a "$LOG_FILE"
ls -la "$BASELINE_OUTPUT"/*.json 2>/dev/null | tee -a "$LOG_FILE" || echo "    (none)" | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "=========================================================================="
echo "                    COMPARISON COMPLETE!"
echo "=========================================================================="
echo ""
echo "Log file: $LOG_FILE"
echo ""
