#!/bin/bash
# =============================================================================
# DockTKinase - Sequential Pipeline Execution with Multiple Protein Models
# =============================================================================
# This script runs the complete pipeline sequentially for each protein model.
# When one model finishes, the next one starts automatically.
#
# Usage:
#   ./scripts/run_all_protein_models.sh [input_file] [output_base_dir]
#
# Example:
#   ./scripts/run_all_protein_models.sh tests/datasets/kinase_non_human_compounds.tsv results/benchmark
#
# Author: DockTKinase Team
# Date: November 25, 2025
# =============================================================================

set -e  # Exit on error

# =============================================================================
# Configuration
# =============================================================================

# Default paths (can be overridden via command line)
INPUT_FILE="${1:-tests/datasets/kinase_non_human_compounds.tsv}"
OUTPUT_BASE="${2:-results/protein_model_benchmark}"

# Device configuration
DEVICE="auto"  # Options: auto, cpu, cuda, mps

# Seed for reproducibility
SEED=42

# Available protein models (ordered by size: smallest to largest)
# Total: 11 models

# ESM-2 Local Models (6 models)
ESM2_MODELS=(
    "esm2_t6_8M_UR50D"      # 320-dim, ~8M params
    "esm2_t12_35M_UR50D"    # 480-dim, ~35M params
    "esm2_t30_150M_UR50D"   # 640-dim, ~150M params
    "esm2_t33_650M_UR50D"   # 1280-dim, ~650M params
    "esm2_t36_3B_UR50D"     # 2560-dim, ~3B params
    "esm2_t48_15B_UR50D"    # 5120-dim, ~15B params
)

# ESM-C API Models (3 models - require ESM_API_KEY)
ESMC_MODELS=(
    "esmc-300m-2024-12"     # 960-dim, ~300M params
    "esmc-600m-2024-12"     # 1152-dim, ~600M params
    "esmc-6b-2024-12"       # 4096-dim, ~6B params (API only)
)

# Structure-based Models (2 models)
STRUCTURE_MODELS=(
    "openfold3"             # 1536-dim, Structure prediction
    "boltz2"                # 384-dim, Boltz-2 embeddings
)

# =============================================================================
# Select which models to run (uncomment/modify as needed)
# =============================================================================

# Option 1: Run ALL 11 models (ESM-2 + ESM-C + Structure)
MODELS_TO_RUN=("${ESM2_MODELS[@]}" "${ESMC_MODELS[@]}" "${STRUCTURE_MODELS[@]}")

# Option 2: Run only ESM-2 local models (no API key required) - uncomment below
# MODELS_TO_RUN=("${ESM2_MODELS[@]}")

# Option 3: Run only ESM-C API models (require ESM_API_KEY) - uncomment below
# MODELS_TO_RUN=("${ESMC_MODELS[@]}")

# Option 4: Run only structure-based models - uncomment below
# MODELS_TO_RUN=("${STRUCTURE_MODELS[@]}")

# =============================================================================
# Logging Configuration
# =============================================================================

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${OUTPUT_BASE}/logs_${TIMESTAMP}"
SUMMARY_FILE="${LOG_DIR}/execution_summary.txt"

# =============================================================================
# Helper Functions
# =============================================================================

print_header() {
    echo ""
    echo "============================================================================="
    echo "$1"
    echo "============================================================================="
    echo ""
}

print_model_info() {
    local model=$1
    local index=$2
    local total=$3
    
    echo ""
    echo "┌─────────────────────────────────────────────────────────────────────────┐"
    echo "│  Model ${index}/${total}: ${model}"
    echo "│  Started: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "└─────────────────────────────────────────────────────────────────────────┘"
    echo ""
}

get_model_dim() {
    local model=$1
    case $model in
        "esm2_t6_8M_UR50D")     echo "320" ;;
        "esm2_t12_35M_UR50D")   echo "480" ;;
        "esm2_t30_150M_UR50D")  echo "640" ;;
        "esm2_t33_650M_UR50D")  echo "1280" ;;
        "esm2_t36_3B_UR50D")    echo "2560" ;;
        "esm2_t48_15B_UR50D")   echo "5120" ;;
        "esmc-300m-2024-12")    echo "960" ;;
        "esmc-600m-2024-12")    echo "1152" ;;
        "esmc-6b-2024-12")      echo "4096" ;;
        "openfold3")            echo "1536" ;;
        "boltz2")               echo "384" ;;
        *)                      echo "unknown" ;;
    esac
}

format_duration() {
    local seconds=$1
    local hours=$((seconds / 3600))
    local minutes=$(((seconds % 3600) / 60))
    local secs=$((seconds % 60))
    printf "%02d:%02d:%02d" $hours $minutes $secs
}

# =============================================================================
# Pre-flight Checks
# =============================================================================

print_header "DockTKinase - Multi-Model Pipeline Execution"

echo "📋 Configuration:"
echo "   Input File:    ${INPUT_FILE}"
echo "   Output Base:   ${OUTPUT_BASE}"
echo "   Device:        ${DEVICE}"
echo "   Seed:          ${SEED}"
echo "   Models:        ${#MODELS_TO_RUN[@]}"
echo "   Log Directory: ${LOG_DIR}"
echo ""

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "❌ Error: Input file not found: ${INPUT_FILE}"
    exit 1
fi

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Warning: Virtual environment not activated."
    echo "   Attempting to activate env/bin/activate..."
    if [ -f "env/bin/activate" ]; then
        source env/bin/activate
        echo "   ✅ Virtual environment activated."
    else
        echo "   ❌ Could not find virtual environment. Continuing anyway..."
    fi
fi

# Check for ESM API key if running API models
for model in "${MODELS_TO_RUN[@]}"; do
    if [[ $model == esmc-* ]] && [ -z "$ESM_API_KEY" ]; then
        echo "⚠️  Warning: ESM_API_KEY not set. API models (esmc-*) may fail."
        break
    fi
done

# Create log directory
mkdir -p "$LOG_DIR"

# =============================================================================
# Main Execution Loop
# =============================================================================

print_header "Starting Sequential Pipeline Execution"

TOTAL_MODELS=${#MODELS_TO_RUN[@]}
SUCCESSFUL=0
FAILED=0
GLOBAL_START=$(date +%s)

# Initialize summary file
cat > "$SUMMARY_FILE" << EOF
================================================================================
DockTKinase - Multi-Model Pipeline Execution Summary
================================================================================
Started:     $(date '+%Y-%m-%d %H:%M:%S')
Input File:  ${INPUT_FILE}
Output Base: ${OUTPUT_BASE}
Device:      ${DEVICE}
Seed:        ${SEED}
Total Models: ${TOTAL_MODELS}

================================================================================
Model Results
================================================================================

EOF

for i in "${!MODELS_TO_RUN[@]}"; do
    MODEL="${MODELS_TO_RUN[$i]}"
    MODEL_INDEX=$((i + 1))
    MODEL_DIM=$(get_model_dim "$MODEL")
    
    # Create output directory for this model
    OUTPUT_DIR="${OUTPUT_BASE}/${MODEL}"
    LOG_FILE="${LOG_DIR}/${MODEL}.log"
    
    print_model_info "$MODEL" "$MODEL_INDEX" "$TOTAL_MODELS"
    
    echo "📊 Model Details:"
    echo "   Embedding Dim: ${MODEL_DIM}"
    echo "   Output Dir:    ${OUTPUT_DIR}"
    echo "   Log File:      ${LOG_FILE}"
    echo ""
    
    # Record start time
    MODEL_START=$(date +%s)
    
    # Run the pipeline
    echo "🚀 Running pipeline..."
    echo ""
    
    # Build the command
    CMD="python run_complete_pipeline.py"
    CMD+=" --input ${INPUT_FILE}"
    CMD+=" --output ${OUTPUT_DIR}"
    CMD+=" --protein-model ${MODEL}"
    CMD+=" --device ${DEVICE}"
    CMD+=" --seed ${SEED}"
    
    # Add API key if available and model requires it
    if [[ $MODEL == esmc-* ]] && [ -n "$ESM_API_KEY" ]; then
        CMD+=" --api ${ESM_API_KEY}"
    fi
    
    echo "Command: ${CMD}"
    echo ""
    
    # Execute with logging
    if $CMD 2>&1 | tee "$LOG_FILE"; then
        MODEL_END=$(date +%s)
        DURATION=$((MODEL_END - MODEL_START))
        DURATION_FMT=$(format_duration $DURATION)
        
        echo ""
        echo "✅ Model ${MODEL} completed successfully!"
        echo "   Duration: ${DURATION_FMT}"
        echo ""
        
        SUCCESSFUL=$((SUCCESSFUL + 1))
        echo "✅ ${MODEL} | Duration: ${DURATION_FMT} | Status: SUCCESS" >> "$SUMMARY_FILE"
    else
        MODEL_END=$(date +%s)
        DURATION=$((MODEL_END - MODEL_START))
        DURATION_FMT=$(format_duration $DURATION)
        
        echo ""
        echo "❌ Model ${MODEL} failed!"
        echo "   Duration: ${DURATION_FMT}"
        echo "   Check log: ${LOG_FILE}"
        echo ""
        
        FAILED=$((FAILED + 1))
        echo "❌ ${MODEL} | Duration: ${DURATION_FMT} | Status: FAILED" >> "$SUMMARY_FILE"
    fi
    
    # Small pause between models (allow GPU memory cleanup)
    if [ $MODEL_INDEX -lt $TOTAL_MODELS ]; then
        echo "⏳ Waiting 10 seconds before next model (GPU memory cleanup)..."
        sleep 10
    fi
done

# =============================================================================
# Final Summary
# =============================================================================

GLOBAL_END=$(date +%s)
TOTAL_DURATION=$((GLOBAL_END - GLOBAL_START))
TOTAL_DURATION_FMT=$(format_duration $TOTAL_DURATION)

print_header "Execution Complete"

echo "📊 Final Summary:"
echo "   Total Models:  ${TOTAL_MODELS}"
echo "   Successful:    ${SUCCESSFUL}"
echo "   Failed:        ${FAILED}"
echo "   Total Time:    ${TOTAL_DURATION_FMT}"
echo ""
echo "📁 Results saved to: ${OUTPUT_BASE}"
echo "📋 Logs saved to:    ${LOG_DIR}"
echo "📄 Summary file:     ${SUMMARY_FILE}"

# Append final summary to file
cat >> "$SUMMARY_FILE" << EOF

================================================================================
Final Summary
================================================================================
Completed:    $(date '+%Y-%m-%d %H:%M:%S')
Total Time:   ${TOTAL_DURATION_FMT}
Successful:   ${SUCCESSFUL}/${TOTAL_MODELS}
Failed:       ${FAILED}/${TOTAL_MODELS}
================================================================================
EOF

# Exit with error code if any model failed
if [ $FAILED -gt 0 ]; then
    echo ""
    echo "⚠️  Warning: ${FAILED} model(s) failed. Check logs for details."
    exit 1
fi

echo ""
echo "🎉 All models completed successfully!"
exit 0
