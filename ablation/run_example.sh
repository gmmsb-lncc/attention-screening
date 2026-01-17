#!/bin/bash
# Example execution script for dual-dataset ablation study
# This demonstrates how to run the complete pipeline for both datasets

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_PYTHON="/media/leon/ssd2tb/docktkinase/env/bin/python"

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}ABLATION STUDY: DUAL-DATASET EXECUTION${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""

# Function to run a command with logging
run_step() {
    local description=$1
    local command=$2
    
    echo -e "${YELLOW}→ ${description}${NC}"
    echo "  Command: ${command}"
    
    if eval ${command}; then
        echo -e "${GREEN}  ✓ Success${NC}"
        echo ""
    else
        echo -e "${RED}  ✗ Failed${NC}"
        exit 1
    fi
}

# Menu
echo "Select execution mode:"
echo "  1. Non-human dataset only (already complete)"
echo "  2. Human dataset only (new)"
echo "  3. Both datasets sequentially"
echo "  4. Classification only (specify dataset)"
echo "  5. Regression only (specify dataset)"
echo ""
read -p "Enter choice [1-5]: " choice

case $choice in
    1)
        DATASET="non_human"
        TASK="both"
        ;;
    2)
        DATASET="human"
        TASK="both"
        ;;
    3)
        DATASET="both"
        TASK="both"
        ;;
    4)
        DATASET="both"
        TASK="classification"
        read -p "Which dataset? [non_human/human/both]: " DATASET
        ;;
    5)
        DATASET="both"
        TASK="regression"
        read -p "Which dataset? [non_human/human/both]: " DATASET
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}Configuration:${NC}"
echo "  Dataset: ${DATASET}"
echo "  Task: ${TASK}"
echo ""
read -p "Proceed? [y/N]: " confirm

if [[ ! $confirm =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}STARTING EXECUTION${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""

# Execute using orchestrator script
run_step "Running ablation study" \
    "${VENV_PYTHON} ${SCRIPT_DIR}/run_ablation_study.py --dataset ${DATASET} --task ${TASK}"

echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}EXECUTION COMPLETE${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""

# Show results location
if [[ $DATASET == "non_human" || $DATASET == "both" ]]; then
    echo -e "${GREEN}Non-Human Results:${NC}"
    echo "  Classification: ${SCRIPT_DIR}/classification/results_non_human/"
    echo "  Regression: ${SCRIPT_DIR}/regression/results_non_human/"
    echo ""
fi

if [[ $DATASET == "human" || $DATASET == "both" ]]; then
    echo -e "${GREEN}Human Results:${NC}"
    echo "  Classification: ${SCRIPT_DIR}/classification/results_human/"
    echo "  Regression: ${SCRIPT_DIR}/regression/results_human/"
    echo ""
fi

echo -e "${GREEN}Next steps:${NC}"
echo "  1. Check figures in results_*/figures/"
echo "  2. Compare metrics in *_summary.csv files"
echo "  3. Analyze cross-species performance"
echo ""
