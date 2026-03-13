#!/usr/bin/env bash
# ===========================================================================
# run_benchmark.sh — Ligand-weight sweep for 8M + MolFormer (non_human)
#
# Usage:
#   bash run_benchmark.sh
#
# Runs train then test for each ligand weight and prints MCC summary.
# ===========================================================================
set -euo pipefail

DATASET="non_human"
EMBEDDING="8M"
LIGAND_MODEL="molformer"
LEVELS=("2" "3")
PATIENCE="30"
WEIGHTS=("1.0" "1.25" "1.5" "2.0" "2.5" "3.0")
OUTPUT_ROOT="results/sweep_non_human_8M_molformer_ligw"

run_phase() {
    local mode="$1"
    local weight="$2"
    local out_dir="$3"

    python semantic_screening_models.py \
        --dataset "${DATASET}" \
        --embedding "${EMBEDDING}" \
        --ligand-model "${LIGAND_MODEL}" \
        --levels "${LEVELS[@]}" \
        --ligand-weight "${weight}" \
        --patience "${PATIENCE}" \
        --output_dir "${out_dir}" \
        "--${mode}"
}

echo "============================================================"
echo " Ligand-weight sweep | dataset=${DATASET} embedding=${EMBEDDING}"
echo " Model setup: ligand=${LIGAND_MODEL}, levels=${LEVELS[*]}"
echo " Weights: ${WEIGHTS[*]}"
echo "============================================================"

for w in "${WEIGHTS[@]}"; do
    out_dir="${OUTPUT_ROOT}/lw_${w}"

    echo ""
    echo "============================================================"
    echo " Weight ${w} — TRAIN (fit=train, eval=val)"
    echo "============================================================"
    run_phase "train" "${w}" "${out_dir}"

    echo ""
    echo "============================================================"
    echo " Weight ${w} — TEST (fit=val, eval=test)"
    echo "============================================================"
    run_phase "test" "${w}" "${out_dir}"
done

echo ""
echo "============================================================"
echo " MCC summary from benchmark_comparison.json"
echo "============================================================"
python - <<'PY'
import json
from pathlib import Path

root = Path("results/sweep_non_human_8M_molformer_ligw")
weights = ["1.0", "1.25", "1.5", "2.0", "2.5", "3.0"]
targets = [
    "level2_meanpool_knn",
    "level2_meanpool_mlp",
    "level3_attnpool_knn",
    "level3_attnpool_mlp",
]

print(f"{'weight':>6}  {'model':<20} {'MCC':>8} {'MCC_std':>8}")
print("-" * 50)

for w in weights:
    p = root / f"lw_{w}" / "benchmark_comparison.json"
    if not p.exists():
        print(f"{w:>6}  {'(missing benchmark_comparison.json)':<20}")
        continue

    with p.open() as fh:
        payload = json.load(fh)

    results = payload.get("results", {})
    for model in targets:
        row = results.get(model, {})
        mcc = row.get("mcc")
        mcc_std = row.get("mcc_std")
        mcc_txt = f"{mcc:.4f}" if isinstance(mcc, (float, int)) else "N/A"
        std_txt = f"{mcc_std:.4f}" if isinstance(mcc_std, (float, int)) else "N/A"
        print(f"{w:>6}  {model:<20} {mcc_txt:>8} {std_txt:>8}")

echo ""
echo "Best candidates should maximize MCC with low MCC_std."
PY

echo ""
echo "============================================================"
echo " Done. Sweep results saved under:"
echo "   ${OUTPUT_ROOT}/lw_*/"
echo "============================================================"
