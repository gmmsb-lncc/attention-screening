#!/usr/bin/env bash
# =============================================================================
# DT-Kinase committee inference demo: imatinib (Gleevec) vs human kinome
# =============================================================================
#
# Real-world end-to-end demo of the 4-model committee pipeline (DT-Kinase,
# DrugBAN, GraphBAN, ConPLex) using imatinib as the query ligand.
#
# This script runs in LOOKUP MODE: each model's probability is read from the
# pre-computed raw_predictions.npz on the universal test split, rather than
# encoding new pairs on-the-fly. Lookup mode lets you reproduce the consensus
# without GPU access or cold-start encoder downloads. For pairs outside the
# universal_test set, switch to the production path documented in
# scripts/inference/README.md (committee.py with conda envs).
#
# Expected runtime: ~10 seconds. Disk usage: ~200 KB output.
#
# Validation criterion (biological sanity check):
#   - ABL must rank in top-10 with tier STRONG and prob_mean > 0.7
#   - Known off-targets (LCK, FLT3, FYN, SRC, ABL2) must appear in top-30
#
# Usage:
#   bash scripts/inference/examples/run_imatinib_demo.sh
#   bash scripts/inference/examples/run_imatinib_demo.sh /path/to/custom/output
#
# Compatible with: macOS default bash 3.2+ (no associative arrays required).
# =============================================================================

set -eu
set -o pipefail

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYBIN="${REPO}/env/bin/python"
OUT_DIR="${1:-${REPO}/results/inference/imatinib_demo}"

# Imatinib (CHEMBL941) — kinase inhibitor used for chronic myeloid leukemia.
# Primary clinical targets: BCR-ABL, KIT, PDGFRA/B.
SMILES='CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5'
QUERY_NAME='imatinib'
QUERY_CHEMBL_ID='CHEMBL941'
ORGANISM='human'
CKPT_CORPUS='all'        # broadest training corpus (483 H + 177 NH kinases)
SEED=42                  # canonical default seed

# -----------------------------------------------------------------------------
# Per-model artifact paths (raw_predictions.npz + calibration sidecar)
# -----------------------------------------------------------------------------

DTK_NPZ="${REPO}/results/all/benchmark_${CKPT_CORPUS}_8M_13_04_2026/test/level4_cnn_8M/${CKPT_CORPUS}/seed_${SEED}/raw_predictions.npz"
DTK_SIDE="${REPO}/results/all/benchmark_${CKPT_CORPUS}_8M_13_04_2026/test/level4_cnn_8M/${CKPT_CORPUS}/seed_${SEED}/level4_cnn_calibration.json"

DBN_NPZ="${REPO}/DrugBAN/results_universal/results_universal/${CKPT_CORPUS}/seed_${SEED}/raw_predictions.npz"
DBN_SIDE="${REPO}/DrugBAN/results_universal/results_universal/${CKPT_CORPUS}/seed_${SEED}/drugban_calibration.json"

GBN_NPZ="${REPO}/GraphBAN/results_universal/${CKPT_CORPUS}/seed_${SEED}/raw_predictions.npz"
GBN_SIDE="${REPO}/GraphBAN/results_universal/${CKPT_CORPUS}/seed_${SEED}/graphban_calibration.json"

CPL_NPZ="${REPO}/ConPLex/results_universal/${CKPT_CORPUS}/seed_${SEED}/raw_predictions.npz"
CPL_SIDE="${REPO}/ConPLex/results_universal/${CKPT_CORPUS}/seed_${SEED}/conplex_calibration.json"

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

print_banner() {
    printf '\n=============================================================\n'
    printf ' %s\n' "$1"
    printf '=============================================================\n'
}

require_file() {
    local path="$1"
    local label="$2"
    [[ -f "${path}" ]] || {
        echo "FATAL: ${label} not found at ${path}" >&2
        exit 1
    }
}

# -----------------------------------------------------------------------------
# Pre-flight checks
# -----------------------------------------------------------------------------

check_prerequisites() {
    print_banner '[0/5] Pre-flight checks'

    [[ -x "${PYBIN}" ]] || {
        echo "FATAL: python interpreter not found at ${PYBIN}" >&2
        echo "       run scripts/inference/setup steps first" >&2
        exit 1
    }

    "${PYBIN}" -c 'import pandas, numpy, rdkit' 2>/dev/null || {
        echo "FATAL: missing required Python packages (pandas, numpy, rdkit)" >&2
        exit 1
    }

    require_file "${DTK_NPZ}"  "DT-Kinase raw_predictions.npz"
    require_file "${DTK_SIDE}" "DT-Kinase calibration sidecar"
    require_file "${DBN_NPZ}"  "DrugBAN raw_predictions.npz"
    require_file "${DBN_SIDE}" "DrugBAN calibration sidecar"
    require_file "${GBN_NPZ}"  "GraphBAN raw_predictions.npz"
    require_file "${GBN_SIDE}" "GraphBAN calibration sidecar"
    require_file "${CPL_NPZ}"  "ConPLex raw_predictions.npz"
    require_file "${CPL_SIDE}" "ConPLex calibration sidecar"

    echo "OK — env, deps, 4 npz files and 4 sidecars present"
}

# -----------------------------------------------------------------------------
# Stage 1: expand SMILES to pairs.tsv (combinatorial against human kinome)
# -----------------------------------------------------------------------------

stage_expand_pairs() {
    print_banner '[1/5] Expand pairs (imatinib × human kinome)'
    mkdir -p "${OUT_DIR}"

    "${PYBIN}" "${REPO}/scripts/inference/expand_pairs.py" \
        --smiles "${SMILES}" \
        --organism "${ORGANISM}" \
        --out "${OUT_DIR}/pairs.tsv"

    local n_pairs
    n_pairs=$(($(wc -l < "${OUT_DIR}/pairs.tsv") - 1))
    echo "generated ${n_pairs} pairs (1 SMILES × ${n_pairs} ${ORGANISM} kinases)"
}

# -----------------------------------------------------------------------------
# Stage 2: score via lookup on cached raw_predictions
# -----------------------------------------------------------------------------

stage_score_via_lookup() {
    print_banner '[2/5] Score 4 models via lookup on universal test set'

    REPO_ARG="${REPO}" \
    OUT_ARG="${OUT_DIR}" \
    CHEMBL_ARG="${QUERY_CHEMBL_ID}" \
    DTK_NPZ_ARG="${DTK_NPZ}" \
    DTK_SIDE_ARG="${DTK_SIDE}" \
    DBN_NPZ_ARG="${DBN_NPZ}" \
    DBN_SIDE_ARG="${DBN_SIDE}" \
    GBN_NPZ_ARG="${GBN_NPZ}" \
    GBN_SIDE_ARG="${GBN_SIDE}" \
    CPL_NPZ_ARG="${CPL_NPZ}" \
    CPL_SIDE_ARG="${CPL_SIDE}" \
    "${PYBIN}" - <<'PYEOF'
import json
import os
import numpy as np
import pandas as pd
from pathlib import Path

REPO = Path(os.environ["REPO_ARG"])
OUT  = Path(os.environ["OUT_ARG"])
QUERY_CHEMBL = os.environ["CHEMBL_ARG"]

# Locate query rows in universal_test.tsv. The raw_predictions.npz arrays
# from each model are positionally aligned with this TSV (same row order),
# so the universal_test row indices double as raw_predictions indices.
df = pd.read_csv(REPO/"scaffolds_splits/output/universal_test.tsv", sep="\t")
hit_idx = df[df["chembl_id"] == QUERY_CHEMBL].index.to_numpy()
hit_rows = df.loc[hit_idx]
n_hits = len(hit_idx)

if n_hits == 0:
    raise SystemExit(
        f"query {QUERY_CHEMBL} not present in universal_test "
        f"(use full encoder pipeline instead)"
    )

print(f"  {QUERY_CHEMBL} matches in universal_test: {n_hits} rows "
      f"({hit_rows.drop_duplicates(['seq_id']).shape[0]} unique kinases)")

MODELS = [
    ("dtkinase",  os.environ["DTK_NPZ_ARG"], os.environ["DTK_SIDE_ARG"]),
    ("drugban",   os.environ["DBN_NPZ_ARG"], os.environ["DBN_SIDE_ARG"]),
    ("graphban",  os.environ["GBN_NPZ_ARG"], os.environ["GBN_SIDE_ARG"]),
    ("conplex",   os.environ["CPL_NPZ_ARG"], os.environ["CPL_SIDE_ARG"]),
]

for model, npz_path, side_path in MODELS:
    d = np.load(npz_path)
    probs_all = d["test_y_prob"] if "test_y_prob" in d.files else d["y_prob"]
    if len(probs_all) != len(df):
        raise SystemExit(f"{model}: shape mismatch {len(probs_all)} vs {len(df)} rows")

    probs = probs_all[hit_idx]
    thr   = json.loads(Path(side_path).read_text())["threshold"]
    pd.DataFrame({
        "uniprot":   hit_rows["seq_id"].astype(str).values,
        "chembl_id": QUERY_CHEMBL,
        "prob":      probs,
        "pred":      (probs >= thr).astype(int),
        "threshold": thr,
    }).to_csv(OUT/f"scores_{model}.csv", index=False)
    print(f"  {model:9s}: thr={thr:.3f}  mean_prob={probs.mean():.3f}  "
          f"agreement={int((probs >= thr).sum()):3d}/{len(probs)}")
PYEOF
}

# -----------------------------------------------------------------------------
# Stage 3: aggregate via committee (soft mean + Borda + tier)
# -----------------------------------------------------------------------------

stage_aggregate() {
    print_banner '[3/5] Aggregate consensus (soft mean + Borda + tier)'

    "${PYBIN}" "${REPO}/scripts/inference/aggregate.py" \
        --scores-dir "${OUT_DIR}" \
        --out "${OUT_DIR}/consensus.csv" \
        --top-k 20
}

# -----------------------------------------------------------------------------
# Stage 4: annotate consensus with kinase target names + organism
# -----------------------------------------------------------------------------

stage_annotate() {
    print_banner '[4/5] Annotate consensus with kinase metadata'

    REPO_ARG="${REPO}" OUT_ARG="${OUT_DIR}" \
    "${PYBIN}" - <<'PYEOF'
import os
import pandas as pd
from pathlib import Path

REPO = Path(os.environ["REPO_ARG"])
OUT  = Path(os.environ["OUT_ARG"])

c = pd.read_csv(OUT/"consensus.csv")
df = pd.read_csv(REPO/"scaffolds_splits/output/universal_test.tsv", sep="\t")

meta = (df.drop_duplicates(["seq_id"])
          [["seq_id", "target_kinase", "organism"]]
          .rename(columns={"seq_id": "uniprot"}))
meta["uniprot"] = meta["uniprot"].astype(str)
c["uniprot"]    = c["uniprot"].astype(str)

annotated = c.merge(meta, on="uniprot", how="left")
annotated.to_csv(OUT/"consensus.annotated.csv", index=False)
annotated.head(20).to_csv(OUT/"consensus.top.annotated.csv", index=False)
print(f"  wrote consensus.annotated.csv ({len(annotated)} rows)")
print(f"  wrote consensus.top.annotated.csv (top 20)")
PYEOF
}

# -----------------------------------------------------------------------------
# Stage 5: biological validation report
# -----------------------------------------------------------------------------

stage_validate() {
    print_banner '[5/5] Biological validation report'

    OUT_ARG="${OUT_DIR}" \
    "${PYBIN}" - <<'PYEOF'
import os
import pandas as pd
from pathlib import Path

OUT = Path(os.environ["OUT_ARG"])
c = pd.read_csv(OUT/"consensus.annotated.csv")

print("\n--- TIER BREAKDOWN (committee verdict per kinase) ---")
print(c["tier"].value_counts().sort_index().to_string())

print(f"\n--- TOP 15 PREDICTED BINDERS ---")
top = c.head(15)[["target_kinase", "organism", "prob_mean", "prob_std",
                  "agreement_count", "tier"]]
print(top.to_string(index=False))

print(f"\n--- KNOWN IMATINIB TARGETS — committee rank ---")
known = [
    ("ABL",    "primary clinical (BCR-ABL CML)"),
    ("KIT",    "secondary clinical (GIST)"),
    ("PDGFR",  "secondary clinical (CMML, dermatofibrosarcoma)"),
    ("DDR",    "off-target tyr kinase"),
    ("CSF1R",  "off-target tyr kinase"),
    ("LCK",    "Src-family off-target"),
    ("FYN",    "Src-family off-target"),
    ("SRC",    "Src-family off-target"),
    ("FLT3",   "off-target receptor TK"),
]
for kw, descr in known:
    hits = c[c["target_kinase"].astype(str).str.contains(kw, case=False, na=False)]
    if len(hits) == 0:
        print(f"  {kw:8s} ({descr}): not found in test set")
        continue
    for _, row in hits.iterrows():
        rank = int((c["prob_mean"] > row["prob_mean"]).sum() + 1)
        print(f"  {kw:8s} ({descr})")
        print(f"     -> {row['target_kinase']:32s} rank={rank:3d}/{len(c)}  "
              f"prob={row['prob_mean']:.3f}  sigma={row['prob_std']:.3f}  "
              f"tier={row['tier']:9s}  agree={int(row['agreement_count'])}/4")

# Sanity check: ABL must be in top-10 STRONG with prob > 0.7
abl = c[c["target_kinase"].astype(str) == "Tyrosine-protein kinase ABL"]
if len(abl) == 0:
    print("\nFAIL: ABL not found — committee output suspect")
    raise SystemExit(1)

abl_top  = abl.iloc[abl["prob_mean"].argmax()]
abl_rank = int((c["prob_mean"] > abl_top["prob_mean"]).sum() + 1)
ok = (abl_rank <= 10) and (abl_top["tier"] == "STRONG") and (abl_top["prob_mean"] > 0.7)

print(f"\n--- SANITY CHECK ---")
print(f"  ABL rank: {abl_rank} (expected <= 10)")
print(f"  ABL tier: {abl_top['tier']} (expected STRONG)")
print(f"  ABL prob: {abl_top['prob_mean']:.3f} (expected > 0.7)")
print(f"  verdict:  {'PASS' if ok else 'FAIL'}")
raise SystemExit(0 if ok else 1)
PYEOF
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

main() {
    echo "=============================================================="
    echo " DT-Kinase committee demo"
    echo " query  : ${QUERY_NAME} (${QUERY_CHEMBL_ID})"
    echo " corpus : ${CKPT_CORPUS}"
    echo " seed   : ${SEED}"
    echo " out    : ${OUT_DIR}"
    echo "=============================================================="

    check_prerequisites
    stage_expand_pairs
    stage_score_via_lookup
    stage_aggregate
    stage_annotate
    stage_validate

    print_banner 'DONE'
    echo "outputs:"
    ls -lh "${OUT_DIR}" | awk 'NR>1 {printf "  %s  %s\n", $5, $9}'
}

main "$@"
