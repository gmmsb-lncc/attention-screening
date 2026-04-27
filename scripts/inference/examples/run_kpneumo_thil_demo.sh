#!/usr/bin/env bash
# =============================================================================
# DT-Kinase out-of-domain demo: K. pneumoniae thiamine monophosphate kinase
# =============================================================================
#
# Real-world demo of inference under EXTRAPOLATION conditions: querying a
# bacterial metabolic kinase (ThiL, EC 2.7.4.16) that lies outside the
# training distribution.
#
# Why this matters: the training set (~660 kinases curated from ChEMBL) is
# dominated by human protein tyrosine/serine kinases used as drug targets.
# ThiL is a metabolic small-molecule kinase that phosphorylates thiamine
# monophosphate to thiamine pyrophosphate (TPP), a different fold and
# substrate from typical drug-target kinases.
#
# Expected behavior under extrapolation:
#   - High mean probability across many ligands (saturated signal)
#   - Pipeline correctly emits "low cosine similarity vs known kinases" warning
#   - Ranking is informative for compound triage but NOT for ThiL specificity
#   - Discriminative validity REQUIRES ortholog negative controls (e.g.,
#     same query against compounds known not to bind kinases)
#
# This demo intentionally uses ONLY DT-Kinase (single-model mode), since:
#   1. baseline conda envs (drugban/graphban/conplex) typically not available
#      on first-time setups
#   2. ConPLex committee output is documented for organisms in the training
#      set; extrapolation invalidates the cosine-similarity calibration
#
# Expected runtime: ~2 minutes on CPU (cold-start MoLFormer download +
# ESM-2 forward + 100 forward passes). ~5 seconds on warm cache.
#
# Usage:
#   bash scripts/inference/examples/run_kpneumo_thil_demo.sh
#   bash scripts/inference/examples/run_kpneumo_thil_demo.sh /custom/out
# =============================================================================

set -eu
set -o pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYBIN="${REPO}/env/bin/python"
OUT_DIR="${1:-${REPO}/results/inference/kpneumo_demo}"

QUERY_NAME='K. pneumoniae thiamine monophosphate kinase (ThiL)'
QUERY_FASTA_HEADER='kpneumo_query'
N_LIGANDS=100
CKPT_CORPUS='all'

# Sequence: K. pneumoniae thiamine monophosphate kinase (322 AA).
# Note: ThiL is a metabolic kinase, NOT a protein tyrosine/serine kinase.
# Predictions for this query lie in the extrapolation regime.
SEQUENCE='MATGEFSLIARYFDRVKSARLDVETGIGDDCALLHIPEKKTLAASITVLAGNHFTPDIDPADLAYKALAVNLSDLAAMGAEPAWLTLALTLPEVDEVWLEAFSDSLFVQLDYYDMQLIGGDTTRGPLSMTLGIHGFVPPGRAMKRAGAKPGDWIYVTGTPGDSAAGLAVLQNRLTVDEPSDADYLLARHLRPMPRVLQGQALRDLATSAIDLSDGLISDLGHILKASGCGARIDLDAMPYSDAMLRQVDSEQALRWALAGGEDYELCFTVPELNRGALDVALGHLGARFTCIGQIAPESEGLQFIRDGKPVALDLKGYDHFA'

DTK_CKPT="${REPO}/results/all/benchmark_${CKPT_CORPUS}_8M_13_04_2026/test/level4_cnn_8M/${CKPT_CORPUS}/seed_42/level4_cnn_model.pt"
DTK_SIDE="${REPO}/results/all/benchmark_${CKPT_CORPUS}_8M_13_04_2026/test/level4_cnn_8M/${CKPT_CORPUS}/seed_42/level4_cnn_calibration.json"
LIG_LIB="${REPO}/data/reference/ligand_library.tsv"

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

print_banner() {
    printf '\n=============================================================\n'
    printf ' %s\n' "$1"
    printf '=============================================================\n'
}

# -----------------------------------------------------------------------------
# Pre-flight
# -----------------------------------------------------------------------------

check_prerequisites() {
    print_banner '[0/4] Pre-flight checks'

    [[ -x "${PYBIN}" ]] || { echo "FATAL: python not at ${PYBIN}" >&2; exit 1; }
    "${PYBIN}" -c 'import pandas, numpy, rdkit, torch' 2>/dev/null \
        || { echo "FATAL: missing pandas/numpy/rdkit/torch" >&2; exit 1; }
    "${PYBIN}" -c 'import transformers' 2>/dev/null \
        || { echo "FATAL: transformers missing — pip install transformers<4.40" >&2; exit 1; }

    [[ -f "${DTK_CKPT}" ]] || { echo "FATAL: DT-K ckpt missing at ${DTK_CKPT}" >&2; exit 1; }
    [[ -f "${DTK_SIDE}" ]] || { echo "FATAL: DT-K calibration missing at ${DTK_SIDE}" >&2; exit 1; }
    [[ -f "${LIG_LIB}"  ]] || { echo "FATAL: ligand library missing at ${LIG_LIB}" >&2; exit 1; }

    echo "OK — env, deps, ckpt, sidecar, ligand library present"
}

# -----------------------------------------------------------------------------
# Stage 1: write FASTA + build pairs.tsv (1 protein × N ligands)
# -----------------------------------------------------------------------------

stage_build_pairs() {
    print_banner "[1/4] Build pairs.tsv (ThiL × ${N_LIGANDS} ChEMBL ligands)"
    mkdir -p "${OUT_DIR}"

    cat > "${OUT_DIR}/query.fa" <<EOF
>${QUERY_FASTA_HEADER} ${QUERY_NAME}
${SEQUENCE}
EOF

    REPO_ARG="${REPO}" OUT_ARG="${OUT_DIR}" \
    HEADER_ARG="${QUERY_FASTA_HEADER}" \
    SEQUENCE_ARG="${SEQUENCE}" \
    LIG_LIB_ARG="${LIG_LIB}" \
    N_ARG="${N_LIGANDS}" \
    "${PYBIN}" - <<'PYEOF'
import os
import pandas as pd
from pathlib import Path

OUT  = Path(os.environ["OUT_ARG"])
N    = int(os.environ["N_ARG"])
LIB  = Path(os.environ["LIG_LIB_ARG"])

lib = pd.read_csv(LIB, sep="\t").head(N)
df = pd.DataFrame({
    "uniprot":   os.environ["HEADER_ARG"],
    "sequence":  os.environ["SEQUENCE_ARG"],
    "chembl_id": lib["chembl_id"],
    "smiles":    lib["smiles"],
    "source":    "kpneumo_demo",
})
df.to_csv(OUT/"pairs.tsv", sep="\t", index=False)
print(f"  wrote {len(df)} pairs (1 protein × {len(df)} ligands)")
print(f"  ligand library subset: head({N}) of {LIB.name}")
PYEOF
}

# -----------------------------------------------------------------------------
# Stage 2: extrapolation warning (cosine sim vs known kinome)
# -----------------------------------------------------------------------------

stage_extrapolation_check() {
    print_banner '[2/4] Extrapolation check (sequence vs kinome reference)'

    REPO_ARG="${REPO}" SEQUENCE_ARG="${SEQUENCE}" \
    "${PYBIN}" - <<'PYEOF'
"""Crude extrapolation check: sequence identity vs known kinome via
N-gram overlap (proxy for cosine sim before encoder is loaded)."""
import os
from pathlib import Path
from collections import Counter

REPO = Path(os.environ["REPO_ARG"])
seq  = os.environ["SEQUENCE_ARG"]

def ngrams(s, n=3):
    return Counter(s[i:i+n] for i in range(len(s) - n + 1))

q_ngrams = ngrams(seq)
total_q = sum(q_ngrams.values())

best = (-1, "")
with open(REPO/"data/reference/kinome_human.fasta") as fh:
    cur_id, cur_seq = None, []
    for line in fh:
        line = line.strip()
        if line.startswith(">"):
            if cur_id and cur_seq:
                ks = "".join(cur_seq)
                kg = ngrams(ks)
                overlap = sum((q_ngrams & kg).values())
                jacc = overlap / max(sum((q_ngrams | kg).values()), 1)
                if jacc > best[0]:
                    best = (jacc, cur_id)
            cur_id, cur_seq = line[1:].split()[0], []
        else:
            cur_seq.append(line)
    if cur_id and cur_seq:
        ks = "".join(cur_seq)
        kg = ngrams(ks)
        overlap = sum((q_ngrams & kg).values())
        jacc = overlap / max(sum((q_ngrams | kg).values()), 1)
        if jacc > best[0]:
            best = (jacc, cur_id)

j, kid = best
print(f"  query length: {len(seq)} AA")
print(f"  best 3-gram Jaccard overlap with human kinome: {j:.3f} (vs seq_id={kid})")
if j < 0.10:
    print("  WARN: very low overlap → query is far OUT-OF-DOMAIN")
    print("        predictions should be interpreted as extrapolation,")
    print("        not as predictions calibrated for this enzyme family.")
elif j < 0.20:
    print("  CAUTION: low overlap → partial domain similarity only.")
else:
    print("  OK: query has substantive overlap with known kinases.")
PYEOF
}

# -----------------------------------------------------------------------------
# Stage 3: score with DT-Kinase (single-model mode for out-of-domain query)
# -----------------------------------------------------------------------------

stage_score_dtkinase() {
    print_banner '[3/4] Score 100 pairs with DT-Kinase (cold-start encoders)'
    echo "  NOTE: first run downloads MoLFormer (~190 MB) + uses local ESM-2 8M"
    echo "  Expected runtime: ~2 min CPU cold; ~30s warm cache"

    "${PYBIN}" "${REPO}/scripts/inference/models/dtkinase_score.py" \
        --pairs "${OUT_DIR}/pairs.tsv" \
        --out   "${OUT_DIR}/scores_dtkinase.csv" \
        --corpus "${CKPT_CORPUS}" \
        --batch-size-lig 32
}

# -----------------------------------------------------------------------------
# Stage 4: report top-K + interpretive notes
# -----------------------------------------------------------------------------

stage_report() {
    print_banner '[4/4] Report top binders + interpretive caveats'

    OUT_ARG="${OUT_DIR}" \
    "${PYBIN}" - <<'PYEOF'
import os
import pandas as pd
from pathlib import Path

OUT = Path(os.environ["OUT_ARG"])
df  = pd.read_csv(OUT/"scores_dtkinase.csv")

print(f"\n--- DISTRIBUTION OF DT-Kinase PROBABILITIES ---")
print(f"  threshold:      {df['threshold'].iloc[0]:.3f}")
print(f"  binders ≥ thr:  {df['pred'].sum()}/{len(df)}")
print(f"  prob mean:      {df['prob'].mean():.3f}")
print(f"  prob quartiles: " + ", ".join(f"{q:.3f}" for q in df['prob'].quantile([0.25, 0.5, 0.75])))
print(f"  prob max:       {df['prob'].max():.3f}")
print(f"  prob min:       {df['prob'].min():.3f}")

print(f"\n--- TOP 15 PREDICTED BINDERS ---")
top = (df.sort_values("prob", ascending=False).head(15)
         [["chembl_id", "prob", "pred", "threshold"]])
print(top.to_string(index=False))
top.to_csv(OUT/"scores_dtkinase.top15.csv", index=False)

# Diagnostic: signal saturation indicator
saturation = (df["prob"] > 0.5).mean()
print(f"\n--- SATURATION DIAGNOSTIC ---")
print(f"  fraction of ligands with prob > 0.5: {saturation:.1%}")
if saturation > 0.7:
    print("  → HIGH SATURATION: model assigns high probability to most ligands.")
    print("    Likely cause: query lies far out-of-domain; predictions reflect")
    print("    'compounds resembling kinase inhibitors' rather than ThiL-specific")
    print("    binders. Use full-library screen with negative controls (kinome")
    print("    decoys) to recover discriminative ranking.")

print("\n--- INTERPRETIVE CAVEATS ---")
print("  1. ThiL is a small-molecule metabolic kinase, NOT a protein kinase.")
print("     The training set (660 kinases) is dominated by drug-target tyrosine")
print("     and serine/threonine protein kinases. Predictions on ThiL are")
print("     extrapolative, not calibrated.")
print("  2. The scoring is single-model (DT-Kinase only). The full 4-model")
print("     committee requires conda envs (drugban, graphban, conplex) which")
print("     are not invoked in this demo. Committee consensus is documented")
print("     for in-domain queries (Anexo B); for out-of-domain queries the")
print("     consensus interpretation does not apply.")
print("  3. Ranking validity for compound triage: the top-K compounds may")
print("     still be informative as 'kinase-like binders' but ThiL-specific")
print("     activity requires biochemical validation (e.g., TMP-AMP coupled")
print("     assay or thermal shift).")
PYEOF
}

# -----------------------------------------------------------------------------
# Stage 5: extract attention maps for top-K hits (single-model mode)
# -----------------------------------------------------------------------------

stage_attention() {
    print_banner '[5/5] Extract attention maps for top hits (PNG + JSON)'
    echo "  Extracts DT-Kinase 3-level attention via forward-hooks."
    echo "  Per-pair output: <pair_id>_attention.{png,json} + <pair_id>_ligand_2d.png"
    echo ""

    "${PYBIN}" "${REPO}/scripts/inference/attention.py" \
        --scores  "${OUT_DIR}/scores_dtkinase.csv" \
        --pairs   "${OUT_DIR}/pairs.tsv" \
        --out-dir "${OUT_DIR}/attention" \
        --top-k   5 \
        --corpus  "${CKPT_CORPUS}" \
        2>&1 | grep -v -E "^\[2[0-3]:" | tail -25
    echo ""
    echo "  attention/ output:"
    find "${OUT_DIR}/attention" -type f 2>/dev/null | head -20 | sed 's/^/    /'
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

main() {
    echo "=============================================================="
    echo " DT-Kinase out-of-domain demo"
    echo " query   : ${QUERY_NAME}"
    echo " corpus  : ${CKPT_CORPUS}"
    echo " ligands : ${N_LIGANDS} (subset of ChEMBL kinase library)"
    echo " out     : ${OUT_DIR}"
    echo "=============================================================="

    check_prerequisites
    stage_build_pairs
    stage_extrapolation_check
    stage_score_dtkinase
    stage_report
    stage_attention

    print_banner 'DONE'
    echo "outputs:"
    ls -lh "${OUT_DIR}" | awk 'NR>1 {printf "  %s  %s\n", $5, $9}'
}

main "$@"
