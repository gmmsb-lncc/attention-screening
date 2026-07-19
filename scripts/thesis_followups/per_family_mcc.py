#!/usr/bin/env python3
"""
Per-family MCC breakdown across 4 models × 3 corpora.

Maps `target_kinase` strings (free-form ChEMBL) to Manning kinome groups
via prioritised substring heuristics. Aligns predictions with test split,
computes MCC per Manning group, aggregates 5-seed mean ± std.

NOTE: requires manual review of unmatched `target_kinase` strings before
publication. Heuristic match coverage is expected to be 80-90%; remainder
should be inspected and either added to MANNING_HEURISTICS or grouped as
"unmapped". Order alignment between `raw_predictions.npz` and test TSV is
assumed (same-row); verify with a sanity check on a known-positive sample.

Usage:
    python3 scripts/thesis_followups/per_family_mcc.py \
        --corpus human --model dtkinase --seed 42

Outputs to results/per_family_mcc/{corpus}_{model}_seed{seed}.csv
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef


REPO = Path(__file__).resolve().parents[2]


# Manning families heuristic. Order matters (priority); longer/more specific
# substrings precede shorter ones to avoid spurious matches. Lookup is
# case-insensitive (str.lower() applied to both pattern and target).
MANNING_HEURISTICS = [
    # === Atypical (PI3K/PIKK family, mTOR, ATM, ATR, DNA-PK, IKK, casein kinase II) ===
    ("Atypical", [
        "PI3-kinase", "PI3 kinase", "PI4-kinase", "PI4 kinase", "PI5-kinase",
        "Phosphatidylinositol", "phosphatidylinositol",
        "phosphoinositide 3-kinase", "PI3K", "PI4K", "PIKK",
        "mTOR", "ATM", "ATR", "DNA-PK", "DNAPK", "PRKDC",
        "DNA-dependent protein kinase", "DNA-activated protein kinase",
        "TBK1", "IKK", "I-kappa-B kinase", "Inhibitor of nuclear factor kappa B kinase",
        "Casein kinase II", "Casein kinase 2", "casein kinase II", "CK2",
        "Bromodomain", "Adenosine kinase",
        "Serine/threonine-protein kinase/endoribonuclease IRE1", "IRE1",
        "Hexokinase", "Fructokinase", "Glucokinase",
        "Eukaryotic translation initiation factor 2", "eIF2A kinase",
        "GCN2", "PERK", "PKR", "HRI",
    ]),
    # === STE (MEK/MAP2K/MAP3K/MAP4K, PAK, MST, GCK, TAO, ASK1, NIK) ===
    ("STE", [
        "Mitogen-activated protein kinase kinase kinase kinase",
        "Mitogen-activated protein kinase kinase kinase",
        "Mitogen-activated protein kinase kinase",
        "Dual specificity mitogen-activated protein kinase kinase",
        "MEK1", "MEK2", "MEK4", "MEK5", "MEK6", "MEK7",
        "MKK", "MAP2K", "MAP3K", "MAP4K",
        "PAK", "p21-activated", "p21 protein", "p21-Cdc42",
        "MST1", "MST2", "MST3", "MST4", "Mammalian sterile",
        "OSR", "GCK", "TAO", "TAOK", "ASK1", "Apoptosis signal-regulating",
        "NIK", "STE20", "Slk",
        "Serine/threonine-protein kinase 10",
        "Serine/threonine-protein kinase 24",
        "Serine/threonine-protein kinase 25",
        "Misshapen", "TNIK", "MAP4K6",
        "PDZ-binding kinase", "PBK",
        "Hippo", "LATS1", "LATS2", "MOB",
    ]),
    # === TKL (tyrosine-kinase-like: RAF, RIPK, IRAK, TGFBR, BMPR, MLK, LRRK, LIMK) ===
    ("TKL", [
        "RAF", "B-Raf", "BRaf", "BRAF", "ARAF", "CRAF", "Raf-1", "raf-1",
        "RIPK", "Receptor-interacting serine",
        "IRAK", "Interleukin-1 receptor-associated kinase",
        "TGFBR", "TGF-beta receptor", "Activin receptor", "ACVR", "BMPR",
        "MLK", "Mixed lineage kinase", "ZAK", "DLK", "LZK",
        "LRRK", "Leucine-rich repeat",
        "LIMK", "LIM domain kinase",
        "TESK", "Testicular protein kinase",
    ]),
    # === TK (tyrosine kinases: receptor + non-receptor) ===
    ("TK", [
        "EGFR", "Epidermal growth factor receptor",
        "ErbB", "ERBB", "HER2", "HER3", "HER4", "Receptor protein-tyrosine kinase erbB",
        "ABL", "BCR-ABL", "Tyrosine-protein kinase ABL",
        "SRC", "FYN", "LCK", "LYN", "HCK", "FGR", "BLK", "YES",
        "BTK", "ITK", "TEC", "FRK", "TXK",
        "FGFR", "Fibroblast growth factor receptor",
        "PDGFR", "Platelet-derived growth factor receptor",
        "VEGFR", "KDR", "Vascular endothelial growth factor receptor",
        "FLT1", "FLT3", "FLT4", "Fms-like",
        "KIT", "Stem cell factor receptor",
        "MET", "RON", "Hepatocyte growth factor receptor",
        "RET", "Ret proto-oncogene",
        "ALK", "Anaplastic lymphoma kinase",
        "ROS1", "ROS proto-oncogene",
        "JAK1", "JAK2", "JAK3", "Janus kinase", "TYK2",
        "SYK", "Spleen tyrosine kinase",
        "ZAP70", "ZAP-70",
        "INSR", "Insulin receptor", "IGF1R", "Insulin-like growth factor",
        "MERTK", "AXL", "TYRO3",
        "TIE1", "TIE2", "TEK",
        "EPHA", "EPHB", "Ephrin", "Eph receptor",
        "DDR1", "DDR2", "Discoidin domain",
        "Focal adhesion kinase", "FAK",
        "PYK2", "Proline-rich tyrosine kinase",
        "Urokinase-type plasminogen activator",
        "tyrosine kinase", "tyrosine-protein kinase", "Tyrosine kinase",
    ]),
    # === CMGC (CDK, MAPK, ERK, JNK, p38, GSK, CLK, DYRK, CDKL, SRPK, NLK) ===
    ("CMGC", [
        "CDK", "Cyclin-dependent kinase", "cyclin-dependent",
        "Cell division protein kinase",
        "Cell division cycle 2", "cdc2", "CDC2",
        "PCTAIRE", "PFTAIRE",
        "MAPK", "Mitogen-activated protein kinase", "MAP kinase",
        "ERK", "Extracellular signal-regulated",
        "JNK", "c-Jun N-terminal kinase", "Stress-activated protein kinase",
        "p38", "p38-alpha", "p38a", "p38 mitogen",
        "GSK", "Glycogen synthase kinase",
        "CLK", "CDC-like kinase",
        "DYRK", "Dual specificity",
        "CDKL", "SRPK", "Serine/arginine-rich",
        "NLK", "Nemo-like",
        "HIPK", "Homeodomain-interacting",
        "ICK", "intestinal cell kinase",
        "MAK",
    ]),
    # === AGC (AKT, PKA, PKC, PDK, S6K, RSK, MSK, ROCK, GRK, SGK, PRK, PKN) ===
    ("AGC", [
        "AKT", "PKB", "Protein kinase B",
        "PKA", "Protein kinase A", "cAMP-dependent",
        "PKG", "cGMP-dependent",
        "PKC", "Protein kinase C",
        "PDK1", "PDPK1", "3-phosphoinositide dependent protein kinase",
        "p70 S6", "p70S6K", "Ribosomal protein S6 kinase",
        "S6K1", "S6K2",
        "RSK", "p90 ribosomal", "MSK1", "MSK2",
        "ROCK", "Rho-associated", "Rho-kinase",
        "GRK", "G protein-coupled receptor kinase", "G-protein coupled receptor kinase",
        "G-protein-coupled receptor kinase", "beta-adrenergic receptor kinase",
        "SGK", "Serum/glucocorticoid",
        "PRK", "PKN1", "PKN2", "Protein kinase N",
        "DMPK", "Myotonic dystrophy",
        "Serine/threonine-protein kinase D", "PKD1", "PKD2", "PKD3",
        "MRCK", "CDC42 binding protein kinase",
        "Citron", "CIT",
    ]),
    # === CAMK (CaMK, AMPK, CHK, MK2/3, MAPKAPK, DAPK, PIM, MELK, MARK, NUAK, BRSK) ===
    ("CAMK", [
        "CaMK", "CAMK", "calcium/calmodulin", "Calcium/calmodulin",
        "CaM kinase", "CaM-kinase",
        "AMPK", "AMP-activated", "AMP-activated protein kinase",
        "CHK1", "CHK2", "Checkpoint kinase",
        "MK2", "MK3", "MK5", "MAPKAPK", "MAPK-activated",
        "DAPK", "Death-associated protein kinase",
        "PIM1", "PIM2", "PIM3", "Proto-oncogene serine/threonine-protein kinase pim",
        "MELK", "Maternal embryonic leucine zipper",
        "DCAMKL", "DCLK",
        "PfCDPK", "calcium-dependent protein kinase",
        "MARK", "MAP/microtubule affinity-regulating",
        "NUAK", "BRSK", "BR serine/threonine",
        "SIK", "Salt-inducible",
        "MNK1", "MNK2", "MAP kinase-interacting",
        "Phosphorylase kinase",
        "TSSK", "Testis-specific",
        "Myosin light chain kinase", "MLCK",
        "Serine/threonine-protein kinase 11", "STK11", "LKB1",
        "Serine/threonine-protein kinase 17A", "Serine/threonine-protein kinase 17B",
        "DRAK1", "DRAK2",
    ]),
    # === CK1 (CK1 alpha/delta/epsilon, TTBK, VRK) ===
    ("CK1", [
        "Casein kinase I", "casein kinase I", "Casein kinase 1", "casein kinase 1",
        "CK1", "CKI alpha", "CKI delta", "CKI epsilon",
        "TTBK", "Tau-tubulin",
        "VRK", "Vaccinia-related",
    ]),
    # === Other (PLK, NEK, AURK, BUB, WEE, GAK, AAK, others not in 8 Manning groups) ===
    ("Other", [
        "PLK1", "PLK2", "PLK3", "PLK4", "Polo-like kinase",
        "NEK1", "NEK2", "NEK3", "NEK4", "NEK6", "NEK7", "NEK9", "NEK11",
        "NIMA-related",
        "AURK", "AURKA", "AURKB", "AURKC", "Aurora kinase", "Aurora-A", "Aurora-B",
        "BUB1", "BUB3", "BUBR",
        "WEE1", "Wee1", "Tyrosine- and threonine-specific cdc2-inhibitory kinase",
        "GAK", "Cyclin-G-associated",
        "AAK", "Adaptor-associated kinase",
        "Haspin", "GSG2",
        "TLK", "Tousled-like",
        "TTK", "Dual specificity protein kinase TTK",
        "MASTL", "Microtubule-associated serine/threonine-protein kinase-like",
        "ULK", "Unc-51-like",
        "PASK", "Per-Arnt-Sim",
        "PINK1", "PTEN-induced",
        "TBK", "TANK-binding",
        "IRE2",
        "BMP-2-inducible", "BIKE", "BMP2K",
        "CDC7", "Cell division cycle 7",
        "Serine/threonine-protein kinase 16",
        "TNNI3K", "Cardiac troponin I-interacting",
        "Serine/threonine-protein kinase 2 ",
    ]),
    # === Plasmodium / Trypanosoma / Leishmania / Toxoplasma (Other parasite) ===
    ("Other_parasite", [
        "PfPK", "Pf PK", "Plasmodium",
        "Trypanosoma", "Leishmania", "Toxoplasma", "Schistosoma",
    ]),
    # === Pseudokinases (functional, not Manning) ===
    ("Pseudokinase", [
        "STRADa", "STRAD-alpha", "STRAD alpha",
        "KSR1", "KSR2",
        "Pseudokinase",
    ]),
]


def map_to_family(target: str) -> str:
    """Return Manning group label or 'unmapped' if no rule matches."""
    if not isinstance(target, str):
        return "unmapped"
    for fam, patterns in MANNING_HEURISTICS:
        for p in patterns:
            if p.lower() in target.lower():
                return fam
    return "unmapped"


PRED_PATHS = {
    ("dtkinase", "human"):     "results/benchmark_human_8M_13_05_2026/test/level4_cnn_8M/human/seed_{seed}/raw_predictions.npz",
    ("dtkinase", "non_human"): "results/benchmark_non_human_8M_13_05_2026_v3/test/level4_cnn_8M/non_human/seed_{seed}/raw_predictions.npz",
    ("dtkinase", "all"):       "results/all/benchmark_all_8M_13_04_2026/test/level4_cnn_8M/all/seed_{seed}/raw_predictions.npz",
    ("drugban", "human"):      "DrugBAN/results_universal/results_universal/human/seed_{seed}/raw_predictions.npz",
    ("drugban", "non_human"):  "DrugBAN/results_universal/results_universal/non_human/seed_{seed}/raw_predictions.npz",
    ("drugban", "all"):        "DrugBAN/results_universal/results_universal/all/seed_{seed}/raw_predictions.npz",
    ("graphban", "human"):     "GraphBAN/results_universal/human/seed_{seed}/raw_predictions.npz",
    ("graphban", "non_human"): "GraphBAN/results_universal/non_human/seed_{seed}/raw_predictions.npz",
    ("graphban", "all"):       "GraphBAN/results_universal/all/seed_{seed}/raw_predictions.npz",
    ("conplex",  "human"):     "ConPLex/results_universal/human/seed_{seed}/raw_predictions.npz",
    ("conplex",  "non_human"): "ConPLex/results_universal/non_human/seed_{seed}/raw_predictions.npz",
    ("conplex",  "all"):       "ConPLex/results_universal/all/seed_{seed}/raw_predictions.npz",
}


CAL_PATHS = {
    ("dtkinase", "human"):     "results/benchmark_human_8M_13_05_2026/test/level4_cnn_8M/human/seed_{seed}/level4_cnn_results.json",
    ("dtkinase", "non_human"): "results/benchmark_non_human_8M_13_05_2026_v3/test/level4_cnn_8M/non_human/seed_{seed}/level4_cnn_results.json",
    ("dtkinase", "all"):       "results/all/benchmark_all_8M_13_04_2026/test/level4_cnn_8M/all/seed_{seed}/level4_cnn_results.json",
    ("drugban", "human"):      "DrugBAN/results_universal/results_universal/human/seed_{seed}/drugban_calibration.json",
    ("drugban", "non_human"):  "DrugBAN/results_universal/results_universal/non_human/seed_{seed}/drugban_calibration.json",
    ("drugban", "all"):        "DrugBAN/results_universal/results_universal/all/seed_{seed}/drugban_calibration.json",
    ("graphban", "human"):     "GraphBAN/results_universal/human/seed_{seed}/graphban_calibration.json",
    ("graphban", "non_human"): "GraphBAN/results_universal/non_human/seed_{seed}/graphban_calibration.json",
    ("graphban", "all"):       "GraphBAN/results_universal/all/seed_{seed}/graphban_calibration.json",
    ("conplex",  "human"):     "ConPLex/results_universal/human/seed_{seed}/conplex_calibration.json",
    ("conplex",  "non_human"): "ConPLex/results_universal/non_human/seed_{seed}/conplex_calibration.json",
    ("conplex",  "all"):       "ConPLex/results_universal/all/seed_{seed}/conplex_calibration.json",
}


TEST_TSV = {
    "human":     "scaffolds_splits/output/human_test.tsv",
    "non_human": "scaffolds_splits/output/non_human_test.tsv",
    "all":       "scaffolds_splits/output/universal_test.tsv",
}


def load_predictions(model: str, corpus: str, seed: int):
    pth = REPO / PRED_PATHS[(model, corpus)].format(seed=seed)
    arr = np.load(pth)
    if "y_true" in arr:
        y_true = arr["y_true"]
        y_prob = arr["y_prob"]
    else:
        y_true = arr["test_y_true"]
        y_prob = arr["test_y_prob"]
    return y_true, y_prob


def load_threshold(model: str, corpus: str, seed: int) -> float:
    import json
    pth = REPO / CAL_PATHS[(model, corpus)].format(seed=seed)
    if not pth.exists():
        return 0.5
    cal = json.loads(pth.read_text())
    if model == "dtkinase":
        return float(cal["Split by Scaffold"]["MLP"]["val_threshold"])
    return float(cal.get("threshold", 0.5))


def per_family_mcc(model: str, corpus: str, seed: int) -> pd.DataFrame:
    test = pd.read_csv(REPO / TEST_TSV[corpus], sep="\t")
    y_true, y_prob = load_predictions(model, corpus, seed)
    thr = load_threshold(model, corpus, seed)

    if len(y_true) != len(test):
        raise ValueError(
            f"Misalignment: predictions n={len(y_true)} vs test TSV n={len(test)}. "
            f"Verify featurisation pipeline preserves row order for {model}/{corpus}/seed={seed}."
        )

    family = test["target_kinase"].apply(map_to_family).values
    y_pred = (y_prob >= thr).astype(int)

    rows = []
    for fam in sorted(set(family)):
        mask = family == fam
        if mask.sum() < 10:
            continue
        try:
            mcc = matthews_corrcoef(y_true[mask], y_pred[mask])
        except Exception:
            mcc = float("nan")
        rows.append({"family": fam, "n": int(mask.sum()),
                      "n_pos": int(y_true[mask].sum()),
                      "mcc": mcc})
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["human","non_human","all"], required=True)
    ap.add_argument("--model", choices=["dtkinase","drugban","graphban","conplex"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="results/per_family_mcc")
    args = ap.parse_args()

    df = per_family_mcc(args.model, args.corpus, args.seed)
    out = REPO / args.out_dir / f"{args.corpus}_{args.model}_seed{args.seed}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
