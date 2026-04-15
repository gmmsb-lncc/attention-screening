#!/usr/bin/env python3
"""
Leakage analysis on Universal Datasets — MLP + Morgan FP
=========================================================

Reproduces Table 13 / Figure 19 of the thesis using the
curated universal datasets (Human + Non-Human).

For each dataset and each partition scenario (S1–S4, Scaffold):
  - 10-fold stratified CV (seed=42)
  - MLP classifier on 2048-bit Morgan fingerprints
  - Reports MCC mean ± std

Output: JSON + TSV with the results for direct LaTeX update.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

# ── paths ──────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent.parent / "scaffolds_splits" / "output"

DATASETS = {
    "Non-Human": {
        "train": BASE / "non_human_train.tsv",
        "val": BASE / "non_human_val.tsv",
        "test": BASE / "non_human_test.tsv",
    },
    "Human": {
        "train": BASE / "human_train.tsv",
        "val": BASE / "human_val.tsv",
        "test": BASE / "human_test.tsv",
    },
}

SCENARIOS = ["S1", "S2", "Scaffold", "S3", "S4"]
N_FOLDS = 10
SEED = 42
FP_BITS = 2048
FP_RADIUS = 2


# ── fingerprints ───────────────────────────────────────────────────────────
def smiles_to_fp(smiles: str) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_BITS)
    return np.array(fp, dtype=np.float32)


def build_fp_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, y, valid_mask) for all rows with valid SMILES."""
    fps, labels, mask = [], [], []
    for _, row in df.iterrows():
        fp = smiles_to_fp(row["canonical_smiles"])
        if fp is not None:
            fps.append(fp)
            labels.append(int(row["label"]))
            mask.append(True)
        else:
            mask.append(False)
    return np.array(fps), np.array(labels), np.array(mask)


# ── scaffold from SMILES ───────────────────────────────────────────────────
def get_scaffold(smiles: str) -> str:
    """Return Murcko scaffold or the SMILES itself as fallback."""
    from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
    try:
        return MurckoScaffoldSmiles(smiles)
    except Exception:
        return smiles


# ── splitting helpers ──────────────────────────────────────────────────────
def split_s1(df, n_folds, seed):
    """S1: random stratified."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(skf.split(np.arange(len(df)), df["label"].values))


def split_by_group(df, group_col, n_folds, seed):
    """S2 (compound), S3 (kinase), Scaffold: group-aware stratified split."""
    groups = df[group_col].values
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(sgkf.split(np.arange(len(df)), df["label"].values, groups))


def split_s4(df, n_folds, seed):
    """S4: double disjoint — new compound AND new kinase.
    We build folds by grouping on compound, then post-filter
    to keep only rows where kinase is also unseen in train."""
    rng = np.random.default_rng(seed)
    compounds = df["chembl_id"].unique()
    rng.shuffle(compounds)

    fold_size = len(compounds) // n_folds
    folds = []
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else len(compounds)
        val_compounds = set(compounds[start:end])

        val_mask = df["chembl_id"].isin(val_compounds)
        train_mask = ~val_mask

        # Post-filter: keep only val rows where kinase is NOT in train
        train_kinases = set(df.loc[train_mask, "target_kinase"].unique())
        val_kinase_ok = ~df["target_kinase"].isin(train_kinases)
        val_final = val_mask & val_kinase_ok

        # Also post-filter train: keep only rows where kinase NOT in val
        val_kinases = set(df.loc[val_final, "target_kinase"].unique())
        train_final = train_mask & ~df["target_kinase"].isin(val_kinases)

        train_idx = np.where(train_final)[0]
        val_idx = np.where(val_final)[0]

        if len(val_idx) > 0 and len(train_idx) > 0:
            folds.append((train_idx, val_idx))

    return folds


def get_splits(df, scenario, n_folds, seed):
    if scenario == "S1":
        return split_s1(df, n_folds, seed)
    elif scenario == "S2":
        return split_by_group(df, "chembl_id", n_folds, seed)
    elif scenario == "Scaffold":
        return split_by_group(df, "scaffold", n_folds, seed)
    elif scenario == "S3":
        return split_by_group(df, "target_kinase", n_folds, seed)
    elif scenario == "S4":
        return split_s4(df, n_folds, seed)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")


# ── MLP training & evaluation ─────────────────────────────────────────────
def train_evaluate_fold(X_train, y_train, X_val, y_val) -> float:
    """Train MLP, return MCC on val."""
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_v = scaler.transform(X_val)

    clf = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=SEED,
        verbose=False,
    )
    clf.fit(X_tr, y_train)
    preds = clf.predict(X_v)
    return float(matthews_corrcoef(y_val, preds))


# ── main ───────────────────────────────────────────────────────────────────
def main():
    output_dir = Path(__file__).resolve().parent / "leakage_universal_results"
    output_dir.mkdir(exist_ok=True)

    all_results = {}

    for ds_name, files in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"  Dataset: {ds_name}")
        print(f"{'='*60}")

        # Load and concatenate all splits into single pool
        dfs = []
        for split_name, fpath in files.items():
            if fpath.exists():
                dfs.append(pd.read_csv(fpath, sep="\t"))
        df = pd.concat(dfs, ignore_index=True)
        print(f"  Total rows: {len(df)}")
        print(f"  Unique compounds: {df['chembl_id'].nunique()}")
        print(f"  Unique kinases: {df['target_kinase'].nunique()}")

        # Ensure scaffold column exists
        if "scaffold" not in df.columns:
            print("  Computing scaffolds...")
            df["scaffold"] = df["canonical_smiles"].apply(get_scaffold)

        # Build fingerprint matrix
        print("  Computing Morgan fingerprints...")
        X_all, y_all, valid_mask = build_fp_matrix(df)
        df_valid = df.loc[valid_mask].reset_index(drop=True)
        print(f"  Valid molecules: {len(df_valid)} / {len(df)}")

        ds_results = {}

        for scenario in SCENARIOS:
            print(f"\n  --- Scenario {scenario} ---")
            try:
                folds = get_splits(df_valid, scenario, N_FOLDS, SEED)
            except Exception as e:
                print(f"  ERROR splitting: {e}")
                ds_results[scenario] = {"mean": float("nan"), "std": float("nan"), "n_folds": 0}
                continue

            mccs = []
            for i, (train_idx, val_idx) in enumerate(folds):
                # Check both classes present
                y_tr = y_all[train_idx]
                y_vl = y_all[val_idx]
                if len(np.unique(y_tr)) < 2 or len(np.unique(y_vl)) < 2:
                    print(f"    Fold {i}: skipped (single class)")
                    continue

                mcc = train_evaluate_fold(X_all[train_idx], y_tr, X_all[val_idx], y_vl)
                mccs.append(mcc)
                print(f"    Fold {i}: MCC = {mcc:.3f}")

            if mccs:
                mean_mcc = float(np.mean(mccs))
                std_mcc = float(np.std(mccs))
                print(f"  {scenario}: MCC = {mean_mcc:.3f} ± {std_mcc:.3f} ({len(mccs)} folds)")
            else:
                mean_mcc = float("nan")
                std_mcc = float("nan")
                print(f"  {scenario}: No valid folds!")

            ds_results[scenario] = {"mean": mean_mcc, "std": std_mcc, "n_folds": len(mccs)}

        all_results[ds_name] = ds_results

    # ── Output ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  SUMMARY (for Table 13)")
    print(f"{'='*60}")
    print(f"{'Scenario':<35} {'MCC Non-Human':<20} {'MCC Human':<20}")
    print("-" * 75)

    rows = []
    for scenario in SCENARIOS:
        nh = all_results.get("Non-Human", {}).get(scenario, {})
        hu = all_results.get("Human", {}).get(scenario, {})
        nh_str = f"{nh.get('mean', float('nan')):.3f} ± {nh.get('std', float('nan')):.3f}"
        hu_str = f"{hu.get('mean', float('nan')):.3f} ± {hu.get('std', float('nan')):.3f}"
        print(f"  {scenario:<33} {nh_str:<20} {hu_str:<20}")
        rows.append({
            "scenario": scenario,
            "mcc_nh_mean": nh.get("mean"), "mcc_nh_std": nh.get("std"),
            "mcc_hu_mean": hu.get("mean"), "mcc_hu_std": hu.get("std"),
        })

    # Save JSON
    json_path = output_dir / "leakage_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nJSON saved: {json_path}")

    # Save TSV
    tsv_path = output_dir / "leakage_results.tsv"
    pd.DataFrame(rows).to_csv(tsv_path, sep="\t", index=False)
    print(f"TSV saved: {tsv_path}")


if __name__ == "__main__":
    main()
