#!/usr/bin/env python3
"""
Leakage analysis on Universal Datasets — GPU-accelerated MLP
=============================================================

Reproduces Table 13 / Figure 19 of the thesis using the
curated universal datasets (Human + Non-Human).

For each dataset and each partition scenario (S1–S4, Scaffold):
  - 10-fold stratified CV (seed=42)
  - MLP classifier on 2048-bit Morgan fingerprints (PyTorch + CUDA)
  - Reports MCC mean ± std

Usage:
  python leakage_universal_mlp.py              # auto-detect GPU
  python leakage_universal_mlp.py --device cpu  # force CPU
"""

from __future__ import annotations

import argparse
import json
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

# ── paths ──────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent.parent / "scaffolds_splits" / "output"

DATASETS = {
    "Non-Human": {
        "train": BASE / "non_human_train.tsv",
        "val":   BASE / "non_human_val.tsv",
        "test":  BASE / "non_human_test.tsv",
    },
    "Human": {
        "train": BASE / "human_train.tsv",
        "val":   BASE / "human_val.tsv",
        "test":  BASE / "human_test.tsv",
    },
}

SCENARIOS = ["S1", "S2", "Scaffold", "S3", "S4"]
N_FOLDS = 10
SEED = 42
FP_BITS = 2048
FP_RADIUS = 2

# ── MLP (PyTorch) ─────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ── fingerprints ───────────────────────────────────────────────────────────
def smiles_to_fp(smiles: str) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_BITS)
    return np.array(fp, dtype=np.float32)


def build_fp_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


# ── scaffold ───────────────────────────────────────────────────────────────
def get_scaffold(smiles: str) -> str:
    from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
    try:
        return MurckoScaffoldSmiles(smiles)
    except Exception:
        return smiles


# ── splitting ──────────────────────────────────────────────────────────────
def split_s1(df, n_folds, seed):
    """S1: random stratified split."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(skf.split(np.arange(len(df)), df["label"].values))


def split_by_group(df, group_col, n_folds, seed):
    """S2 (compound), S3 (kinase), Scaffold: group-disjoint split.

    Uses GroupKFold (guarantees group disjunction) instead of
    StratifiedGroupKFold (which failed to produce valid folds
    for Scaffold due to class imbalance within groups).
    """
    groups = df[group_col].fillna("__NO_SCAFFOLD__").astype(str).values
    n_groups = len(np.unique(groups))

    # Reduce n_folds if fewer groups than requested folds
    effective_folds = min(n_folds, n_groups)
    if effective_folds < 2:
        raise ValueError(f"Only {n_groups} unique groups for '{group_col}', need ≥2")

    gkf = GroupKFold(n_splits=effective_folds)
    return list(gkf.split(np.arange(len(df)), df["label"].values, groups))


def split_s4(df, n_folds, seed):
    """S4: double disjoint — new compound AND new kinase in val.

    For datasets with few kinases (e.g. Non-Human: 112), uses fewer
    folds to ensure enough kinases can be held out per fold.
    """
    rng = np.random.default_rng(seed)
    n_kinases = df["target_kinase"].nunique()

    # Adaptive fold count: need at least ~10 kinases per fold for S4
    effective_folds = min(n_folds, max(2, n_kinases // 10))

    compounds = df["chembl_id"].unique()
    rng.shuffle(compounds)

    # Split kinases into folds first (ensures kinase disjunction)
    kinases = df["target_kinase"].unique()
    rng.shuffle(kinases)
    kinase_fold_size = len(kinases) // effective_folds

    folds = []
    for i in range(effective_folds):
        # Kinase partition
        k_start = i * kinase_fold_size
        k_end = k_start + kinase_fold_size if i < effective_folds - 1 else len(kinases)
        val_kinases = set(kinases[k_start:k_end])

        # Compound partition (different from kinase partition)
        c_fold_size = len(compounds) // effective_folds
        c_start = i * c_fold_size
        c_end = c_start + c_fold_size if i < effective_folds - 1 else len(compounds)
        val_compounds = set(compounds[c_start:c_end])

        # Val: both kinase AND compound must be in val partition
        val_mask = df["chembl_id"].isin(val_compounds) & df["target_kinase"].isin(val_kinases)
        # Train: neither kinase NOR compound is in val partition
        train_mask = ~df["chembl_id"].isin(val_compounds) & ~df["target_kinase"].isin(val_kinases)

        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]

        if len(val_idx) > 10 and len(train_idx) > 10:
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


# ── GPU training ───────────────────────────────────────────────────────────
def train_evaluate_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
) -> float:
    """Train MLP on GPU, return MCC on val."""
    # Standardize
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_v = scaler.transform(X_val)

    # Tensors → GPU
    X_tr_t = torch.from_numpy(X_tr.astype(np.float32)).to(device)
    y_tr_t = torch.from_numpy(y_train.astype(np.float32)).to(device)
    X_v_t = torch.from_numpy(X_v.astype(np.float32)).to(device)

    # Class weight for balancing
    n_pos = max((y_train == 1).sum(), 1)
    n_neg = max((y_train == 0).sum(), 1)
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)

    # Model
    model = MLP(X_tr.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    # DataLoader
    train_ds = TensorDataset(X_tr_t, y_tr_t)
    train_dl = DataLoader(train_ds, batch_size=2048, shuffle=True, drop_last=False)

    # Train
    model.train()
    best_loss = float("inf")
    patience, patience_counter = 10, 0
    best_state = None

    for epoch in range(100):
        epoch_loss = 0.0
        for xb, yb in train_dl:
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        scheduler.step()
        epoch_loss /= len(train_ds)

        if epoch_loss < best_loss - 1e-4:
            best_loss = epoch_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Eval
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(X_v_t)
        preds = (logits > 0.0).cpu().numpy().astype(int)

    return float(matthews_corrcoef(y_val, preds))


# ── main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", help="cuda / cpu / auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    output_dir = Path(__file__).resolve().parent / "leakage_universal_results"
    output_dir.mkdir(exist_ok=True)

    all_results = {}

    for ds_name, files in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"  Dataset: {ds_name}")
        print(f"{'='*60}")

        dfs = []
        for fpath in files.values():
            if fpath.exists():
                dfs.append(pd.read_csv(fpath, sep="\t"))
            else:
                print(f"  WARNING: {fpath} not found, skipping")
        df = pd.concat(dfs, ignore_index=True)
        print(f"  Total rows: {len(df):,}")
        print(f"  Unique compounds: {df['chembl_id'].nunique():,}")
        print(f"  Unique kinases: {df['target_kinase'].nunique()}")

        if "scaffold" not in df.columns:
            print("  Computing scaffolds...")
            df["scaffold"] = df["canonical_smiles"].apply(get_scaffold)

        print("  Computing Morgan fingerprints...")
        X_all, y_all, valid_mask = build_fp_matrix(df)
        df_valid = df.loc[valid_mask].reset_index(drop=True)
        print(f"  Valid molecules: {len(df_valid):,} / {len(df):,}")

        ds_results = {}

        for scenario in SCENARIOS:
            print(f"\n  --- Scenario {scenario} ---")
            try:
                folds = get_splits(df_valid, scenario, N_FOLDS, SEED)
                print(f"  Generated {len(folds)} folds")
            except Exception as e:
                print(f"  ERROR splitting: {e}")
                traceback.print_exc()
                ds_results[scenario] = {"mean": float("nan"), "std": float("nan"), "n_folds": 0}
                continue

            mccs = []
            for i, (train_idx, val_idx) in enumerate(folds):
                y_tr = y_all[train_idx]
                y_vl = y_all[val_idx]
                if len(np.unique(y_tr)) < 2 or len(np.unique(y_vl)) < 2:
                    print(f"    Fold {i}: skipped (single class in {'train' if len(np.unique(y_tr))<2 else 'val'})")
                    continue

                mcc = train_evaluate_fold(X_all[train_idx], y_tr, X_all[val_idx], y_vl, device)
                mccs.append(mcc)
                print(f"    Fold {i}: MCC = {mcc:.3f}  (train={len(train_idx):,}, val={len(val_idx):,})")

            if mccs:
                mean_mcc = float(np.mean(mccs))
                std_mcc = float(np.std(mccs))
                print(f"  {scenario}: MCC = {mean_mcc:.3f} ± {std_mcc:.3f} ({len(mccs)} folds)")
            else:
                mean_mcc, std_mcc = float("nan"), float("nan")
                print(f"  {scenario}: No valid folds!")

            ds_results[scenario] = {"mean": round(mean_mcc, 3), "std": round(std_mcc, 3), "n_folds": len(mccs)}

        all_results[ds_name] = ds_results

    # ── Output ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  SUMMARY — Table 13")
    print(f"{'='*60}")
    print(f"  {'Scenario':<35} {'MCC Non-Human':<20} {'MCC Human':<20}")
    print("  " + "-" * 75)

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

    json_path = output_dir / "leakage_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nJSON: {json_path}")

    tsv_path = output_dir / "leakage_results.tsv"
    pd.DataFrame(rows).to_csv(tsv_path, sep="\t", index=False)
    print(f"TSV:  {tsv_path}")


if __name__ == "__main__":
    main()
