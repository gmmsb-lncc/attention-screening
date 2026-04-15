#!/usr/bin/env python3
"""
Leakage analysis on Universal Datasets — GPU-accelerated MLP
=============================================================

Reproduces Table 13 / Figure 19 of the thesis using the
curated universal datasets (Human + Non-Human).

For each dataset and each partition scenario (S1–S4, Scaffold):
  - Single train/test split following the scenario's partition rules
  - MLP classifier on 2048-bit Morgan fingerprints (PyTorch + CUDA)
  - 10 seeds for variance estimation
  - Reports MCC mean ± std

The KEY difference from CV: each scenario produces a DIFFERENT
train/test partition from the same pool. S1 allows compound/scaffold
leakage; Scaffold ensures structural disjunction; S4 ensures both
compound AND kinase disjunction. The test set difficulty increases.

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
from sklearn.model_selection import train_test_split
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
N_SEEDS = 10
BASE_SEED = 42
FP_BITS = 2048
FP_RADIUS = 2
TEST_FRACTION = 0.20  # 80/20 split

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


# ── partition strategies ───────────────────────────────────────────────────
def partition_s1(df, seed):
    """S1: random stratified split. Compounds/scaffolds CAN leak."""
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=TEST_FRACTION,
        stratify=df["label"].values, random_state=seed,
    )
    return train_idx, test_idx


def partition_by_group(df, group_col, seed):
    """Group-disjoint split: entities in test never appear in train."""
    rng = np.random.default_rng(seed)
    groups = df[group_col].fillna("__NA__").astype(str).values
    unique_groups = np.unique(groups)
    rng.shuffle(unique_groups)

    # Select ~20% of groups for test
    n_test_groups = max(1, int(len(unique_groups) * TEST_FRACTION))
    test_groups = set(unique_groups[:n_test_groups])

    test_mask = np.isin(groups, list(test_groups))
    train_idx = np.where(~test_mask)[0]
    test_idx = np.where(test_mask)[0]
    return train_idx, test_idx


def partition_s4(df, seed):
    """S4: double disjoint — compounds AND kinases in test are new."""
    rng = np.random.default_rng(seed)

    compounds = df["chembl_id"].unique()
    kinases = df["target_kinase"].unique()
    rng.shuffle(compounds)
    rng.shuffle(kinases)

    n_test_compounds = max(1, int(len(compounds) * TEST_FRACTION))
    n_test_kinases = max(1, int(len(kinases) * TEST_FRACTION))

    test_compounds = set(compounds[:n_test_compounds])
    test_kinases = set(kinases[:n_test_kinases])

    # Test: both compound AND kinase must be in test partition
    test_mask = df["chembl_id"].isin(test_compounds) & df["target_kinase"].isin(test_kinases)
    # Train: neither compound NOR kinase is in test partition
    train_mask = ~df["chembl_id"].isin(test_compounds) & ~df["target_kinase"].isin(test_kinases)

    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]
    return train_idx, test_idx


def partition(df, scenario, seed):
    if scenario == "S1":
        return partition_s1(df, seed)
    elif scenario == "S2":
        return partition_by_group(df, "chembl_id", seed)
    elif scenario == "Scaffold":
        return partition_by_group(df, "scaffold", seed)
    elif scenario == "S3":
        return partition_by_group(df, "target_kinase", seed)
    elif scenario == "S4":
        return partition_s4(df, seed)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")


# ── GPU training ───────────────────────────────────────────────────────────
def train_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
    seed: int,
) -> float:
    """Train MLP on GPU, return MCC on external test."""
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    X_tr_t = torch.from_numpy(X_tr.astype(np.float32)).to(device)
    y_tr_t = torch.from_numpy(y_train.astype(np.float32)).to(device)
    X_te_t = torch.from_numpy(X_te.astype(np.float32)).to(device)

    n_pos = max((y_train == 1).sum(), 1)
    n_neg = max((y_train == 0).sum(), 1)
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)

    torch.manual_seed(seed)
    model = MLP(X_tr.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    train_ds = TensorDataset(X_tr_t, y_tr_t)
    train_dl = DataLoader(train_ds, batch_size=2048, shuffle=True, drop_last=False)

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

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(X_te_t)
        preds = (logits > 0.0).cpu().numpy().astype(int)

    return float(matthews_corrcoef(y_test, preds))


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

        # Load and combine into single pool
        dfs = []
        for fpath in files.values():
            if fpath.exists():
                dfs.append(pd.read_csv(fpath, sep="\t"))
            else:
                print(f"  WARNING: {fpath} not found")
        df = pd.concat(dfs, ignore_index=True)
        print(f"  Total rows: {len(df):,}")
        print(f"  Unique compounds: {df['chembl_id'].nunique():,}")
        print(f"  Unique kinases: {df['target_kinase'].nunique()}")

        # Ensure scaffold column
        if "scaffold" not in df.columns:
            print("  Computing scaffolds...")
            df["scaffold"] = df["canonical_smiles"].apply(get_scaffold)

        # Build fingerprint matrix
        print("  Computing Morgan fingerprints...")
        X_all, y_all, valid_mask = build_fp_matrix(df)
        df_valid = df.loc[valid_mask].reset_index(drop=True)
        print(f"  Valid molecules: {len(df_valid):,} / {len(df):,}")

        ds_results = {}

        for scenario in SCENARIOS:
            print(f"\n  --- Scenario {scenario} ---")
            mccs = []

            for s in range(N_SEEDS):
                seed = BASE_SEED + s
                try:
                    train_idx, test_idx = partition(df_valid, scenario, seed)
                except Exception as e:
                    print(f"    Seed {seed}: ERROR partitioning: {e}")
                    continue

                y_tr = y_all[train_idx]
                y_te = y_all[test_idx]

                # Skip if single class in train or test
                if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                    print(f"    Seed {seed}: skipped (single class)")
                    continue

                if len(test_idx) < 10:
                    print(f"    Seed {seed}: skipped (test too small: {len(test_idx)})")
                    continue

                mcc = train_evaluate(
                    X_all[train_idx], y_tr,
                    X_all[test_idx], y_te,
                    device, seed,
                )
                mccs.append(mcc)
                print(f"    Seed {seed}: MCC = {mcc:.3f}  "
                      f"(train={len(train_idx):,}, test={len(test_idx):,})")

            if mccs:
                mean_mcc = float(np.mean(mccs))
                std_mcc = float(np.std(mccs))
                print(f"  {scenario}: MCC = {mean_mcc:.3f} ± {std_mcc:.3f} "
                      f"({len(mccs)} seeds)")
            else:
                mean_mcc, std_mcc = float("nan"), float("nan")
                print(f"  {scenario}: No valid runs!")

            ds_results[scenario] = {
                "mean": round(mean_mcc, 3),
                "std": round(std_mcc, 3),
                "n_seeds": len(mccs),
            }

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
