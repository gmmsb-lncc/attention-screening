#!/usr/bin/env python3
"""
Leakage analysis on Universal Datasets — GPU-accelerated MLP
=============================================================

Reproduces Table 13 / Figure 19 of the thesis using the
curated universal datasets (Human + Non-Human).

Features:
  - Protein: ESM-2-8M mean-pooled embedding (320-dim), loaded from pre-computed .npy
  - Ligand:  Morgan fingerprint (2048-bit)
  - Concatenated → MLP classifier (PyTorch + CUDA)

For each dataset and each partition scenario (S1–S4, Scaffold):
  - Single train/test split following the scenario's partition rules
  - 10 seeds for variance estimation
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

# ── paths ──────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SPLITS_DIR = REPO_ROOT / "scaffolds_splits" / "output"
RESULTS_DIR = REPO_ROOT / "results"

DATASETS = {
    "Non-Human": {
        "train": SPLITS_DIR / "non_human_train.tsv",
        "val":   SPLITS_DIR / "non_human_val.tsv",
        "test":  SPLITS_DIR / "non_human_test.tsv",
    },
    "Human": {
        "train": SPLITS_DIR / "human_train.tsv",
        "val":   SPLITS_DIR / "human_val.tsv",
        "test":  SPLITS_DIR / "human_test.tsv",
    },
}

# Pre-computed ESM-2 8M protein matrices: {seq_id}_matrix.npy → [seq_len, 320]
PROTEIN_MATRIX_DIRS = {
    "Non-Human": [
        RESULTS_DIR / "protein_model_benchmark_non_human_v2" / "esm2_t6_8M_UR50D" / "build" / "protein_matrices",
    ],
    "Human": [
        RESULTS_DIR / "protein_model_benchmark_human_v2" / "esm2_t6_8M_UR50D" / "build" / "protein_matrices",
    ],
}

SCENARIOS = ["S1", "S2", "Scaffold", "S3", "S4"]
N_SEEDS = 10
BASE_SEED = 42
FP_BITS = 2048
FP_RADIUS = 2
TEST_FRACTION = 0.10     # ~10% of pool for test
HOLDOUT_FRACTION = 0.20  # 20% held out → split into 10% val + 10% test

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


# ── Load pre-computed ESM-2 protein embeddings ─────────────────────────────
def load_protein_embeddings(
    unique_seq_ids: list,
    protein_dirs: list[Path],
) -> dict[int, np.ndarray]:
    """Load pre-computed ESM-2 matrices and mean-pool to 320-dim vectors.

    Files: {seq_id}_matrix.npy → [seq_len, embed_dim]
    Returns: {seq_id: mean_pooled_vector [embed_dim]}
    """
    cache = {}
    missing = 0

    for seq_id in unique_seq_ids:
        loaded = False
        for pdir in protein_dirs:
            fpath = pdir / f"{seq_id}_matrix.npy"
            if fpath.exists():
                matrix = np.load(fpath)  # [seq_len, 320]
                cache[seq_id] = matrix.mean(axis=0).astype(np.float32)  # [320]
                loaded = True
                break
        if not loaded:
            missing += 1

    if missing > 0:
        print(f"  WARNING: {missing}/{len(unique_seq_ids)} seq_ids not found in protein_matrices")

    return cache


# ── fingerprints ───────────────────────────────────────────────────────────
def smiles_to_fp(smiles: str) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_BITS)
    return np.array(fp, dtype=np.float32)


def build_feature_matrix(
    df: pd.DataFrame,
    protein_cache: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build concatenated [ESM-2 mean-pool | Morgan FP] matrix."""
    features, labels, mask = [], [], []
    for _, row in df.iterrows():
        fp = smiles_to_fp(row["canonical_smiles"])
        seq_id = row["seq_id"]
        prot_emb = protein_cache.get(seq_id) if not pd.isna(seq_id) else None

        if fp is not None and prot_emb is not None:
            concat = np.concatenate([prot_emb, fp])  # [320 + 2048]
            features.append(concat)
            labels.append(int(row["label"]))
            mask.append(True)
        else:
            mask.append(False)
    return np.array(features, dtype=np.float32), np.array(labels), np.array(mask)


# ── scaffold ───────────────────────────────────────────────────────────────
def get_scaffold(smiles: str) -> str:
    from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
    try:
        return MurckoScaffoldSmiles(smiles)
    except Exception:
        return smiles


# ── partition strategies ───────────────────────────────────────────────────
# All scenarios produce 80% train / 10% val (discarded) / 10% test.
# Scaffold uses the official universal split. Others select 20% held-out
# by scenario rules, then split it 50/50 for val and test.

def _split_holdout_to_val_test(held_out_idx, seed):
    """Split held-out indices 50/50 into val and test."""
    rng = np.random.default_rng(seed + 999)
    rng.shuffle(held_out_idx)
    mid = len(held_out_idx) // 2
    val_idx = held_out_idx[:mid]
    test_idx = held_out_idx[mid:]
    return val_idx, test_idx


def partition_s1(df, seed):
    """S1: random stratified split → 80/10/10."""
    train_idx, held_idx = train_test_split(
        np.arange(len(df)), test_size=HOLDOUT_FRACTION,
        stratify=df["label"].values, random_state=seed,
    )
    val_idx, test_idx = _split_holdout_to_val_test(held_idx, seed)
    return train_idx, val_idx, test_idx


def partition_by_group(df, group_col, seed):
    """Group-disjoint split → 80/10/10."""
    rng = np.random.default_rng(seed)
    groups = df[group_col].fillna("__NA__").astype(str).values
    unique_groups = np.unique(groups)
    rng.shuffle(unique_groups)

    # Select ~20% of groups for held-out
    n_held_groups = max(1, int(len(unique_groups) * HOLDOUT_FRACTION))
    held_groups = list(unique_groups[:n_held_groups])

    # Split held groups 50/50 → val groups + test groups
    mid = len(held_groups) // 2
    val_groups = set(held_groups[:mid])
    test_groups = set(held_groups[mid:])

    held_mask = np.isin(groups, held_groups)
    val_mask = np.isin(groups, list(val_groups))
    test_mask = np.isin(groups, list(test_groups))
    train_idx = np.where(~held_mask)[0]
    val_idx = np.where(val_mask)[0]
    test_idx = np.where(test_mask)[0]
    return train_idx, val_idx, test_idx


def partition_scaffold_universal(df, seed):
    """Scaffold: use the EXACT universal scaffold split (train.tsv → train, test.tsv → test).

    This is deterministic — the partition is always the same (the official one).
    Only the MLP initialization varies across seeds.
    Requires '_split_origin' column with values 'train', 'val', 'test'.
    """
    # train = original train rows ONLY; test = original test rows
    # val = original val rows — used for early stopping
    train_idx = np.where(df["_split_origin"] == "train")[0]
    val_idx = np.where(df["_split_origin"] == "val")[0]
    test_idx = np.where(df["_split_origin"] == "test")[0]
    return train_idx, val_idx, test_idx


def partition_s4(df, seed):
    """S4: double disjoint → 80/10/10."""
    rng = np.random.default_rng(seed)

    compounds = df["chembl_id"].unique()
    kinases = df["target_kinase"].unique()
    rng.shuffle(compounds)
    rng.shuffle(kinases)

    # 20% held-out for compounds and kinases
    n_held_compounds = max(1, int(len(compounds) * HOLDOUT_FRACTION))
    n_held_kinases = max(1, int(len(kinases) * HOLDOUT_FRACTION))

    held_compounds = compounds[:n_held_compounds]
    held_kinases = kinases[:n_held_kinases]

    # Split held compounds/kinases 50/50 → val + test
    mid_c = len(held_compounds) // 2
    mid_k = len(held_kinases) // 2
    test_compounds = set(held_compounds[mid_c:])
    test_kinases = set(held_kinases[mid_k:])
    val_compounds = set(held_compounds[:mid_c])
    val_kinases = set(held_kinases[:mid_k])

    test_mask = df["chembl_id"].isin(test_compounds) & df["target_kinase"].isin(test_kinases)
    val_mask = df["chembl_id"].isin(val_compounds) & df["target_kinase"].isin(val_kinases)
    train_mask = ~df["chembl_id"].isin(set(held_compounds)) & ~df["target_kinase"].isin(set(held_kinases))

    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    test_idx = np.where(test_mask)[0]
    return train_idx, val_idx, test_idx


def partition(df, scenario, seed):
    if scenario == "S1":
        return partition_s1(df, seed)
    elif scenario == "S2":
        return partition_by_group(df, "chembl_id", seed)
    elif scenario == "Scaffold":
        return partition_scaffold_universal(df, seed)
    elif scenario == "S3":
        return partition_by_group(df, "target_kinase", seed)
    elif scenario == "S4":
        return partition_s4(df, seed)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")


# ── GPU training ─────────────────────────────────────────────────────────────
def train_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
    seed: int,
) -> float:
    """Train MLP on GPU with early stopping on val loss. Return MCC on test."""
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_va = scaler.transform(X_val)
    X_te = scaler.transform(X_test)

    X_tr_t = torch.from_numpy(X_tr.astype(np.float32)).to(device)
    y_tr_t = torch.from_numpy(y_train.astype(np.float32)).to(device)
    X_va_t = torch.from_numpy(X_va.astype(np.float32)).to(device)
    y_va_t = torch.from_numpy(y_val.astype(np.float32)).to(device)
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

    best_val_loss = float("inf")
    patience, patience_counter = 10, 0
    best_state = None

    for epoch in range(100):
        # ---- Train ----
        model.train()
        for xb, yb in train_dl:
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # ---- Val loss (early stopping) ----
        model.eval()
        with torch.no_grad():
            val_logits = model(X_va_t)
            val_loss = criterion(val_logits, y_va_t).item()

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
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

        # Load and combine into single pool, tracking origin
        dfs = []
        for split_name, fpath in files.items():
            if fpath.exists():
                tmp = pd.read_csv(fpath, sep="\t")
                tmp["_split_origin"] = split_name  # "train", "val", or "test"
                dfs.append(tmp)
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

        # ── Load pre-computed ESM-2 protein embeddings ─────────────────
        unique_seq_ids = df["seq_id"].dropna().unique().tolist()
        protein_dirs = PROTEIN_MATRIX_DIRS.get(ds_name, [])
        print(f"  Unique seq_ids: {len(unique_seq_ids)}")
        print(f"  Protein matrix dirs: {[str(p) for p in protein_dirs]}")

        protein_cache = load_protein_embeddings(unique_seq_ids, protein_dirs)
        print(f"  Loaded {len(protein_cache)} protein embeddings")

        if len(protein_cache) == 0:
            print("  ERROR: No protein embeddings found! Check protein_matrices path.")
            print("  Skipping this dataset.")
            continue

        # Sample an embedding to get dimension
        sample_emb = next(iter(protein_cache.values()))
        prot_dim = sample_emb.shape[0]
        feat_dim = prot_dim + FP_BITS
        print(f"  Protein dim: {prot_dim}, Feature dim: {feat_dim} (ESM-2 + FP)")

        # ── Build feature matrix ───────────────────────────────────────
        print("  Building feature matrix...")
        X_all, y_all, valid_mask = build_feature_matrix(df, protein_cache)
        df_valid = df.loc[valid_mask].reset_index(drop=True)
        print(f"  Valid samples: {len(df_valid):,} / {len(df):,}")

        ds_results = {}

        for scenario in SCENARIOS:
            print(f"\n  --- Scenario {scenario} ---")
            mccs = []

            for s in range(N_SEEDS):
                seed = BASE_SEED + s
                try:
                    train_idx, val_idx, test_idx = partition(df_valid, scenario, seed)
                except Exception as e:
                    print(f"    Seed {seed}: ERROR: {e}")
                    continue

                y_tr = y_all[train_idx]
                y_va = y_all[val_idx]
                y_te = y_all[test_idx]

                if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                    print(f"    Seed {seed}: skipped (single class in train/test)")
                    continue
                if len(test_idx) < 10:
                    print(f"    Seed {seed}: skipped (test too small: {len(test_idx)})")
                    continue
                if len(val_idx) < 10:
                    print(f"    Seed {seed}: skipped (val too small: {len(val_idx)})")
                    continue

                mcc = train_evaluate(
                    X_all[train_idx], y_tr,
                    X_all[val_idx], y_va,
                    X_all[test_idx], y_te,
                    device, seed,
                )
                mccs.append(mcc)
                print(f"    Seed {seed}: MCC = {mcc:.3f}  "
                      f"(train={len(train_idx):,}, val={len(val_idx):,}, test={len(test_idx):,})")

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
