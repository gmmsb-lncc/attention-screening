#!/usr/bin/env python3
"""
Leakage analysis on Universal Datasets — GPU-accelerated MLP
=============================================================

Reproduces Table 13 / Figure 19 of the thesis using the
curated universal datasets (Human + Non-Human).

Features:
  - Protein: ESM-2-8M mean-pooled embedding (320-dim)
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
TEST_FRACTION = 0.20
ESM_MODEL_NAME = "esm2_t6_8M_UR50D"
ESM_DIM = 320  # embedding dimension for 8M model

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


# ── ESM-2 embeddings ──────────────────────────────────────────────────────
def compute_esm_embeddings(
    unique_seqs: list[str],
    device: torch.device,
    batch_size: int = 8,
) -> dict[str, np.ndarray]:
    """Compute mean-pooled ESM-2 embeddings for each unique sequence."""
    import esm

    print(f"  Loading ESM-2 model ({ESM_MODEL_NAME})...")
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model = model.eval().to(device)

    embeddings = {}
    total = len(unique_seqs)

    for i in range(0, total, batch_size):
        batch_seqs = unique_seqs[i:i + batch_size]
        data = [(f"seq_{i+j}", seq) for j, seq in enumerate(batch_seqs)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)

        with torch.no_grad():
            results = model(tokens, repr_layers=[6], return_contacts=False)
            # results["representations"][6] shape: [batch, seq_len, 320]
            reps = results["representations"][6]

        for j, seq in enumerate(batch_seqs):
            # Mean-pool over sequence (excluding BOS/EOS tokens)
            seq_len = len(seq)
            embedding = reps[j, 1:seq_len + 1, :].mean(dim=0).cpu().numpy()
            embeddings[seq] = embedding

        if (i // batch_size) % 10 == 0:
            print(f"    ESM-2 progress: {min(i + batch_size, total)}/{total}")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()
    return embeddings


# ── fingerprints ───────────────────────────────────────────────────────────
def smiles_to_fp(smiles: str) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_BITS)
    return np.array(fp, dtype=np.float32)


def build_feature_matrix(
    df: pd.DataFrame,
    esm_cache: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build concatenated [ESM-2 mean-pool (320) | Morgan FP (2048)] matrix."""
    features, labels, mask = [], [], []
    for _, row in df.iterrows():
        fp = smiles_to_fp(row["canonical_smiles"])
        seq = row["seq"]
        esm_emb = esm_cache.get(seq)

        if fp is not None and esm_emb is not None:
            concat = np.concatenate([esm_emb, fp])  # [320 + 2048] = [2368]
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

    test_mask = df["chembl_id"].isin(test_compounds) & df["target_kinase"].isin(test_kinases)
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

        # ── Compute ESM-2 embeddings for all unique sequences ──────────
        unique_seqs = df["seq"].dropna().unique().tolist()
        print(f"  Unique protein sequences: {len(unique_seqs)}")

        esm_cache_path = output_dir / f"esm2_cache_{ds_name.lower().replace('-', '_')}.npz"
        if esm_cache_path.exists():
            print(f"  Loading cached ESM-2 embeddings from {esm_cache_path}")
            cached = np.load(esm_cache_path, allow_pickle=True)
            esm_cache = {k: cached[k] for k in cached.files}
        else:
            print(f"  Computing ESM-2 embeddings for {len(unique_seqs)} sequences...")
            esm_cache = compute_esm_embeddings(unique_seqs, device)
            print(f"  Caching ESM-2 embeddings to {esm_cache_path}")
            np.savez_compressed(esm_cache_path, **esm_cache)

        print(f"  ESM-2 cache size: {len(esm_cache)} sequences, dim={ESM_DIM}")

        # ── Build feature matrix ───────────────────────────────────────
        print("  Building feature matrix [ESM-2 (320) | Morgan FP (2048)]...")
        X_all, y_all, valid_mask = build_feature_matrix(df, esm_cache)
        df_valid = df.loc[valid_mask].reset_index(drop=True)
        print(f"  Valid samples: {len(df_valid):,} / {len(df):,}")
        print(f"  Feature dim: {X_all.shape[1]} (ESM-2: {ESM_DIM} + FP: {FP_BITS})")

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
