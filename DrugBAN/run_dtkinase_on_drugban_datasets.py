#!/usr/bin/env python3
"""
DT-Kinase on DrugBAN Datasets — Cross-Architecture Benchmark
============================================================

Trains the DT-Kinase Level4CNN (variant v7: multi-head dot-product interaction
maps + 2D CNN + hierarchical attention pooling) on the three DrugBAN datasets
(Human, BioSNAP, BindingDB) using:

  * Random split (7:1:2)    — matches DrugBAN "in-domain" evaluation
  * Cold pair split         — matches DrugBAN "cross-domain" evaluation

Both split types are pre-computed by the DrugBAN repository under:
  DrugBAN/src/datasets/{dataset}/random/{train,val,test}.csv
  DrugBAN/src/datasets/{dataset}/cluster/{train,val,test}.csv  (cold pair)

The script mirrors the full DT-Kinase pipeline:
  1. ESM-2 per-residue protein matrices  [seq_len, prot_dim]
  2. MolFormer per-token ligand matrices  [n_tokens, 768]
  3. InteractionMapCNN v7 — multi-head projection + 2D CNN +
     hierarchical attention pool + linear classifier
  4. Platt calibration (on train) + MCC-optimal threshold (on val)
  5. 5-seed evaluation → mean ± std

Results are saved to:
  DrugBAN/results/dtkinase/{dataset}/{split_type}/results.json

Usage:
  python run_dtkinase_on_drugban_datasets.py --dataset human --split random
  python run_dtkinase_on_drugban_datasets.py --dataset biosnap --split cold
  python run_dtkinase_on_drugban_datasets.py --dataset all --split all
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Resolve paths: script is in DrugBAN/, semantic-screening is one level up
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent          # DrugBAN/
REPO_ROOT   = SCRIPT_DIR.parent                        # semantic-screening/
DRUGBAN_DATASETS = SCRIPT_DIR / "src" / "datasets"
RESULTS_DIR = SCRIPT_DIR / "results" / "drugban_dataset_testes"

# Add benchmark code to sys.path
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------
DATASET_META = {
    "human": {
        "csv_dir": DRUGBAN_DATASETS / "human",
        "n_train_expected": 4197,
    },
    "biosnap": {
        "csv_dir": DRUGBAN_DATASETS / "biosnap",
        "n_train_expected": 19224,
    },
    "bindingdb": {
        "csv_dir": DRUGBAN_DATASETS / "bindingdb",
        "n_train_expected": 34000,
    },
}

SPLIT_SUBDIRS = {
    "random": "random",
    "cold":   "cluster",   # DrugBAN calls cold pair "cluster" in its directory
}

# Datasets that support cold pair split (cluster subdir present)
COLD_SPLIT_SUPPORTED = {"biosnap", "bindingdb"}

CANONICAL_SEEDS = [42, 123, 456, 789, 1024]

# ---------------------------------------------------------------------------
# Embedding dimensions
# ---------------------------------------------------------------------------
ESM2_MODELS = {
    "8M":   "esm2_t6_8M_UR50D",
    "150M": "esm2_t30_150M_UR50D",
    "650M": "esm2_t33_650M_UR50D",
}
ESM2_DIMS = {
    "esm2_t6_8M_UR50D":   320,
    "esm2_t30_150M_UR50D": 640,
    "esm2_t33_650M_UR50D": 1280,
}
MOLFORMER_DIM = 768


# ===========================================================================
# Step 1 — data loading
# ===========================================================================

import pandas as pd

def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and types for DrugBAN CSVs."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl == "smiles":            col_map[c] = "smiles"
        elif cl == "protein":         col_map[c] = "protein_seq"
        elif cl in ("y", "label"):   col_map[c] = "label"
    df = df.rename(columns=col_map)
    df["label"] = df["label"].astype(float).astype(int)
    return df[["smiles", "protein_seq", "label"]]


def load_random_split(dataset: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load random split (7:1:2) for a DrugBAN dataset."""
    base = DATASET_META[dataset]["csv_dir"] / "random"
    train_df = _normalize_df(pd.read_csv(base / "train.csv"))
    val_df   = _normalize_df(pd.read_csv(base / "val.csv"))
    test_df  = _normalize_df(pd.read_csv(base / "test.csv"))
    return train_df, val_df, test_df


def load_cold_split(dataset: str, val_fraction: float = 0.2, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load cold pair (cluster) split for a DrugBAN dataset.

    DrugBAN cold split protocol:
      - Training:   source_train.csv  (source domain, all pairs)
                  + target_train.csv  (target domain, seen drugs/proteins)
      - Validation: stratified 20% of target_train (held out from training)
      - Test:       target_test.csv   (target domain, unseen drugs/proteins)

    Only BioSNAP and BindingDB have cluster splits.
    """
    if dataset not in COLD_SPLIT_SUPPORTED:
        raise ValueError(
            f"Cold pair split not available for '{dataset}'. "
            f"Available: {sorted(COLD_SPLIT_SUPPORTED)}"
        )
    base = DATASET_META[dataset]["csv_dir"] / "cluster"
    src_train   = _normalize_df(pd.read_csv(base / "source_train.csv"))
    tgt_train   = _normalize_df(pd.read_csv(base / "target_train.csv"))
    test_df     = _normalize_df(pd.read_csv(base / "target_test.csv"))

    # Stratified split of target_train → val set
    from sklearn.model_selection import train_test_split
    tgt_train_fit, val_df = train_test_split(
        tgt_train, test_size=val_fraction, stratify=tgt_train["label"],
        random_state=seed,
    )
    # Full train = source_train + remaining target_train
    train_df = pd.concat([src_train, tgt_train_fit], ignore_index=True)

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def load_split(dataset: str, split_type: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Dispatch to the correct split loader."""
    if split_type == "random":
        return load_random_split(dataset)
    elif split_type == "cold":
        return load_cold_split(dataset)
    else:
        raise ValueError(f"Unknown split_type: '{split_type}'. Choose 'random' or 'cold'.")


# ===========================================================================
# Step 2 — Protein embedding: ESM-2 per-residue matrices
# ===========================================================================

def compute_protein_matrices(
    sequences: List[str],
    esm_model_name: str,
    device: torch.device,
    cache_dir: Path,
    batch_size: int = 8,
) -> Dict[str, np.ndarray]:
    """Generate ESM-2 per-residue matrices for all unique protein sequences.

    Output shape per sequence: [seq_len, prot_dim]
    Files are cached at cache_dir/{seq_hash}_matrix.npy
    """
    import hashlib
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Deduplicate sequences
    unique_seqs = list(dict.fromkeys(sequences))
    print(f"  Computing ESM-2 matrices for {len(unique_seqs)} unique proteins "
          f"(model: {esm_model_name})...")

    # Build hash map: sequence → cache key
    def _seq_hash(seq: str) -> str:
        return hashlib.sha256(seq.encode()).hexdigest()[:16]

    seq_to_hash = {seq: _seq_hash(seq) for seq in unique_seqs}

    # Check cache
    missing = [
        seq for seq, h in seq_to_hash.items()
        if not (cache_dir / f"{h}_matrix.npy").exists()
    ]
    print(f"  Cache: {len(unique_seqs) - len(missing)} found, {len(missing)} to compute.")

    if missing:
        # HuggingFace ESM-2 model IDs
        HF_ESM_MODELS = {
            "esm2_t6_8M_UR50D":    "facebook/esm2_t6_8M_UR50D",
            "esm2_t30_150M_UR50D": "facebook/esm2_t30_150M_UR50D",
            "esm2_t33_650M_UR50D": "facebook/esm2_t33_650M_UR50D",
        }

        try:
            from transformers import EsmTokenizer, EsmModel
        except ImportError:
            raise ImportError(
                "transformers not found. Install with: pip install transformers"
            )

        hf_model_id = HF_ESM_MODELS.get(esm_model_name, esm_model_name)
        tokenizer = EsmTokenizer.from_pretrained(hf_model_id)
        esm_model = EsmModel.from_pretrained(hf_model_id).eval().to(device)

        from tqdm import tqdm
        for i in tqdm(range(0, len(missing), batch_size),
                      desc="  ESM-2 encoding", unit="batch"):
            batch_seqs = missing[i:i + batch_size]
            enc = tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1022,  # ESM-2 max is 1024 incl. special tokens
            )
            input_ids      = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            with torch.no_grad():
                out = esm_model(input_ids=input_ids, attention_mask=attention_mask)

            # last_hidden_state: [B, L_padded, D]  (includes BOS/EOS pads)
            hidden = out.last_hidden_state  # [B, L, D]
            for j, seq in enumerate(batch_seqs):
                # ESM tokenizer: <cls> at pos 0, <eos> at pos L+1
                # mask_j.sum() = len(seq) + 2 (cls + residues + eos)
                mask_j = attention_mask[j].bool().cpu()  # [L_padded]
                seq_len = min(len(seq), int(mask_j.sum().item()) - 2)  # subtract cls,eos
                seq_len = max(seq_len, 1)
                mat = hidden[j, 1 : seq_len + 1, :].float().cpu().numpy()  # [seq_len, D]
                h = seq_to_hash[seq]
                np.save(cache_dir / f"{h}_matrix.npy", mat)

        del esm_model, tokenizer
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        from tqdm import tqdm  # ensure tqdm imported even in cache-hit path


    # Build output dict: sequence → matrix
    result = {}
    for seq in unique_seqs:
        h = seq_to_hash[seq]
        result[seq] = np.load(cache_dir / f"{h}_matrix.npy")
    return result


# ===========================================================================
# Step 3 — Ligand embedding: MolFormer per-token matrices
# ===========================================================================

def compute_ligand_matrices(
    smiles_list: List[str],
    device: torch.device,
    cache_dir: Path,
) -> Dict[str, np.ndarray]:
    """Generate MolFormer per-token matrices for all unique SMILES.

    Output shape per SMILES: [n_tokens, 768]
    Files are cached at cache_dir/{smiles_hash}_matrix.npy
    """
    import hashlib
    cache_dir.mkdir(parents=True, exist_ok=True)

    unique_smiles = list(dict.fromkeys(smiles_list))
    print(f"  Computing MolFormer matrices for {len(unique_smiles)} unique SMILES...")

    def _smiles_hash(smi: str) -> str:
        return hashlib.sha256(smi.encode()).hexdigest()[:16]

    smi_to_hash = {smi: _smiles_hash(smi) for smi in unique_smiles}

    missing = [
        smi for smi, h in smi_to_hash.items()
        if not (cache_dir / f"{h}_matrix.npy").exists()
    ]
    print(f"  Cache: {len(unique_smiles) - len(missing)} found, {len(missing)} to compute.")

    if missing:
        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError:
            raise ImportError("transformers not found. Install with: pip install transformers")

        print("  Loading MolFormer-XL model (ibm/MolFormer-XL-both-10pct)...")
        tokenizer = AutoTokenizer.from_pretrained(
            "ibm/MolFormer-XL-both-10pct", trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            "ibm/MolFormer-XL-both-10pct", trust_remote_code=True,
            output_hidden_states=True,
        )
        model = model.eval().to(device)

        BATCH = 64
        from tqdm import tqdm
        for i in tqdm(range(0, len(missing), BATCH),
                      desc="  MolFormer encoding", unit="batch"):
            batch_smi = missing[i:i + BATCH]
            enc = tokenizer(
                batch_smi,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            input_ids      = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)

            # Last hidden state: [B, seq_tokens, 768]
            hidden = out.last_hidden_state  # or out.hidden_states[-1]

            for j, smi in enumerate(batch_smi):
                # Keep only valid (non-padding) token positions
                mask = attention_mask[j].bool().cpu()
                mat = hidden[j][mask].float().cpu().numpy()   # [n_valid_tokens, 768]
                h = smi_to_hash[smi]
                np.save(cache_dir / f"{h}_matrix.npy", mat)

        del model, tokenizer
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result = {}
    for smi in unique_smiles:
        h = smi_to_hash[smi]
        result[smi] = np.load(cache_dir / f"{h}_matrix.npy")
    return result


# ===========================================================================
# Step 4 — Dataset + DataLoader (inline, mirrors matrix_utils.py)
# ===========================================================================

from torch.utils.data import Dataset, DataLoader

class DrugBANMatrixDataset(Dataset):
    """Pair dataset mapping (protein_seq, smiles) → (protein_matrix, ligand_matrix, label)."""

    def __init__(
        self,
        df: pd.DataFrame,
        prot_matrices: Dict[str, np.ndarray],
        lig_matrices: Dict[str, np.ndarray],
        prot_dim: int = 320,
        lig_dim: int = MOLFORMER_DIM,
    ) -> None:
        self._df = df.reset_index(drop=True)
        self._prot = prot_matrices
        self._lig = lig_matrices
        self._prot_dim = prot_dim
        self._lig_dim = lig_dim

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int):
        row = self._df.iloc[idx]
        prot_mat = self._prot.get(
            row["protein_seq"],
            np.zeros((1, self._prot_dim), dtype=np.float32),  # safe fallback
        )
        lig_mat = self._lig.get(
            row["smiles"],
            np.zeros((1, self._lig_dim), dtype=np.float32),   # safe fallback
        )
        label = int(row["label"])
        return prot_mat.astype(np.float32), lig_mat.astype(np.float32), label



def _pad_and_mask(
    matrices: Tuple[np.ndarray, ...],
) -> Tuple[np.ndarray, np.ndarray]:
    """Zero-pad a list of 2D matrices to common length → (padded, mask)."""
    max_len = max(m.shape[0] for m in matrices)
    dim = matrices[0].shape[1]
    padded = np.zeros((len(matrices), max_len, dim), dtype=np.float32)
    mask   = np.ones((len(matrices), max_len), dtype=bool)
    for i, mat in enumerate(matrices):
        L = mat.shape[0]
        padded[i, :L] = mat
        mask[i, L:]   = False
    return padded, mask


def _collate(batch):
    prot_mats, lig_mats, labels = zip(*batch)
    p, pm = _pad_and_mask(prot_mats)
    l, lm = _pad_and_mask(lig_mats)
    return {
        "protein_matrix": torch.from_numpy(p),
        "ligand_matrix":  torch.from_numpy(l),
        "protein_mask":   torch.from_numpy(pm.astype(np.float32)),
        "ligand_mask":    torch.from_numpy(lm.astype(np.float32)),
        "label":          torch.tensor(labels, dtype=torch.float32),
    }


def build_loaders(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
    prot_mats: Dict[str, np.ndarray],
    lig_mats:  Dict[str, np.ndarray],
    batch_size: int = 64,
    prot_dim: int = 320,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    kw = {
        "collate_fn": _collate,
        "pin_memory": torch.cuda.is_available(),
        "num_workers": min(4, os.cpu_count() or 0) if torch.cuda.is_available() else 0,
    }
    ds_kwargs = dict(prot_dim=prot_dim, lig_dim=MOLFORMER_DIM)
    train_loader = DataLoader(
        DrugBANMatrixDataset(train_df, prot_mats, lig_mats, **ds_kwargs),
        batch_size=batch_size, shuffle=True, **kw,
    )
    val_loader = DataLoader(
        DrugBANMatrixDataset(val_df, prot_mats, lig_mats, **ds_kwargs),
        batch_size=batch_size, shuffle=False, **kw,
    )
    test_loader = DataLoader(
        DrugBANMatrixDataset(test_df, prot_mats, lig_mats, **ds_kwargs),
        batch_size=batch_size, shuffle=False, **kw,
    )
    return train_loader, val_loader, test_loader



# ===========================================================================
# Step 5 — Import InteractionMapCNN v7 from the existing codebase
# ===========================================================================

def load_model(protein_dim: int, variant: str = "v7",
               num_heads: int = 8, head_dim: int = 32,
               cnn_channels: int = 64, dropout: float = 0.3) -> nn.Module:
    """Import and instantiate InteractionMapCNN from the main codebase."""
    try:
        from benchmark.levels.level4_cnn import InteractionMapCNN
        model = InteractionMapCNN(
            protein_dim=protein_dim,
            ligand_dim=MOLFORMER_DIM,
            num_heads=num_heads,
            head_dim=head_dim,
            cnn_channels=cnn_channels,
            dropout=dropout,
            variant=variant,
        )
        return model
    except ImportError as e:
        raise ImportError(
            f"Could not import InteractionMapCNN: {e}\n"
            f"Make sure semantic-screening/ is in sys.path: {REPO_ROOT}"
        )


# ===========================================================================
# Step 6 — Training loop (mirrors level4_cnn._train_interaction_cnn)
# ===========================================================================

from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
    average_precision_score,
)
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm


def _set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _pos_weight(labels: np.ndarray) -> float:
    n_pos = max(int((labels == 1).sum()), 1)
    n_neg = max(int((labels == 0).sum()), 1)
    return float(np.clip(n_neg / n_pos, 1.0, 20.0))


def _extract_labels_from_loader(loader: DataLoader) -> np.ndarray:
    all_labels = []
    for batch in loader:
        all_labels.append(batch["label"].numpy())
    return np.concatenate(all_labels).astype(int)


@torch.inference_mode()
def _predict_raw(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    model_dtype: torch.dtype,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (raw_sigmoid_probabilities, true_labels) — no calibration."""
    model.eval()
    probs_all, labels_all = [], []
    for batch in loader:
        p  = batch["protein_matrix"].to(device=device, dtype=model_dtype)
        l  = batch["ligand_matrix"].to(device=device, dtype=model_dtype)
        pm = batch["protein_mask"].to(device)
        lm = batch["ligand_mask"].to(device)
        logits = model(p, l, pm, lm).float().cpu().numpy().ravel()  # [B]
        proba  = 1.0 / (1.0 + np.exp(-logits))  # stable sigmoid
        probs_all.append(proba)
        labels_all.append(batch["label"].numpy().ravel())
    return np.concatenate(probs_all), np.concatenate(labels_all).astype(int)



def _best_mcc_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> Tuple[float, float]:
    """Two-pass threshold sweep maximizing MCC."""
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return 0.5, 0.0
    grid = np.linspace(0.01, 0.99, 100)
    anchors = np.unique(np.clip(y_proba, 0.01, 0.99))
    thresholds = np.unique(np.concatenate([grid, anchors]))
    best_thr, best_mcc = 0.5, -1.0
    for thr in thresholds:
        mcc = float(matthews_corrcoef(y_true, (y_proba >= thr).astype(int)))
        if mcc > best_mcc:
            best_mcc, best_thr = mcc, float(thr)
    # Fine pass
    for thr in np.linspace(max(0.01, best_thr - 0.05), min(0.99, best_thr + 0.05), 100):
        mcc = float(matthews_corrcoef(y_true, (y_proba >= thr).astype(int)))
        if mcc > best_mcc:
            best_mcc, best_thr = mcc, float(thr)
    return best_thr, best_mcc


def _compute_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
) -> Dict:
    """Compute all evaluation metrics. Handles edge cases (single-class splits)."""
    y_pred = (y_proba >= threshold).astype(int)
    n_classes = len(np.unique(y_true))

    # AUROC / AUPRC require both classes present
    if n_classes < 2:
        auroc = float("nan")
        auprc = float("nan")
    else:
        auroc = float(roc_auc_score(y_true, y_proba))
        auprc = float(average_precision_score(y_true, y_proba))

    return {
        "auroc":       auroc,
        "auprc":       auprc,
        "accuracy":    float(accuracy_score(y_true, y_pred)),
        "mcc":         float(matthews_corrcoef(y_true, y_pred)),
        "f1":          float(f1_score(y_true, y_pred, zero_division=0)),
        "precision":   float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":      float(recall_score(y_true, y_pred, zero_division=0)),
        # DrugBAN paper reports sensitivity/specificity
        "sensitivity": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "specificity": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "threshold":   float(threshold),
    }


def train_one_seed(
    train_loader:  DataLoader,
    val_loader:    DataLoader,
    test_loader:   DataLoader,
    protein_dim:   int,
    seed:          int,
    train_labels:  Optional[np.ndarray] = None,  # pre-computed; extracted from loader if None
    epochs:        int = 100,
    lr:            float = 5e-5,
    batch_size:    int = 64,
    variant:       str = "v7",
    num_heads:     int = 8,
    head_dim:      int = 32,
    cnn_channels:  int = 64,
    dropout:       float = 0.3,
    weight_decay:  float = 0.02,
    patience:      int = 10,
    device:        Optional[torch.device] = None,
    use_double:    bool = False,
) -> Dict:
    """Train InteractionMapCNN for one seed, return test metrics."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _set_seed(seed)

    model_dtype = torch.float64 if use_double else torch.float32
    model = load_model(
        protein_dim=protein_dim,
        variant=variant,
        num_heads=num_heads,
        head_dim=head_dim,
        cnn_channels=cnn_channels,
        dropout=dropout,
    ).to(device=device, dtype=model_dtype)

    # pos_weight from training labels (use pre-computed if provided)
    _labels = train_labels if train_labels is not None else _extract_labels_from_loader(train_loader)
    pw = _pos_weight(_labels)
    pw_tensor = torch.tensor([pw], device=device, dtype=model_dtype)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )

    use_amp = (device.type == "cuda") and (not use_double)
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_auroc = -1.0
    best_state: Optional[Dict] = None
    no_improve = 0

    pbar = tqdm(range(1, epochs + 1), desc=f"  Seed {seed}", leave=False)
    for epoch in pbar:
        # --- Training ---
        model.train()
        epoch_loss = 0.0
        n_batches  = 0
        for batch in train_loader:
            p  = batch["protein_matrix"].to(device=device, dtype=model_dtype)
            l  = batch["ligand_matrix"].to(device=device, dtype=model_dtype)
            pm = batch["protein_mask"].to(device)
            lm = batch["ligand_mask"].to(device)
            y  = batch["label"].to(device=device, dtype=model_dtype)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(p, l, pm, lm).squeeze(-1)
                loss   = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            n_batches  += 1
        scheduler.step()

        # --- Validation AUROC (DrugBAN selection criterion) ---
        val_proba, val_labels = _predict_raw(model, val_loader, device, model_dtype)
        if len(np.unique(val_labels)) > 1:
            val_auroc = float(roc_auc_score(val_labels, val_proba))
        else:
            val_auroc = 0.0

        pbar.set_postfix({
            "loss": f"{epoch_loss / max(n_batches, 1):.4f}",
            "val_auroc": f"{val_auroc:.4f}",
        })

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_state     = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve     = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                tqdm.write(f"    Early stop at epoch {epoch} (patience={patience})")
                break

    # --- Restore best model ---
    if best_state is not None:
        model.load_state_dict(best_state)

    # --- Platt scaling calibration (fitted on train raw sigmoid probs) ---
    # atanh(sigmoid) ≈ logit/2 — maps (0,1) → ℝ for LogisticRegression input
    train_proba, train_labels_np = _predict_raw(model, train_loader, device, model_dtype)
    train_logit_feats = np.arctanh(np.clip(train_proba, 1e-7, 1 - 1e-7)).reshape(-1, 1)
    calibrator = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
    calibrator.fit(train_logit_feats, train_labels_np)

    def _apply_calibration(raw_proba: np.ndarray) -> np.ndarray:
        """Apply Platt calibration to an array of raw sigmoid probabilities."""
        feats = np.arctanh(np.clip(raw_proba, 1e-7, 1 - 1e-7)).reshape(-1, 1)
        return calibrator.predict_proba(feats)[:, 1]

    # --- Val: find MCC-optimal threshold on calibrated probabilities ---
    val_proba_raw, val_labels_cal = _predict_raw(model, val_loader, device, model_dtype)
    val_proba_cal = _apply_calibration(val_proba_raw)
    val_threshold, _ = _best_mcc_threshold(val_labels_cal, val_proba_cal)

    # --- Test: calibrate then apply val threshold ---
    test_proba_raw, test_labels = _predict_raw(model, test_loader, device, model_dtype)
    test_proba = _apply_calibration(test_proba_raw)
    test_metrics = _compute_metrics(test_labels, test_proba, val_threshold)
    test_metrics["best_val_auroc"] = round(best_val_auroc, 6)
    test_metrics["seed"]           = seed

    return test_metrics



# ===========================================================================
# Step 7 — Multi-seed evaluation
# ===========================================================================

def run_benchmark(
    dataset:     str,
    split_type:  str,
    esm_variant: str = "8M",
    seeds:       List[int] = None,
    epochs:      int = 100,
    lr:          float = 5e-5,
    batch_size:  int = 64,
    variant:     str = "v7",
    num_heads:   int = 8,
    head_dim:    int = 32,
    cnn_channels: int = 64,
    dropout:     float = 0.3,
    weight_decay: float = 0.02,
    patience:    int = 10,
    use_double:  bool = True,
    force:       bool = False,
) -> Dict:
    """Full benchmark for one (dataset, split_type) combination."""
    if seeds is None:
        seeds = CANONICAL_SEEDS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f" Dataset: {dataset}  |  Split: {split_type}  |  Device: {device}")
    print(f" ESM-2: {esm_variant}  |  Variant: {variant}  |  Seeds: {seeds}")
    print(f"{'='*60}\n")

    # Output directory
    out_dir = RESULTS_DIR / dataset / split_type
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"

    if results_path.exists() and not force:
        print(f"  Results already exist: {results_path}  (use --force to rerun)")
        with open(results_path) as fh:
            return json.load(fh)

    # --- Load CSVs ---
    print("Loading split CSVs...")
    train_df, val_df, test_df = load_split(dataset, split_type)
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Combine all sequences/smiles for embedding computation
    all_seqs   = pd.concat([train_df, val_df, test_df])["protein_seq"].tolist()
    all_smiles = pd.concat([train_df, val_df, test_df])["smiles"].tolist()

    # Cache directories
    esm_cache  = out_dir / f"cache_esm2_{esm_variant}"
    mol_cache  = out_dir / "cache_molformer"

    # --- ESM-2 embeddings ---
    esm_model_name = ESM2_MODELS[esm_variant]
    print("\nStep 2: Protein matrices (ESM-2)...")
    prot_matrices = compute_protein_matrices(
        sequences=all_seqs,
        esm_model_name=esm_model_name,
        device=device,
        cache_dir=esm_cache,
        batch_size=8,
    )

    # --- MolFormer embeddings ---
    print("\nStep 3: Ligand matrices (MolFormer)...")
    lig_matrices = compute_ligand_matrices(
        smiles_list=all_smiles,
        device=device,
        cache_dir=mol_cache,
    )

    protein_dim = ESM2_DIMS[esm_model_name]

    # Pre-extract train labels from DataFrame (no DataLoader needed)
    train_labels_np = train_df["label"].to_numpy(dtype=int)


    # --- Multi-seed training ---
    all_seed_results = []
    for seed in seeds:
        print(f"\n{'─'*40}")
        print(f" Training seed {seed}/{seeds[-1]} ...")
        t0 = time.time()

        train_loader, val_loader, test_loader = build_loaders(
            train_df, val_df, test_df,
            prot_matrices, lig_matrices,
            batch_size=batch_size,
            prot_dim=protein_dim,
        )

        seed_result = train_one_seed(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            protein_dim=protein_dim,
            seed=seed,
            train_labels=train_labels_np,   # pass pre-extracted; avoids one loader pass per seed
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            variant=variant,
            num_heads=num_heads,
            head_dim=head_dim,
            cnn_channels=cnn_channels,
            dropout=dropout,
            weight_decay=weight_decay,
            patience=patience,
            device=device,
            use_double=use_double,
        )
        elapsed = time.time() - t0
        seed_result["elapsed_seconds"] = round(elapsed, 1)
        all_seed_results.append(seed_result)
        print(f"  Seed {seed}: AUROC={seed_result['auroc']:.4f} | "
              f"MCC={seed_result['mcc']:.4f} | "
              f"AUPRC={seed_result['auprc']:.4f} ({elapsed:.0f}s)")

    # --- Aggregate ---
    metric_keys = ["auroc", "auprc", "accuracy", "mcc", "f1",
                   "precision", "recall", "sensitivity", "specificity"]
    agg = {}
    for k in metric_keys:
        vals = [r[k] for r in all_seed_results]
        agg[k] = {
            "mean":  float(np.mean(vals)),
            "std":   float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "values": vals,
        }

    output = {
        "dataset":       dataset,
        "split_type":    split_type,
        "model":         "DT-Kinase Level4CNN",
        "variant":       variant,
        "esm_variant":   esm_variant,
        "seeds":         seeds,
        "epochs":        epochs,
        "lr":            lr,
        "batch_size":    batch_size,
        "num_heads":     num_heads,
        "head_dim":      head_dim,
        "cnn_channels":  cnn_channels,
        "dropout":       dropout,
        "weight_decay":  weight_decay,
        "n_train":       len(train_df),
        "n_val":         len(val_df),
        "n_test":        len(test_df),
        "aggregated":    agg,
        "per_seed":      all_seed_results,
        "timestamp":     time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    with open(results_path, "w") as fh:
        json.dump(output, fh, indent=2)
    print(f"\n  Results saved → {results_path}")

    # Print summary table
    print(f"\n{'─'*50}")
    print(f" FINAL RESULTS: {dataset} / {split_type}")
    print(f"{'─'*50}")
    for k in metric_keys:
        m = agg[k]["mean"]
        s = agg[k]["std"]
        print(f"  {k:<15}: {m:.4f} ± {s:.4f}")
    print(f"{'─'*50}")

    return output


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train DT-Kinase Level4CNN on DrugBAN datasets"
    )
    parser.add_argument(
        "--dataset", choices=["human", "biosnap", "bindingdb", "all"],
        default="human",
        help="Which DrugBAN dataset to use (default: human)",
    )
    parser.add_argument(
        "--split", choices=["random", "cold", "all"],
        default="random",
        help="Split protocol: random (7:1:2) | cold (clustering) | all (default: random)",
    )
    parser.add_argument(
        "--esm", choices=["8M", "150M", "650M"],
        default="8M",
        help="ESM-2 model size (default: 8M = esm2_t6_8M_UR50D)",
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Max training epochs per seed (default: 100, same as DrugBAN paper)",
    )
    parser.add_argument(
        "--lr", type=float, default=5e-5,
        help="Learning rate (default: 5e-5, same as DrugBAN paper)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size (default: 64, same as DrugBAN paper)",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=CANONICAL_SEEDS,
        help="Random seeds (default: 42 123 456 789 1024)",
    )
    parser.add_argument(
        "--patience", type=int, default=10,
        help="Early stopping patience in epochs (default: 10 = 10%% of 100)",
    )
    parser.add_argument(
        "--variant", default="v7",
        help="InteractionMapCNN variant (default: v7 = dot-product)",
    )
    parser.add_argument(
        "--num-heads", type=int, default=8,
        help="Number of interaction map heads (default: 8)",
    )
    parser.add_argument(
        "--head-dim", type=int, default=32,
        help="Projection head dimension (default: 32)",
    )
    parser.add_argument(
        "--cnn-channels", type=int, default=64,
        help="CNN output channels (default: 64)",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.3,
        help="Dropout rate (default: 0.3)",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.02,
        help="AdamW weight decay (default: 0.02)",
    )
    parser.add_argument(
        "--no-double", action="store_true",
        help="Use float32 instead of float64 (faster, less precise)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Rerun even if results.json already exists",
    )
    args = parser.parse_args()

    datasets    = ["human", "biosnap", "bindingdb"] if args.dataset == "all" else [args.dataset]
    split_types = ["random", "cold"]                if args.split   == "all" else [args.split]

    all_results = {}
    for ds in datasets:
        for sp in split_types:
            # Human dataset has no cluster/cold split
            if sp == "cold" and ds not in COLD_SPLIT_SUPPORTED:
                print(f"\n[SKIP] Cold split not available for '{ds}' — skipping.")
                continue
            key = f"{ds}/{sp}"
            try:
                result = run_benchmark(
                    dataset=ds,
                    split_type=sp,
                    esm_variant=args.esm,
                    seeds=args.seeds,
                    epochs=args.epochs,
                    lr=args.lr,
                    batch_size=args.batch_size,
                    variant=args.variant,
                    num_heads=args.num_heads,
                    head_dim=args.head_dim,
                    cnn_channels=args.cnn_channels,
                    dropout=args.dropout,
                    weight_decay=args.weight_decay,
                    patience=args.patience,
                    use_double=(not args.no_double),
                    force=args.force,
                )
                all_results[key] = result["aggregated"]
            except Exception as e:
                print(f"\nERROR on {key}: {e}")
                import traceback; traceback.print_exc()
                all_results[key] = {"error": str(e)}

    # --- Final comparison table ---
    print(f"\n{'='*70}")
    print(" BENCHMARK SUMMARY — DT-Kinase vs DrugBAN reference values")
    print(f"{'='*70}")
    print(f"  {'Dataset/Split':<25} {'AUROC':>10} {'AUPRC':>10} {'MCC':>10}")
    print(f"  {'─'*55}")

    # DrugBAN paper reference values (Table 1, random split, in-domain)
    drugban_ref = {
        "human/random":   {"auroc": 0.990, "auprc": 0.989},
        "biosnap/random": {"auroc": 0.917, "auprc": 0.934},
        "bindingdb/random": {"auroc": 0.960, "auprc": 0.969},
    }
    for key, agg in all_results.items():
        if "error" in agg:
            print(f"  {key:<25} ERROR: {agg['error'][:30]}")
            continue
        auroc = agg.get("auroc", {}).get("mean", 0)
        auprc = agg.get("auprc", {}).get("mean", 0)
        mcc   = agg.get("mcc",   {}).get("mean", 0)
        print(f"  {key:<25} {auroc:>10.4f} {auprc:>10.4f} {mcc:>10.4f}")

    if drugban_ref:
        print(f"\n  {'─'*55}")
        print("  DrugBAN reference (random, from paper Table 1):")
        for key, ref in drugban_ref.items():
            print(f"  {key:<25} {ref['auroc']:>10.4f} {ref['auprc']:>10.4f} {'N/A':>10}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
