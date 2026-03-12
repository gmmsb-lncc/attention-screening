"""BAN-Kinase-Network (BKN): GraphBAN with ESM-2 + MoLFormer + Attention Pooling.

Variant of GraphBAN/run_baseline.py with three encoder substitutions:
  1. Protein: ESM-2 650M (esm2_t33_650M_UR50D) instead of ESM-1b
  2. Drug:    MoLFormer-XL (ibm/MoLFormer-XL-both-10pct, 768-d) instead of ChemBERTa (384-d)
  3. Pooling: CLS-guided attention pooling for both encoders instead of mean/CLS-token

Architecture and training protocol are identical to GraphBAN/run_baseline.py:
  - Same scaffold splits (Bemis-Murcko 80/10/10)
  - Same 5 canonical seeds {42, 123, 456, 789, 1024}
  - Same teacher GAE embeddings for knowledge distillation
  - Same domain adaptation (CDAN)
  - Threshold calibrated on validation MCC, applied to test (no leakage)

The key implementation challenge: GraphBAN's trainer.py hardcodes the drug
embedding reshape as (batch, 1, 384). This module patches it dynamically at
import time by reading trainer.py source, substituting 384 → DRUG_EMB_DIM=768,
and exec()-ing the modified source. The original file on disk is never modified.

Usage:
    python run_bkn.py --dataset non_human
    python run_bkn.py --dataset human --seeds 42 123 456
    python run_bkn.py --dataset all --max-epoch 10  # quick test
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
import warnings
from pathlib import Path

# Apply DGL graphbolt compatibility shim before any DGL import
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import dgl_compat  # noqa: F401, E402 — must be before dgl

# Suppress torch.tensor(sourceTensor) deprecation warnings from GraphBAN's
# trainer.py (upstream code, not modified here).
warnings.filterwarnings(
    "ignore",
    message="To copy construct from a tensor.*use sourceTensor.clone",
    category=UserWarning,
)

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = SCRIPT_DIR.parent
GRAPHBAN_DIR = REPO_ROOT / "GraphBAN"
GRAPHBAN_INDUCTIVE = GRAPHBAN_DIR / "src" / "inductive_mode"

# Canonical seeds — same as GraphBAN baseline and DT-Kinase
CANONICAL_SEEDS = [42, 123, 456, 789, 1024]
DEFAULT_TEACHER_EPOCHS = 10

# MoLFormer drug embedding dimension
DRUG_EMB_DIM = 768

# ---------------------------------------------------------------------------
# GraphBAN dynamic loader — patches trainer.py for 768-d drug embeddings
# ---------------------------------------------------------------------------

def _load_trainer_patched(inductive_dir: Path, drug_emb_dim: int = DRUG_EMB_DIM) -> type:
    """Load GraphBAN's Trainer class with drug embedding dimension patched.

    Reads trainer.py source, replaces the hardcoded reshape(_, 1, 384) with
    reshape(_, 1, drug_emb_dim), and exec()s the result. The file on disk is
    never modified, so GraphBAN/run_baseline.py continues to work unaffected.
    """
    trainer_path = inductive_dir / "trainer.py"
    if not trainer_path.exists():
        raise FileNotFoundError(
            f"trainer.py not found at {trainer_path}\n"
            "Clone GraphBAN src to GraphBAN/src/ first:\n"
            "  git clone https://github.com/peizhenbai/GraphBAN "
            f"{GRAPHBAN_DIR / 'src'}"
        )

    source = trainer_path.read_text(encoding="utf-8")

    # Replace ONLY the reshape calls that use the drug embedding dimension.
    # Pattern matches: torch.reshape(sm, (sm.shape[0], 1, 384))
    #              and: torch.reshape(smt, (smt.shape[0], 1, 384))
    # (sm = source drug, smt = target drug in domain adaptation)
    n_replaced = 0
    for var in ("sm", "smt"):
        pattern = rf'torch\.reshape\({var}\s*,\s*\({var}\.shape\[0\]\s*,\s*1\s*,\s*384\)\)'
        replacement = f"torch.reshape({var}, ({var}.shape[0], 1, {drug_emb_dim}))"
        new_source, count = re.subn(pattern, replacement, source)
        if count:
            source = new_source
            n_replaced += count

    if n_replaced == 0:
        # Fallback: older trainer.py uses positional reshape without keyword
        source, n_replaced = re.subn(
            r'(?<=, 1, )384(?=\))',
            str(drug_emb_dim),
            source,
        )
        if n_replaced == 0:
            warnings.warn(
                "BKN: could not find '384' reshape in trainer.py. "
                f"The Trainer may still expect 384-d drug embeddings (have {drug_emb_dim}-d). "
                "Check trainer.py manually.",
                stacklevel=2,
            )

    # Build a fresh module from the patched source
    mod = importlib.util.module_from_spec(
        importlib.util.spec_from_file_location("trainer_bkn", str(trainer_path))
    )
    # Inject the inductive_mode directory into the module's namespace so that
    # its own relative imports (from models import ...) resolve correctly.
    mod.__file__ = str(trainer_path)
    mod.__package__ = ""
    # sys.path must include inductive_mode for trainer's own imports
    if str(inductive_dir) not in sys.path:
        sys.path.insert(0, str(inductive_dir))
    exec(compile(source, str(trainer_path), "exec"), mod.__dict__)  # noqa: S102

    return mod.__dict__["Trainer"]


def setup_bkn_imports() -> dict:
    """Import GraphBAN modules with BKN patches applied."""
    if not GRAPHBAN_INDUCTIVE.exists():
        raise RuntimeError(
            f"GraphBAN inductive src not found at {GRAPHBAN_INDUCTIVE}\n"
            "Run: bash GraphBAN/setup_env.sh"
        )

    if str(GRAPHBAN_INDUCTIVE) not in sys.path:
        sys.path.insert(0, str(GRAPHBAN_INDUCTIVE))

    try:
        from configs import get_cfg_defaults
        from dataloader import DTIDataset, DTIDataset2, MultiDataLoader
        from models import GraphBAN, binary_cross_entropy, cross_entropy_logits
        from utils import graph_collate_func, graph_collate_func2, mkdir, set_seed
        from domain_adaptator import Discriminator

        # Load trainer with 768-d patch (does NOT modify trainer.py on disk)
        Trainer = _load_trainer_patched(GRAPHBAN_INDUCTIVE, DRUG_EMB_DIM)

        return {
            "get_cfg_defaults": get_cfg_defaults,
            "DTIDataset": DTIDataset,
            "DTIDataset2": DTIDataset2,
            "MultiDataLoader": MultiDataLoader,
            "GraphBAN": GraphBAN,
            "binary_cross_entropy": binary_cross_entropy,
            "cross_entropy_logits": cross_entropy_logits,
            "Trainer": Trainer,
            "graph_collate_func": graph_collate_func,
            "graph_collate_func2": graph_collate_func2,
            "mkdir": mkdir,
            "set_seed": set_seed,
            "Discriminator": Discriminator,
        }
    except ImportError as e:
        print(f"ERROR: GraphBAN import failed: {e}")
        print("Run: bash GraphBAN/setup_env.sh")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Feature extraction — ESM-2 + MoLFormer with CLS-guided attention pooling
# ---------------------------------------------------------------------------

def _cls_guided_attention_pool(
    tokens: torch.Tensor,
    cls: torch.Tensor,
    dim: int,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """CLS-guided attention pooling.

    Uses the CLS token representation as a query that attends over all token
    representations (keys/values), producing a single context-aware vector.
    This captures the global context (CLS) conditioned on local information.

    Args:
        tokens: [L, D] — all token representations (including CLS at index 0).
        cls:    [D]    — CLS token representation (query).
        dim:    D      — embedding dimension (for scaling).
        mask:   [L]    — boolean or int mask (1=valid, 0=padding). If None,
                         all tokens are used.

    Returns:
        [D] — pooled representation.
    """
    # Attention scores: dot(CLS, token) / sqrt(D) → [L]
    scores = torch.matmul(tokens, cls) / (dim ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    weights = torch.softmax(scores, dim=0)          # [L]
    return (weights.unsqueeze(1) * tokens).sum(0)   # [D]


def extract_esm2_features(df: pd.DataFrame, device: torch.device) -> pd.DataFrame:
    """Extract ESM-2 650M protein embeddings (1280-d, CLS-guided attention pooling).

    Replaces GraphBAN's ESM-1b mean-pool with:
      - Model: esm2_t33_650M_UR50D (same 1280-d hidden dim as ESM-1b)
      - Pooling: CLS-guided attention pooling over residue representations
        (CLS token from layer 33 used as query; residues 1..L as keys/values)
    """
    import esm

    print("\n  Loading ESM-2 650M (esm2_t33_650M_UR50D)...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model = model.eval().to(device)

    df = df.copy()
    # ESM-2 max sequence length is 1022 (same as ESM-1b)
    df["Protein"] = df["Protein"].apply(lambda x: x[:1022] if len(x) > 1022 else x)

    pro_list = df["Protein"].unique()
    print(f"  Extracting ESM-2 features for {len(pro_list)} unique proteins "
          f"(CLS-guided attention pooling)...")

    dictionary = {}
    data_tmp = [(f"protein{i}", p) for i, p in enumerate(pro_list)]

    batch_size = 5
    for i in tqdm(range(0, len(data_tmp), batch_size), desc="  ESM-2"):
        batch = data_tmp[i : i + batch_size]
        if not batch:
            continue
        _, _, batch_tokens = batch_converter(batch)
        batch_tokens = batch_tokens.to(device)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
        token_reps = results["representations"][33]  # [B, L+2, 1280]

        for j, (_, seq) in enumerate(batch):
            # CLS token is at position 0; residues at 1..len(seq)
            residues = token_reps[j, 1 : len(seq) + 1]  # [L, 1280]
            cls = token_reps[j, 0]                       # [1280]
            emb = _cls_guided_attention_pool(
                residues, cls, dim=1280,
            ).cpu().numpy()
            dictionary[seq] = emb

    esm_df = pd.DataFrame(list(dictionary.items()), columns=["Protein", "esm"])
    df = pd.merge(df, esm_df, on="Protein", how="left")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("  ESM-2 feature extraction complete.")
    return df


def extract_molformer_features(df: pd.DataFrame, device: torch.device) -> pd.DataFrame:
    """Extract MoLFormer-XL drug embeddings (768-d, CLS-guided attention pooling).

    Replaces GraphBAN's ChemBERTa CLS-token with:
      - Model: ibm/MoLFormer-XL-both-10pct (768-d)
      - Pooling: CLS-guided attention pooling over all (non-padding) token reps
        (CLS token at position 0 used as query; molecule tokens as keys/values)
    """
    from transformers import AutoModel, AutoTokenizer

    model_name = "ibm/MoLFormer-XL-both-10pct"
    print(f"\n  Loading MoLFormer-XL ({model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model = model.to(device).eval()

    df_unique = df.drop_duplicates(subset="SMILES").copy()
    print(f"  Extracting MoLFormer features for {len(df_unique)} unique SMILES "
          f"(CLS-guided attention pooling)...")

    emblist = []
    for _, row in tqdm(df_unique.iterrows(), total=len(df_unique), desc="  MoLFormer"):
        encodings = tokenizer(
            row["SMILES"],
            return_tensors="pt",
            padding="max_length",
            max_length=202,      # MoLFormer default max length
            truncation=True,
        )
        encodings = {k: v.to(device) for k, v in encodings.items()}
        with torch.no_grad():
            output = model(**encodings)
        token_reps = output.last_hidden_state[0]        # [L, 768]
        mask = encodings["attention_mask"][0]           # [L], 1=valid 0=pad
        cls = token_reps[0]                             # [768] — CLS token

        emb = _cls_guided_attention_pool(
            token_reps, cls, dim=768, mask=mask,
        ).cpu().numpy().astype(np.float64)
        emblist.append(emb)

    df_unique["fcfp"] = emblist
    df = pd.merge(df, df_unique[["SMILES", "fcfp"]], on="SMILES", how="left")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("  MoLFormer feature extraction complete.")
    return df


def extract_features_cached(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    device: torch.device,
    cache_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Extract ESM-2 + MoLFormer features with caching (once for all seeds)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "features_extracted_bkn.pkl"

    if cache_file.exists():
        print("\n  Loading cached BKN features...")
        import pickle
        with open(cache_file, "rb") as f:
            train_df, val_df, test_df = pickle.load(f)
        print("  Cache loaded.")
        return train_df, val_df, test_df

    print("\n  Extracting features (ESM-2 + MoLFormer + AttentionPool)...")
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    all_df = extract_esm2_features(all_df, device)
    all_df = extract_molformer_features(all_df, device)

    # Split back preserving original indices
    n_train = len(train_df)
    n_val = len(val_df)
    train_df = all_df.iloc[:n_train].reset_index(drop=True)
    val_df = all_df.iloc[n_train : n_train + n_val].reset_index(drop=True)
    test_df = all_df.iloc[n_train + n_val :].reset_index(drop=True)

    import pickle
    with open(cache_file, "wb") as f:
        pickle.dump((train_df, val_df, test_df), f, protocol=4)
    print(f"  Features cached to {cache_file}")

    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Teacher GAE embeddings (identical to GraphBAN baseline — no change needed)
# ---------------------------------------------------------------------------

def generate_teacher_embeddings(
    train_csv: Path,
    seed: int,
    output_parquet: Path,
    epochs: int = DEFAULT_TEACHER_EPOCHS,
) -> None:
    """Generate teacher GAE embeddings for a single seed.

    Delegates to GraphBAN's generate_teacher_embeddings.py script.
    Identical to GraphBAN/run_baseline.py — teacher uses only the graph
    structure (not ESM/MoLFormer features), so it is unchanged.
    """
    script = GRAPHBAN_INDUCTIVE / "generate_teacher_embeddings.py"
    if not script.exists():
        raise FileNotFoundError(
            f"generate_teacher_embeddings.py not found at {script}"
        )

    cmd = [
        sys.executable, str(script),
        "--train_csv", str(train_csv),
        "--seed", str(seed),
        "--output", str(output_parquet),
        "--epochs", str(epochs),
    ]
    print(f"\n  Generating teacher embeddings (seed={seed}, epochs={epochs})...")
    result = subprocess.run(cmd, capture_output=False, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Teacher embedding generation failed (exit {result.returncode})")
    print(f"    Teacher embeddings saved: {output_parquet}")


# ---------------------------------------------------------------------------
# Fair evaluation protocol (same as GraphBAN/run_baseline.py)
# ---------------------------------------------------------------------------

def _collect_predictions(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    n_class: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect (y_true, y_prob) from a data loader using the given model.

    Uses DRUG_EMB_DIM=768 (MoLFormer) instead of 384 (ChemBERTa).
    """
    from models import binary_cross_entropy, cross_entropy_logits

    y_true, y_prob = [], []
    with torch.no_grad():
        model.eval()
        for batch in data_loader:
            v_d, sm, v_p, esm_feat, labels = batch
            sm = torch.tensor(sm, dtype=torch.float32)
            sm = torch.reshape(sm, (sm.shape[0], 1, DRUG_EMB_DIM))   # 768-d MoLFormer
            esm_feat = torch.tensor(esm_feat, dtype=torch.float32)
            esm_feat = torch.reshape(esm_feat, (sm.shape[0], 1, 1280))
            v_d = v_d.to(device)
            sm = sm.to(device)
            v_p = v_p.to(device)
            esm_feat = esm_feat.to(device)
            labels = labels.float().to(device)
            _, _, _, score = model(v_d, sm, v_p, esm_feat, device)

            if n_class == 1:
                n, _ = binary_cross_entropy(score, labels)
            else:
                n, _ = cross_entropy_logits(score, labels)

            y_prob.extend(n.to("cpu").tolist())
            y_true.extend(labels.to("cpu").tolist())

    return np.array(y_true), np.array(y_prob)


def optimize_threshold_on_validation(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "mcc",
) -> tuple[float, float]:
    """Optimize decision threshold on validation predictions (no test leakage).

    Same vectorised sweep as GraphBAN/run_baseline.py.
    """
    if len(y_true) == 0:
        return 0.5, 0.0

    order = np.argsort(y_prob, kind="mergesort")[::-1]
    probs_sorted = y_prob[order]
    labels_sorted = y_true[order]

    total_pos = float((labels_sorted == 1).sum())
    total_neg = float((labels_sorted == 0).sum())

    tp_cum = np.cumsum(labels_sorted == 1, dtype=np.float64)
    fp_cum = np.cumsum(labels_sorted == 0, dtype=np.float64)

    last_indices = np.r_[np.where(np.diff(probs_sorted) != 0)[0], len(probs_sorted) - 1]
    tp = tp_cum[last_indices]
    fp = fp_cum[last_indices]
    fn = total_pos - tp
    tn = total_neg - fp
    thresholds = probs_sorted[last_indices]

    sentinel = np.nextafter(float(np.max(probs_sorted)), np.inf)
    tp = np.concatenate(([0.0], tp))
    fp = np.concatenate(([0.0], fp))
    fn = np.concatenate(([total_pos], fn))
    tn = np.concatenate(([total_neg], tn))
    thresholds = np.concatenate(([sentinel], thresholds))

    if metric == "mcc":
        denom_sq = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        valid = denom_sq > 0
        scores = np.zeros_like(denom_sq)
        scores[valid] = (tp[valid] * tn[valid] - fp[valid] * fn[valid]) / np.sqrt(denom_sq[valid])
    elif metric == "f1":
        denom = (2 * tp) + fp + fn
        scores = np.where(denom > 0, (2 * tp) / denom, 0.0)
    else:
        raise ValueError(f"Unsupported metric: {metric!r}")

    best_score = float(np.nanmax(scores))
    tie_idx = np.where(np.isclose(scores, best_score, rtol=1e-9, atol=1e-12))[0]
    best_idx = int(tie_idx[np.argmin(np.abs(thresholds[tie_idx] - 0.5))])

    return float(thresholds[best_idx]), best_score


def compute_metrics_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict:
    """Compute all metrics at a given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "threshold": float(threshold),
    }


# ---------------------------------------------------------------------------
# Per-epoch MCC injection (same monkey-patch as GraphBAN/run_baseline.py)
# ---------------------------------------------------------------------------

def _patch_trainer_for_mcc_logging(
    trainer_obj,
    val_gen: torch.utils.data.DataLoader,
    device: torch.device,
    n_class: int,
) -> None:
    """Inject per-epoch Val MCC line after GraphBAN's own AUROC/AUPRC line."""
    original_test = trainer_obj.test

    def _test_with_mcc(*args, **kwargs):
        result = original_test(*args, **kwargs)
        try:
            y_true, y_prob = _collect_predictions(
                trainer_obj.model, val_gen, device, n_class,
            )
            _, val_mcc = optimize_threshold_on_validation(y_true, y_prob, metric="mcc")
            print(f"  → Val MCC={val_mcc:.4f}")
        except Exception:
            pass
        return result

    trainer_obj.test = _test_with_mcc


# ---------------------------------------------------------------------------
# Single-seed training
# ---------------------------------------------------------------------------

def train_single_seed(
    cfg,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    teacher_parquet: Path,
    seed: int,
    output_dir: Path,
    device: torch.device,
    modules: dict,
) -> dict:
    """Train BKN for a single seed and return metrics.

    Protocol identical to GraphBAN/run_baseline.py:
      1. Load teacher embeddings and attach to training data
      2. Create datasets/dataloaders with actual val split
      3. Train (model selected by val AUROC)
      4. Collect val predictions → MCC-optimal threshold (no test leakage)
      5. Collect train + test predictions → apply val threshold
      6. Return {train, val, test} metric dicts
    """
    print(f"\n{'─'*50}")
    print(f"  Seed: {seed}")
    print(f"{'─'*50}")

    modules["set_seed"](seed)

    seed_output = output_dir / f"seed_{seed}"
    modules["mkdir"](str(seed_output))

    cfg.defrost()
    cfg.SOLVER.SEED = seed
    cfg.RESULT.OUTPUT_DIR = str(seed_output)
    cfg.freeze()

    train_emb = pd.read_parquet(teacher_parquet)
    train_emb["Array"] = train_emb.apply(lambda row: np.array(row), axis=1)
    train_emb.drop(train_emb.columns.difference(["Array"]), axis=1, inplace=True)

    train_df_seed = train_df.copy()
    train_df_seed["teacher_emb"] = train_emb["Array"].values

    print(f"  Data: train={len(train_df_seed)}, val={len(val_df)}, test={len(test_df)}")

    train_dataset = modules["DTIDataset2"](train_df_seed.index.values, train_df_seed)
    val_dataset = modules["DTIDataset"](val_df.index.values, val_df)
    test_dataset = modules["DTIDataset"](test_df.index.values, test_df)

    params_train = {
        "batch_size": cfg.SOLVER.BATCH_SIZE,
        "shuffle": True,
        "num_workers": cfg.SOLVER.NUM_WORKERS,
        "drop_last": True,
        "collate_fn": modules["graph_collate_func2"],
    }
    params_eval = {
        "batch_size": cfg.SOLVER.BATCH_SIZE,
        "shuffle": False,
        "num_workers": cfg.SOLVER.NUM_WORKERS,
        "drop_last": False,
        "collate_fn": modules["graph_collate_func"],
    }

    training_generator = torch.utils.data.DataLoader(train_dataset, **params_train)
    val_generator = torch.utils.data.DataLoader(val_dataset, **params_eval)
    test_generator = torch.utils.data.DataLoader(test_dataset, **params_eval)

    model = modules["GraphBAN"](**cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    if cfg.DA.USE:
        source_generator = torch.utils.data.DataLoader(train_dataset, **params_train)
        target_generator = torch.utils.data.DataLoader(
            val_dataset, **{**params_eval, "shuffle": True, "drop_last": True}
        )
        n_batches = max(len(source_generator), len(target_generator))
        multi_generator = modules["MultiDataLoader"](
            dataloaders=[source_generator, target_generator],
            n_batches=n_batches,
        )
        domain_dmm = modules["Discriminator"](
            input_size=cfg.DA.RANDOM_DIM,
            n_class=cfg.DECODER.BINARY,
        ).to(device)
        opt_da = torch.optim.Adam(domain_dmm.parameters(), lr=cfg.SOLVER.DA_LR)
    else:
        multi_generator = None
        domain_dmm = None
        opt_da = None

    n_class = cfg.DECODER.BINARY
    train_loader = multi_generator if cfg.DA.USE else training_generator
    trainer = modules["Trainer"](
        model, opt, device,
        train_loader,
        val_generator,
        test_generator,
        opt_da=opt_da,
        discriminator=domain_dmm,
        experiment=None,
        **cfg,
    )

    _patch_trainer_for_mcc_logging(trainer, val_generator, device, n_class)

    t0 = time.time()
    result = trainer.train()
    elapsed = time.time() - t0

    if isinstance(result, tuple):
        result_metrics = result[0] if isinstance(result[0], dict) else {}
    elif isinstance(result, dict):
        result_metrics = result
    else:
        result_metrics = {}

    # --- Fair evaluation protocol ---
    print("  Collecting validation predictions for threshold optimization...")
    val_y_true, val_y_prob = _collect_predictions(
        trainer.best_model, val_generator, device, n_class,
    )

    val_threshold, val_best_mcc = optimize_threshold_on_validation(
        val_y_true, val_y_prob, metric="mcc",
    )
    print(f"  Val-optimized threshold={val_threshold:.4f} (val MCC={val_best_mcc:.4f})")

    print("  Collecting train predictions...")
    train_eval_gen = torch.utils.data.DataLoader(train_dataset, **params_eval)
    train_y_true, train_y_prob = _collect_predictions(
        trainer.best_model, train_eval_gen, device, n_class,
    )

    print("  Collecting test predictions...")
    test_y_true, test_y_prob = _collect_predictions(
        trainer.best_model, test_generator, device, n_class,
    )

    train_metrics = compute_metrics_at_threshold(train_y_true, train_y_prob, val_threshold)
    val_metrics = compute_metrics_at_threshold(val_y_true, val_y_prob, val_threshold)
    test_metrics = compute_metrics_at_threshold(test_y_true, test_y_prob, val_threshold)

    native_threshold = result_metrics.get("thred_optim", 0.5)
    native_test_metrics = compute_metrics_at_threshold(
        test_y_true, test_y_prob, native_threshold,
    )

    metrics = {
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "val_threshold": val_threshold,
        "threshold_source": "validation_mcc",
        "training_time_s": round(elapsed, 1),
        "best_epoch": result_metrics.get("best_epoch", -1),
        "model_selection": "val_auroc",
        "graphban_native": {
            "threshold": native_threshold,
            "threshold_source": "test_f1_optimal (GraphBAN original — NOT used for comparison)",
            "mcc": native_test_metrics["mcc"],
            "auroc": result_metrics.get("auroc", native_test_metrics["auroc"]),
            "auprc": result_metrics.get("auprc", None),
        },
    }

    print(f"  Results (seed={seed}, threshold={val_threshold:.4f}):")
    print(f"    {'Split':<6}  {'MCC':>7}  {'AUROC':>7}  {'F1':>7}  {'Acc':>7}")
    print(f"    {'─'*38}")
    for split_name, split_m in [("Train", train_metrics), ("Val", val_metrics), ("Test", test_metrics)]:
        print(f"    {split_name:<6}  {split_m['mcc']:>7.4f}  {split_m['auroc']:>7.4f}  "
              f"{split_m['f1']:>7.4f}  {split_m['accuracy']:>7.4f}")
    print(f"    Time: {elapsed:.1f}s  Best epoch: {metrics['best_epoch']}")

    np.savez(
        seed_output / "raw_predictions.npz",
        train_y_true=train_y_true,
        train_y_prob=train_y_prob,
        val_y_true=val_y_true,
        val_y_prob=val_y_prob,
        test_y_true=test_y_true,
        test_y_prob=test_y_prob,
    )

    return metrics


# ---------------------------------------------------------------------------
# Aggregation and reporting
# ---------------------------------------------------------------------------

def aggregate_results(all_metrics: list[dict]) -> dict:
    """Compute mean +/- std across seeds for train/val/test splits."""
    metric_names = ["accuracy", "f1", "precision", "recall", "mcc", "auroc"]
    agg: dict = {}

    for split in ["train", "val", "test"]:
        agg[split] = {}
        for m in metric_names:
            values = [r[split][m] for r in all_metrics]
            agg[split][m] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "values": values,
            }

    agg["val_threshold"] = {
        "mean": float(np.mean([r["val_threshold"] for r in all_metrics])),
        "std": float(np.std([r["val_threshold"] for r in all_metrics])),
        "values": [r["val_threshold"] for r in all_metrics],
    }
    agg["training_time_s"] = {
        "mean": float(np.mean([r["training_time_s"] for r in all_metrics])),
    }
    native_mcc_vals = [
        r["graphban_native"]["mcc"] for r in all_metrics if "graphban_native" in r
    ]
    if native_mcc_vals:
        agg["graphban_native_mcc"] = {
            "mean": float(np.mean(native_mcc_vals)),
            "std": float(np.std(native_mcc_vals)),
            "values": native_mcc_vals,
        }
    return agg


def print_summary_table(agg: dict, dataset: str, seeds: list[int]) -> None:
    """Print a formatted summary table showing train/val/test metrics."""
    print(f"\n{'='*72}")
    print(f"  BKN (ESM-2 + MoLFormer + AttnPool) — {dataset} ({len(seeds)} seeds)")
    print(f"{'='*72}")
    print(f"  {'Metric':<12} {'Train Mean':>11} {'± Std':>7}  {'Val Mean':>9} {'± Std':>7}  {'Test Mean':>10} {'± Std':>7}")
    print(f"  {'─'*68}")
    for m in ["mcc", "auroc", "f1", "accuracy", "precision", "recall"]:
        tr = agg["train"][m]
        vl = agg["val"][m]
        te = agg["test"][m]
        print(f"  {m.upper():<12} {tr['mean']:>11.4f} {tr['std']:>7.4f}  "
              f"{vl['mean']:>9.4f} {vl['std']:>7.4f}  "
              f"{te['mean']:>10.4f} {te['std']:>7.4f}")
    print(f"  {'─'*68}")
    if "graphban_native_mcc" in agg:
        native = agg["graphban_native_mcc"]
        print(f"  {'MCC (native)':<12} {'N/A':>11} {'':>7}  {'N/A':>9} {'':>7}  "
              f"{native['mean']:>10.4f} {native['std']:>7.4f}  (test-set threshold, not used)")
    print(f"  Avg training time: {agg['training_time_s']['mean']:.1f}s per seed")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="BKN: GraphBAN with ESM-2 + MoLFormer + Attention Pooling"
    )
    parser.add_argument(
        "--dataset",
        choices=["non_human", "human", "all"],
        default="non_human",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=CANONICAL_SEEDS,
    )
    parser.add_argument("--max-epoch", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--teacher-epochs",
        type=int,
        default=DEFAULT_TEACHER_EPOCHS,
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-da", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    # Dataset path (shared with GraphBAN)
    dataset_path = GRAPHBAN_DIR / "datasets" / "kinase" / args.dataset / "scaffold"
    if not dataset_path.exists() or not (dataset_path / "train.csv").exists():
        print(f"Dataset not found at {dataset_path}")
        print(f"Auto-preparing data for '{args.dataset}'...")
        sys.path.insert(0, str(GRAPHBAN_DIR))
        from prepare_data import prepare_dataset
        prepare_dataset(args.dataset, GRAPHBAN_DIR)
        if not (dataset_path / "train.csv").exists():
            print("ERROR: Data preparation failed.")
            sys.exit(1)
        print("Data prepared successfully.\n")

    modules = setup_bkn_imports()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = modules["get_cfg_defaults"]()
    config_path = SCRIPT_DIR / "configs" / "kinase_bkn.yaml"
    if config_path.exists():
        cfg.merge_from_file(str(config_path))

    if args.max_epoch is not None:
        cfg.defrost(); cfg.SOLVER.MAX_EPOCH = args.max_epoch; cfg.freeze()
    if args.batch_size is not None:
        cfg.defrost(); cfg.SOLVER.BATCH_SIZE = args.batch_size; cfg.freeze()
    if args.num_workers > 0:
        cfg.defrost(); cfg.SOLVER.NUM_WORKERS = args.num_workers; cfg.freeze()
    if args.no_da:
        cfg.defrost(); cfg.DA.USE = False; cfg.DA.TASK = False; cfg.freeze()

    output_base = args.output_dir or (SCRIPT_DIR / "results" / args.dataset)
    output_base = Path(output_base).resolve()
    output_base.mkdir(parents=True, exist_ok=True)

    print(f"Model:    BKN (ESM-2 650M + MoLFormer-XL 768-d + AttentionPool)")
    print(f"Dataset:  {args.dataset} (scaffold split)")
    print(f"Seeds:    {args.seeds}")
    print(f"Epochs:   {cfg.SOLVER.MAX_EPOCH}  |  Batch: {cfg.SOLVER.BATCH_SIZE}")
    print(f"DA:       {'enabled' if cfg.DA.USE else 'disabled'}")
    print(f"Output:   {output_base}")
    print(f"Device:   {device}")

    train_csv = dataset_path / "train.csv"
    val_csv = dataset_path / "val.csv"
    test_csv = dataset_path / "test.csv"

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)
    print(f"\nData: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    cache_dir = output_base / "feature_cache"
    train_df, val_df, test_df = extract_features_cached(
        train_df, val_df, test_df, device, cache_dir,
    )

    all_metrics = []
    for seed in args.seeds:
        teacher_dir = output_base / f"seed_{seed}" / "teacher"
        teacher_dir.mkdir(parents=True, exist_ok=True)
        teacher_parquet = teacher_dir / "teacher_embeddings.parquet"

        if not teacher_parquet.exists():
            generate_teacher_embeddings(
                train_csv, seed, teacher_parquet, epochs=args.teacher_epochs,
            )
        else:
            print(f"\n    Using cached teacher embeddings: {teacher_parquet}")

        metrics = train_single_seed(
            cfg=cfg.clone(),
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            teacher_parquet=teacher_parquet,
            seed=seed,
            output_dir=output_base,
            device=device,
            modules=modules,
        )
        all_metrics.append({"seed": seed, **metrics})

    agg = aggregate_results(all_metrics)
    print_summary_table(agg, args.dataset, args.seeds)

    results_file = output_base / "bkn_results.json"
    output_data = {
        "model": "BKN",
        "variant": "ESM-2-650M + MoLFormer-XL-768d + CLS-Attention-Pooling",
        "dataset": args.dataset,
        "split": "scaffold",
        "seeds": args.seeds,
        "methodology": {
            "protein_encoder": "ESM-2 esm2_t33_650M_UR50D (1280-d, frozen)",
            "drug_encoder": "MoLFormer-XL ibm/MoLFormer-XL-both-10pct (768-d, frozen)",
            "pooling": "CLS-guided attention pooling for both encoders",
            "model_selection": "validation AUROC",
            "threshold_optimization": "validation MCC-optimal (no test leakage)",
            "teacher": f"GAE on bipartite CPI graph (256-d, {args.teacher_epochs} epochs)",
            "trainer_patch": (
                f"GraphBAN trainer.py patched in-memory (exec) to reshape drug "
                f"embedding as (batch, 1, {DRUG_EMB_DIM}) instead of (batch, 1, 384). "
                "Original file on disk is NOT modified."
            ),
        },
        "config": {
            "max_epoch": cfg.SOLVER.MAX_EPOCH,
            "batch_size": cfg.SOLVER.BATCH_SIZE,
            "lr": cfg.SOLVER.LR,
            "domain_adaptation": cfg.DA.USE,
            "drug_emb_dim": DRUG_EMB_DIM,
        },
        "aggregate": agg,
        "per_seed": all_metrics,
    }

    with open(results_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to {results_file}")
    print("Done!")


if __name__ == "__main__":
    main()
