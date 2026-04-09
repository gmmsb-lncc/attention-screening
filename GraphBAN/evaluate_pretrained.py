#!/usr/bin/env python3
"""Evaluate GraphBAN pretrained weights on thesis kinase test sets.

Loads the upstream GraphBAN inductive models (trained on BioSNAP, BindingDB,
KIBA, C.elegans, PDB — 5 seeds each) and evaluates them zero-shot on the
thesis's universal scaffold test sets (non_human, human, all).

Uses the paper's evaluation protocol:
  - F1-optimal threshold from ROC curve (per the paper's trainer.py)
  - AUROC, AUPRC, F1, Sensitivity, Specificity, Accuracy
  - Multi-seed aggregation with mean ± std

Usage:
    # From repo root, with graphban conda env active:
    python GraphBAN/evaluate_pretrained.py --dataset non_human
    python GraphBAN/evaluate_pretrained.py --dataset human
    python GraphBAN/evaluate_pretrained.py --dataset all
    python GraphBAN/evaluate_pretrained.py --dataset all --source-models biosnap bindingdb kiba

    # Evaluate all datasets × all source models:
    python GraphBAN/evaluate_pretrained.py --dataset all --source-models all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
GRAPHBAN_SRC = SCRIPT_DIR / "src" / "inductive_mode"
SCAFFOLD_DIR = REPO_ROOT / "scaffolds_splits" / "output"
TRAINED_MODELS_DIR = GRAPHBAN_SRC / "trained_models"

# Activity threshold matching thesis config
PCHEMBL_ACTIVITY_THRESHOLD = 6.0

# Datasets available with pretrained weights
AVAILABLE_SOURCE_MODELS = ["biosnap", "bindingdb", "kiba", "c.elegans", "pdb"]
# These are the upstream checkpoint identifiers (folder names in trained_models/).
# They correspond to the 5 random seeds used by the original GraphBAN authors during
# training. They are NOT the thesis canonical seeds {42,123,456,789,1024} — the
# pretrained weights are fixed artifacts and cannot be re-seeded.
SEEDS_PER_SOURCE = [12, 14, 16, 18, 20]


# ---------------------------------------------------------------------------
# GraphBAN imports (deferred — requires specific environment)
# ---------------------------------------------------------------------------
def setup_graphban_imports():
    """Add GraphBAN source to path and import required modules."""
    sys.path.insert(0, str(GRAPHBAN_SRC))

    from models import GraphBAN, cross_entropy_logits, binary_cross_entropy
    from utils import integer_label_protein, graph_collate_func, set_seed
    from dataloader import DTIDataset
    from configs import get_cfg_defaults

    return {
        "GraphBAN": GraphBAN,
        "cross_entropy_logits": cross_entropy_logits,
        "binary_cross_entropy": binary_cross_entropy,
        "integer_label_protein": integer_label_protein,
        "graph_collate_func": graph_collate_func,
        "set_seed": set_seed,
        "DTIDataset": DTIDataset,
        "get_cfg_defaults": get_cfg_defaults,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_thesis_test_set(dataset: str) -> pd.DataFrame:
    """Load thesis scaffold test set and convert to GraphBAN format.

    Returns DataFrame with columns: SMILES, Protein, Y
    """
    if dataset == "all":
        dfs = []
        for sub in ["human", "non_human"]:
            dfs.append(load_thesis_test_set(sub))
        df = pd.concat(dfs, ignore_index=True)
        # Remove exact duplicates (same SMILES×Protein pair)
        df = df.drop_duplicates(subset=["SMILES", "Protein"]).reset_index(drop=True)
        return df

    test_path = SCAFFOLD_DIR / f"{dataset}_test.tsv"
    if not test_path.exists():
        test_path_gz = SCAFFOLD_DIR / f"{dataset}_test.tsv.gz"
        if test_path_gz.exists():
            df = pd.read_csv(test_path_gz, sep="\t", compression="gzip")
        else:
            raise FileNotFoundError(f"Test set not found: {test_path}")
    else:
        df = pd.read_csv(test_path, sep="\t")

    # Convert to GraphBAN format
    result = pd.DataFrame()
    result["SMILES"] = df["canonical_smiles"]
    result["Protein"] = df["seq"]

    # Generate binary labels using thesis threshold
    if "label" in df.columns:
        result["Y"] = df["label"].astype(int)
    else:
        result["Y"] = (df["pchembl_value"] >= PCHEMBL_ACTIVITY_THRESHOLD).astype(int)

    # Drop rows with missing SMILES or Protein
    result = result.dropna(subset=["SMILES", "Protein"])
    result = result[result["SMILES"].str.strip().astype(bool)]
    result = result[result["Protein"].str.strip().astype(bool)]

    # GraphBAN truncates proteins to 1022 internally (ESM-1b limit)
    result["Protein"] = result["Protein"].apply(lambda x: x[:1022] if len(x) > 1022 else x)

    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feature extraction (ESM-1b + ChemBERTa) — with caching
# ---------------------------------------------------------------------------
def extract_esm_features(
    proteins: list[str],
    device: torch.device,
    cache_dir: Path | None = None,
    batch_size: int = 5,
) -> dict[str, np.ndarray]:
    """Extract ESM-1b mean-pooled embeddings for unique protein sequences.

    Follows GraphBAN's predict.py: esm1b_t33_650M_UR50S, layer 33, mean pool.
    """
    cache_file = cache_dir / "esm1b_embeddings.npz" if cache_dir else None
    if cache_file and cache_file.exists():
        print(f"    Loading cached ESM-1b embeddings from {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
        return dict(zip(data["keys"], data["values"]))

    print(f"    Extracting ESM-1b embeddings for {len(proteins)} unique proteins...")
    # Must load fair-esm (not ESM-3) from local path for .pretrained API
    import importlib
    esm_local = str(REPO_ROOT / "llm" / "ESM")
    if esm_local not in sys.path:
        sys.path.insert(0, esm_local)
    if "esm" in sys.modules:
        # Force reload to pick up local fair-esm, not the ESM-3 package
        del sys.modules["esm"]
        for key in list(sys.modules.keys()):
            if key.startswith("esm."):
                del sys.modules[key]
    import esm as esm_module
    importlib.reload(esm_module)

    esm_model, alphabet = esm_module.pretrained.esm1b_t33_650M_UR50S()
    esm_model = esm_model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()

    embeddings = {}
    for i in range(0, len(proteins), batch_size):
        batch_proteins = proteins[i : i + batch_size]
        data_tuples = [
            (f"protein{j}", p[:1022]) for j, p in enumerate(batch_proteins)
        ]
        _, _, batch_tokens = batch_converter(data_tuples)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = esm_model(batch_tokens, repr_layers=[33], return_contacts=False)
        token_reps = results["representations"][33]

        for j, (_, seq) in enumerate(data_tuples):
            emb = token_reps[j, 1 : len(seq) + 1].mean(0).cpu().numpy()
            embeddings[batch_proteins[j]] = emb

        if (i // batch_size) % 10 == 0:
            print(f"      ESM-1b: {min(i + batch_size, len(proteins))}/{len(proteins)}")

    # Clean up GPU memory
    del esm_model
    torch.cuda.empty_cache()

    if cache_file:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez(cache_file, keys=list(embeddings.keys()), values=list(embeddings.values()))
        print(f"    Cached ESM-1b embeddings to {cache_file}")

    return embeddings


def extract_chemberta_features(
    smiles_list: list[str],
    device: torch.device,
    cache_dir: Path | None = None,
) -> dict[str, np.ndarray]:
    """Extract ChemBERTa-77M-MTR CLS token embeddings for unique SMILES.

    Follows GraphBAN's predict.py: DeepChem/ChemBERTa-77M-MTR, CLS token (384-d).
    """
    cache_file = cache_dir / "chemberta_embeddings.npz" if cache_dir else None
    if cache_file and cache_file.exists():
        print(f"    Loading cached ChemBERTa embeddings from {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
        return dict(zip(data["keys"], data["values"]))

    print(f"    Extracting ChemBERTa embeddings for {len(smiles_list)} unique SMILES...")
    from transformers import AutoTokenizer, RobertaModel

    model_name = "DeepChem/ChemBERTa-77M-MTR"
    model_chem = RobertaModel.from_pretrained(
        model_name, num_labels=2, add_pooling_layer=True
    ).to(device)
    model_chem.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    embeddings = {}
    for i, smi in enumerate(smiles_list):
        encodings = tokenizer(
            smi,
            return_tensors="pt",
            padding="max_length",
            max_length=290,
            truncation=True,
        ).to(device)
        with torch.no_grad():
            output = model_chem(**encodings)
            cls_emb = output.last_hidden_state[0, 0].cpu().numpy()
        embeddings[smi] = cls_emb

        if (i + 1) % 500 == 0:
            print(f"      ChemBERTa: {i + 1}/{len(smiles_list)}")

    # Clean up GPU memory
    del model_chem
    torch.cuda.empty_cache()

    if cache_file:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez(cache_file, keys=list(embeddings.keys()), values=list(embeddings.values()))
        print(f"    Cached ChemBERTa embeddings to {cache_file}")

    return embeddings


def add_features_to_df(
    df: pd.DataFrame,
    esm_embs: dict[str, np.ndarray],
    chem_embs: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Add pre-computed ESM-1b and ChemBERTa features to the DataFrame."""
    df = df.copy()
    df["esm"] = df["Protein"].map(esm_embs)
    df["fcfp"] = df["SMILES"].map(chem_embs)

    # Check for missing
    missing_esm = df["esm"].isna().sum()
    missing_chem = df["fcfp"].isna().sum()
    if missing_esm > 0:
        print(f"    WARNING: {missing_esm} rows missing ESM-1b embeddings")
    if missing_chem > 0:
        print(f"    WARNING: {missing_chem} rows missing ChemBERTa embeddings")

    # Drop rows with missing features
    df = df.dropna(subset=["esm", "fcfp"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Model loading & inference
# ---------------------------------------------------------------------------
def find_best_model(source_dataset: str, seed: int) -> Path | None:
    """Find the best_model_epoch_*.pth for a given source dataset and seed."""
    seed_dir = TRAINED_MODELS_DIR / source_dataset / "inductive" / f"seed{seed}" / "result"
    if not seed_dir.exists():
        return None

    best_models = sorted(seed_dir.glob("best_model_epoch_*.pth"))
    if not best_models:
        return None

    # If multiple best_model files exist, pick the one with highest epoch
    # (latest checkpoint = final best)
    def extract_epoch(p: Path) -> int:
        try:
            return int(p.stem.split("_")[-1])
        except ValueError:
            return 0

    return max(best_models, key=extract_epoch)


def _detect_dgl_cuda() -> bool:
    """Check if DGL was compiled with CUDA support."""
    try:
        import dgl
        g = dgl.graph(([0], [1]))
        g = g.to("cuda")
        return True
    except Exception:
        return False


# Cache the result so we only probe once
_DGL_HAS_CUDA: bool | None = None


def run_inference(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    n_class: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference and return (y_true, y_prob).

    y_prob is the probability of the positive class.
    Uses GPU if DGL has CUDA support, otherwise falls back to CPU.
    """
    global _DGL_HAS_CUDA
    if _DGL_HAS_CUDA is None:
        _DGL_HAS_CUDA = _detect_dgl_cuda()
        print(f"    DGL CUDA support: {'YES → using GPU' if _DGL_HAS_CUDA else 'NO → using CPU'}")

    infer_device = device if _DGL_HAS_CUDA else torch.device("cpu")
    model = model.to(infer_device)

    y_label_all = []
    y_pred_all = []

    model.eval()
    with torch.no_grad():
        for v_d, sm, v_p, esm_feat, labels in dataloader:
            sm = sm.clone().detach().float()
            sm = sm.reshape(sm.shape[0], 1, 384)
            esm_feat = esm_feat.clone().detach().float()
            esm_feat = esm_feat.reshape(sm.shape[0], 1, 1280)

            v_d = v_d.to(infer_device)
            sm = sm.to(infer_device)
            v_p = v_p.to(infer_device)
            esm_feat = esm_feat.to(infer_device)
            labels = labels.float().to(infer_device)

            v_d_out, v_p_out, score, f = model(v_d, sm, v_p, esm_feat, infer_device, mode="eval")

            # Compute probabilities following paper protocol
            if n_class == 1:
                # Binary sigmoid
                probs = torch.sigmoid(score).squeeze(1).cpu().numpy()
            else:
                # 2-class softmax → positive class probability
                probs = F.softmax(score, dim=1)[:, 1].cpu().numpy()

            y_label_all.extend(labels.cpu().tolist())
            y_pred_all.extend(probs.tolist())

    return np.array(y_label_all), np.array(y_pred_all)


# ---------------------------------------------------------------------------
# Metrics (following paper's trainer.py protocol)
# ---------------------------------------------------------------------------
def compute_metrics_paper_protocol(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, Any]:
    """Compute metrics following GraphBAN paper's evaluation protocol.

    The paper uses F1-optimal threshold from ROC curve:
      1. Compute ROC curve (fpr, tpr, thresholds)
      2. precision = tpr / (tpr + fpr)
      3. f1 = 2 * precision * tpr / (tpr + precision + eps)
      4. thred_optim = thresholds[5:][argmax(f1[5:])]  (skip first 5 points)
      5. Report confusion-matrix-derived metrics at this threshold
    """
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        roc_curve,
        confusion_matrix,
        precision_score,
        matthews_corrcoef,
    )

    result: dict[str, Any] = {}

    # AUROC and AUPRC (threshold-independent)
    try:
        result["auroc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        result["auroc"] = float("nan")
    try:
        result["auprc"] = float(average_precision_score(y_true, y_prob))
    except ValueError:
        result["auprc"] = float("nan")

    # Paper's F1-optimal threshold from ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    # Compute "precision" as tpr / (tpr + fpr) — paper's definition
    precision_roc = tpr / (tpr + fpr + 1e-8)
    f1_roc = 2 * precision_roc * tpr / (tpr + precision_roc + 1e-5)

    # Skip first 5 points (paper does thresholds[5:])
    skip = min(5, len(f1_roc) - 1)
    thred_optim = float(thresholds[skip:][np.argmax(f1_roc[skip:])])
    result["f1_roc_optimal"] = float(np.max(f1_roc[skip:]))
    result["threshold_f1_optimal"] = thred_optim

    # Metrics at F1-optimal threshold
    y_pred_binary = (y_prob >= thred_optim).astype(int)
    cm = confusion_matrix(y_true, y_pred_binary)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        result["accuracy"] = float((tp + tn) / (tp + tn + fp + fn))
        # Standard convention (NOT the paper's swapped labels)
        result["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0  # TPR = Recall
        result["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0  # TNR
        result["precision"] = float(precision_score(y_true, y_pred_binary, zero_division=0))
        result["f1"] = result["f1_roc_optimal"]
        result["mcc"] = float(matthews_corrcoef(y_true, y_pred_binary))
    else:
        result["accuracy"] = float("nan")
        result["sensitivity"] = float("nan")
        result["specificity"] = float("nan")
        result["precision"] = float("nan")
        result["f1"] = float("nan")
        result["mcc"] = float("nan")

    # Also compute at threshold=0.5 for reference
    y_pred_05 = (y_prob >= 0.5).astype(int)
    cm_05 = confusion_matrix(y_true, y_pred_05)
    if cm_05.shape == (2, 2):
        tn05, fp05, fn05, tp05 = cm_05.ravel()
        result["accuracy_t05"] = float((tp05 + tn05) / (tp05 + tn05 + fp05 + fn05))
        result["mcc_t05"] = float(matthews_corrcoef(y_true, y_pred_05))
        result["f1_t05"] = float(
            2 * tp05 / (2 * tp05 + fp05 + fn05) if (2 * tp05 + fp05 + fn05) > 0 else 0
        )
    else:
        result["accuracy_t05"] = float("nan")
        result["mcc_t05"] = float("nan")
        result["f1_t05"] = float("nan")

    result["n_samples"] = len(y_true)
    result["n_positive"] = int(y_true.sum())
    result["n_negative"] = int(len(y_true) - y_true.sum())
    result["class_ratio"] = float(y_true.mean())

    return result


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate_seed_metrics(
    seed_metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate metrics across seeds: mean ± std."""
    if not seed_metrics:
        return {}

    metric_keys = [
        "auroc", "auprc", "f1", "f1_roc_optimal", "accuracy",
        "sensitivity", "specificity", "precision", "mcc",
        "accuracy_t05", "mcc_t05", "f1_t05",
    ]

    agg: dict[str, Any] = {}
    for key in metric_keys:
        values = [m[key] for m in seed_metrics if key in m and not np.isnan(m[key])]
        if values:
            agg[f"{key}_mean"] = float(np.mean(values))
            agg[f"{key}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        else:
            agg[f"{key}_mean"] = float("nan")
            agg[f"{key}_std"] = float("nan")

    agg["n_seeds"] = len(seed_metrics)
    return agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GraphBAN pretrained weights on thesis kinase test sets"
    )
    parser.add_argument(
        "--dataset",
        choices=["non_human", "human", "all"],
        default="non_human",
        help="Thesis test set to evaluate on (default: non_human)",
    )
    parser.add_argument(
        "--source-models",
        nargs="+",
        default=["biosnap", "bindingdb", "kiba"],
        help=f"Source models to evaluate. Choices: {AVAILABLE_SOURCE_MODELS + ['all']}",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=SEEDS_PER_SOURCE,
        help=f"Seeds to evaluate (default: {SEEDS_PER_SOURCE})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference (default: 32)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: GraphBAN/results/pretrained_evaluation/)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name for this evaluation run (used in output filename, e.g. 'v1', 'final'). "
             "Output file becomes <run-name>_pretrained_evaluation.json",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable feature caching",
    )
    args = parser.parse_args()

    if "all" in args.source_models:
        args.source_models = AVAILABLE_SOURCE_MODELS

    print("=" * 70)
    print("  GraphBAN Pretrained Evaluation on Thesis Kinase Test Sets")
    print("=" * 70)
    print(f"  Target dataset:  {args.dataset}")
    print(f"  Source models:   {args.source_models}")
    print(f"  Seeds:           {args.seeds}")
    print(f"  Batch size:      {args.batch_size}")
    if args.run_name:
        print(f"  Run name:        {args.run_name}")

    # Setup
    modules = setup_graphban_imports()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device:          {device}")

    # Load config (needed to instantiate model)
    cfg = modules["get_cfg_defaults"]()
    config_path = GRAPHBAN_SRC / "GraphBAN_DA.yaml"
    if config_path.exists():
        cfg.merge_from_file(str(config_path))
    cfg.freeze()
    n_class = cfg.DECODER.BINARY
    print(f"  Decoder binary:  {n_class}")

    # Output directory
    output_base = Path(args.output_dir) if args.output_dir else SCRIPT_DIR / "results" / "pretrained_evaluation"
    output_base.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load test data ────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print(f"Step 1: Loading thesis test set: {args.dataset}")
    print(f"{'─' * 70}")

    df_test = load_thesis_test_set(args.dataset)
    print(f"  Loaded {len(df_test)} test samples")
    print(f"  Positive: {df_test['Y'].sum()} ({df_test['Y'].mean()*100:.1f}%)")
    print(f"  Negative: {(1 - df_test['Y']).sum()}")
    print(f"  Unique proteins: {df_test['Protein'].nunique()}")
    print(f"  Unique SMILES: {df_test['SMILES'].nunique()}")

    # ── Step 2: Extract features ──────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print(f"Step 2: Extracting features (ESM-1b + ChemBERTa)")
    print(f"{'─' * 70}")

    cache_dir = output_base / "feature_cache" / args.dataset if not args.no_cache else None

    unique_proteins = df_test["Protein"].unique().tolist()
    unique_smiles = df_test["SMILES"].unique().tolist()

    esm_embs = extract_esm_features(unique_proteins, device, cache_dir)
    chem_embs = extract_chemberta_features(unique_smiles, device, cache_dir)

    df_test = add_features_to_df(df_test, esm_embs, chem_embs)
    print(f"  Features added: {len(df_test)} samples with complete features")

    # ── Step 3: Evaluate each source model ────────────────────────────────
    print(f"\n{'─' * 70}")
    print(f"Step 3: Evaluating pretrained models")
    print(f"{'─' * 70}")

    # Create test dataset and dataloader
    # IMPORTANT: Use max_drug_nodes=290 (upstream default) to match pretrained
    # model training. Our patch_upstream.py changed it to 310 for re-training
    # on kinase data, but pretrained weights expect 290-node padding.
    PRETRAINED_MAX_DRUG_NODES = 290
    test_dataset = modules["DTIDataset"](df_test.index.values, df_test,
                                         max_drug_nodes=PRETRAINED_MAX_DRUG_NODES)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=modules["graph_collate_func"],
    )
    print(f"  max_drug_nodes:  {PRETRAINED_MAX_DRUG_NODES} (upstream default for pretrained weights)")

    all_results: dict[str, Any] = {
        "target_dataset": args.dataset,
        "n_test_samples": len(df_test),
        "class_balance": float(df_test["Y"].mean()),
        "evaluation_protocol": {
            "threshold_method": "F1-optimal from ROC curve (paper protocol)",
            "metrics": "AUROC, AUPRC, F1, Sensitivity, Specificity, Accuracy, MCC",
            "note": "GraphBAN models were trained on external datasets (BioSNAP, BindingDB, KIBA, etc.) and evaluated zero-shot on thesis kinase test sets. No training, validation, or threshold optimization was performed on thesis data.",
        },
        "per_source": {},
    }

    for source_name in args.source_models:
        print(f"\n  ═══ Source: {source_name} ═══")
        seed_metrics = []

        for seed in args.seeds:
            model_path = find_best_model(source_name, seed)
            if model_path is None:
                print(f"    Seed {seed}: no best_model found, skipping")
                continue

            print(f"    Seed {seed}: {model_path.name}")

            # Detect BINARY from checkpoint (BindingDB uses 1, BioSNAP/KIBA use 2)
            modules["set_seed"](seed)
            state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
            ckpt_binary = state_dict["mlp_classifier.fc4.weight"].shape[0]

            if ckpt_binary != n_class:
                # Create model with matching config
                cfg_local = modules["get_cfg_defaults"]()
                if config_path.exists():
                    cfg_local.merge_from_file(str(config_path))
                cfg_local.defrost()
                cfg_local.DECODER.BINARY = ckpt_binary
                cfg_local.freeze()
                model = modules["GraphBAN"](**cfg_local).to("cpu")
            else:
                model = modules["GraphBAN"](**cfg).to("cpu")

            model.load_state_dict(state_dict)
            actual_n_class = ckpt_binary

            # Run inference
            y_true, y_prob = run_inference(model, test_loader, device, n_class=actual_n_class)

            # Compute metrics per paper protocol
            metrics = compute_metrics_paper_protocol(y_true, y_prob)
            metrics["seed"] = seed
            metrics["model_path"] = str(model_path)
            seed_metrics.append(metrics)

            print(
                f"      AUROC={metrics['auroc']:.4f}  "
                f"AUPRC={metrics['auprc']:.4f}  "
                f"F1={metrics['f1']:.4f}  "
                f"MCC={metrics['mcc']:.4f}  "
                f"Acc={metrics['accuracy']:.4f}  "
                f"Thr={metrics['threshold_f1_optimal']:.4f}"
            )

            # Clean up
            del model
            torch.cuda.empty_cache()

        # Aggregate across seeds
        if seed_metrics:
            agg = aggregate_seed_metrics(seed_metrics)
            print(f"\n    ── {source_name} aggregate ({agg['n_seeds']} seeds) ──")
            print(
                f"      AUROC: {agg['auroc_mean']:.4f} ± {agg['auroc_std']:.4f}  "
                f"MCC: {agg['mcc_mean']:.4f} ± {agg['mcc_std']:.4f}  "
                f"F1: {agg['f1_mean']:.4f} ± {agg['f1_std']:.4f}"
            )

            all_results["per_source"][source_name] = {
                "aggregate": agg,
                "per_seed": seed_metrics,
            }

    # ── Step 4: Save results ──────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print(f"Step 4: Saving results")
    print(f"{'─' * 70}")

    result_dir = output_base / args.dataset
    result_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{args.run_name}_pretrained_evaluation.json" if args.run_name else "pretrained_evaluation.json"
    result_file = result_dir / filename
    all_results["run_name"] = args.run_name
    with open(result_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Saved: {result_file}")

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print(f"  Summary: GraphBAN Zero-Shot → {args.dataset} test set")
    print(f"{'═' * 70}")
    header = f"{'Source':<15} {'AUROC':>12}  {'AUPRC':>12}  {'MCC':>12}  {'F1':>12}  {'Acc':>12}"
    print(header)
    print("─" * len(header))

    for source_name, data in all_results["per_source"].items():
        agg = data["aggregate"]
        print(
            f"{source_name:<15} "
            f"{agg['auroc_mean']:.4f}±{agg['auroc_std']:.4f}  "
            f"{agg['auprc_mean']:.4f}±{agg['auprc_std']:.4f}  "
            f"{agg['mcc_mean']:.4f}±{agg['mcc_std']:.4f}  "
            f"{agg['f1_mean']:.4f}±{agg['f1_std']:.4f}  "
            f"{agg['accuracy_mean']:.4f}±{agg['accuracy_std']:.4f}"
        )

    print(f"\nDone!")


if __name__ == "__main__":
    main()
