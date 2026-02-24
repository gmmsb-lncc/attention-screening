#!/usr/bin/env python3
"""
Split Comparison Analysis: Main Entry Point
===========================================

This script is the main entry point for split comparison analysis with baseline
models (KNN and MLP). It mirrors the CLI flow used by
crossattention_split_analysis_main.py, while keeping model choices specific to
the baseline comparison.
"""

import argparse
import json
import os
import time
import traceback
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import faiss
from scipy.stats import wilcoxon
from joblib import dump
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, matthews_corrcoef, f1_score,
    precision_score, recall_score, roc_auc_score
)
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from crossattention_split_analysis.config import (
    DATASET_PATHS, DEFAULT_AFFINITY_THRESHOLD,
    AVAILABLE_SCENARIOS, DEFAULT_SCENARIOS
)
from crossattention_split_analysis.utils.json_io import read_json, write_json
from crossattention_split_analysis.visualization.leakage_analysis import (
    run_leakage_diagnostics,
)
from crossattention_split_analysis.visualization.image_readme import (
    write_images_readme,
)

DEFAULT_SPLIT_SEED = 42
DEFAULT_N_FOLDS = 10
DEFAULT_TEST_FRACTION = 0.10
DEFAULT_S4_RESTARTS = 2048
MIN_TEST_SAMPLES = 50
DEFAULT_SPLIT_MODE = "single_90_10"
SPLIT_PROTOCOL_VERSION = "single_split_v1"

# Plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 12


def faiss_knn_predict(X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray, k: int = 5):
    """KNN classification via FAISS with cosine similarity and distance-weighted voting.

    Uses L2-normalized inner product (equivalent to cosine similarity).
    Implements distance-weighted voting equivalent to sklearn weights='distance'.

    Returns (predictions, probabilities) where probabilities are the normalized
    class scores for the positive class (used for AUROC computation).
    """
    X_train_f32 = np.ascontiguousarray(X_train, dtype=np.float32)
    X_test_f32 = np.ascontiguousarray(X_test, dtype=np.float32)
    faiss.normalize_L2(X_train_f32)
    faiss.normalize_L2(X_test_f32)

    index = faiss.IndexFlatIP(X_train_f32.shape[1])
    index.add(X_train_f32)
    similarities, indices = index.search(X_test_f32, k)

    # Distance-weighted voting: weight = similarity (higher = closer)
    # Clamp similarities to avoid negative weights from numerical noise
    weights = np.maximum(similarities, 0.0)

    classes = np.unique(y_train)
    n_test = X_test_f32.shape[0]

    # Vectorized weighted voting
    neighbor_labels = y_train[indices]  # (n_test, k)
    class_scores = np.zeros((n_test, len(classes)), dtype=np.float64)
    for ci, cls in enumerate(classes):
        mask = (neighbor_labels == cls)
        class_scores[:, ci] = np.where(mask, weights, 0.0).sum(axis=1)

    best_class_idx = class_scores.argmax(axis=1)
    predictions = classes[best_class_idx]

    # Compute probability for positive class (class=1) for AUROC
    row_sums = class_scores.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-12)  # avoid division by zero
    proba_positive = class_scores[:, -1] / row_sums.ravel()  # last class = 1

    return predictions, proba_positive


def load_dataset(filepath: str, keep_monotonic: bool = False,
                 filter_monotonic_compounds: bool = False):
    """Load dataset and create binary labels using pChEMBL threshold.

    By default, removes kinases with monotonic activity profiles (100% active
    or 100% inactive), as they provide no discriminative signal and inflate
    metrics. Use keep_monotonic=True to retain them.

    filter_monotonic_compounds: if True, also removes compounds that are 100%
    active (pan-active) or 100% inactive (pan-inactive) across all kinases
    they were tested against. These "monotonic compounds" are trivial cases.
    """
    df = pd.read_csv(filepath, sep='\t')

    threshold = DEFAULT_AFFINITY_THRESHOLD.threshold_pchembl  # 6.0

    # Calculate pChEMBL from standard_value when missing
    # NOTE: assumes standard_value is in nM. pChEMBL = 9 - log10(nM)
    n_missing = df['pchembl_value'].isna().sum()
    if n_missing > 0:
        print(f"  Calculating pChEMBL for {n_missing} samples with missing values...")
        mask_missing = df['pchembl_value'].isna()
        std_vals = df.loc[mask_missing, 'standard_value']
        # Sanity check: standard_value should be positive and in nM range
        if (std_vals <= 0).any():
            n_invalid = (std_vals <= 0).sum()
            print(f"  WARNING: {n_invalid} rows with standard_value <= 0, skipping imputation for those")
            mask_valid = mask_missing & (df['standard_value'] > 0)
            df.loc[mask_valid, 'pchembl_value'] = 9 - np.log10(df.loc[mask_valid, 'standard_value'])
        else:
            df.loc[mask_missing, 'pchembl_value'] = 9 - np.log10(std_vals)
        # Warn if values suggest wrong units (e.g. molar instead of nM)
        computed = df.loc[mask_missing, 'pchembl_value'].dropna()
        if len(computed) > 0 and (computed > 15).any():
            print(f"  WARNING: {(computed > 15).sum()} computed pChEMBL values > 15, "
                  f"check that standard_value is in nM (not M or uM)")

    df['label'] = (df['pchembl_value'] >= threshold).astype(int)

    # Filter monotonic kinases (100% active or 100% inactive)
    if not keep_monotonic:
        kin_rates = df.groupby('target_kinase')['label'].mean()
        mono_kinases = set(kin_rates[(kin_rates == 0.0) | (kin_rates == 1.0)].index)
        if mono_kinases:
            n_before = len(df)
            k_before = df['target_kinase'].nunique()
            df = df[~df['target_kinase'].isin(mono_kinases)].reset_index(drop=True)
            n_removed = n_before - len(df)
            k_removed = k_before - df['target_kinase'].nunique()
            print(f"  Monotonic kinase filter: removed {k_removed} kinases ({n_removed} samples, "
                  f"{100*n_removed/n_before:.1f}%) with 100% single-class profiles")

    # Filter monotonic compounds (pan-active or pan-inactive across all tested kinases)
    if filter_monotonic_compounds:
        # Only consider compounds tested against >1 kinase (single-kinase compounds are not "pan")
        compound_kinase_counts = df.groupby('chembl_id')['target_kinase'].nunique()
        multi_kinase_compounds = set(compound_kinase_counts[compound_kinase_counts > 1].index)

        # For multi-kinase compounds, compute activity rate
        df_multi = df[df['chembl_id'].isin(multi_kinase_compounds)]
        if len(df_multi) > 0:
            compound_rates = df_multi.groupby('chembl_id')['label'].mean()
            mono_compounds = set(compound_rates[(compound_rates == 0.0) | (compound_rates == 1.0)].index)

            if mono_compounds:
                n_before = len(df)
                c_before = df['chembl_id'].nunique()
                df = df[~df['chembl_id'].isin(mono_compounds)].reset_index(drop=True)
                n_removed = n_before - len(df)
                c_removed = c_before - df['chembl_id'].nunique()
                pan_active = sum(1 for c in mono_compounds if compound_rates[c] == 1.0)
                pan_inactive = len(mono_compounds) - pan_active
                print(f"  Monotonic compound filter: removed {c_removed} compounds ({n_removed} samples, "
                      f"{100*n_removed/n_before:.1f}%): {pan_active} pan-active, {pan_inactive} pan-inactive")

    # Report class distribution
    n_active = df['label'].sum()
    n_inactive = len(df) - n_active
    threshold_nm_equiv = 10 ** (9 - threshold)
    print(f"  Affinity threshold: pChEMBL >= {threshold:.1f} (equivalent to <= {threshold_nm_equiv:.0f} nM)")
    print(f"  Class distribution: {n_active} active ({100*n_active/len(df):.1f}%), "
          f"{n_inactive} inactive ({100*n_inactive/len(df):.1f}%)")

    return df


_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def compute_morgan_fingerprints(smiles_list: list, desc: str = "Computing fingerprints"):
    """Compute Morgan fingerprints for a list of SMILES. Returns float32."""
    fingerprints = []
    valid_indices = []
    n_invalid = 0

    for i, smi in enumerate(tqdm(smiles_list, desc=desc, disable=len(smiles_list) < 1000)):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = _MORGAN_GEN.GetFingerprintAsNumPy(mol).astype(np.float32)
            fingerprints.append(fp)
            valid_indices.append(i)
        else:
            n_invalid += 1

    if n_invalid > 0:
        print(f"  WARNING: {n_invalid}/{len(smiles_list)} SMILES failed RDKit parsing "
              f"({100*n_invalid/len(smiles_list):.1f}% dropped)")

    return np.array(fingerprints, dtype=np.float32) if fingerprints else np.empty((0, 2048), dtype=np.float32), valid_indices


def prepare_features(df: pd.DataFrame, all_kinases: list, fp_cache: dict = None):
    """Prepare features: Morgan FP + One-Hot Kinase (float32).

    If fp_cache is provided, reuses precomputed fingerprints keyed by chembl_id.
    """
    n_kinases = len(all_kinases)
    kinase_to_idx = {k: i for i, k in enumerate(all_kinases)}

    if fp_cache is not None:
        # Use cached fingerprints: vectorized lookup by chembl_id
        chembl_ids = df['chembl_id'].values
        smiles = df['canonical_smiles'].values
        fps_list = []
        valid_idx_pos = []
        n = len(df)

        for pos in tqdm(range(n), desc="    Preparing features", disable=n < 10000):
            cid = chembl_ids[pos]
            if cid in fp_cache:
                fps_list.append(fp_cache[cid])
                valid_idx_pos.append(pos)
            else:
                mol = Chem.MolFromSmiles(smiles[pos])
                if mol is not None:
                    fp_arr = _MORGAN_GEN.GetFingerprintAsNumPy(mol).astype(np.float32)
                    fp_cache[cid] = fp_arr
                    fps_list.append(fp_arr)
                    valid_idx_pos.append(pos)

        if not fps_list:
            return np.empty((0, 2048 + n_kinases), dtype=np.float32), np.array([]), []

        fps = np.stack(fps_list)
    else:
        fps, valid_idx_pos = compute_morgan_fingerprints(df['canonical_smiles'].tolist())
        if len(fps) == 0:
            return np.empty((0, 2048 + n_kinases), dtype=np.float32), np.array([]), []

    kinase_oh = np.zeros((len(valid_idx_pos), n_kinases), dtype=np.float32)
    for j, pos in enumerate(valid_idx_pos):
        kinase = df.iloc[pos]['target_kinase']
        if kinase in kinase_to_idx:
            kinase_oh[j, kinase_to_idx[kinase]] = 1.0

    X = np.hstack([fps, kinase_oh])
    y = df.iloc[valid_idx_pos]['label'].values

    return X, y, valid_idx_pos


# =============================================================================
# TRAINING AND EVALUATION
# =============================================================================

def _compute_metrics(y_true, y_pred, y_proba=None):
    """Compute all classification metrics."""
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'mcc': float(matthews_corrcoef(y_true, y_pred)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None and len(np.unique(y_true)) == 2:
        try:
            metrics['auroc'] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            metrics['auroc'] = float('nan')
    else:
        metrics['auroc'] = float('nan')
    return metrics


def train_and_evaluate(
    df: pd.DataFrame,
    train_idx,
    test_idx,
    seed: int = 42,
    fp_cache: dict = None,
    save_models: bool = False,
    model_output_dir: str = None,
    scenario_id: str = None,
    dropped_idx=None,
    dataset_metadata: dict = None,
    save_knn_features: bool = False,
):
    """Train KNN (FAISS) and MLP and return metrics.

    NOTE on blind-target condition: When unseen kinases appear in the test set
    (kinase and new_compound_new_kinase scenarios), their one-hot encoding is
    all-zeros. This means the model has NO target identity information for those
    samples. The performance drop in these scenarios reflects BOTH data leakage
    removal AND loss of target representation. Models using sequence embeddings
    (e.g., ESM-2) would retain some generalization signal for novel kinases.

    NOTE on MLP early stopping: sklearn MLPClassifier carves a random 10%
    validation split from training data for early stopping. This internal split
    does NOT respect compound/kinase boundaries, which is a known limitation
    of using sklearn for this evaluation. It affects only when training stops,
    not the test evaluation integrity.
    """

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # IMPORTANT: one-hot kinase space is defined from TRAIN only.
    # Unseen kinases in test therefore map to all-zero vectors (blind-target).
    train_kinases = list(train_df['target_kinase'].unique())
    X_train, y_train, train_valid = prepare_features(train_df, train_kinases, fp_cache)
    X_test, y_test, test_valid = prepare_features(test_df, train_kinases, fp_cache)

    if len(X_test) == 0 or len(X_train) == 0:
        return None
    if len(np.unique(y_train)) < 2:
        print("    WARNING: training fold has single class; skipping fold.")
        return None

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    results = {}

    # KNN via FAISS (cosine similarity with distance-weighted voting)
    print("    Training KNN (FAISS)...")
    t0 = time.time()
    knn_k = 5
    knn_pred, knn_proba = faiss_knn_predict(X_train_scaled, y_train, X_test_scaled, k=knn_k)
    knn_time = time.time() - t0
    results['KNN'] = _compute_metrics(y_test, knn_pred, knn_proba)
    print(f"    KNN done in {format_time(knn_time)}")

    # MLP
    print("    Training MLP...")
    t0 = time.time()
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=seed
    )
    mlp.fit(X_train_scaled, y_train)
    mlp_pred = mlp.predict(X_test_scaled)
    mlp_proba = mlp.predict_proba(X_test_scaled)[:, 1]
    mlp_time = time.time() - t0
    results['MLP'] = _compute_metrics(y_test, mlp_pred, mlp_proba)
    print(f"    MLP done in {format_time(mlp_time)}")

    artifact_paths = {}
    if save_models and model_output_dir and scenario_id:
        scenario_dir = Path(model_output_dir) / scenario_id
        scenario_dir.mkdir(parents=True, exist_ok=True)

        scaler_path = scenario_dir / "scaler.joblib"
        mlp_path = scenario_dir / "mlp_model.joblib"
        train_kinases_path = scenario_dir / "train_kinases.json"
        split_indices_path = scenario_dir / "split_indices.npz"
        knn_payload_path = scenario_dir / "knn_payload.npz"
        metadata_path = scenario_dir / "metadata.json"

        dump(scaler, scaler_path)
        dump(mlp, mlp_path)

        with train_kinases_path.open("w", encoding="utf-8") as f:
            json.dump(train_kinases, f, indent=2)

        dropped_idx_arr = np.asarray(dropped_idx if dropped_idx is not None else [], dtype=np.int64)
        np.savez_compressed(
            split_indices_path,
            train_idx_raw=np.asarray(train_idx, dtype=np.int64),
            test_idx_raw=np.asarray(test_idx, dtype=np.int64),
            dropped_idx_raw=dropped_idx_arr,
            train_valid_pos=np.asarray(train_valid, dtype=np.int64),
            test_valid_pos=np.asarray(test_valid, dtype=np.int64),
        )

        if save_knn_features:
            np.savez_compressed(
                knn_payload_path,
                X_train_scaled=X_train_scaled,
                y_train=np.asarray(y_train, dtype=np.int8),
                knn_k=np.asarray([knn_k], dtype=np.int32),
            )
            knn_payload_saved = True
        else:
            # For KNN, the "model" is the reference training set; indices above are sufficient to rebuild it.
            np.savez_compressed(
                knn_payload_path,
                knn_k=np.asarray([knn_k], dtype=np.int32),
                y_train=np.asarray(y_train, dtype=np.int8),
            )
            knn_payload_saved = False

        metadata = {
            "scenario_id": scenario_id,
            "seed": int(seed),
            "train_rows_raw": int(len(train_idx)),
            "test_rows_raw": int(len(test_idx)),
            "dropped_rows_raw": int(len(dropped_idx_arr)),
            "train_rows_used": int(len(X_train)),
            "test_rows_used": int(len(X_test)),
            "knn_k": int(knn_k),
            "knn_payload_has_features": bool(knn_payload_saved),
            "models": {
                "mlp": str(mlp_path),
                "scaler": str(scaler_path),
                "train_kinases": str(train_kinases_path),
                "split_indices": str(split_indices_path),
                "knn_payload": str(knn_payload_path),
            },
            "metrics": {
                "KNN": results["KNN"],
                "MLP": results["MLP"],
            },
        }
        if dataset_metadata:
            metadata["dataset"] = dataset_metadata

        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        artifact_paths = {
            "scenario_dir": str(scenario_dir),
            "mlp_model": str(mlp_path),
            "scaler": str(scaler_path),
            "train_kinases": str(train_kinases_path),
            "split_indices": str(split_indices_path),
            "knn_payload": str(knn_payload_path),
            "metadata": str(metadata_path),
        }
        print(f"    Model artifacts saved: {scenario_dir}")

    results["_artifacts"] = artifact_paths
    return results


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

# Scenario configuration: map CLI key -> (display_name, plot_key)
ALL_SCENARIOS_CONFIG = {
    'new_compound_new_kinase': (
        'New Compound + New Kinase (true generalization, S4)',
        'New Comp.\n+ New Kinase'
    ),
    'scaffold': (
        'Split by Scaffold (cold-scaffold)',
        'Split by\nScaffold'
    ),
    'compound': (
        'Split by Compound (cold-drug, S2)',
        'Split by\nCompound'
    ),
    'kinase': (
        'Split by Kinase (cold-target, S3)',
        'Split by\nKinase'
    ),
    'random': (
        'Random Split (with leakage, S1)',
        'Random Split\n(Original)'
    ),
}


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}min"
    return f"{seconds / 3600:.1f}h"


def _build_compound_fp_map(df: pd.DataFrame):
    """Build mapping: compound_id -> Morgan fingerprint (bool vector)."""
    unique_comp = df[['chembl_id', 'canonical_smiles']].drop_duplicates(subset=['chembl_id'])
    smiles = unique_comp['canonical_smiles'].tolist()
    fps, valid_idx = compute_morgan_fingerprints(smiles, desc="Building FP map (unique compounds)")
    fp_map = {}
    for fp_i, src_i in enumerate(valid_idx):
        comp_id = unique_comp.iloc[src_i]['chembl_id']
        fp_map[comp_id] = fps[fp_i].astype(bool)
    return fp_map


def _get_murcko_scaffold(smiles: str) -> str:
    """Return Murcko scaffold SMILES for a compound. Returns 'UNKNOWN' on failure."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 'UNKNOWN'
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return 'UNKNOWN'


def _build_compound_scaffold_map(df: pd.DataFrame) -> dict:
    """Build mapping: chembl_id -> Murcko scaffold SMILES."""
    unique_comp = df[['chembl_id', 'canonical_smiles']].drop_duplicates(subset=['chembl_id'])
    scaffold_map = {}
    for _, row in tqdm(unique_comp.iterrows(), total=len(unique_comp),
                       desc="Computing Murcko scaffolds", disable=len(unique_comp) < 1000):
        scaffold_map[row['chembl_id']] = _get_murcko_scaffold(row['canonical_smiles'])
    n_scaffolds = len(set(scaffold_map.values()) - {'UNKNOWN'})
    print(f"  Murcko scaffolds: {n_scaffolds} unique scaffolds from {len(scaffold_map)} compounds")
    return scaffold_map


def resolve_dataset_input_path(
    dataset_type: str,
    input_path: str = None,
    use_without_test_input: bool = True,
    without_test_dir: str = "scaffolds_splits/output",
) -> str:
    """Resolve dataset path, preferring scaffold-split non-test inputs when requested."""
    if input_path:
        return input_path

    if use_without_test_input and dataset_type in ("human", "non_human"):
        candidate = os.path.join(without_test_dir, f"{dataset_type}_input_without_test.tsv")
        if os.path.exists(candidate):
            return candidate
        print(f"  WARNING: without-test input not found: {candidate}. Falling back to default dataset path.")

    return DATASET_PATHS.get(dataset_type)


def _split_quality_loss(
    y_train: np.ndarray,
    y_test: np.ndarray,
    test_fraction_achieved: float,
    target_test_fraction: float,
    dropped_fraction: float = 0.0,
) -> float:
    """Loss for selecting train/test splits with class support and target ratio."""
    if len(y_train) == 0 or len(y_test) == 0:
        return 1e9

    train_classes = set(np.unique(y_train).tolist())
    test_classes = set(np.unique(y_test).tolist())
    missing = int(0 not in train_classes) + int(1 not in train_classes) + int(0 not in test_classes) + int(1 not in test_classes)

    train_pos = float((y_train == 1).mean())
    test_pos = float((y_test == 1).mean())
    class_rate_gap = abs(train_pos - test_pos)

    return float(
        abs(test_fraction_achieved - target_test_fraction)
        + 2.0 * class_rate_gap
        + 10.0 * missing
        + 0.5 * dropped_fraction
    )


def _target_s4_probability(test_fraction: float) -> float:
    """Solve p for strict S4 where test=(p^2), train=((1-p)^2) over used rows."""
    t = float(np.clip(test_fraction, 1e-6, 1.0 - 1e-6))
    if abs(t - 0.5) < 1e-12:
        return 0.5
    root = np.sqrt(t * (1.0 - t))
    p1 = (t + root) / (2.0 * t - 1.0)
    p2 = (t - root) / (2.0 * t - 1.0)
    candidates = [p for p in (p1, p2) if 0.0 < p < 1.0]
    if not candidates:
        return 0.25
    return min(candidates, key=lambda x: abs(x - 0.25))


def _generate_single_split(
    df: pd.DataFrame,
    scenario_id: str,
    seed: int,
    test_fraction: float,
    scaffold_map: dict = None,
    s4_restarts: int = DEFAULT_S4_RESTARTS,
):
    """Generate one train/test split (~90/10 by default) for a scenario."""
    if test_fraction <= 0.0 or test_fraction >= 1.0:
        raise ValueError("test_fraction must be in (0, 1)")

    n_samples = len(df)
    labels = df["label"].to_numpy()
    rng = np.random.default_rng(seed)

    # Scenarios with a single grouping dimension.
    if scenario_id in ("random", "compound", "kinase", "scaffold"):
        # Use 10 buckets for 10% test by default; robust for other fractions too.
        n_buckets = max(2, int(round(1.0 / test_fraction)))
        if scenario_id == "random":
            fold_assignments = _assign_stratified_folds(labels, n_buckets, rng)
        elif scenario_id == "compound":
            fold_assignments = _assign_group_folds(df["chembl_id"].values, n_buckets, rng)
        elif scenario_id == "kinase":
            fold_assignments = _assign_group_folds(df["target_kinase"].values, n_buckets, rng)
        else:
            if scaffold_map is None:
                raise ValueError("scaffold scenario requires scaffold_map")
            scaffolds = df["chembl_id"].map(scaffold_map).fillna("UNKNOWN").values
            fold_assignments = _assign_group_folds(scaffolds, n_buckets, rng)

        best = None
        best_loss = float("inf")
        for fold_i in range(n_buckets):
            test_idx = np.where(fold_assignments == fold_i)[0]
            train_idx = np.where(fold_assignments != fold_i)[0]
            if len(test_idx) < MIN_TEST_SAMPLES or len(train_idx) == 0:
                continue
            loss = _split_quality_loss(
                y_train=labels[train_idx],
                y_test=labels[test_idx],
                test_fraction_achieved=len(test_idx) / n_samples,
                target_test_fraction=test_fraction,
            )
            if loss < best_loss:
                best_loss = loss
                best = (train_idx, test_idx)

        if best is None:
            raise RuntimeError(f"Could not generate a valid single split for scenario={scenario_id}")

        train_idx, test_idx = best
        return {
            "train_idx": train_idx,
            "test_idx": test_idx,
            "dropped_idx": np.array([], dtype=np.int64),
            "test_fraction_used": float(len(test_idx) / max(len(train_idx) + len(test_idx), 1)),
            "test_fraction_total": float(len(test_idx) / n_samples),
        }

    # Strict S4: test = compounds_in_test AND kinases_in_test
    if scenario_id != "new_compound_new_kinase":
        raise ValueError(f"Unknown scenario_id: {scenario_id}")

    comp_codes, comp_uniques = pd.factorize(df["chembl_id"].astype(str), sort=False)
    kin_codes, kin_uniques = pd.factorize(df["target_kinase"].astype(str), sort=False)

    p0 = _target_s4_probability(test_fraction)
    best = None
    best_loss = float("inf")

    for restart in range(max(1, s4_restarts)):
        rrng = np.random.default_rng(seed + 7919 * restart)
        p_comp = float(np.clip(p0 * rrng.uniform(0.8, 1.2), 0.05, 0.9))
        p_kin = float(np.clip(p0 * rrng.uniform(0.8, 1.2), 0.05, 0.9))

        comp_test = rrng.random(len(comp_uniques)) < p_comp
        kin_test = rrng.random(len(kin_uniques)) < p_kin

        row_comp_test = comp_test[comp_codes]
        row_kin_test = kin_test[kin_codes]

        test_mask = row_comp_test & row_kin_test
        train_mask = (~row_comp_test) & (~row_kin_test)
        dropped_mask = ~(test_mask | train_mask)

        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        dropped_idx = np.where(dropped_mask)[0]

        if len(test_idx) < MIN_TEST_SAMPLES or len(train_idx) == 0:
            continue

        used = len(train_idx) + len(test_idx)
        test_frac_used = len(test_idx) / max(used, 1)
        dropped_frac = len(dropped_idx) / n_samples
        loss = _split_quality_loss(
            y_train=labels[train_idx],
            y_test=labels[test_idx],
            test_fraction_achieved=test_frac_used,
            target_test_fraction=test_fraction,
            dropped_fraction=dropped_frac,
        )

        if loss < best_loss:
            best_loss = loss
            best = (train_idx, test_idx, dropped_idx, test_frac_used)

    if best is None:
        raise RuntimeError("Could not generate a valid strict S4 single split")

    train_idx, test_idx, dropped_idx, test_frac_used = best
    return {
        "train_idx": train_idx,
        "test_idx": test_idx,
        "dropped_idx": dropped_idx,
        "test_fraction_used": float(test_frac_used),
        "test_fraction_total": float(len(test_idx) / n_samples),
    }


# =============================================================================
# K-FOLD CROSS-VALIDATION
# =============================================================================

def _assign_stratified_folds(labels, n_folds, rng):
    """Assign samples to folds with stratification by class label.

    Within each class, samples are shuffled and cyclically assigned to folds,
    ensuring approximately equal class proportions in each fold.
    """
    labels = np.asarray(labels)
    fold_assignments = np.empty(len(labels), dtype=np.int64)

    for cls in np.unique(labels):
        cls_indices = np.where(labels == cls)[0].copy()
        rng.shuffle(cls_indices)
        for i, idx in enumerate(cls_indices):
            fold_assignments[idx] = i % n_folds

    return fold_assignments


def _assign_group_folds(groups, n_folds, rng):
    """Assign samples to folds by group using greedy bin-packing.

    All members of a group go to the same fold. Groups are shuffled for
    randomness, then sorted by size descending. Each group is assigned to
    the fold with the fewest samples so far, producing balanced fold sizes.
    """
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)

    # Compute group sizes
    group_sizes = {}
    for g in unique_groups:
        group_sizes[g] = int((groups == g).sum())

    # Shuffle for randomness, then stable-sort by size descending
    perm = rng.permutation(len(unique_groups))
    shuffled = unique_groups[perm]
    sorted_groups = sorted(shuffled, key=lambda g: group_sizes[g], reverse=True)

    # Greedy bin-packing: assign each group to the fold with fewest samples
    fold_counts = np.zeros(n_folds, dtype=np.int64)
    group_to_fold = {}

    for g in sorted_groups:
        target_fold = int(np.argmin(fold_counts))
        group_to_fold[g] = target_fold
        fold_counts[target_fold] += group_sizes[g]

    # Assign fold to each sample
    fold_assignments = np.empty(len(groups), dtype=np.int64)
    for i, g in enumerate(groups):
        fold_assignments[i] = group_to_fold[g]

    return fold_assignments


def _generate_cv_folds(df, scenario_id, n_folds, seed, scaffold_map=None):
    """Generate k-fold cross-validation splits for a given scenario.

    Returns list of (fold_i, train_idx, val_idx, test_idx) tuples.
    Folds with < MIN_TEST_SAMPLES test samples or single-class test sets
    are skipped with a warning.

    For non-S4 scenarios: test = fold i, val = fold (i+1)%k, train = rest.
    For S4 (new_compound_new_kinase): compounds and kinases are independently
    assigned to folds; test = rows where BOTH are in fold i, train = rows
    where NEITHER is in fold i, val = orphans.
    """
    rng = np.random.default_rng(seed)
    labels = df['label'].values
    folds = []

    if scenario_id == 'new_compound_new_kinase':
        # Double-group: compounds AND kinases independently assigned to folds
        compounds = df['chembl_id'].values
        kinases = df['target_kinase'].values
        rng_comp = np.random.default_rng(seed)
        rng_kin = np.random.default_rng(seed + 1)
        comp_folds = _assign_group_folds(compounds, n_folds, rng_comp)
        kin_folds = _assign_group_folds(kinases, n_folds, rng_kin)

        for i in range(n_folds):
            comp_in = (comp_folds == i)
            kin_in = (kin_folds == i)
            test_mask = comp_in & kin_in
            train_mask = (~comp_in) & (~kin_in)
            val_mask = ~(test_mask | train_mask)  # orphans

            test_idx = np.where(test_mask)[0]
            train_idx = np.where(train_mask)[0]
            val_idx = np.where(val_mask)[0]

            if len(test_idx) < MIN_TEST_SAMPLES:
                print(f"  WARNING: Fold {i} skipped (test_size={len(test_idx)} < {MIN_TEST_SAMPLES})")
                continue
            if len(train_idx) == 0 or len(val_idx) == 0:
                print(f"  WARNING: Fold {i} skipped (empty train/val partition)")
                continue
            y_train = labels[train_idx]
            y_test = labels[test_idx]
            y_val = labels[val_idx]
            if len(np.unique(y_train)) < 2:
                print(f"  WARNING: Fold {i} skipped (single class in train)")
                continue
            if len(np.unique(y_test)) < 2:
                print(f"  WARNING: Fold {i} skipped (single class in test)")
                continue
            if len(np.unique(y_val)) < 2:
                print(f"  WARNING: Fold {i} skipped (single class in val)")
                continue

            folds.append((i, train_idx, val_idx, test_idx))
    else:
        # Standard scenarios: assign to folds, test=fold_i, val=fold_(i+1)%k
        if scenario_id == 'random':
            fold_assignments = _assign_stratified_folds(labels, n_folds, rng)
        elif scenario_id == 'compound':
            fold_assignments = _assign_group_folds(df['chembl_id'].values, n_folds, rng)
        elif scenario_id == 'kinase':
            fold_assignments = _assign_group_folds(df['target_kinase'].values, n_folds, rng)
        elif scenario_id == 'scaffold':
            if scaffold_map is None:
                raise ValueError("scaffold scenario requires scaffold_map")
            scaffolds = df['chembl_id'].map(scaffold_map).fillna('UNKNOWN').values
            fold_assignments = _assign_group_folds(scaffolds, n_folds, rng)
        else:
            raise ValueError(f"Unknown scenario_id: {scenario_id}")

        for i in range(n_folds):
            val_fold = (i + 1) % n_folds
            test_idx = np.where(fold_assignments == i)[0]
            val_idx = np.where(fold_assignments == val_fold)[0]
            train_idx = np.where(
                (fold_assignments != i) & (fold_assignments != val_fold)
            )[0]

            if len(test_idx) < MIN_TEST_SAMPLES:
                print(f"  WARNING: Fold {i} skipped (test_size={len(test_idx)} < {MIN_TEST_SAMPLES})")
                continue
            if len(train_idx) == 0 or len(val_idx) == 0:
                print(f"  WARNING: Fold {i} skipped (empty train/val partition)")
                continue
            y_train = labels[train_idx]
            y_test = labels[test_idx]
            y_val = labels[val_idx]
            if len(np.unique(y_train)) < 2:
                print(f"  WARNING: Fold {i} skipped (single class in train)")
                continue
            if len(np.unique(y_test)) < 2:
                print(f"  WARNING: Fold {i} skipped (single class in test)")
                continue
            if len(np.unique(y_val)) < 2:
                print(f"  WARNING: Fold {i} skipped (single class in val)")
                continue

            folds.append((i, train_idx, val_idx, test_idx))

    return folds


# =============================================================================
# COMPARISON RUNNER
# =============================================================================

def run_comparison(
    df: pd.DataFrame,
    output_dir: str = '.',
    seed: int = DEFAULT_SPLIT_SEED,
    n_folds: int = DEFAULT_N_FOLDS,
    scenarios: list = None,
    checkpoint_path: str = None,
    force: bool = False,
    split_mode: str = DEFAULT_SPLIT_MODE,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    save_models: bool = True,
    save_knn_features: bool = False,
    model_output_dir: str = None,
    dataset_metadata: dict = None,
    s4_restarts: int = DEFAULT_S4_RESTARTS,
):
    """Run comparison across split scenarios.

    split_mode="single_90_10": one train/test split per scenario.
    split_mode="kfold_cv": legacy k-fold protocol with train/val/test folds.
    """

    valid_modes = {"single_90_10", "kfold_cv"}
    if split_mode not in valid_modes:
        raise ValueError(f"Unknown split_mode={split_mode!r}. Valid options: {sorted(valid_modes)}")
    if split_mode == "kfold_cv" and n_folds < 2:
        raise ValueError("n_folds must be >= 2 when split_mode='kfold_cv'")
    if split_mode == "single_90_10" and not (0.0 < test_fraction < 1.0):
        raise ValueError("test_fraction must be in (0, 1) when split_mode='single_90_10'")
    if scenarios is None:
        scenarios = DEFAULT_SCENARIOS
    if save_models and not model_output_dir:
        model_output_dir = os.path.join(output_dir, "models")
    if save_models and model_output_dir:
        os.makedirs(model_output_dir, exist_ok=True)

    print("=" * 70)
    print("COMPARISON: RANDOM SPLIT VS CORRECT SPLIT")
    print("=" * 70)
    print(f"Model seed (fixed): {seed}")
    print(f"Split mode: {split_mode}")
    if split_mode == "kfold_cv":
        print(f"k-folds: {n_folds}")
    else:
        print(f"Single-split target: train={1.0-test_fraction:.0%}, test={test_fraction:.0%}")
        print(f"S4 restarts: {s4_restarts}")
    print(f"Scenarios: {scenarios}")
    print(f"Save models: {save_models}")
    if save_models:
        print(f"Model artifacts dir: {model_output_dir}")

    scenarios_config = [
        (s, ALL_SCENARIOS_CONFIG[s][0], ALL_SCENARIOS_CONFIG[s][1])
        for s in scenarios if s in ALL_SCENARIOS_CONFIG
    ]
    if not scenarios_config:
        print("  No valid scenarios selected. Nothing to run.")
        return {}, {}

    all_results = {}
    split_stats = {}
    completed_scenarios = []

    # Checkpoint resume (scenario-level)
    if checkpoint_path:
        checkpoint_data = read_json(checkpoint_path)
        if checkpoint_data and not force:
            cfg = checkpoint_data.get('config', {})
            cfg_mode = cfg.get('split_mode', 'kfold_cv')
            cfg_test_fraction = cfg.get('test_fraction', None)
            cfg_matches = (
                cfg.get('seed') == seed and
                cfg_mode == split_mode and
                cfg.get('scenarios') == [s for s, _, _ in scenarios_config]
            )
            if split_mode == "kfold_cv":
                cfg_matches = cfg_matches and (cfg.get('n_folds') == n_folds)
            else:
                cfg_matches = cfg_matches and isinstance(cfg_test_fraction, (int, float)) \
                    and abs(float(cfg_test_fraction) - float(test_fraction)) < 1e-12
            if cfg_matches:
                all_results = checkpoint_data.get('all_results', {})
                split_stats = checkpoint_data.get('split_stats', {})
                completed_scenarios = checkpoint_data.get('completed_scenarios', [])
                completed_scenarios = [
                    s for s in completed_scenarios if s in all_results and s in split_stats
                ]
                if completed_scenarios:
                    print(
                        f"[CHECKPOINT] Resuming comparison from {checkpoint_path} "
                        f"({len(completed_scenarios)}/{len(scenarios_config)} scenarios ready)"
                    )
            else:
                print("[CHECKPOINT] Existing comparison checkpoint ignored (configuration mismatch).")

    if len(completed_scenarios) == len(scenarios_config):
        print("[CHECKPOINT] All scenarios already completed. Reusing cached comparison results.")
        return all_results, split_stats

    # Build fp_cache for model features
    fp_map = _build_compound_fp_map(df)
    fp_cache = {cid: fp.astype(np.float32) for cid, fp in fp_map.items()}
    print(f"Fingerprint cache: {len(fp_cache)} unique compounds precomputed")

    # Build scaffold map if scaffold scenario is requested
    scaffold_map = None
    scenario_ids = [s for s, _, _ in scenarios_config]
    if 'scaffold' in scenario_ids:
        scaffold_map = _build_compound_scaffold_map(df)

    for scenario_pos, (scenario_id, scenario_name, scenario_key) in enumerate(scenarios_config, 1):
        print(f"\n{'-' * 50}")
        print(f"SCENARIO [{scenario_pos}/{len(scenarios_config)}]: {scenario_name}")
        print("-" * 50)

        if scenario_key in completed_scenarios and scenario_key in all_results and scenario_key in split_stats:
            print("  [checkpoint] Scenario already computed. Skipping recalculation.")
            continue

        eval_rounds = []
        if split_mode == "single_90_10":
            split_payload = _generate_single_split(
                df=df,
                scenario_id=scenario_id,
                seed=seed,
                test_fraction=test_fraction,
                scaffold_map=scaffold_map,
                s4_restarts=s4_restarts,
            )
            train_idx = split_payload["train_idx"]
            test_idx = split_payload["test_idx"]
            dropped_idx = split_payload["dropped_idx"]
            eval_rounds = [(
                0,
                train_idx,
                np.array([], dtype=np.int64),
                test_idx,
                dropped_idx,
                split_payload,
            )]
            print(
                "  Single split ready: "
                f"train={len(train_idx)}, test={len(test_idx)}, dropped={len(dropped_idx)}, "
                f"test_used={split_payload['test_fraction_used']:.4f}, "
                f"test_total={split_payload['test_fraction_total']:.4f}"
            )
        else:
            cv_folds = _generate_cv_folds(df, scenario_id, n_folds, seed, scaffold_map)
            print(f"  Valid folds: {len(cv_folds)}/{n_folds}")
            eval_rounds = [
                (fold_i, train_idx, val_idx, test_idx, np.array([], dtype=np.int64), None)
                for (fold_i, train_idx, val_idx, test_idx) in cv_folds
            ]
            if not eval_rounds:
                print(f"  WARNING: No valid folds for scenario {scenario_id}. Skipping.")
                continue

        fold_results = {}
        fold_split_stats = []

        for round_pos, (round_i, train_idx, val_idx, test_idx, dropped_idx, split_payload) in enumerate(eval_rounds, 1):
            round_label = "Split" if split_mode == "single_90_10" else "Fold"
            print(f"\n  {round_label} {round_pos}/{len(eval_rounds)} (id={round_i})")

            train_compounds = set(df.iloc[train_idx]['chembl_id'])
            test_compounds = set(df.iloc[test_idx]['chembl_id'])
            leaked = train_compounds & test_compounds

            stats = {
                'train_size': len(train_idx),
                'val_size': len(val_idx),
                'test_size': len(test_idx),
                'dropped_size': len(dropped_idx),
                'test_compounds': len(test_compounds),
                'leaked_compounds': len(leaked),
                'leak_pct': 100 * len(leaked) / len(test_compounds) if test_compounds else 0
            }
            if len(df) > 0:
                stats['test_fraction_total'] = float(len(test_idx) / len(df))
                stats['dropped_fraction_total'] = float(len(dropped_idx) / len(df))
            used_rows = len(train_idx) + len(test_idx)
            stats['test_fraction_used'] = float(len(test_idx) / used_rows) if used_rows > 0 else 0.0
            if scenario_id in ('kinase', 'new_compound_new_kinase'):
                train_kinases = set(df.iloc[train_idx]['target_kinase'])
                test_kinases_set = set(df.iloc[test_idx]['target_kinase'])
                stats['test_kinases'] = len(test_kinases_set)
                stats['leaked_kinases'] = len(train_kinases & test_kinases_set)
            fold_split_stats.append(stats)

            print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}, Dropped: {len(dropped_idx)}")
            print(f"  Test compounds: {len(test_compounds)}, Leaked: {len(leaked)} ({stats['leak_pct']:.1f}%)")
            if split_payload is not None:
                print(f"  Test fraction (used rows): {split_payload['test_fraction_used']:.4f}")
                print(f"  Test fraction (all rows):  {split_payload['test_fraction_total']:.4f}")
            if 'leaked_kinases' in stats:
                print(f"  Test kinases: {stats['test_kinases']}, Leaked kinases: {stats['leaked_kinases']}")

            per_run_metadata = dict(dataset_metadata or {})
            per_run_metadata.update({
                "scenario_id": scenario_id,
                "scenario_name": scenario_name,
                "scenario_key": scenario_key,
                "round_id": int(round_i),
                "split_mode": split_mode,
                "test_fraction_target": float(test_fraction),
                "test_fraction_used": float(stats['test_fraction_used']),
                "test_fraction_total": float(stats.get('test_fraction_total', 0.0)),
            })
            artifact_scenario_id = scenario_id if split_mode == "single_90_10" else f"{scenario_id}/fold_{round_i}"
            results = train_and_evaluate(
                df,
                train_idx,
                test_idx,
                seed=seed,
                fp_cache=fp_cache,
                save_models=save_models,
                model_output_dir=model_output_dir,
                scenario_id=artifact_scenario_id,
                dropped_idx=dropped_idx,
                dataset_metadata=per_run_metadata,
                save_knn_features=save_knn_features,
            )
            if results is not None:
                fold_results[round_i] = {
                    'metrics': results
                }
                knn_auroc = f", AUROC={results['KNN']['auroc']:.4f}" if not np.isnan(results['KNN'].get('auroc', float('nan'))) else ""
                mlp_auroc = f", AUROC={results['MLP']['auroc']:.4f}" if not np.isnan(results['MLP'].get('auroc', float('nan'))) else ""
                print(f"  KNN: Acc={results['KNN']['accuracy']:.4f}, MCC={results['KNN']['mcc']:.4f}, F1={results['KNN']['f1']:.4f}{knn_auroc}")
                print(f"  MLP: Acc={results['MLP']['accuracy']:.4f}, MCC={results['MLP']['mcc']:.4f}, F1={results['MLP']['f1']:.4f}{mlp_auroc}")

        # Aggregate across folds
        if fold_results:
            # Aggregate split stats for plotting/summary
            agg_split = {}
            for k in [
                'train_size', 'val_size', 'test_size', 'dropped_size',
                'test_compounds', 'leaked_compounds', 'leak_pct',
                'test_kinases', 'leaked_kinases',
                'test_fraction_used', 'test_fraction_total', 'dropped_fraction_total',
            ]:
                vals = [s[k] for s in fold_split_stats if k in s]
                if vals:
                    agg_split[k] = int(round(np.mean(vals))) if k.endswith('size') or 'compounds' in k or 'kinases' in k else float(np.mean(vals))
                    agg_split[f'{k}_std'] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            split_stats[scenario_key] = agg_split

            if len(fold_results) == 1:
                # Single fold: use metrics directly
                single = list(fold_results.values())[0]['metrics']
                all_results[scenario_key] = single
            else:
                # Multi-fold: mean +/- std
                aggregated = {}
                for model in ['KNN', 'MLP']:
                    model_agg = {}
                    for metric in ['accuracy', 'mcc', 'f1', 'precision', 'recall', 'auroc']:
                        values = [r['metrics'][model][metric] for r in fold_results.values()
                                  if not np.isnan(r['metrics'][model].get(metric, float('nan')))]
                        if values:
                            model_agg[metric] = float(np.mean(values))
                            model_agg[f'{metric}_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                        else:
                            model_agg[metric] = float('nan')
                            model_agg[f'{metric}_std'] = 0.0
                    model_agg['fold_results'] = {
                        str(f_idx): {
                            'metrics': f_data['metrics'][model]
                        }
                        for f_idx, f_data in fold_results.items()
                    }
                    model_agg['n_folds'] = len(fold_results)
                    aggregated[model] = model_agg
                aggregated['_artifacts_by_round'] = {
                    str(f_idx): f_data['metrics'].get('_artifacts', {})
                    for f_idx, f_data in fold_results.items()
                }
                all_results[scenario_key] = aggregated

                # Print aggregate summary
                print(f"\n  --- Aggregate ({len(fold_results)} folds) ---")
                for model in ['KNN', 'MLP']:
                    m = aggregated[model]
                    auroc_str = f", AUROC={m['auroc']:.4f}" if not np.isnan(m.get('auroc', float('nan'))) else ""
                    print(f"  {model}: Acc={m['accuracy']:.4f}+/-{m['accuracy_std']:.4f}, "
                          f"MCC={m['mcc']:.4f}+/-{m['mcc_std']:.4f}, "
                          f"F1={m['f1']:.4f}+/-{m['f1_std']:.4f}{auroc_str}")

            if scenario_key not in completed_scenarios:
                completed_scenarios.append(scenario_key)
            if checkpoint_path:
                write_json(checkpoint_path, {
                    'status': 'running',
                    'config': {
                        'seed': seed,
                        'split_mode': split_mode,
                        'test_fraction': float(test_fraction),
                        'n_folds': n_folds,
                        'scenarios': [s for s, _, _ in scenarios_config],
                        'split_protocol_version': SPLIT_PROTOCOL_VERSION,
                    },
                    'completed_scenarios': completed_scenarios,
                    'all_results': all_results,
                    'split_stats': split_stats
                })
                print(
                    f"  [checkpoint] Saved {len(completed_scenarios)}/{len(scenarios_config)} scenarios "
                    f"-> {checkpoint_path}"
                )

    if checkpoint_path:
        write_json(checkpoint_path, {
            'status': 'completed' if len(completed_scenarios) == len(scenarios_config) else 'partial',
            'config': {
                'seed': seed,
                'split_mode': split_mode,
                'test_fraction': float(test_fraction),
                'n_folds': n_folds,
                'scenarios': [s for s, _, _ in scenarios_config],
                'split_protocol_version': SPLIT_PROTOCOL_VERSION,
            },
            'completed_scenarios': completed_scenarios,
            'all_results': all_results,
            'split_stats': split_stats
        })

    return all_results, split_stats


def _get_metric(all_results: dict, scenario_key: str, model: str, metric: str):
    """Extract metric value from results dict (works for single and multi-fold)."""
    return all_results[scenario_key][model][metric]


def _get_metric_std(all_results: dict, scenario_key: str, model: str, metric: str):
    """Extract metric std from results dict (returns 0 for single-fold)."""
    return all_results[scenario_key][model].get(f'{metric}_std', 0.0)


def _is_multi_round(all_results: dict):
    """Check if results contain multi-fold aggregation."""
    first_scenario = list(all_results.values())[0]
    knn = first_scenario.get('KNN', {})
    return 'n_folds' in knn


def plot_inflated_vs_real(all_results: dict, output_dir: str = '.', prefix: str = ''):
    """
    Create chart comparing Inflated Performance vs Real Performance.
    Shows the drop in MCC and Accuracy between random split and generalization.
    """

    inflated_key = 'Random Split\n(Original)'
    real_key = 'New Comp.\n+ New Kinase'
    if inflated_key not in all_results or real_key not in all_results:
        print("Skipping inflated-vs-real plot: requires both random and new_compound_new_kinase scenarios.")
        return
    multi_round = _is_multi_round(all_results)

    models = ['KNN', 'MLP']
    metrics_data = {}
    for model in models:
        for label, key in [('inflated', inflated_key), ('real', real_key)]:
            for metric in ['mcc', 'accuracy']:
                val = _get_metric(all_results, key, model, metric)
                std = _get_metric_std(all_results, key, model, metric) if multi_round else 0
                metrics_data[f'{model}_{metric}_{label}'] = val
                metrics_data[f'{model}_{metric}_{label}_std'] = std

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    x = np.arange(2)  # KNN, MLP
    width = 0.35

    for ax_idx, (metric, metric_label) in enumerate([('mcc', 'MCC (Matthews Correlation Coefficient)'),
                                                       ('accuracy', 'Accuracy')]):
        ax = axes[ax_idx]

        inflated_vals = [metrics_data[f'{m}_{metric}_inflated'] for m in models]
        real_vals = [metrics_data[f'{m}_{metric}_real'] for m in models]
        inflated_stds = [metrics_data[f'{m}_{metric}_inflated_std'] for m in models]
        real_stds = [metrics_data[f'{m}_{metric}_real_std'] for m in models]

        yerr1 = inflated_stds if multi_round else None
        yerr2 = real_stds if multi_round else None

        bars1 = ax.bar(x - width/2, inflated_vals, width,
                        yerr=yerr1, capsize=4,
                        label=f'{metric_label.split(" ")[0]} Reported (Random Split)',
                        color='#3498db', edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x + width/2, real_vals, width,
                        yerr=yerr2, capsize=4,
                        label=f'{metric_label.split(" ")[0]} Real (Generalization)',
                        color='#e74c3c', edgecolor='black', linewidth=1.5)

        ax.set_ylabel(metric_label, fontsize=12)
        ax.set_title(f'Inflated vs Real Performance\n({metric_label.split(" ")[0]})', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['KNN', 'MLP'], fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.set_ylim(0, 1.0)

        # Add values on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                            xy=(bar.get_x() + bar.get_width()/2, height),
                            xytext=(0, 5), textcoords="offset points",
                            ha='center', va='bottom', fontsize=12, fontweight='bold',
                            color='#2c3e50')

        # Drop arrows and percentage
        for i, (inf, real) in enumerate(zip(inflated_vals, real_vals)):
            if inf > 0:
                drop_pct = 100 * (inf - real) / inf
            else:
                drop_pct = 0
            mid_y = (inf + real) / 2

            ax.annotate('',
                        xy=(i + width/2, real + 0.02),
                        xytext=(i - width/2, inf - 0.02),
                        arrowprops=dict(arrowstyle='->', color='#8e44ad', lw=2.5))

            offset_y = 0.08 if metric == 'mcc' else 0.06
            ax.text(i, mid_y + offset_y, f'↓ {drop_pct:.0f}%',
                    ha='center', va='center', fontsize=14, fontweight='bold',
                    color='#8e44ad',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                             edgecolor='#8e44ad', alpha=0.9))

        # Reference line
        ref_label = 'MCC = 0.5' if metric == 'mcc' else 'Random = 0.5'
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.text(1.5, 0.52, ref_label, fontsize=9, color='gray', ha='right')

    plt.tight_layout()
    filename = f'{prefix}07_inflated_vs_real_performance.png' if prefix else '07_inflated_vs_real_performance.png'
    plt.savefig(f'{output_dir}/{filename}', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {output_dir}/{filename}")


# =============================================================================
# MAIN
# =============================================================================

def run_single_dataset(
    dataset_type: str,
    output_dir: str,
    force: bool = False,
    seed: int = DEFAULT_SPLIT_SEED,
    n_folds: int = DEFAULT_N_FOLDS,
    scenarios: list = None,
    keep_monotonic: bool = False,
    filter_monotonic_compounds: bool = False,
    split_mode: str = DEFAULT_SPLIT_MODE,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    input_path: str = None,
    use_without_test_input: bool = True,
    without_test_dir: str = "scaffolds_splits/output",
    save_models: bool = True,
    save_knn_features: bool = False,
    model_output_dir: str = None,
    s4_restarts: int = DEFAULT_S4_RESTARTS,
):
    """Run analysis for a single dataset type."""
    data_path = resolve_dataset_input_path(
        dataset_type=dataset_type,
        input_path=input_path,
        use_without_test_input=use_without_test_input,
        without_test_dir=without_test_dir,
    )
    if not data_path:
        print(f"Error: Unknown dataset type '{dataset_type}'")
        return None

    if model_output_dir is None:
        model_output_dir = os.path.join(output_dir, "models")

    # Output is already dataset-scoped (e.g. .../non_human, .../human, .../all).
    # Keep filenames canonical without dataset prefix.
    prefix = ""

    # Cache and checkpoint files
    json_file = os.path.join(output_dir, f'{prefix}split_comparison_results.json')
    leakage_checkpoint_file = os.path.join(output_dir, f'{prefix}leakage_diagnostics_checkpoint.json')
    comparison_checkpoint_file = os.path.join(output_dir, f'{prefix}comparison_progress_checkpoint.json')

    if os.path.exists(json_file) and not force:
        print(f"\n[CACHE] Results already exist for {dataset_type}: {json_file}")
        print(f"        Use --force to recalculate.")
        return None

    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"ANALYSIS: {dataset_type.upper()}")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Input path: {data_path}")
    print(f"Split mode: {split_mode}")
    if split_mode == "single_90_10":
        print(f"Train/Test target: {1.0-test_fraction:.0%}/{test_fraction:.0%}")
    else:
        print(f"k-folds: {n_folds}")
    print(f"Model artifacts directory: {model_output_dir}")
    print(f"Save models: {save_models} (save_knn_features={save_knn_features})")

    if force:
        for ckpt in [leakage_checkpoint_file, comparison_checkpoint_file]:
            if os.path.exists(ckpt):
                try:
                    os.remove(ckpt)
                    print(f"[FORCE] Removed checkpoint: {ckpt}")
                except OSError:
                    pass

    print(f"\n[1/4] Loading dataset: {dataset_type}...")
    try:
        df = load_dataset(data_path, keep_monotonic=keep_monotonic,
                          filter_monotonic_compounds=filter_monotonic_compounds)
    except FileNotFoundError:
        print(f"  ERROR: File not found: {data_path}")
        return None

    print(f"  Total: {len(df)} rows, {df['chembl_id'].nunique()} compounds, {df['target_kinase'].nunique()} kinases")

    # Generate (or reuse) legacy leakage diagnostics (plots 01-05)
    print("\n[2/4] Leakage diagnostics (01-05)")
    print("\n" + "-" * 50)
    print("GENERATING LEAKAGE DIAGNOSTICS (01-05)")
    print("-" * 50)
    leakage_artifacts = None
    if split_mode == "single_90_10":
        leakage_test_fraction = float(test_fraction)
        leakage_val_fraction = 0.0
    else:
        leakage_test_fraction = 1.0 / n_folds
        leakage_val_fraction = 1.0 / n_folds if n_folds > 2 else 0.0

    if not force:
        leakage_ckpt = read_json(leakage_checkpoint_file)
        if leakage_ckpt:
            cfg_matches = (
                leakage_ckpt.get('dataset') == dataset_type and
                leakage_ckpt.get('seed') == seed and
                leakage_ckpt.get('split_mode', 'kfold_cv') == split_mode and
                abs(float(leakage_ckpt.get('test_fraction', -1.0)) - leakage_test_fraction) < 1e-12 and
                abs(float(leakage_ckpt.get('val_fraction', -1.0)) - leakage_val_fraction) < 1e-12
            )
            if split_mode == "kfold_cv":
                cfg_matches = cfg_matches and leakage_ckpt.get('n_folds') == n_folds
            cached_artifacts = leakage_ckpt.get('artifacts', {})
            missing = [
                p for p in cached_artifacts.values()
                if p and not os.path.exists(p)
            ]
            if cfg_matches and not missing and cached_artifacts:
                leakage_artifacts = cached_artifacts
                print(f"[CHECKPOINT] Reusing leakage diagnostics from {leakage_checkpoint_file}")
            elif cfg_matches and missing:
                print(f"[CHECKPOINT] Leakage checkpoint found but {len(missing)} artifact(s) missing. Recomputing...")
            elif not cfg_matches:
                print("[CHECKPOINT] Leakage checkpoint ignored (configuration mismatch).")

    if leakage_artifacts is None:
        leakage_artifacts = run_leakage_diagnostics(
            df=df,
            output_dir=output_dir,
            prefix=prefix,
            seed=seed,
            test_fraction=leakage_test_fraction,
            val_fraction=leakage_val_fraction,
            knn_k=5,
            similarity_sample_size=500
        )
        write_json(leakage_checkpoint_file, {
            'status': 'completed',
            'dataset': dataset_type,
            'seed': seed,
            'n_folds': n_folds,
            'split_mode': split_mode,
            'test_fraction': leakage_test_fraction,
            'val_fraction': leakage_val_fraction,
            'artifacts': leakage_artifacts
        })
        print(f"[CHECKPOINT] Leakage diagnostics saved: {leakage_checkpoint_file}")

    for artifact_key, artifact_path in leakage_artifacts.items():
        if artifact_path:
            print(f"  [{artifact_key}] {artifact_path}")

    print("\n[3/4] Split comparison (06-07)")
    dataset_metadata = {
        "dataset_type": dataset_type,
        "input_path": data_path,
        "split_mode": split_mode,
        "test_fraction_target": float(test_fraction),
        "monotonic_kinase_filter": not keep_monotonic,
        "monotonic_compound_filter": filter_monotonic_compounds,
    }
    all_results, split_stats = run_comparison(
        df,
        output_dir,
        seed=seed,
        n_folds=n_folds,
        scenarios=scenarios,
        checkpoint_path=comparison_checkpoint_file,
        force=force,
        split_mode=split_mode,
        test_fraction=test_fraction,
        save_models=save_models,
        save_knn_features=save_knn_features,
        model_output_dir=model_output_dir,
        dataset_metadata=dataset_metadata,
        s4_restarts=s4_restarts,
    )
    if not all_results:
        print("  No results generated.")
        return None

    print("\n[4/4] Finalizing artifacts and report")
    # Generate plots
    print("\n" + "-" * 50)
    print("GENERATING PLOTS")
    print("-" * 50)

    plot_comparison(all_results, split_stats, output_dir, prefix)

    # Write README explaining the logic behind all generated images
    readme_path = write_images_readme(
        output_dir=output_dir,
        dataset_type=dataset_type,
        seed=seed,
        n_folds=n_folds,
        scenarios=scenarios if scenarios else DEFAULT_SCENARIOS,
        split_protocol_version=SPLIT_PROTOCOL_VERSION,
        affinity_threshold_pchembl=DEFAULT_AFFINITY_THRESHOLD.threshold_pchembl,
        leakage_artifacts=leakage_artifacts,
        prefix=prefix
    )
    print(f"README saved: {readme_path}")

    # Save JSON results
    json_results = {}
    multi_round = _is_multi_round(all_results)
    for scenario_key in all_results:
        scenario_json = {}
        for model in ['KNN', 'MLP']:
            m = all_results[scenario_key][model]
            scenario_json[model] = {}
            for metric in ['accuracy', 'mcc', 'f1', 'precision', 'recall', 'auroc']:
                val = m.get(metric)
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    scenario_json[model][metric] = val
            if multi_round:
                for metric in ['accuracy', 'mcc', 'f1', 'precision', 'recall', 'auroc']:
                    std_val = m.get(f'{metric}_std', 0.0)
                    if not (isinstance(std_val, float) and np.isnan(std_val)):
                        scenario_json[model][f'{metric}_std'] = std_val
                scenario_json[model]['n_folds'] = m.get('n_folds', 1)
                if 'fold_results' in m:
                    scenario_json[model]['fold_results'] = m['fold_results']
        if '_artifacts' in all_results[scenario_key]:
            scenario_json['_artifacts'] = all_results[scenario_key]['_artifacts']
        if '_artifacts_by_round' in all_results[scenario_key]:
            scenario_json['_artifacts_by_round'] = all_results[scenario_key]['_artifacts_by_round']
        json_results[scenario_key.replace('\n', ' ')] = scenario_json

    json_file = os.path.join(output_dir, f'{prefix}split_comparison_results.json')
    with open(json_file, 'w') as f:
        json.dump({
            'dataset': dataset_type,
            'input_path': data_path,
            'split_mode': split_mode,
            'test_fraction': float(test_fraction),
            'split_protocol_version': SPLIT_PROTOCOL_VERSION,
            'model_seed': seed,
            'n_folds': n_folds,
            'model_artifacts_dir': model_output_dir if save_models else None,
            'save_models': save_models,
            'save_knn_features': save_knn_features,
            'monotonic_kinase_filter': not keep_monotonic,
            'monotonic_compound_filter': filter_monotonic_compounds,
            'scenarios': scenarios if scenarios else DEFAULT_SCENARIOS,
            'affinity_threshold_pchembl': DEFAULT_AFFINITY_THRESHOLD.threshold_pchembl,
            'auxiliary_artifacts': leakage_artifacts,
            'checkpoints': {
                'leakage_diagnostics': leakage_checkpoint_file,
                'split_comparison_progress': comparison_checkpoint_file
            },
            'documentation': {
                'readme': readme_path
            },
            'results': json_results,
            'split_stats': {k.replace('\n', ' '): v for k, v in split_stats.items()}
        }, f, indent=2)
    print(f"Results saved: {json_file}")

    # Conclusion
    print("\n" + "-" * 50)
    print(f"CONCLUSION ({dataset_type.upper()})")
    print("-" * 50)

    random_key = 'Random Split\n(Original)'
    new_key = 'New Comp.\n+ New Kinase'
    if random_key in all_results and new_key in all_results:
        orig_knn_mcc = _get_metric(all_results, random_key, 'KNN', 'mcc')
        orig_mlp_mcc = _get_metric(all_results, random_key, 'MLP', 'mcc')
        new_knn_mcc = _get_metric(all_results, new_key, 'KNN', 'mcc')
        new_mlp_mcc = _get_metric(all_results, new_key, 'MLP', 'mcc')

        print(f"\nKNN MCC drop:  {orig_knn_mcc:.3f} -> {new_knn_mcc:.3f} (delta = {new_knn_mcc - orig_knn_mcc:+.3f})")
        print(f"MLP MCC drop:  {orig_mlp_mcc:.3f} -> {new_mlp_mcc:.3f} (delta = {new_mlp_mcc - orig_mlp_mcc:+.3f})")

        if multi_round:
            orig_knn_std = _get_metric_std(all_results, random_key, 'KNN', 'mcc')
            new_knn_std = _get_metric_std(all_results, new_key, 'KNN', 'mcc')
            orig_mlp_std = _get_metric_std(all_results, random_key, 'MLP', 'mcc')
            new_mlp_std = _get_metric_std(all_results, new_key, 'MLP', 'mcc')
            print(f"  KNN: {orig_knn_mcc:.3f}+/-{orig_knn_std:.3f} -> {new_knn_mcc:.3f}+/-{new_knn_std:.3f}")
            print(f"  MLP: {orig_mlp_mcc:.3f}+/-{orig_mlp_std:.3f} -> {new_mlp_mcc:.3f}+/-{new_mlp_std:.3f}")

            # Wilcoxon signed-rank test (paired comparison across folds)
            random_data = all_results[random_key]
            new_data = all_results[new_key]
            for model in ['KNN', 'MLP']:
                if 'fold_results' in random_data[model] and 'fold_results' in new_data[model]:
                    r_keys = sorted(random_data[model]['fold_results'].keys(), key=int)
                    n_keys = sorted(new_data[model]['fold_results'].keys(), key=int)
                    if len(r_keys) >= 5 and len(n_keys) >= 5:
                        r_mcc = [random_data[model]['fold_results'][k]['metrics']['mcc'] for k in r_keys]
                        n_mcc = [new_data[model]['fold_results'][k]['metrics']['mcc'] for k in n_keys]
                        try:
                            stat, p_val = wilcoxon(r_mcc, n_mcc, alternative='greater')
                            print(f"  {model} Wilcoxon signed-rank (random > new_comp_new_kin): "
                                  f"stat={stat:.1f}, p={p_val:.4f}"
                                  f"{' ***' if p_val < 0.01 else ' **' if p_val < 0.05 else ' (n.s.)'}")
                        except ValueError as e:
                            print(f"  {model} Wilcoxon test: could not compute ({e})")
                    else:
                        print(f"  {model} Wilcoxon test: requires >= 5 folds (have {min(len(r_keys), len(n_keys))})")

        # Document blind-target limitation
        print(f"\n  NOTE: One-hot kinase encoding creates a 'blind target' condition for")
        print(f"  cold-kinase scenarios (kinase, new_compound_new_kinase). Unseen kinases")
        print(f"  receive all-zero vectors, so the performance drop reflects BOTH leakage")
        print(f"  removal AND loss of target representation.")
    else:
        print("\nMCC drop summary skipped: requires both random and new_compound_new_kinase scenarios.")

    return all_results


def plot_comparison(all_results: dict, split_stats: dict, output_dir: str = '.', prefix: str = ''):
    """Generate comparison plots."""

    scenarios = list(all_results.keys())
    multi_round = _is_multi_round(all_results)

    knn_acc = [_get_metric(all_results, s, 'KNN', 'accuracy') for s in scenarios]
    knn_mcc = [_get_metric(all_results, s, 'KNN', 'mcc') for s in scenarios]
    mlp_acc = [_get_metric(all_results, s, 'MLP', 'accuracy') for s in scenarios]
    mlp_mcc = [_get_metric(all_results, s, 'MLP', 'mcc') for s in scenarios]

    knn_acc_std = [_get_metric_std(all_results, s, 'KNN', 'accuracy') for s in scenarios] if multi_round else None
    knn_mcc_std = [_get_metric_std(all_results, s, 'KNN', 'mcc') for s in scenarios] if multi_round else None
    mlp_acc_std = [_get_metric_std(all_results, s, 'MLP', 'accuracy') for s in scenarios] if multi_round else None
    mlp_mcc_std = [_get_metric_std(all_results, s, 'MLP', 'mcc') for s in scenarios] if multi_round else None

    test_sizes = [split_stats[s]['test_size'] for s in scenarios]
    test_compounds = [split_stats[s]['test_compounds'] for s in scenarios]

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    x = np.arange(len(scenarios))
    width = 0.35

    # Plot 1: Accuracy
    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, knn_acc, width, label='KNN', color='#3498db', edgecolor='black',
                     yerr=knn_acc_std, capsize=4)
    bars2 = ax1.bar(x + width/2, mlp_acc, width, label='MLP', color='#e74c3c', edgecolor='black',
                     yerr=mlp_acc_std, capsize=4)

    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Accuracy by Evaluation Scenario', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, fontsize=10)
    ax1.legend(loc='upper right', fontsize=11)
    ax1.set_ylim(0, 1.15)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    for i, (ts, tc) in enumerate(zip(test_sizes, test_compounds)):
        ax1.annotate(f'n={ts}\n({tc} comp.)',
                    xy=(i, 0), xytext=(0, -25),
                    textcoords="offset points",
                    ha='center', va='top', fontsize=9, color='gray')

    # Plot 2: MCC
    ax2 = axes[1]
    bars3 = ax2.bar(x - width/2, knn_mcc, width, label='KNN', color='#3498db', edgecolor='black',
                     yerr=knn_mcc_std, capsize=4)
    bars4 = ax2.bar(x + width/2, mlp_mcc, width, label='MLP', color='#e74c3c', edgecolor='black',
                     yerr=mlp_mcc_std, capsize=4)

    ax2.set_ylabel('MCC', fontsize=12)
    ax2.set_title('MCC by Evaluation Scenario', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios, fontsize=10)
    ax2.legend(loc='upper right', fontsize=11)
    ax2.set_ylim(0, 1.0)

    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    for i, (ts, tc) in enumerate(zip(test_sizes, test_compounds)):
        ax2.annotate(f'n={ts}\n({tc} comp.)',
                    xy=(i, 0), xytext=(0, -25),
                    textcoords="offset points",
                    ha='center', va='top', fontsize=9, color='gray')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    filename = f'{prefix}06_split_comparison.png' if prefix else '06_split_comparison.png'
    plt.savefig(f'{output_dir}/{filename}', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved: {output_dir}/{filename}")

    # Additional plot: Inflated vs Real
    plot_inflated_vs_real(all_results, output_dir, prefix)


def print_comparative_summary(all_dataset_results: dict):
    """Print comparative summary across datasets."""
    if not all_dataset_results:
        return

    print("\n" + "=" * 90)
    print("COMPARATIVE SUMMARY: ALL DATASETS")
    print("=" * 90)

    print(f"\n{'Scenario + Model':<42} ", end="")
    for dataset in all_dataset_results.keys():
        print(f"{dataset:>15}", end="")
    print()
    print("-" * (42 + 15 * len(all_dataset_results)))

    first_result = list(all_dataset_results.values())[0]
    if not first_result:
        return

    scenarios = list(first_result.keys())
    for scenario in scenarios:
        scenario_clean = scenario.replace('\n', ' ')
        for model in ['KNN', 'MLP']:
            label = f"{scenario_clean} [{model}]"
            print(f"{label:<42} ", end="")
            for dataset, results in all_dataset_results.items():
                if results and scenario in results:
                    metrics = results[scenario].get(model, {})
                    mcc = metrics.get('mcc', 0.0)
                    mcc_std = metrics.get('mcc_std', 0.0)
                    if mcc_std > 0:
                        print(f"{mcc:.3f}±{mcc_std:.3f}".rjust(15), end="")
                    else:
                        print(f"{mcc:.3f}".rjust(15), end="")
                else:
                    print(f"{'N/A':>15}", end="")
            print()

    print("=" * 90)
    print("(Values shown: MCC ± std)")


def main():
    parser = argparse.ArgumentParser(
        description="Split Comparison Analysis for Data Leakage Assessment (KNN/MLP)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single dataset with default scenarios (single 90/10 split per scenario)
  %(prog)s --dataset non_human

  # Run all datasets (non_human, human, all)
  %(prog)s --run_all

  # Run specific scenarios only
  %(prog)s --dataset non_human --scenarios random,compound

  # Legacy 5-fold CV mode (60/20/20 train/val/test)
  %(prog)s --dataset human --n_folds 5

  # Explicit 90/10 split using without-test inputs and saved model artifacts
  %(prog)s --dataset non_human --scenarios all --split_mode single_90_10 --test_fraction 0.10

  # Debug mode for detailed error traces
  %(prog)s --dataset all --debug

Available scenarios (Pahikkala et al. 2015 framework):
  - new_compound_new_kinase : S4 - both compound and kinase unseen (hardest)
  - compound                : S2 - compound unseen (cold-drug)
  - kinase                  : S3 - kinase unseen (cold-target)
  - random                  : S1 - random split (baseline with leakage)
        """
    )

    parser.add_argument(
        '--dataset', '-d',
        choices=['human', 'non_human', 'all'],
        help='Dataset type to use'
    )
    parser.add_argument(
        '--run_all',
        action='store_true',
        help='Run analysis for all supported datasets (non_human, human, all)'
    )
    parser.add_argument(
        '--scenarios', '-s',
        type=str,
        default=None,
        help='Comma-separated list of scenarios to run (default: all). '
             'Use "all" to run all scenarios. '
             'Options: new_compound_new_kinase, scaffold, compound, kinase, random'
    )
    parser.add_argument(
        '--output_dir', '-o',
        default='./results/split_comparison_analysis',
        help='Output directory for results (default: ./results/split_comparison_analysis)'
    )
    parser.add_argument(
        '--input_path',
        type=str,
        default=None,
        help='Optional explicit dataset TSV path. If provided, it overrides dataset defaults.'
    )
    parser.add_argument(
        '--without_test_dir',
        type=str,
        default='scaffolds_splits/output',
        help='Directory containing *_input_without_test.tsv files (default: scaffolds_splits/output)'
    )
    parser.add_argument(
        '--use_without_test_input',
        dest='use_without_test_input',
        action='store_true',
        default=True,
        help='Prefer *_input_without_test.tsv for human/non_human datasets when available (default: enabled).'
    )
    parser.add_argument(
        '--no_without_test_input',
        dest='use_without_test_input',
        action='store_false',
        help='Disable automatic use of *_input_without_test.tsv and use default dataset paths.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=DEFAULT_SPLIT_SEED,
        help=f'Fixed random seed for model + CV fold assignment (default: {DEFAULT_SPLIT_SEED})'
    )
    parser.add_argument(
        '--split_mode',
        choices=['single_90_10', 'kfold_cv'],
        default=DEFAULT_SPLIT_MODE,
        help=f"Split protocol to use (default: {DEFAULT_SPLIT_MODE})"
    )
    parser.add_argument(
        '--test_fraction',
        type=float,
        default=DEFAULT_TEST_FRACTION,
        help=f'Test fraction for single-split mode (default: {DEFAULT_TEST_FRACTION:.2f})'
    )
    parser.add_argument(
        '--n_folds',
        type=int,
        default=DEFAULT_N_FOLDS,
        help=f'Number of CV folds for split_mode=kfold_cv (default: {DEFAULT_N_FOLDS}). '
             'k=10 gives 80/10/10 train/val/test, k=5 gives 60/20/20.'
    )
    parser.add_argument(
        '--s4_restarts',
        type=int,
        default=DEFAULT_S4_RESTARTS,
        help=f'Number of random restarts used to optimize strict S4 single-split quality (default: {DEFAULT_S4_RESTARTS}).'
    )
    parser.add_argument(
        '--keep_monotonic',
        action='store_true',
        help='Keep kinases with monotonic activity profiles (100%% active or inactive). '
             'By default these are removed as they provide no discriminative signal.'
    )
    parser.add_argument(
        '--filter_monotonic_compounds',
        action='store_true',
        help='Remove compounds with monotonic activity profiles (pan-active or pan-inactive). '
             'These are compounds that are 100%% active or 100%% inactive across all kinases tested.'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force recalculation even if results exist'
    )
    parser.add_argument(
        '--save_models',
        dest='save_models',
        action='store_true',
        default=True,
        help='Persist model/scaler/split artifacts for each scenario (default: enabled).'
    )
    parser.add_argument(
        '--no_save_models',
        dest='save_models',
        action='store_false',
        help='Disable model artifact persistence.'
    )
    parser.add_argument(
        '--model_output_dir',
        type=str,
        default=None,
        help='Base directory for model artifacts. Default: <output_dir>/<dataset>/models'
    )
    parser.add_argument(
        '--save_knn_features',
        action='store_true',
        help='Also save scaled KNN training feature matrices (larger artifacts).'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with full error tracebacks'
    )

    args = parser.parse_args()

    if not args.run_all and args.dataset is None:
        parser.error("--dataset is required unless --run_all is specified")
    if args.run_all and args.input_path:
        parser.error("--input_path cannot be used with --run_all (it applies to a single dataset only)")
    if args.split_mode == 'kfold_cv' and args.n_folds < 2:
        parser.error("--n_folds must be >= 2 when --split_mode kfold_cv")
    if args.split_mode == 'single_90_10' and not (0.0 < args.test_fraction < 1.0):
        parser.error("--test_fraction must be in (0, 1) when --split_mode single_90_10")
    if args.s4_restarts < 1:
        parser.error("--s4_restarts must be >= 1")

    if args.scenarios:
        if args.scenarios.strip().lower() == 'all':
            scenarios = DEFAULT_SCENARIOS
        else:
            scenarios = [s.strip() for s in args.scenarios.split(',')]
            for s in scenarios:
                if s not in AVAILABLE_SCENARIOS:
                    parser.error(f"Unknown scenario: {s}. Available: {list(AVAILABLE_SCENARIOS.keys())}")
    else:
        scenarios = DEFAULT_SCENARIOS

    if args.run_all:
        datasets_to_run = ['non_human', 'human', 'all']
    else:
        datasets_to_run = [args.dataset]

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("SPLIT COMPARISON ANALYSIS (KNN/MLP)")
    print("=" * 70)
    print(f"Datasets:         {datasets_to_run}")
    print(f"Scenarios:        {scenarios}")
    print(f"Output root dir:  {args.output_dir}")
    print(f"Split protocol:   {SPLIT_PROTOCOL_VERSION}")
    print(f"Split mode:       {args.split_mode}")
    print(f"Model seed:       {args.seed} (fixed)")
    if args.split_mode == "kfold_cv":
        train_pct = (args.n_folds - 2) / args.n_folds
        val_pct = 1.0 / args.n_folds
        test_pct = 1.0 / args.n_folds
        print(f"k-folds:          {args.n_folds}")
        print(f"Split ratio:      {train_pct:.0%}/{val_pct:.0%}/{test_pct:.0%} (train/val/test)")
    else:
        train_pct = 1.0 - args.test_fraction
        print(f"Split ratio:      {train_pct:.0%}/{args.test_fraction:.0%} (train/test)")
        print(f"S4 restarts:      {args.s4_restarts}")
    print(f"Without-test input: {args.use_without_test_input} (dir={args.without_test_dir})")
    if args.input_path:
        print(f"Input override:   {args.input_path}")
    print(f"Save models:      {args.save_models}")
    if args.save_models:
        model_root = args.model_output_dir if args.model_output_dir else "<dataset_output_dir>/models"
        print(f"Model output dir: {model_root}")
    print(f"Save KNN feats:   {args.save_knn_features}")
    mono_kin_status = 'OFF (keeping all)' if args.keep_monotonic else 'ON (removing 100% single-class kinases)'
    mono_cmp_status = 'ON (removing pan-active/pan-inactive)' if args.filter_monotonic_compounds else 'OFF (keeping all)'
    print(f"Monotonic kinase filter:   {mono_kin_status}")
    print(f"Monotonic compound filter: {mono_cmp_status}")
    print(f"Force recalc:     {args.force}")
    print(f"Debug mode:       {args.debug}")
    print("=" * 70)

    all_dataset_results = {}
    all_dataset_times = {}
    total_start_time = time.time()

    dataset_output_dirs = {}
    for i, dataset_type in enumerate(datasets_to_run, 1):
        print(f"\n{'#' * 70}")
        print(f"# [{i}/{len(datasets_to_run)}] Running analysis for dataset={dataset_type}...")
        print(f"{'#' * 70}")
        dataset_output_dir = os.path.join(args.output_dir, dataset_type)
        dataset_output_dirs[dataset_type] = dataset_output_dir
        if args.model_output_dir:
            dataset_model_output_dir = os.path.join(args.model_output_dir, dataset_type)
        else:
            dataset_model_output_dir = os.path.join(dataset_output_dir, "models")
        print(f"# Output: {dataset_output_dir}")
        print(f"# Model artifacts: {dataset_model_output_dir}")

        dataset_start_time = time.time()
        try:
            result = run_single_dataset(
                dataset_type, dataset_output_dir, args.force,
                seed=args.seed,
                n_folds=args.n_folds,
                scenarios=scenarios,
                keep_monotonic=args.keep_monotonic,
                filter_monotonic_compounds=args.filter_monotonic_compounds,
                split_mode=args.split_mode,
                test_fraction=args.test_fraction,
                input_path=args.input_path if not args.run_all else None,
                use_without_test_input=args.use_without_test_input,
                without_test_dir=args.without_test_dir,
                save_models=args.save_models,
                save_knn_features=args.save_knn_features,
                model_output_dir=dataset_model_output_dir,
                s4_restarts=args.s4_restarts,
            )
            dataset_time = time.time() - dataset_start_time
            all_dataset_times[dataset_type] = dataset_time
            if result is None:
                print(f"  Skipped {dataset_type} (results already exist or data not found)")
                all_dataset_results[dataset_type] = None
            else:
                print(
                    f"  Completed {dataset_type} analysis in {format_time(dataset_time)} "
                    f"(output: {dataset_output_dir})"
                )
                all_dataset_results[dataset_type] = result
        except Exception as e:
            dataset_time = time.time() - dataset_start_time
            all_dataset_times[dataset_type] = dataset_time
            all_dataset_results[dataset_type] = None

            print(f"  Error running {dataset_type} analysis: {e}")
            if args.debug:
                print("\n" + "=" * 50)
                print("DEBUG TRACEBACK:")
                print("=" * 50)
                traceback.print_exc()
                print("=" * 50 + "\n")
            continue

    total_time = time.time() - total_start_time

    print("\n" + "=" * 70)
    print("EXECUTION TIME SUMMARY")
    print("=" * 70)
    for dataset, dataset_time in all_dataset_times.items():
        status = "OK" if all_dataset_results.get(dataset) else "FAIL/SKIP"
        print(f"  [{status}] {dataset}: {format_time(dataset_time)}")
    print("-" * 70)
    print(f"  TOTAL: {format_time(total_time)}")
    print("=" * 70)

    if len(datasets_to_run) > 1:
        valid_results = {k: v for k, v in all_dataset_results.items() if v is not None}
        if valid_results:
            print_comparative_summary(valid_results)

    print("\nAnalysis complete!")
    print(f"Results saved under: {args.output_dir}")
    for dataset_type in datasets_to_run:
        print(f"  - {dataset_type}: {dataset_output_dirs.get(dataset_type, os.path.join(args.output_dir, dataset_type))}")


if __name__ == '__main__':
    main()
