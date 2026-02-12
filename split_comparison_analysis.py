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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import faiss
from scipy.stats import wilcoxon
from sklearn.model_selection import train_test_split
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
    AVAILABLE_SCENARIOS, DEFAULT_SCENARIOS, DEFAULT_SEEDS
)

DEFAULT_SPLIT_SEED = 42
DEFAULT_N_ROUNDS = 5
DEFAULT_TEST_FRACTION = 0.10
DEFAULT_VAL_FRACTION = 0.10
DEFAULT_N_SPLIT_CANDIDATES = 25
MIN_TEST_SAMPLES = 50
SPLIT_PROTOCOL_VERSION = "80_10_10_balanced_tanimoto_candidates_v4"
MAX_DIVERSITY_COMPOUNDS = 8000
# Well-separated seeds for multi-round experiments (avoids correlated splits)
ROUND_SEEDS = DEFAULT_SEEDS  # [42, 123, 456, 789, 1024]

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


def load_dataset(filepath: str, keep_monotonic: bool = False):
    """Load dataset and create binary labels using pChEMBL threshold.

    By default, removes kinases with monotonic activity profiles (100% active
    or 100% inactive), as they provide no discriminative signal and inflate
    metrics. Use keep_monotonic=True to retain them.
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


def train_and_evaluate(df: pd.DataFrame, train_idx, test_idx, all_kinases: list,
                       seed: int = 42, fp_cache: dict = None):
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

    X_train, y_train, train_valid = prepare_features(train_df, all_kinases, fp_cache)
    X_test, y_test, test_valid = prepare_features(test_df, all_kinases, fp_cache)

    if len(X_test) == 0 or len(X_train) == 0:
        return None

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    results = {}

    # KNN via FAISS (cosine similarity with distance-weighted voting)
    print("    Training KNN (FAISS)...")
    t0 = time.time()
    knn_pred, knn_proba = faiss_knn_predict(X_train_scaled, y_train, X_test_scaled, k=5)
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


def _protein_group_column(df: pd.DataFrame) -> str:
    """Return the best available protein-group column for balancing."""
    candidates = [
        'protein_class', 'protein_family', 'kinase_group', 'kinase_family',
        'target_family', 'target_class'
    ]
    for col in candidates:
        if col in df.columns:
            return col
    return 'target_kinase'


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


def _pairwise_tanimoto_mean_sampled(fp_matrix: np.ndarray, rng: np.random.Generator, n_pairs: int = 5000):
    """
    Estimate mean pairwise Tanimoto similarity by random pair sampling.
    Uses sampled all-vs-all pairs for scalability on large sets.
    """
    n = fp_matrix.shape[0]
    if n < 2:
        return 0.0

    n_pairs = int(min(max(1000, n_pairs), n * (n - 1)))
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    neq = i != j
    if not np.any(neq):
        return 0.0
    i = i[neq]
    j = j[neq]

    a = fp_matrix[i]
    b = fp_matrix[j]
    inter = np.logical_and(a, b).sum(axis=1).astype(np.float64)
    union = np.logical_or(a, b).sum(axis=1).astype(np.float64)
    sim = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    return float(sim.mean()) if len(sim) else 0.0


def _estimate_internal_diversity(df: pd.DataFrame, idx: np.ndarray, fp_map: dict, rng: np.random.Generator):
    """Estimate internal chemical diversity (1 - mean Tanimoto) on unique compounds."""
    compounds = pd.unique(df.iloc[idx]['chembl_id'])
    compounds = [c for c in compounds if c in fp_map]
    n_total_compounds = len(compounds)
    if n_total_compounds < 2:
        return 0.0, n_total_compounds
    if n_total_compounds > MAX_DIVERSITY_COMPOUNDS:
        sample_idx = rng.choice(n_total_compounds, size=MAX_DIVERSITY_COMPOUNDS, replace=False)
        compounds = [compounds[i] for i in sample_idx]
    fp_matrix = np.stack([fp_map[c] for c in compounds], axis=0)
    mean_tanimoto = _pairwise_tanimoto_mean_sampled(fp_matrix, rng=rng, n_pairs=5000)
    return 1.0 - mean_tanimoto, n_total_compounds


def _distribution_l1(values: np.ndarray, reference_probs: dict):
    """L1 distance between empirical distribution(values) and reference_probs."""
    if len(values) == 0:
        return 1.0
    counts = pd.Series(values).value_counts(normalize=True)
    dist = 0.0
    for key, p_ref in reference_probs.items():
        p = float(counts.get(key, 0.0))
        dist += abs(p - p_ref)
    return float(0.5 * dist)


def _generate_candidate_split(df: pd.DataFrame, scenario_id: str, seed: int,
                               test_fraction: float, val_fraction: float = 0.10,
                               scaffold_map: dict = None):
    """
    Generate a candidate 80/10/10 split for one scenario.
    Returns (train_idx, val_idx, test_idx, metadata).
    """
    n = len(df)
    indices = np.arange(n, dtype=np.int64)
    labels = np.asarray(df['label'].values)
    rng = np.random.default_rng(seed)
    holdout_fraction = test_fraction + val_fraction

    if scenario_id == 'random':
        # Stratified 80/20, then split the 20% into val/test
        train_idx, holdout_idx = train_test_split(
            indices, test_size=holdout_fraction, stratify=labels, random_state=seed
        )
        holdout_labels = labels[holdout_idx]
        relative_test = test_fraction / holdout_fraction
        val_idx, test_idx = train_test_split(
            holdout_idx, test_size=relative_test, stratify=holdout_labels, random_state=seed + 1
        )
        return train_idx, val_idx, test_idx, {'strategy': 'stratified_random_80_10_10'}

    compounds = np.array(df['chembl_id'].unique())

    if scenario_id == 'scaffold':
        if scaffold_map is None:
            raise ValueError("scaffold scenario requires scaffold_map")
        row_scaffolds = df['chembl_id'].map(scaffold_map).fillna('UNKNOWN')
        unique_scaffolds = np.array(row_scaffolds.unique())
        rng.shuffle(unique_scaffolds)
        n_test = max(1, int(round(len(unique_scaffolds) * test_fraction)))
        n_val = max(1, int(round(len(unique_scaffolds) * val_fraction)))
        test_scaffolds = set(unique_scaffolds[:n_test])
        val_scaffolds = set(unique_scaffolds[n_test:n_test + n_val])
        test_mask = row_scaffolds.isin(test_scaffolds).values
        val_mask = row_scaffolds.isin(val_scaffolds).values
        train_mask = ~(test_mask | val_mask)
        return (np.where(train_mask)[0], np.where(val_mask)[0], np.where(test_mask)[0],
                {'strategy': 'scaffold_split_80_10_10',
                 'n_test_scaffolds': n_test, 'n_val_scaffolds': n_val,
                 'n_total_scaffolds': len(unique_scaffolds)})

    if scenario_id == 'compound':
        rng.shuffle(compounds)
        n_test = max(1, int(round(len(compounds) * test_fraction)))
        n_val = max(1, int(round(len(compounds) * val_fraction)))
        test_compounds = set(compounds[:n_test])
        val_compounds = set(compounds[n_test:n_test + n_val])
        test_mask = df['chembl_id'].isin(test_compounds).values
        val_mask = df['chembl_id'].isin(val_compounds).values
        train_mask = ~(test_mask | val_mask)
        return (np.where(train_mask)[0], np.where(val_mask)[0], np.where(test_mask)[0],
                {'strategy': 'compound_split_80_10_10'})

    if scenario_id == 'kinase':
        kinases = np.array(df['target_kinase'].unique())
        rng.shuffle(kinases)
        n_test = max(1, int(round(len(kinases) * test_fraction)))
        n_val = max(1, int(round(len(kinases) * val_fraction)))
        test_kinases = set(kinases[:n_test])
        val_kinases = set(kinases[n_test:n_test + n_val])
        test_mask = df['target_kinase'].isin(test_kinases).values
        val_mask = df['target_kinase'].isin(val_kinases).values
        train_mask = ~(test_mask | val_mask)
        return (np.where(train_mask)[0], np.where(val_mask)[0], np.where(test_mask)[0],
                {'strategy': 'kinase_split_80_10_10'})

    if scenario_id == 'new_compound_new_kinase':
        kinases = np.array(df['target_kinase'].unique())
        rng.shuffle(compounds)
        rng.shuffle(kinases)

        # Broader candidate fractions; scoring selects the best near target test size
        frac_min = max(0.06, test_fraction * 0.6)
        frac_max = min(0.85, test_fraction * 3.0)
        frac_comp = float(rng.uniform(frac_min, frac_max))
        frac_kin = float(rng.uniform(frac_min, frac_max))

        n_test_comp = max(1, int(round(len(compounds) * frac_comp)))
        n_test_kin = max(1, int(round(len(kinases) * frac_kin)))
        test_compounds = set(compounds[:n_test_comp])
        test_kinases = set(kinases[:n_test_kin])

        comp_test = df['chembl_id'].isin(test_compounds).values
        kin_test = df['target_kinase'].isin(test_kinases).values
        test_mask = comp_test & kin_test
        train_mask = (~comp_test) & (~kin_test)
        # Orphan rows (new compound + old kinase, or old compound + new kinase) = val
        val_mask = ~(train_mask | test_mask)

        return (np.where(train_mask)[0], np.where(val_mask)[0], np.where(test_mask)[0],
                {'strategy': 'new_compound_new_kinase_80_10_10',
                 'frac_compound_holdout': frac_comp, 'frac_kinase_holdout': frac_kin,
                 'n_test_compounds_target': n_test_comp, 'n_test_kinases_target': n_test_kin})

    raise ValueError(f"Unknown scenario_id: {scenario_id}")


def _validate_split(df: pd.DataFrame, scenario_id: str, train_idx: np.ndarray,
                     val_idx: np.ndarray, test_idx: np.ndarray):
    """Validate hard split constraints and minimum viability."""
    if len(train_idx) == 0 or len(test_idx) == 0:
        return False, "empty_train_or_test", {}
    if len(val_idx) == 0:
        return False, "empty_val", {}
    if len(test_idx) < MIN_TEST_SAMPLES:
        return False, "test_too_small", {'test_size': len(test_idx)}

    train_set = set(train_idx.tolist())
    val_set = set(val_idx.tolist())
    test_set = set(test_idx.tolist())
    if train_set & test_set:
        return False, "index_overlap_train_test", {}
    if train_set & val_set:
        return False, "index_overlap_train_val", {}
    if val_set & test_set:
        return False, "index_overlap_val_test", {}

    y_test = np.asarray(df.iloc[test_idx]['label'].values)
    n_pos = int(y_test.sum())
    n_neg = int(len(y_test) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return False, "single_class_test", {'n_pos': n_pos, 'n_neg': n_neg}

    y_val = np.asarray(df.iloc[val_idx]['label'].values)
    if int(y_val.sum()) == 0 or int(len(y_val) - y_val.sum()) == 0:
        return False, "single_class_val", {'val_n_pos': int(y_val.sum()), 'val_n_neg': int(len(y_val) - y_val.sum())}

    n_test_kinases = int(df.iloc[test_idx]['target_kinase'].nunique())
    if n_test_kinases < 2:
        return False, "too_few_test_kinases", {'test_kinases': n_test_kinases}

    train_comp = set(df.iloc[train_idx]['chembl_id'])
    test_comp = set(df.iloc[test_idx]['chembl_id'])
    leaked_comp = train_comp & test_comp

    if scenario_id in ('compound', 'new_compound_new_kinase') and leaked_comp:
        return False, "compound_leakage", {'leaked_compounds': len(leaked_comp)}

    leaked_kin = set()
    if scenario_id in ('kinase', 'new_compound_new_kinase'):
        train_kin = set(df.iloc[train_idx]['target_kinase'])
        test_kin = set(df.iloc[test_idx]['target_kinase'])
        leaked_kin = train_kin & test_kin
        if leaked_kin:
            return False, "kinase_leakage", {'leaked_kinases': len(leaked_kin)}

    diagnostics = {
        'train_size': len(train_idx),
        'val_size': len(val_idx),
        'test_size': len(test_idx),
        'test_positive_rate': float(y_test.mean()),
        'val_positive_rate': float(y_val.mean()),
        'test_compounds': len(test_comp),
        'test_kinases': n_test_kinases,
        'leaked_compounds': len(leaked_comp),
    }
    if scenario_id in ('kinase', 'new_compound_new_kinase'):
        diagnostics['test_kinases'] = len(set(df.iloc[test_idx]['target_kinase']))
        diagnostics['leaked_kinases'] = len(leaked_kin)
    return True, "ok", diagnostics


def _split_quality_score(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    protein_group_col: str,
    global_label_rate: float,
    global_protein_probs: dict,
    global_diversity: float,
    fp_map: dict,
    target_test_fraction: float,
    rng: np.random.Generator
):
    """Compute split quality score: lower is better."""
    y_train = np.asarray(df.iloc[train_idx]['label'].values)
    y_test = np.asarray(df.iloc[test_idx]['label'].values)
    label_dev = abs(float(y_test.mean()) - global_label_rate)
    label_dev_train = abs(float(y_train.mean()) - global_label_rate)

    protein_test = df.iloc[test_idx][protein_group_col].fillna('UNKNOWN').astype(str).values
    protein_train = df.iloc[train_idx][protein_group_col].fillna('UNKNOWN').astype(str).values
    protein_l1_test = _distribution_l1(protein_test, global_protein_probs)
    protein_l1_train = _distribution_l1(protein_train, global_protein_probs)

    test_fraction = len(test_idx) / len(df)
    test_fraction_dev = abs(test_fraction - target_test_fraction)

    div_test, n_comp_test = _estimate_internal_diversity(df, test_idx, fp_map, rng)
    div_train, n_comp_train = _estimate_internal_diversity(df, train_idx, fp_map, rng)
    div_dev_test = abs(div_test - global_diversity)
    div_dev_train = abs(div_train - global_diversity)

    # Weighted composite score:
    # - hard emphasis on distribution stability in test
    # - moderate emphasis on target test fraction and train representativity
    score = (
        2.0 * test_fraction_dev +
        2.0 * label_dev +
        0.8 * label_dev_train +
        3.0 * protein_l1_test +
        1.0 * protein_l1_train +
        2.0 * div_dev_test +
        0.8 * div_dev_train
    )

    quality = {
        'score': float(score),
        'test_fraction': float(test_fraction),
        'target_test_fraction': float(target_test_fraction),
        'test_fraction_dev': float(test_fraction_dev),
        'label_rate_train': float(y_train.mean()),
        'label_rate_test': float(y_test.mean()),
        'label_rate_global': float(global_label_rate),
        'label_dev_test': float(label_dev),
        'label_dev_train': float(label_dev_train),
        'protein_l1_test_vs_global': float(protein_l1_test),
        'protein_l1_train_vs_global': float(protein_l1_train),
        'diversity_test': float(div_test),
        'diversity_train': float(div_train),
        'diversity_global': float(global_diversity),
        'diversity_dev_test': float(div_dev_test),
        'diversity_dev_train': float(div_dev_train),
        'n_compounds_test_for_diversity': int(n_comp_test),
        'n_compounds_train_for_diversity': int(n_comp_train),
    }
    return float(score), quality


def _select_best_split_candidate(
    df: pd.DataFrame,
    scenario_id: str,
    round_seed: int,
    n_candidates: int,
    test_fraction: float,
    val_fraction: float,
    protein_group_col: str,
    global_label_rate: float,
    global_protein_probs: dict,
    global_diversity: float,
    fp_map: dict,
    scaffold_map: dict = None
):
    """Select best split candidate for one round by constrained quality score."""
    best = None
    invalid_reasons = {}

    for cand_i in range(n_candidates):
        candidate_seed = int(round_seed + 10007 * cand_i)
        train_idx, val_idx, test_idx, meta = _generate_candidate_split(
            df, scenario_id=scenario_id, seed=candidate_seed,
            test_fraction=test_fraction, val_fraction=val_fraction,
            scaffold_map=scaffold_map
        )
        valid, reason, diag = _validate_split(df, scenario_id, train_idx, val_idx, test_idx)
        if not valid:
            invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1
            continue

        rng = np.random.default_rng(candidate_seed + 17)
        score, quality = _split_quality_score(
            df=df,
            train_idx=train_idx,
            test_idx=test_idx,
            protein_group_col=protein_group_col,
            global_label_rate=global_label_rate,
            global_protein_probs=global_protein_probs,
            global_diversity=global_diversity,
            fp_map=fp_map,
            target_test_fraction=test_fraction,
            rng=rng
        )
        candidate = {
            'candidate_seed': candidate_seed,
            'train_idx': train_idx,
            'val_idx': val_idx,
            'test_idx': test_idx,
            'score': score,
            'quality': quality,
            'diagnostics': diag,
            'metadata': meta,
        }
        if best is None or score < best['score']:
            best = candidate

    if best is None:
        raise RuntimeError(
            f"No valid split found for scenario={scenario_id} at round_seed={round_seed}. "
            f"Invalid reason counts: {invalid_reasons}"
        )
    return best


def run_comparison(
    df: pd.DataFrame,
    output_dir: str = '.',
    seed: int = DEFAULT_SPLIT_SEED,
    n_rounds: int = DEFAULT_N_ROUNDS,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    n_split_candidates: int = DEFAULT_N_SPLIT_CANDIDATES,
    scenarios: list = None
):
    """Run comparison across split scenarios with multi-round partitioning."""

    if n_rounds < 1:
        raise ValueError("n_rounds must be >= 1")
    if not (0.05 <= test_fraction <= 0.5):
        raise ValueError("test_fraction must be in [0.05, 0.5]")
    if n_split_candidates < 1:
        raise ValueError("n_split_candidates must be >= 1")
    if scenarios is None:
        scenarios = DEFAULT_SCENARIOS

    print("=" * 70)
    print("COMPARISON: RANDOM SPLIT VS CORRECT SPLIT")
    print("=" * 70)
    # Use well-separated seeds to avoid correlated splits
    split_seeds = ROUND_SEEDS[:n_rounds] if n_rounds <= len(ROUND_SEEDS) else ROUND_SEEDS + [seed + i for i in range(n_rounds - len(ROUND_SEEDS))]
    print(f"Model seed (fixed): {seed}")
    print(f"Split rounds: {n_rounds}")
    print(f"Split seeds by round: {split_seeds}")
    print(f"Target split: {1-test_fraction-val_fraction:.0%}/{val_fraction:.0%}/{test_fraction:.0%} (train/val/test)")
    print(f"Split candidates per round: {n_split_candidates}")
    print(f"Scenarios: {scenarios}")

    scenarios_config = [
        (s, ALL_SCENARIOS_CONFIG[s][0], ALL_SCENARIOS_CONFIG[s][1])
        for s in scenarios if s in ALL_SCENARIOS_CONFIG
    ]
    if not scenarios_config:
        print("  No valid scenarios selected. Nothing to run.")
        return {}, {}

    protein_group_col = _protein_group_column(df)
    global_label_rate = float(df['label'].mean())
    global_protein_probs = (
        df[protein_group_col].fillna('UNKNOWN').astype(str).value_counts(normalize=True).to_dict()
    )
    fp_map = _build_compound_fp_map(df)
    global_rng = np.random.default_rng(seed)
    if len(fp_map) >= 2:
        global_compounds = list(fp_map.keys())
        if len(global_compounds) > MAX_DIVERSITY_COMPOUNDS:
            sel = global_rng.choice(len(global_compounds), size=MAX_DIVERSITY_COMPOUNDS, replace=False)
            global_compounds = [global_compounds[i] for i in sel]
        global_fp = np.stack([fp_map[c] for c in global_compounds], axis=0)
        global_diversity = 1.0 - _pairwise_tanimoto_mean_sampled(global_fp, rng=global_rng, n_pairs=10000)
    else:
        global_diversity = 0.0
    print(f"Protein-group column for balancing: {protein_group_col}")
    print(f"Global internal diversity (unique compounds): {global_diversity:.4f}")

    # Build fp_cache for model features (float32) from the already-computed fp_map (bool)
    fp_cache = {cid: fp.astype(np.float32) for cid, fp in fp_map.items()}
    print(f"Fingerprint cache: {len(fp_cache)} unique compounds precomputed")

    # Build scaffold map if scaffold scenario is requested
    scaffold_map = None
    scenario_ids = [s for s, _, _ in scenarios_config]
    if 'scaffold' in scenario_ids:
        scaffold_map = _build_compound_scaffold_map(df)

    all_kinases = list(df['target_kinase'].unique())
    all_results = {}
    split_stats = {}

    for scenario_id, scenario_name, scenario_key in scenarios_config:
        print(f"\n{'-' * 50}")
        print(f"SCENARIO: {scenario_name}")
        print("-" * 50)

        round_results = {}
        round_split_stats = []

        for round_idx, split_seed in enumerate(split_seeds, 1):
            print(f"\n  Round {round_idx}/{n_rounds} (round_seed={split_seed})")
            best = _select_best_split_candidate(
                df=df,
                scenario_id=scenario_id,
                round_seed=split_seed,
                n_candidates=n_split_candidates,
                test_fraction=test_fraction,
                val_fraction=val_fraction,
                protein_group_col=protein_group_col,
                global_label_rate=global_label_rate,
                global_protein_probs=global_protein_probs,
                global_diversity=global_diversity,
                fp_map=fp_map,
                scaffold_map=scaffold_map
            )
            train_idx = best['train_idx']
            val_idx = best['val_idx']
            test_idx = best['test_idx']

            train_compounds = set(df.iloc[train_idx]['chembl_id'])
            test_compounds = set(df.iloc[test_idx]['chembl_id'])
            leaked = train_compounds & test_compounds

            stats = {
                'train_size': len(train_idx),
                'val_size': len(val_idx),
                'test_size': len(test_idx),
                'test_compounds': len(test_compounds),
                'leaked_compounds': len(leaked),
                'leak_pct': 100 * len(leaked) / len(test_compounds) if test_compounds else 0
            }
            if scenario_id in ('kinase', 'new_compound_new_kinase'):
                train_kinases = set(df.iloc[train_idx]['target_kinase'])
                test_kinases_set = set(df.iloc[test_idx]['target_kinase'])
                stats['test_kinases'] = len(test_kinases_set)
                stats['leaked_kinases'] = len(train_kinases & test_kinases_set)
            round_split_stats.append(stats)

            print(
                f"  Selected candidate_seed={best['candidate_seed']} | "
                f"score={best['score']:.4f} | "
                f"test_frac={best['quality']['test_fraction']:.3f} "
                f"(target={best['quality']['target_test_fraction']:.3f})"
            )
            print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
            print(f"  Test compounds: {len(test_compounds)}, Leaked: {len(leaked)} ({stats['leak_pct']:.1f}%)")
            print(
                f"  Balance diagnostics: label_dev={best['quality']['label_dev_test']:.4f}, "
                f"protein_l1={best['quality']['protein_l1_test_vs_global']:.4f}, "
                f"div_dev={best['quality']['diversity_dev_test']:.4f}"
            )

            results = train_and_evaluate(df, train_idx, test_idx, all_kinases, seed=seed, fp_cache=fp_cache)
            if results is not None:
                round_results[round_idx] = {
                    'round_seed': split_seed,
                    'candidate_seed': best['candidate_seed'],
                    'split_quality': best['quality'],
                    'split_diagnostics': best['diagnostics'],
                    'split_metadata': best['metadata'],
                    'metrics': results
                }
                knn_auroc = f", AUROC={results['KNN']['auroc']:.4f}" if not np.isnan(results['KNN'].get('auroc', float('nan'))) else ""
                mlp_auroc = f", AUROC={results['MLP']['auroc']:.4f}" if not np.isnan(results['MLP'].get('auroc', float('nan'))) else ""
                print(f"  KNN: Acc={results['KNN']['accuracy']:.4f}, MCC={results['KNN']['mcc']:.4f}, F1={results['KNN']['f1']:.4f}{knn_auroc}")
                print(f"  MLP: Acc={results['MLP']['accuracy']:.4f}, MCC={results['MLP']['mcc']:.4f}, F1={results['MLP']['f1']:.4f}{mlp_auroc}")

        # Aggregate across rounds
        if round_results:
            # Aggregate split stats for plotting/summary
            agg_split = {}
            for k in ['train_size', 'val_size', 'test_size', 'test_compounds', 'leaked_compounds', 'leak_pct', 'test_kinases', 'leaked_kinases']:
                vals = [s[k] for s in round_split_stats if k in s]
                if vals:
                    agg_split[k] = int(round(np.mean(vals))) if k.endswith('size') or 'compounds' in k or 'kinases' in k else float(np.mean(vals))
                    agg_split[f'{k}_std'] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            split_stats[scenario_key] = agg_split

            if len(round_results) == 1:
                # Single round: use metrics directly
                single = list(round_results.values())[0]['metrics']
                all_results[scenario_key] = single
            else:
                # Multi-round: mean +/- std
                aggregated = {}
                for model in ['KNN', 'MLP']:
                    model_agg = {}
                    for metric in ['accuracy', 'mcc', 'f1', 'precision', 'recall', 'auroc']:
                        values = [r['metrics'][model][metric] for r in round_results.values()
                                  if not np.isnan(r['metrics'][model].get(metric, float('nan')))]
                        if values:
                            model_agg[metric] = float(np.mean(values))
                            model_agg[f'{metric}_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                        else:
                            model_agg[metric] = float('nan')
                            model_agg[f'{metric}_std'] = 0.0
                    model_agg['round_results'] = {
                        str(r_idx): {
                            'round_seed': r_data['round_seed'],
                            'candidate_seed': r_data['candidate_seed'],
                            'split_quality': r_data['split_quality'],
                            'split_diagnostics': r_data['split_diagnostics'],
                            'split_metadata': r_data['split_metadata'],
                            'metrics': r_data['metrics'][model]
                        }
                        for r_idx, r_data in round_results.items()
                    }
                    quality_scores = [r['split_quality']['score'] for r in round_results.values()]
                    model_agg['split_quality_score_mean'] = float(np.mean(quality_scores))
                    model_agg['split_quality_score_std'] = float(np.std(quality_scores, ddof=1)) if len(quality_scores) > 1 else 0.0
                    model_agg['n_rounds'] = len(round_results)
                    aggregated[model] = model_agg
                all_results[scenario_key] = aggregated

                # Print aggregate summary
                print(f"\n  --- Aggregate ({len(round_results)} rounds) ---")
                for model in ['KNN', 'MLP']:
                    m = aggregated[model]
                    auroc_str = f", AUROC={m['auroc']:.4f}" if not np.isnan(m.get('auroc', float('nan'))) else ""
                    print(f"  {model}: Acc={m['accuracy']:.4f}+/-{m['accuracy_std']:.4f}, "
                          f"MCC={m['mcc']:.4f}+/-{m['mcc_std']:.4f}, "
                          f"F1={m['f1']:.4f}+/-{m['f1_std']:.4f}{auroc_str}")

    return all_results, split_stats


def _get_metric(all_results: dict, scenario_key: str, model: str, metric: str):
    """Extract metric value from results dict (works for single and multi-round)."""
    return all_results[scenario_key][model][metric]


def _get_metric_std(all_results: dict, scenario_key: str, model: str, metric: str):
    """Extract metric std from results dict (returns 0 for single-round)."""
    return all_results[scenario_key][model].get(f'{metric}_std', 0.0)


def _is_multi_round(all_results: dict):
    """Check if results contain multi-round aggregation."""
    first_scenario = list(all_results.values())[0]
    knn = first_scenario.get('KNN', {})
    return ('n_rounds' in knn) or ('n_seeds' in knn)


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
    n_rounds: int = DEFAULT_N_ROUNDS,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    n_split_candidates: int = DEFAULT_N_SPLIT_CANDIDATES,
    scenarios: list = None,
    keep_monotonic: bool = False
):
    """Run analysis for a single dataset type."""
    data_path = DATASET_PATHS.get(dataset_type)
    if not data_path:
        print(f"Error: Unknown dataset type '{dataset_type}'")
        return None

    prefix = f"{dataset_type}_" if dataset_type != 'non_human' else ""

    # Check cache
    json_file = os.path.join(output_dir, f'{prefix}split_comparison_results.json')
    if os.path.exists(json_file) and not force:
        print(f"\n[CACHE] Results already exist for {dataset_type}: {json_file}")
        print(f"        Use --force to recalculate.")
        return None

    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"ANALYSIS: {dataset_type.upper()}")
    print("=" * 70)

    print(f"\nLoading dataset: {dataset_type}...")
    try:
        df = load_dataset(data_path, keep_monotonic=keep_monotonic)
    except FileNotFoundError:
        print(f"  ERROR: File not found: {data_path}")
        return None

    print(f"  Total: {len(df)} rows, {df['chembl_id'].nunique()} compounds, {df['target_kinase'].nunique()} kinases")

    all_results, split_stats = run_comparison(
        df,
        output_dir,
        seed=seed,
        n_rounds=n_rounds,
        test_fraction=test_fraction,
        val_fraction=val_fraction,
        n_split_candidates=n_split_candidates,
        scenarios=scenarios
    )
    if not all_results:
        print("  No results generated.")
        return None

    # Generate plots
    print("\n" + "-" * 50)
    print("GENERATING PLOTS")
    print("-" * 50)

    plot_comparison(all_results, split_stats, output_dir, prefix)

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
                scenario_json[model]['n_rounds'] = m.get('n_rounds', 1)
                if 'round_results' in m:
                    scenario_json[model]['round_results'] = m['round_results']
        json_results[scenario_key.replace('\n', ' ')] = scenario_json

    json_file = os.path.join(output_dir, f'{prefix}split_comparison_results.json')
    with open(json_file, 'w') as f:
        json.dump({
            'dataset': dataset_type,
            'split_protocol_version': SPLIT_PROTOCOL_VERSION,
            'model_seed': seed,
            'n_rounds': n_rounds,
            'split_seeds': ROUND_SEEDS[:n_rounds] if n_rounds <= len(ROUND_SEEDS) else ROUND_SEEDS + [seed + i for i in range(n_rounds - len(ROUND_SEEDS))],
            'target_test_fraction': test_fraction,
            'target_val_fraction': val_fraction,
            'target_split_ratio': f'{1-test_fraction-val_fraction:.0%}/{val_fraction:.0%}/{test_fraction:.0%}',
            'monotonic_kinase_filter': not keep_monotonic,
            'n_split_candidates': n_split_candidates,
            'scenarios': scenarios if scenarios else DEFAULT_SCENARIOS,
            'affinity_threshold_pchembl': DEFAULT_AFFINITY_THRESHOLD.threshold_pchembl,
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

            # Wilcoxon signed-rank test (paired comparison across rounds)
            random_data = all_results[random_key]
            new_data = all_results[new_key]
            for model in ['KNN', 'MLP']:
                if 'round_results' in random_data[model] and 'round_results' in new_data[model]:
                    r_keys = sorted(random_data[model]['round_results'].keys())
                    n_keys = sorted(new_data[model]['round_results'].keys())
                    if len(r_keys) >= 5 and len(n_keys) >= 5:
                        r_mcc = [random_data[model]['round_results'][k]['metrics']['mcc'] for k in r_keys]
                        n_mcc = [new_data[model]['round_results'][k]['metrics']['mcc'] for k in n_keys]
                        try:
                            stat, p_val = wilcoxon(r_mcc, n_mcc, alternative='greater')
                            print(f"  {model} Wilcoxon signed-rank (random > new_comp_new_kin): "
                                  f"stat={stat:.1f}, p={p_val:.4f}"
                                  f"{' ***' if p_val < 0.01 else ' **' if p_val < 0.05 else ' (n.s.)'}")
                        except ValueError as e:
                            print(f"  {model} Wilcoxon test: could not compute ({e})")
                    else:
                        print(f"  {model} Wilcoxon test: requires >= 5 rounds (have {min(len(r_keys), len(n_keys))})")

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
  # Run single dataset with default scenarios
  %(prog)s --dataset non_human

  # Run all datasets (non_human, human, all)
  %(prog)s --run_all

  # Run specific scenarios only
  %(prog)s --dataset non_human --scenarios random,compound

  # Fixed model seed + 5 split rounds (default)
  %(prog)s --dataset human --seed 42 --n_rounds 5

  # More rigorous split selection (more candidate partitions)
  %(prog)s --dataset non_human --n_rounds 5 --n_split_candidates 40 --test_fraction 0.20

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
             'Options: new_compound_new_kinase, scaffold, compound, kinase, random'
    )
    parser.add_argument(
        '--output_dir', '-o',
        default='./results/split_comparison_analysis',
        help='Output directory for results (default: ./results/split_comparison_analysis)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=DEFAULT_SPLIT_SEED,
        help=f'Fixed random seed for model + split-seed base (default: {DEFAULT_SPLIT_SEED})'
    )
    parser.add_argument(
        '--n_rounds',
        type=int,
        default=DEFAULT_N_ROUNDS,
        help=f'Number of split rounds to aggregate (default: {DEFAULT_N_ROUNDS})'
    )
    parser.add_argument(
        '--test_fraction',
        type=float,
        default=DEFAULT_TEST_FRACTION,
        help=f'Target test fraction for candidate selection (default: {DEFAULT_TEST_FRACTION})'
    )
    parser.add_argument(
        '--val_fraction',
        type=float,
        default=DEFAULT_VAL_FRACTION,
        help=f'Target validation fraction (default: {DEFAULT_VAL_FRACTION})'
    )
    parser.add_argument(
        '--keep_monotonic',
        action='store_true',
        help='Keep kinases with monotonic activity profiles (100%% active or inactive). '
             'By default these are removed as they provide no discriminative signal.'
    )
    parser.add_argument(
        '--n_split_candidates',
        type=int,
        default=DEFAULT_N_SPLIT_CANDIDATES,
        help=f'Candidate splits evaluated per round (default: {DEFAULT_N_SPLIT_CANDIDATES})'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force recalculation even if results exist'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with full error tracebacks'
    )

    args = parser.parse_args()

    if not args.run_all and args.dataset is None:
        parser.error("--dataset is required unless --run_all is specified")
    if args.n_rounds < 1:
        parser.error("--n_rounds must be >= 1")
    if args.n_split_candidates < 1:
        parser.error("--n_split_candidates must be >= 1")
    if not (0.05 <= args.test_fraction <= 0.5):
        parser.error("--test_fraction must be in [0.05, 0.5]")
    if not (0.0 <= args.val_fraction <= 0.5):
        parser.error("--val_fraction must be in [0.0, 0.5]")
    if args.test_fraction + args.val_fraction >= 1.0:
        parser.error("test_fraction + val_fraction must be < 1.0")

    if args.scenarios:
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
    train_frac = 1 - args.test_fraction - args.val_fraction
    print(f"Datasets:         {datasets_to_run}")
    print(f"Scenarios:        {scenarios}")
    print(f"Output directory: {args.output_dir}")
    print(f"Split protocol:   {SPLIT_PROTOCOL_VERSION}")
    print(f"Model seed:       {args.seed} (fixed)")
    print(f"Split rounds:     {args.n_rounds}")
    display_seeds = ROUND_SEEDS[:args.n_rounds] if args.n_rounds <= len(ROUND_SEEDS) else ROUND_SEEDS + [args.seed + i for i in range(args.n_rounds - len(ROUND_SEEDS))]
    print(f"Split seeds:      {display_seeds}")
    print(f"Split ratio:      {train_frac:.0%}/{args.val_fraction:.0%}/{args.test_fraction:.0%} (train/val/test)")
    print(f"Split candidates: {args.n_split_candidates}/round")
    mono_status = 'OFF (keeping all)' if args.keep_monotonic else 'ON (removing 100% single-class kinases)'
    print(f"Monotonic filter: {mono_status}")
    print(f"Force recalc:     {args.force}")
    print(f"Debug mode:       {args.debug}")
    print("=" * 70)

    all_dataset_results = {}
    all_dataset_times = {}
    total_start_time = time.time()

    for i, dataset_type in enumerate(datasets_to_run, 1):
        print(f"\n{'#' * 70}")
        print(f"# [{i}/{len(datasets_to_run)}] Running analysis for dataset={dataset_type}...")
        print(f"{'#' * 70}")

        dataset_start_time = time.time()
        try:
            result = run_single_dataset(
                dataset_type, args.output_dir, args.force,
                seed=args.seed,
                n_rounds=args.n_rounds,
                test_fraction=args.test_fraction,
                val_fraction=args.val_fraction,
                n_split_candidates=args.n_split_candidates,
                scenarios=scenarios,
                keep_monotonic=args.keep_monotonic
            )
            dataset_time = time.time() - dataset_start_time
            all_dataset_times[dataset_type] = dataset_time
            if result is None:
                print(f"  Skipped {dataset_type} (results already exist or data not found)")
                all_dataset_results[dataset_type] = None
            else:
                print(f"  Completed {dataset_type} analysis in {format_time(dataset_time)}")
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
    print(f"Results saved in: {args.output_dir}")


if __name__ == '__main__':
    main()
