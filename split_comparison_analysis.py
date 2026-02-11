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
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, matthews_corrcoef
from rdkit import Chem
from rdkit.Chem import AllChem
import warnings
warnings.filterwarnings('ignore')

from crossattention_split_analysis.config import (
    DATASET_PATHS, DEFAULT_AFFINITY_THRESHOLD,
    AVAILABLE_SCENARIOS, DEFAULT_SCENARIOS
)

DEFAULT_SPLIT_SEED = 42
DEFAULT_N_ROUNDS = 5
DEFAULT_TEST_FRACTION = 0.2
DEFAULT_N_SPLIT_CANDIDATES = 25
MIN_TEST_SAMPLES = 100
SPLIT_PROTOCOL_VERSION = "80_20_balanced_tanimoto_candidates_v2"
MAX_DIVERSITY_COMPOUNDS = 8000

# Plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 12


def load_dataset(filepath: str):
    """Load dataset and create binary labels using pChEMBL threshold."""
    df = pd.read_csv(filepath, sep='\t')

    threshold = DEFAULT_AFFINITY_THRESHOLD.threshold_pchembl  # 6.0

    # Calculate pChEMBL from standard_value when missing
    n_missing = df['pchembl_value'].isna().sum()
    if n_missing > 0:
        print(f"  Calculating pChEMBL for {n_missing} samples with missing values...")
        mask_missing = df['pchembl_value'].isna()
        df.loc[mask_missing, 'pchembl_value'] = 9 - np.log10(df.loc[mask_missing, 'standard_value'])

    df['label'] = (df['pchembl_value'] >= threshold).astype(int)

    # Report class distribution
    n_active = df['label'].sum()
    n_inactive = len(df) - n_active
    threshold_nm_equiv = 10 ** (9 - threshold)
    print(f"  Affinity threshold: pChEMBL >= {threshold:.1f} (equivalent to <= {threshold_nm_equiv:.0f} nM)")
    print(f"  Class distribution: {n_active} active ({100*n_active/len(df):.1f}%), "
          f"{n_inactive} inactive ({100*n_inactive/len(df):.1f}%)")

    return df


def compute_morgan_fingerprints(smiles_list: list, radius: int = 2, n_bits: int = 2048):
    """Compute Morgan fingerprints for a list of SMILES."""
    fingerprints = []
    valid_indices = []

    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            fingerprints.append(np.array(fp))
            valid_indices.append(i)

    return np.array(fingerprints), valid_indices


def prepare_features(df: pd.DataFrame, all_kinases: list):
    """Prepare features: Morgan FP + One-Hot Kinase."""
    fps, valid_idx = compute_morgan_fingerprints(df['canonical_smiles'].tolist())

    kinase_to_idx = {k: i for i, k in enumerate(all_kinases)}

    def one_hot_kinase(kinase):
        vec = np.zeros(len(all_kinases))
        if kinase in kinase_to_idx:
            vec[kinase_to_idx[kinase]] = 1
        return vec

    kinase_oh = np.array([one_hot_kinase(k) for k in df.iloc[valid_idx]['target_kinase']])
    X = np.hstack([fps, kinase_oh])
    y = df.iloc[valid_idx]['label'].values

    return X, y, valid_idx


# =============================================================================
# TRAINING AND EVALUATION
# =============================================================================

def train_and_evaluate(df: pd.DataFrame, train_idx, test_idx, all_kinases: list, seed: int = 42):
    """Train KNN and MLP and return metrics."""

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    X_train, y_train, train_valid = prepare_features(train_df, all_kinases)
    X_test, y_test, test_valid = prepare_features(test_df, all_kinases)

    if len(X_test) == 0 or len(X_train) == 0:
        return None

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # KNN
    print("    Training KNN...")
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='cosine', n_jobs=-1)
    knn.fit(X_train_scaled, y_train)
    knn_pred = knn.predict(X_test_scaled)
    results['KNN'] = {
        'accuracy': accuracy_score(y_test, knn_pred),
        'mcc': matthews_corrcoef(y_test, knn_pred)
    }

    # MLP
    print("    Training MLP...")
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
    results['MLP'] = {
        'accuracy': accuracy_score(y_test, mlp_pred),
        'mcc': matthews_corrcoef(y_test, mlp_pred)
    }

    return results


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

# Scenario configuration: map CLI key -> (display_name, plot_key)
ALL_SCENARIOS_CONFIG = {
    'new_compound_new_kinase': (
        'New Compound + New Kinase (true generalization)',
        'New Comp.\n+ New Kinase'
    ),
    'compound': (
        'Split by Compound (no leakage)',
        'Split by\nCompound'
    ),
    'random': (
        'Random Split (with leakage)',
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
    fps, valid_idx = compute_morgan_fingerprints(smiles)
    fp_map = {}
    for fp_i, src_i in enumerate(valid_idx):
        comp_id = unique_comp.iloc[src_i]['chembl_id']
        fp_map[comp_id] = fps[fp_i].astype(bool)
    return fp_map


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


def _generate_candidate_split(df: pd.DataFrame, scenario_id: str, seed: int, test_fraction: float):
    """
    Generate a candidate split for one scenario.
    Returns (train_idx, val_idx, test_idx, metadata).
    """
    n = len(df)
    indices = np.arange(n, dtype=np.int64)
    labels = np.asarray(df['label'].values)
    rng = np.random.default_rng(seed)

    if scenario_id == 'random':
        train_idx, test_idx = train_test_split(
            indices, test_size=test_fraction, stratify=labels, random_state=seed
        )
        val_idx = np.array([], dtype=np.int64)
        return train_idx, val_idx, test_idx, {'strategy': 'stratified_random'}

    compounds = np.array(df['chembl_id'].unique())

    if scenario_id == 'compound':
        rng.shuffle(compounds)
        n_test_comp = max(1, int(round(len(compounds) * test_fraction)))
        test_compounds = set(compounds[:n_test_comp])
        test_mask = df['chembl_id'].isin(test_compounds).values
        train_mask = ~test_mask
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        val_idx = np.array([], dtype=np.int64)
        return train_idx, val_idx, test_idx, {'strategy': 'new_compound_80_20'}

    if scenario_id == 'new_compound_new_kinase':
        kinases = np.array(df['target_kinase'].unique())
        rng.shuffle(compounds)
        rng.shuffle(kinases)

        # Use broader candidate fractions; scoring selects the best candidate near target test size.
        frac_min = max(0.10, test_fraction * 0.6)
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
        val_mask = ~(train_mask | test_mask)

        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]
        test_idx = np.where(test_mask)[0]
        return train_idx, val_idx, test_idx, {
            'strategy': 'new_compound_new_kinase_with_holdout_sets',
            'frac_compound_holdout': frac_comp,
            'frac_kinase_holdout': frac_kin,
            'n_test_compounds_target': n_test_comp,
            'n_test_kinases_target': n_test_kin,
        }

    raise ValueError(f"Unknown scenario_id: {scenario_id}")


def _validate_split(df: pd.DataFrame, scenario_id: str, train_idx: np.ndarray, test_idx: np.ndarray):
    """Validate hard split constraints and minimum viability."""
    if len(train_idx) == 0 or len(test_idx) == 0:
        return False, "empty_train_or_test", {}
    if len(test_idx) < MIN_TEST_SAMPLES:
        return False, "test_too_small", {'test_size': len(test_idx)}

    train_set = set(train_idx.tolist())
    test_set = set(test_idx.tolist())
    if train_set & test_set:
        return False, "index_overlap", {}

    y_test = np.asarray(df.iloc[test_idx]['label'].values)
    n_pos = int(y_test.sum())
    n_neg = int(len(y_test) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return False, "single_class_test", {'n_pos': n_pos, 'n_neg': n_neg}

    n_test_kinases = int(df.iloc[test_idx]['target_kinase'].nunique())
    if n_test_kinases < 2:
        return False, "too_few_test_kinases", {'test_kinases': n_test_kinases}

    train_comp = set(df.iloc[train_idx]['chembl_id'])
    test_comp = set(df.iloc[test_idx]['chembl_id'])
    leaked_comp = train_comp & test_comp

    if scenario_id in ('compound', 'new_compound_new_kinase') and leaked_comp:
        return False, "compound_leakage", {'leaked_compounds': len(leaked_comp)}

    leaked_kin = set()
    if scenario_id == 'new_compound_new_kinase':
        train_kin = set(df.iloc[train_idx]['target_kinase'])
        test_kin = set(df.iloc[test_idx]['target_kinase'])
        leaked_kin = train_kin & test_kin
        if leaked_kin:
            return False, "kinase_leakage", {'leaked_kinases': len(leaked_kin)}

    diagnostics = {
        'train_size': len(train_idx),
        'test_size': len(test_idx),
        'test_positive_rate': float(y_test.mean()),
        'test_compounds': len(test_comp),
        'test_kinases': n_test_kinases,
        'leaked_compounds': len(leaked_comp),
    }
    if scenario_id == 'new_compound_new_kinase':
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
    protein_group_col: str,
    global_label_rate: float,
    global_protein_probs: dict,
    global_diversity: float,
    fp_map: dict
):
    """Select best split candidate for one round by constrained quality score."""
    best = None
    invalid_reasons = {}

    for cand_i in range(n_candidates):
        candidate_seed = int(round_seed + 10007 * cand_i)
        train_idx, val_idx, test_idx, meta = _generate_candidate_split(
            df, scenario_id=scenario_id, seed=candidate_seed, test_fraction=test_fraction
        )
        valid, reason, diag = _validate_split(df, scenario_id, train_idx, test_idx)
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
    split_seeds = [seed + i for i in range(n_rounds)]
    print(f"Model seed (fixed): {seed}")
    print(f"Split rounds: {n_rounds}")
    print(f"Split seeds by round: {split_seeds}")
    print(f"Target test fraction: {test_fraction:.2f}")
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
                protein_group_col=protein_group_col,
                global_label_rate=global_label_rate,
                global_protein_probs=global_protein_probs,
                global_diversity=global_diversity,
                fp_map=fp_map
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
            if scenario_key == 'New Comp.\n+ New Kinase':
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

            results = train_and_evaluate(df, train_idx, test_idx, all_kinases, seed=seed)
            if results is not None:
                round_results[round_idx] = {
                    'round_seed': split_seed,
                    'candidate_seed': best['candidate_seed'],
                    'split_quality': best['quality'],
                    'split_diagnostics': best['diagnostics'],
                    'split_metadata': best['metadata'],
                    'metrics': results
                }
                print(f"  KNN: Acc={results['KNN']['accuracy']:.4f}, MCC={results['KNN']['mcc']:.4f}")
                print(f"  MLP: Acc={results['MLP']['accuracy']:.4f}, MCC={results['MLP']['mcc']:.4f}")

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
                    for metric in ['accuracy', 'mcc']:
                        values = [r['metrics'][model][metric] for r in round_results.values()]
                        model_agg[metric] = float(np.mean(values))
                        model_agg[f'{metric}_std'] = float(np.std(values, ddof=1))
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
                    print(f"  {model}: Acc={m['accuracy']:.4f}+/-{m['accuracy_std']:.4f}, "
                          f"MCC={m['mcc']:.4f}+/-{m['mcc_std']:.4f}")

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
    n_split_candidates: int = DEFAULT_N_SPLIT_CANDIDATES,
    scenarios: list = None
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
        df = load_dataset(data_path)
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
            scenario_json[model] = {
                'accuracy': m['accuracy'],
                'mcc': m['mcc'],
            }
            if multi_round:
                scenario_json[model]['accuracy_std'] = m.get('accuracy_std', 0.0)
                scenario_json[model]['mcc_std'] = m.get('mcc_std', 0.0)
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
            'split_seeds': [seed + i for i in range(n_rounds)],
            'target_test_fraction': test_fraction,
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

Available scenarios:
  - new_compound_new_kinase : Hardest - both compound and kinase unseen
  - compound                : Medium - compound unseen, kinase may overlap
  - random                  : Easiest - random split (baseline with leakage)
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
             'Options: new_compound_new_kinase, compound, random'
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
    print(f"Datasets:         {datasets_to_run}")
    print(f"Scenarios:        {scenarios}")
    print(f"Output directory: {args.output_dir}")
    print(f"Split protocol:   {SPLIT_PROTOCOL_VERSION}")
    print(f"Model seed:       {args.seed} (fixed)")
    print(f"Split rounds:     {args.n_rounds}")
    print(f"Split seeds:      {[args.seed + i for i in range(args.n_rounds)]}")
    print(f"Target test frac: {args.test_fraction:.2f}")
    print(f"Split candidates: {args.n_split_candidates}/round")
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
                n_split_candidates=args.n_split_candidates,
                scenarios=scenarios
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
