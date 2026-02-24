"""Data splitting strategies for evaluation scenarios.

All split functions use local numpy random generators (np.random.default_rng)
to ensure reproducibility without affecting global random state.
"""

from typing import Tuple, Callable, List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def split_by_scaffold(
    df: pd.DataFrame,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split by scaffold 80/10/10 (no scaffold overlap across train/val/test).

    Requires a precomputed 'scaffold' column (as produced by scaffold_split.py).
    This function is kept primarily for compatibility/debug; the main pipeline
    consumes precomputed Sc train/val/test files directly.
    """
    if 'scaffold' not in df.columns:
        raise ValueError("split_by_scaffold requires a precomputed 'scaffold' column")

    rng = np.random.default_rng(seed)
    scaffolds = np.array(df['scaffold'].fillna('UNKNOWN').astype(str).unique())
    rng.shuffle(scaffolds)

    n = len(scaffolds)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)

    train_scaffolds = set(scaffolds[:train_end])
    val_scaffolds = set(scaffolds[train_end:val_end])
    test_scaffolds = set(scaffolds[val_end:])

    train_mask = df['scaffold'].isin(train_scaffolds).values
    val_mask = df['scaffold'].isin(val_scaffolds).values
    test_mask = df['scaffold'].isin(test_scaffolds).values

    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    test_idx = np.where(test_mask)[0]

    return train_idx, val_idx, test_idx


def split_random(
    df: pd.DataFrame,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified random 80/10/10 split (allows data leakage).

    Both compounds and kinases may appear in multiple sets.
    This is the easiest scenario and serves as an upper bound.

    WARNING: This split allows data leakage - use only as baseline.

    Args:
        df: DataFrame with 'label' column for stratification
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    indices = np.arange(len(df), dtype=np.int64)
    # Ensure labels are numpy array (avoids PyArrow issues)
    labels = np.asarray(df['label'].values)

    # Use sklearn's train_test_split with explicit random_state
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.2, stratify=labels, random_state=seed
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=labels[temp_idx], random_state=seed
    )

    return train_idx, val_idx, test_idx


def split_by_compound(
    df: pd.DataFrame,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split by compound 80/10/10 (no compound leakage).

    Compounds in test are unseen during training, but kinases may overlap.
    This evaluates generalization to new compounds.

    Args:
        df: DataFrame with 'chembl_id' column
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    # Use local generator
    rng = np.random.default_rng(seed)

    # Get unique compounds as numpy array
    compounds = np.array(df['chembl_id'].unique())
    rng.shuffle(compounds)

    n = len(compounds)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)

    train_compounds = set(compounds[:train_end])
    val_compounds = set(compounds[train_end:val_end])
    test_compounds = set(compounds[val_end:])

    # Use np.where for robust index extraction (avoids PyArrow issues)
    train_mask = df['chembl_id'].isin(train_compounds).values
    val_mask = df['chembl_id'].isin(val_compounds).values
    test_mask = df['chembl_id'].isin(test_compounds).values

    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    test_idx = np.where(test_mask)[0]

    return train_idx, val_idx, test_idx


def split_new_compound_new_kinase(
    df: pd.DataFrame,
    seed: int = 42,
    test_kinase_frac: float = 0.15,
    test_compound_frac: float = 0.15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split with new compounds AND new kinases (true generalization).

    Both compounds and kinases in test are completely unseen during training.
    This is the hardest scenario and evaluates true generalization capability.

    Args:
        df: DataFrame with 'chembl_id' and 'target_kinase' columns
        seed: Random seed for reproducibility
        test_kinase_frac: Fraction of kinases to hold out for test
        test_compound_frac: Fraction of compounds to hold out for test

    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    # Use local generator
    rng = np.random.default_rng(seed)

    # Get unique values as numpy arrays (avoids PyArrow issues)
    compounds = np.array(df['chembl_id'].unique())
    kinases = np.array(df['target_kinase'].unique())

    # Shuffle using local generator
    rng.shuffle(compounds)
    rng.shuffle(kinases)

    # Select test kinases and compounds
    n_test_kinases = max(1, int(len(kinases) * test_kinase_frac))
    test_kinases = set(kinases[:n_test_kinases])
    train_kinases = set(kinases[n_test_kinases:])

    n_test_compounds = max(1, int(len(compounds) * test_compound_frac))
    test_compounds = set(compounds[:n_test_compounds])
    train_compounds = set(compounds[n_test_compounds:])

    # Create masks as numpy arrays (avoids PyArrow issues)
    train_mask = (
        df['chembl_id'].isin(train_compounds).values &
        df['target_kinase'].isin(train_kinases).values
    )
    test_mask = (
        df['chembl_id'].isin(test_compounds).values &
        df['target_kinase'].isin(test_kinases).values
    )
    val_mask = ~train_mask & ~test_mask

    # Use np.where for robust index extraction
    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    test_idx = np.where(test_mask)[0]

    # Recursively increase fractions if test set is too small
    if len(test_idx) < 50:
        print(f"  WARNING: Test set too small ({len(test_idx)}), increasing fractions...")
        return split_new_compound_new_kinase(
            df, seed,
            test_kinase_frac * 1.5,
            test_compound_frac * 1.5
        )

    return train_idx, val_idx, test_idx


def get_scenarios() -> List[Tuple[str, Callable]]:
    """
    Get all supported evaluation scenarios.

    Returns:
        List of (scenario_name, split_function) tuples
    """
    return [
        ('Split by Scaffold', split_by_scaffold)
    ]
