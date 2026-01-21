"""Data splitting strategies for evaluation scenarios."""

from typing import Tuple, Callable
import numpy as np
import pandas as pd


def split_random(df: pd.DataFrame, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Random 80/10/10 split (allows data leakage).

    Both compounds and kinases may appear in multiple sets.
    This is the easiest scenario and serves as an upper bound.

    Args:
        df: DataFrame with samples
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    np.random.seed(seed)
    indices = np.random.permutation(len(df))
    n = len(df)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)
    return indices[:train_end], indices[train_end:val_end], indices[val_end:]


def split_by_compound(df: pd.DataFrame, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    np.random.seed(seed)
    compounds = np.array(df['chembl_id'].unique())
    np.random.shuffle(compounds)
    n = len(compounds)

    train_compounds = set(compounds[:int(0.8 * n)])
    val_compounds = set(compounds[int(0.8 * n):int(0.9 * n)])
    test_compounds = set(compounds[int(0.9 * n):])

    train_idx = df[df['chembl_id'].isin(train_compounds)].index.values
    val_idx = df[df['chembl_id'].isin(val_compounds)].index.values
    test_idx = df[df['chembl_id'].isin(test_compounds)].index.values

    return train_idx, val_idx, test_idx


def split_new_compound_new_kinase(
    df: pd.DataFrame,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split with new compounds AND new kinases (true generalization).

    Both compounds and kinases in test are completely unseen during training.
    This is the hardest scenario and evaluates true generalization capability.

    Note: This split may discard samples where compound and kinase sets don't align.
    Typically uses ~60-70% of data.

    Args:
        df: DataFrame with 'chembl_id' and 'target_kinase' columns
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    np.random.seed(seed)

    compounds = np.array(df['chembl_id'].unique())
    kinases = np.array(df['target_kinase'].unique())
    np.random.shuffle(compounds)
    np.random.shuffle(kinases)

    # 80/10/10 split for both compounds and kinases
    test_kinases = set(kinases[:int(0.10 * len(kinases))])
    val_kinases = set(kinases[int(0.10 * len(kinases)):int(0.20 * len(kinases))])
    train_kinases = set(kinases[int(0.20 * len(kinases)):])

    test_compounds = set(compounds[:int(0.10 * len(compounds))])
    val_compounds = set(compounds[int(0.10 * len(compounds)):int(0.20 * len(compounds))])
    train_compounds = set(compounds[int(0.20 * len(compounds)):])

    # Only use samples where both compound and kinase belong to the same set
    train_mask = df['chembl_id'].isin(train_compounds) & df['target_kinase'].isin(train_kinases)
    val_mask = df['chembl_id'].isin(val_compounds) & df['target_kinase'].isin(val_kinases)
    test_mask = df['chembl_id'].isin(test_compounds) & df['target_kinase'].isin(test_kinases)

    train_idx = df[train_mask].index.values
    val_idx = df[val_mask].index.values
    test_idx = df[test_mask].index.values

    # Warn if significant data is lost
    total_used = len(train_idx) + len(val_idx) + len(test_idx)
    if total_used < len(df) * 0.7:
        print(f"  WARNING: Only {total_used}/{len(df)} samples used "
              f"({100*total_used/len(df):.1f}%)")

    return train_idx, val_idx, test_idx


def get_scenarios() -> list[Tuple[str, Callable]]:
    """
    Get all evaluation scenarios ordered from hardest to easiest.

    Returns:
        List of (scenario_name, split_function) tuples
    """
    return [
        ('New Comp + New Kinase', split_new_compound_new_kinase),
        ('Split by Compound', split_by_compound),
        ('Random Split', split_random)
    ]
