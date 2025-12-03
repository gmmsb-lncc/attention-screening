#!/usr/bin/env python3
"""
Utility Functions - DockTKinase Regression
===========================================

Helper functions for regression pipeline.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path


def prepare_regression_targets(df, priority=None, verbose=True, keep_all=False, use_pchembl=True):
    """
    Prepares regression targets with measurement prioritization.
    
    If a compound has multiple measurements, uses only the highest priority one.
    
    Args:
        df: DataFrame with 'standard_type' and 'standard_value' columns
        priority: List of measurement priorities (default: ['Ki', 'Kd', 'IC50'])
        verbose: Show information
        keep_all: If True, keeps ALL measurements (no duplicates removed).
                  Use True when you already have pre-defined splits.
        use_pchembl: If True, uses pChEMBL value (-log10 of value in Molar).
                     This is RECOMMENDED for regression as Ki/IC50 values
                     vary over several orders of magnitude.
                     If pchembl_value doesn't exist, calculates automatically.
        
    Returns:
        y: Array with values (pChEMBL if use_pchembl=True, else nM)
        df_filtered: Filtered DataFrame (1 row per compound if keep_all=False)
        measure_types: Array with measurement type used for each sample
        kept_indices: Indices of rows kept from original DataFrame
    """
    # FIX: Mutable default argument - criar nova lista a cada chamada
    if priority is None:
        priority = ['Ki', 'Kd', 'IC50']
    
    if 'standard_type' not in df.columns or 'standard_value' not in df.columns:
        raise ValueError("DataFrame deve ter colunas 'standard_type' e 'standard_value'")
    
    # Filter only valid measurements
    valid_mask = df['standard_type'].isin(priority)
    df_valid = df[valid_mask].copy()
    df_valid['_original_index'] = df_valid.index  # Save original indices
    
    if verbose:
        print(f'📊 Preparing regression targets...')
        print(f'   Original samples: {len(df):,}')
        print(f'   Samples with valid measurements: {len(df_valid):,}')
    
    # Criar coluna de prioridade
    priority_map = {measure: idx for idx, measure in enumerate(priority)}
    df_valid['_priority'] = df_valid['standard_type'].map(priority_map)
    
    if keep_all:
        # Mode: keep ALL measurements (no duplicates removed)
        # Sort by priority for consistent order
        df_filtered = df_valid.sort_values('_priority')
        
        if verbose:
            print(f'   ℹ️  Mode keep_all=True: keeping all measurements')
    else:
        # Default mode: remove duplicates by (molregno, seq_id)
        # If 'molregno' exists, group by compound and get highest priority measurement
        if 'molregno' in df_valid.columns and 'seq_id' in df_valid.columns:
            # Sort by priority and get first occurrence of each (molregno, seq_id) pair
            df_valid = df_valid.sort_values('_priority')
            df_filtered = df_valid.groupby(['molregno', 'seq_id'], as_index=False).first()
            
            n_duplicates = len(df_valid) - len(df_filtered)
            if verbose and n_duplicates > 0:
                print(f'   Duplicatas removidas: {n_duplicates:,}')
        else:
            df_filtered = df_valid.copy()
    
    # Extract targets and original indices
    measure_types = df_filtered['standard_type'].values
    kept_indices = df_filtered['_original_index'].values
    
    # Choose between pChEMBL (logarithmic) or raw value (nM)
    if use_pchembl:
        # Use pchembl_value if available, else calculate from standard_value
        if 'pchembl_value' in df_filtered.columns:
            # Tentar usar pchembl_value existente
            pchembl_series = pd.to_numeric(df_filtered['pchembl_value'], errors='coerce')
            standard_series = pd.to_numeric(df_filtered['standard_value'], errors='coerce')
            
            # Calcular pchembl para valores faltantes: pChEMBL = -log10(nM * 1e-9) = 9 - log10(nM)
            missing_mask = pchembl_series.isna() & standard_series.notna() & (standard_series > 0)
            pchembl_series.loc[missing_mask] = 9 - np.log10(standard_series.loc[missing_mask])
            
            # Filter samples with valid values
            valid_mask = pchembl_series.notna()
            if valid_mask.sum() < len(df_filtered):
                n_invalid = len(df_filtered) - valid_mask.sum()
                if verbose:
                    print(f'   ⚠️  Removing {n_invalid} samples without valid pChEMBL')
                df_filtered = df_filtered[valid_mask].copy()
                pchembl_series = pchembl_series[valid_mask]
                measure_types = measure_types[valid_mask.values]
                kept_indices = kept_indices[valid_mask.values]
            
            y = pchembl_series.values
        else:
            # Calcular pchembl a partir de standard_value
            standard_series = pd.to_numeric(df_filtered['standard_value'], errors='coerce')
            valid_mask = standard_series.notna() & (standard_series > 0)
            
            if valid_mask.sum() < len(df_filtered):
                n_invalid = len(df_filtered) - valid_mask.sum()
                if verbose:
                    print(f'   ⚠️  Removing {n_invalid} samples with invalid standard_value')
                df_filtered = df_filtered[valid_mask].copy()
                standard_series = standard_series[valid_mask]
                measure_types = measure_types[valid_mask.values]
                kept_indices = kept_indices[valid_mask.values]
            
            # pChEMBL = -log10(nM * 1e-9) = 9 - log10(nM)
            y = 9 - np.log10(standard_series.values)
        
        value_unit = 'pChEMBL'
    else:
        # Usar valor bruto em nM (comportamento anterior)
        y = df_filtered['standard_value'].values
        value_unit = 'nM'
    
    # Remove temporary columns
    columns_to_drop = ['_priority', '_original_index']
    df_filtered = df_filtered.drop(columns=columns_to_drop, errors='ignore')
    
    if verbose:
        print(f'   ✅ Final samples: {len(y):,}')
        print(f'\n   Distribution by measurement type:')
        for measure in priority:
            count = (measure_types == measure).sum()
            pct = (count / len(measure_types)) * 100
            print(f'      {measure}: {count:,} ({pct:.1f}%)')
        
        print(f'\n   Value statistics ({value_unit}):')
        print(f'      Min:    {y.min():.2f}')
        print(f'      Median: {np.median(y):.2f}')
        print(f'      Mean:   {y.mean():.2f}')
        print(f'      Max:    {y.max():.2f}')
        
        if use_pchembl:
            print(f'\n   ℹ️  Using pChEMBL scale (logarithmic) - recommended for regression')
    
    return y, df_filtered, measure_types, kept_indices


def load_embeddings_cache(cache_path):
    """
    Loads embeddings from cache.
    
    Args:
        cache_path: Path to .npz file
        
    Returns:
        embeddings: Embeddings array
        metadata: Dict with metadata (if available)
    """
    cache_path = Path(cache_path)
    
    if not cache_path.exists():
        raise FileNotFoundError(f'Embeddings cache not found: {cache_path}')
    
    data = np.load(cache_path, allow_pickle=True)
    embeddings = data['embeddings']
    
    # Tentar carregar metadados
    metadata = {}
    if 'metadata' in data:
        metadata = data['metadata'].item()
    
    return embeddings, metadata


def save_embeddings_cache(embeddings, sequences, cache_path, model_name=None):
    """
    Saves embeddings to cache.
    
    Args:
        embeddings: Embeddings array
        sequences: List of sequences
        cache_path: Path to save
        model_name: ESM-2 model name
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        'n_samples': len(embeddings),
        'embedding_dim': embeddings.shape[1],
        'model_name': model_name
    }
    
    np.savez_compressed(
        cache_path,
        embeddings=embeddings,
        sequences=sequences,
        metadata=metadata
    )


def load_split_indices(stats_file):
    """
    Loads split indices from the classification stats file.
    
    This ensures regression uses EXACTLY the same splits.
    
    Args:
        stats_file: Path to pipeline_stats.json from classification
        
    Returns:
        idx_train, idx_val, idx_test: Arrays with indices
        
    Raises:
        FileNotFoundError: If file doesn't exist
        KeyError: If file doesn't have split information
    """
    stats_file = Path(stats_file)
    
    if not stats_file.exists():
        raise FileNotFoundError(
            f'Stats file not found: {stats_file}\n'
            f'Run the classification pipeline first.'
        )
    
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    
    # Check if has split information
    if 'split_indices' not in stats:
        raise KeyError(
            'Stats file does not contain split information.\n'
            'Update the classification pipeline to save split indices.'
        )
    
    split_info = stats['split_indices']
    idx_train = np.array(split_info['train'])
    idx_val = np.array(split_info['val'])
    idx_test = np.array(split_info['test'])
    
    return idx_train, idx_val, idx_test


def save_split_indices(idx_train, idx_val, idx_test, stats_file):
    """
    Saves split indices to stats file.
    
    Args:
        idx_train, idx_val, idx_test: Arrays with indices
        stats_file: Path to JSON file
    """
    stats_file = Path(stats_file)
    
    # Load existing stats or create new
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            stats = json.load(f)
    else:
        stats = {}
    
    # Add split indices
    stats['split_indices'] = {
        'train': idx_train.tolist(),
        'val': idx_val.tolist(),
        'test': idx_test.tolist()
    }
    
    # Salvar
    stats_file.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)


if __name__ == '__main__':
    # Test prepare_regression_targets
    df_test = pd.DataFrame({
        'molregno': [1, 1, 2, 3],
        'seq_id': [10, 10, 20, 30],
        'standard_type': ['IC50', 'Ki', 'Kd', 'Ki'],
        'standard_value': [500, 300, 800, 1200],
        'pchembl_value': [6.3, 6.52, None, 5.92]  # Some with pchembl, others without
    })
    
    print('=== Test with use_pchembl=True (default) ===')
    y, df_filtered, types, indices = prepare_regression_targets(df_test, keep_all=False, use_pchembl=True)
    print('\nResultados:')
    print(f'y (pChEMBL): {y}')
    print(f'types: {types}')
    print(f'indices: {indices}')
    
    print('\n=== Teste com use_pchembl=False (valores brutos em nM) ===')
    y2, df_filtered2, types2, indices2 = prepare_regression_targets(df_test, keep_all=False, use_pchembl=False)
    print('\nResultados:')
    print(f'y (nM): {y2}')
    print(f'types: {types2}')
    
    print('\n=== Teste sem coluna pchembl_value (deve calcular automaticamente) ===')
    df_test_no_pchembl = pd.DataFrame({
        'molregno': [1, 2, 3],
        'seq_id': [10, 20, 30],
        'standard_type': ['Ki', 'Ki', 'IC50'],
        'standard_value': [100, 1000, 10000]  # Esperado: pChEMBL = 7.0, 6.0, 5.0
    })
    y3, df_filtered3, types3, indices3 = prepare_regression_targets(df_test_no_pchembl, use_pchembl=True)
    print('\nResultados:')
    print(f'y (pChEMBL calculado): {y3}')
    print(f'Esperado: [7.0, 6.0, 5.0]')
