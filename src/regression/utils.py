#!/usr/bin/env python3
"""
Funções Utilitárias - Regressão DockTKinase
============================================

Funções auxiliares para pipeline de regressão.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path


def prepare_regression_targets(df, priority=None, verbose=True, keep_all=False):
    """
    Prepara targets de regressão com priorização de medidas.
    
    Se um composto tem múltiplas medidas, usa apenas a de maior prioridade.
    
    Args:
        df: DataFrame com colunas 'standard_type' e 'standard_value'
        priority: Lista de prioridade das medidas (default: ['Ki', 'Kd', 'IC50'])
        verbose: Mostrar informações
        keep_all: Se True, mantém TODAS as medidas (sem remover duplicatas).
                  Use True quando já tiver splits pré-definidos.
        
    Returns:
        y: Array com valores em nM
        df_filtered: DataFrame filtrado (1 linha por composto se keep_all=False)
        measure_types: Array com tipo de medida usada para cada amostra
        kept_indices: Índices das linhas mantidas no DataFrame original
    """
    # FIX: Mutable default argument - criar nova lista a cada chamada
    if priority is None:
        priority = ['Ki', 'Kd', 'IC50']
    
    if 'standard_type' not in df.columns or 'standard_value' not in df.columns:
        raise ValueError("DataFrame deve ter colunas 'standard_type' e 'standard_value'")
    
    # Filtrar apenas medidas válidas
    valid_mask = df['standard_type'].isin(priority)
    df_valid = df[valid_mask].copy()
    df_valid['_original_index'] = df_valid.index  # Guardar índices originais
    
    if verbose:
        print(f'📊 Preparando targets de regressão...')
        print(f'   Amostras originais: {len(df):,}')
        print(f'   Amostras com medidas válidas: {len(df_valid):,}')
    
    # Criar coluna de prioridade
    priority_map = {measure: idx for idx, measure in enumerate(priority)}
    df_valid['_priority'] = df_valid['standard_type'].map(priority_map)
    
    if keep_all:
        # Modo: manter TODAS as medidas (não remover duplicatas)
        # Ordenar por prioridade para ter ordem consistente
        df_filtered = df_valid.sort_values('_priority')
        
        if verbose:
            print(f'   ℹ️  Modo keep_all=True: mantendo todas as medidas')
    else:
        # Modo padrão: remover duplicatas por (molregno, seq_id)
        # Se existir 'molregno', agrupar por composto e pegar medida de maior prioridade
        if 'molregno' in df_valid.columns and 'seq_id' in df_valid.columns:
            # Ordenar por prioridade e pegar primeira ocorrência de cada par (molregno, seq_id)
            df_valid = df_valid.sort_values('_priority')
            df_filtered = df_valid.groupby(['molregno', 'seq_id'], as_index=False).first()
            
            n_duplicates = len(df_valid) - len(df_filtered)
            if verbose and n_duplicates > 0:
                print(f'   Duplicatas removidas: {n_duplicates:,}')
        else:
            df_filtered = df_valid.copy()
    
    # Extrair targets e índices originais
    y = df_filtered['standard_value'].values
    measure_types = df_filtered['standard_type'].values
    kept_indices = df_filtered['_original_index'].values
    
    # Remover colunas temporárias
    columns_to_drop = ['_priority', '_original_index']
    df_filtered = df_filtered.drop(columns=columns_to_drop, errors='ignore')
    
    if verbose:
        print(f'   ✅ Amostras finais: {len(y):,}')
        print(f'\n   Distribuição por tipo de medida:')
        for measure in priority:
            count = (measure_types == measure).sum()
            pct = (count / len(measure_types)) * 100
            print(f'      {measure}: {count:,} ({pct:.1f}%)')
        
        print(f'\n   Estatísticas dos valores (nM):')
        print(f'      Min:    {y.min():.2f}')
        print(f'      Mediana: {np.median(y):.2f}')
        print(f'      Média:  {y.mean():.2f}')
        print(f'      Max:    {y.max():.2f}')
    
    return y, df_filtered, measure_types, kept_indices


def load_embeddings_cache(cache_path):
    """
    Carrega embeddings do cache.
    
    Args:
        cache_path: Path para arquivo .npz
        
    Returns:
        embeddings: Array de embeddings
        metadata: Dict com metadados (se disponível)
    """
    cache_path = Path(cache_path)
    
    if not cache_path.exists():
        raise FileNotFoundError(f'Cache de embeddings não encontrado: {cache_path}')
    
    data = np.load(cache_path, allow_pickle=True)
    embeddings = data['embeddings']
    
    # Tentar carregar metadados
    metadata = {}
    if 'metadata' in data:
        metadata = data['metadata'].item()
    
    return embeddings, metadata


def save_embeddings_cache(embeddings, sequences, cache_path, model_name=None):
    """
    Salva embeddings em cache.
    
    Args:
        embeddings: Array de embeddings
        sequences: Lista de sequências
        cache_path: Path para salvar
        model_name: Nome do modelo ESM-2
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
    Carrega índices dos splits do arquivo de stats da classificação.
    
    Isso garante que a regressão use EXATAMENTE os mesmos splits.
    
    Args:
        stats_file: Path para pipeline_stats.json da classificação
        
    Returns:
        idx_train, idx_val, idx_test: Arrays com índices
        
    Raises:
        FileNotFoundError: Se arquivo não existir
        KeyError: Se arquivo não tiver informações de split
    """
    stats_file = Path(stats_file)
    
    if not stats_file.exists():
        raise FileNotFoundError(
            f'Arquivo de stats não encontrado: {stats_file}\n'
            f'Execute o pipeline de classificação primeiro.'
        )
    
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    
    # Verificar se tem informações de split
    if 'split_indices' not in stats:
        raise KeyError(
            'Arquivo de stats não contém informações de split.\n'
            'Atualize o pipeline de classificação para salvar índices dos splits.'
        )
    
    split_info = stats['split_indices']
    idx_train = np.array(split_info['train'])
    idx_val = np.array(split_info['val'])
    idx_test = np.array(split_info['test'])
    
    return idx_train, idx_val, idx_test


def save_split_indices(idx_train, idx_val, idx_test, stats_file):
    """
    Salva índices dos splits no arquivo de stats.
    
    Args:
        idx_train, idx_val, idx_test: Arrays com índices
        stats_file: Path para arquivo JSON
    """
    stats_file = Path(stats_file)
    
    # Carregar stats existentes ou criar novo
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            stats = json.load(f)
    else:
        stats = {}
    
    # Adicionar índices de split
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
        'standard_value': [500, 300, 800, 1200]
    })
    
    print('=== Teste com keep_all=False (remove duplicatas) ===')
    y, df_filtered, types, indices = prepare_regression_targets(df_test, keep_all=False)
    print('\nResultados:')
    print(f'y: {y}')
    print(f'types: {types}')
    print(f'indices: {indices}')
    print(f'df_filtered:\n{df_filtered}')
    
    print('\n=== Teste com keep_all=True (mantém todas) ===')
    y2, df_filtered2, types2, indices2 = prepare_regression_targets(df_test, keep_all=True)
    print('\nResultados:')
    print(f'y: {y2}')
    print(f'types: {types2}')
    print(f'indices: {indices2}')
    print(f'df_filtered:\n{df_filtered2}')
