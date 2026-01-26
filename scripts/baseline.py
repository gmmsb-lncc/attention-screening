#!/usr/bin/env python3
"""
DockTKinase - Baseline Pipeline (Random Split)
===============================================

Script para comparação de desempenho entre:
- Split ALEATÓRIO (este script) → baseline
- Split ESTRATIFICADO (pipeline padrão) → otimizado

Objetivo: Verificar a real eficiência da estratificação por clustering
comparando métricas de modelos treinados com split aleatório.

Configuração:
- Seed fixa: 420 (para reprodutibilidade)
- Split: 80/10/10 ALEATÓRIO (sem estratificação por clusters)
- Modelos ESM-2: 8M, 150M, 3B (embeddings pré-calculados)
- Classificadores: KNN, MLP (apenas esses dois)

Uso:
    python baseline.py
    
    # Ou especificando diretório de embeddings
    python baseline.py --embeddings-dir results/protein_model_benchmark_non_human_v2

Autor: DockTKinase Team
Data: Janeiro 2025
"""

import os
import sys
import json
import time
import warnings
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

# Suprimir warnings
warnings.filterwarnings('ignore', message='X does not have valid feature names')
warnings.filterwarnings('ignore', message='.*was fitted with feature names.*')

# Adicionar path do projeto
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

# Imports sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    matthews_corrcoef, balanced_accuracy_score
)


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

# Seed fixa para baseline
RANDOM_SEED = 420

# Modelos ESM-2 a utilizar (embeddings pré-calculados)
ESM_MODELS = [
    'esm2_t6_8M_UR50D',      # 8M parâmetros, 320-dim
    'esm2_t30_150M_UR50D',   # 150M parâmetros, 640-dim
    'esm2_t36_3B_UR50D',     # 3B parâmetros, 2560-dim
]

# Dimensões dos embeddings por modelo
ESM_DIMS = {
    'esm2_t6_8M_UR50D': 320,
    'esm2_t30_150M_UR50D': 640,
    'esm2_t36_3B_UR50D': 2560,
}

# Proporções do split
TEST_SIZE = 0.1
VAL_SIZE = 0.1


# =============================================================================
# MODELOS: KNN e MLP
# =============================================================================

def create_knn_model(random_state: int = RANDOM_SEED) -> Pipeline:
    """
    Cria modelo KNN com StandardScaler.
    
    Configuração otimizada para embeddings de alta dimensão.
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',      # Ponderar por distância
            algorithm='auto',        # Escolher melhor algoritmo
            leaf_size=30,
            metric='cosine',         # Similaridade de cosseno para embeddings
            n_jobs=-1                # Usar todos os cores
        ))
    ])


def create_mlp_model(random_state: int = RANDOM_SEED) -> Pipeline:
    """
    Cria modelo MLP com StandardScaler.
    
    Arquitetura: 2 camadas ocultas (256, 128 neurônios).
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation='relu',
            solver='adam',
            alpha=0.0001,            # Regularização L2
            batch_size='auto',
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=random_state,
            verbose=False
        ))
    ])


def get_models(random_state: int = RANDOM_SEED) -> Dict[str, Pipeline]:
    """Retorna dicionário com os modelos KNN e MLP."""
    return {
        'KNN': create_knn_model(random_state),
        'MLP': create_mlp_model(random_state),
    }


# =============================================================================
# DATA LOADING
# =============================================================================

def load_embeddings_and_labels(
    embeddings_path: Path,
    labels_path: Path
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carrega embeddings e labels, filtrando labels inválidos.
    
    Args:
        embeddings_path: Caminho para embedding_matrix.npy
        labels_path: Caminho para binary_labels.npy
        
    Returns:
        Tuple (embeddings, labels) filtrados
    """
    print(f"   📂 Carregando: {embeddings_path.name}")
    embeddings = np.load(embeddings_path, allow_pickle=True)
    labels = np.load(labels_path, allow_pickle=True)
    
    # Garantir shape correto
    if embeddings.ndim == 1:
        embeddings = np.vstack(embeddings)
    if labels.ndim > 1:
        labels = labels.ravel()
    
    # Filtrar labels inválidos (-1)
    valid_mask = np.isin(labels, [0, 1])
    n_invalid = (~valid_mask).sum()
    
    if n_invalid > 0:
        print(f"   ⚠️  Removendo {n_invalid} amostras com labels inválidos")
        embeddings = embeddings[valid_mask]
        labels = labels[valid_mask].astype(int)
    
    print(f"   ✅ Carregado: {embeddings.shape[0]:,} amostras, {embeddings.shape[1]} dimensões")
    
    return embeddings, labels


def random_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    random_state: int = RANDOM_SEED
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Divide dados em treino/validação/teste de forma ALEATÓRIA.
    
    IMPORTANTE: NÃO usa estratificação por clusters!
    Apenas mantém proporção de classes (stratify=y).
    
    Args:
        X: Embeddings
        y: Labels
        test_size: Proporção do teste (default: 0.1)
        val_size: Proporção da validação (default: 0.1)
        random_state: Seed aleatória (default: 420)
        
    Returns:
        Tuple (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Primeiro split: separar teste
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Mantém proporção de classes, mas SEM clustering
    )
    
    # Segundo split: separar validação do treino
    val_size_adjusted = val_size / (1 - test_size)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


# =============================================================================
# TRAINING & EVALUATION
# =============================================================================

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calcula métricas de classificação.
    
    Args:
        y_true: Labels verdadeiros
        y_pred: Labels preditos
        y_proba: Probabilidades (opcional, para ROC-AUC)
        
    Returns:
        Dict com métricas
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
    }
    
    # ROC-AUC se probabilidades disponíveis
    if y_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics['roc_auc'] = 0.0
    
    return metrics


def train_and_evaluate(
    model: Pipeline,
    model_name: str,
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Treina modelo e avalia em validação e teste.
    
    Args:
        model: Pipeline sklearn
        model_name: Nome do modelo
        X_train, X_val, X_test: Features
        y_train, y_val, y_test: Labels
        verbose: Mostrar progresso
        
    Returns:
        Dict com métricas de treino, validação e teste
    """
    start_time = time.time()
    
    if verbose:
        print(f"\n   🔧 Treinando {model_name}...")
    
    # Treinar
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Predições
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    # Probabilidades (se disponível)
    y_train_proba = None
    y_val_proba = None
    y_test_proba = None
    
    if hasattr(model, 'predict_proba'):
        try:
            y_train_proba = model.predict_proba(X_train)[:, 1]
            y_val_proba = model.predict_proba(X_val)[:, 1]
            y_test_proba = model.predict_proba(X_test)[:, 1]
        except Exception:
            pass
    
    # Calcular métricas
    train_metrics = calculate_metrics(y_train, y_train_pred, y_train_proba)
    val_metrics = calculate_metrics(y_val, y_val_pred, y_val_proba)
    test_metrics = calculate_metrics(y_test, y_test_pred, y_test_proba)
    
    if verbose:
        print(f"      ✅ Concluído em {train_time:.2f}s")
        print(f"      📊 Val ROC-AUC: {val_metrics.get('roc_auc', 0):.4f}")
        print(f"      📊 Test ROC-AUC: {test_metrics.get('roc_auc', 0):.4f}")
    
    return {
        'model_name': model_name,
        'train_time_seconds': train_time,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
    }


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_baseline_for_model(
    esm_model: str,
    embeddings_dir: Path,
    output_dir: Path,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Executa baseline para um modelo ESM-2 específico.
    
    Args:
        esm_model: Nome do modelo ESM-2
        embeddings_dir: Diretório com embeddings pré-calculados
        output_dir: Diretório de saída
        verbose: Mostrar progresso
        
    Returns:
        Dict com resultados
    """
    print(f"\n{'='*70}")
    print(f"🧬 Modelo: {esm_model} ({ESM_DIMS.get(esm_model, '?')}-dim)")
    print(f"{'='*70}")
    
    # Caminhos
    model_dir = embeddings_dir / esm_model / 'build'
    embeddings_path = model_dir / 'embedding_matrix.npy'
    labels_path = model_dir / 'binary_labels.npy'
    
    # Verificar existência
    if not embeddings_path.exists():
        print(f"   ❌ Embeddings não encontrados: {embeddings_path}")
        return {'error': f'Embeddings not found: {embeddings_path}'}
    
    if not labels_path.exists():
        print(f"   ❌ Labels não encontrados: {labels_path}")
        return {'error': f'Labels not found: {labels_path}'}
    
    # Carregar dados
    X, y = load_embeddings_and_labels(embeddings_path, labels_path)
    
    # Split ALEATÓRIO (seed 420)
    print(f"\n   🎲 Split ALEATÓRIO (seed={RANDOM_SEED})")
    print(f"      Proporção: 80% treino / 10% validação / 10% teste")
    
    X_train, X_val, X_test, y_train, y_val, y_test = random_split(
        X, y,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
        random_state=RANDOM_SEED
    )
    
    # Estatísticas do split
    print(f"\n   📊 Distribuição:")
    print(f"      Treino:    {len(X_train):,} amostras ({len(X_train)/len(X)*100:.1f}%)")
    print(f"                 Positivos: {(y_train==1).sum():,} ({(y_train==1).mean()*100:.1f}%)")
    print(f"      Validação: {len(X_val):,} amostras ({len(X_val)/len(X)*100:.1f}%)")
    print(f"                 Positivos: {(y_val==1).sum():,} ({(y_val==1).mean()*100:.1f}%)")
    print(f"      Teste:     {len(X_test):,} amostras ({len(X_test)/len(X)*100:.1f}%)")
    print(f"                 Positivos: {(y_test==1).sum():,} ({(y_test==1).mean()*100:.1f}%)")
    
    # Treinar modelos
    print(f"\n   🏋️ Treinando modelos (KNN, MLP)...")
    
    models = get_models(random_state=RANDOM_SEED)
    results = {}
    
    for model_name, model in models.items():
        result = train_and_evaluate(
            model, model_name,
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            verbose=verbose
        )
        results[model_name] = result
    
    # Compilar resultados
    return {
        'esm_model': esm_model,
        'embedding_dim': ESM_DIMS.get(esm_model, X.shape[1]),
        'n_samples': len(X),
        'split': {
            'method': 'random',
            'seed': RANDOM_SEED,
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test),
            'test_proportion': TEST_SIZE,
            'val_proportion': VAL_SIZE,
        },
        'class_distribution': {
            'total_positive': int((y == 1).sum()),
            'total_negative': int((y == 0).sum()),
            'positive_ratio': float((y == 1).mean()),
        },
        'models': results,
    }


def run_baseline_pipeline(
    embeddings_dir: Path,
    output_dir: Path,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Executa pipeline baseline completo para todos os modelos ESM-2.
    
    Args:
        embeddings_dir: Diretório com embeddings pré-calculados
        output_dir: Diretório de saída
        verbose: Mostrar progresso
        
    Returns:
        Dict com todos os resultados
    """
    start_time = time.time()
    
    print("="*70)
    print("🎯 BASELINE PIPELINE - SPLIT ALEATÓRIO (SEM ESTRATIFICAÇÃO)")
    print("="*70)
    print()
    print(f"📋 Configuração:")
    print(f"   Seed: {RANDOM_SEED}")
    print(f"   Split: {int((1-TEST_SIZE-VAL_SIZE)*100)}% treino / {int(VAL_SIZE*100)}% val / {int(TEST_SIZE*100)}% teste")
    print(f"   Modelos ESM-2: {', '.join(ESM_MODELS)}")
    print(f"   Classificadores: KNN, MLP")
    print(f"   Embeddings: {embeddings_dir}")
    print(f"   Output: {output_dir}")
    print()
    
    # Criar diretório de saída
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Executar para cada modelo ESM-2
    all_results = {
        'pipeline': 'baseline_random_split',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'random_seed': RANDOM_SEED,
            'test_size': TEST_SIZE,
            'val_size': VAL_SIZE,
            'esm_models': ESM_MODELS,
            'classifiers': ['KNN', 'MLP'],
            'embeddings_dir': str(embeddings_dir),
        },
        'results': {}
    }
    
    for esm_model in ESM_MODELS:
        result = run_baseline_for_model(
            esm_model,
            embeddings_dir,
            output_dir,
            verbose=verbose
        )
        all_results['results'][esm_model] = result
    
    total_time = time.time() - start_time
    all_results['total_time_seconds'] = total_time
    
    # Salvar resultados
    results_file = output_dir / 'baseline_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print()
    print("="*70)
    print("✅ BASELINE PIPELINE CONCLUÍDO")
    print("="*70)
    print()
    
    # Resumo
    print("📊 RESUMO DOS RESULTADOS (Test ROC-AUC):")
    print("-"*70)
    print(f"{'Modelo ESM-2':<25} {'KNN':<12} {'MLP':<12}")
    print("-"*70)
    
    for esm_model in ESM_MODELS:
        result = all_results['results'].get(esm_model, {})
        if 'error' in result:
            print(f"{esm_model:<25} {'ERROR':<12} {'ERROR':<12}")
            continue
        
        models = result.get('models', {})
        knn_auc = models.get('KNN', {}).get('test_metrics', {}).get('roc_auc', 0)
        mlp_auc = models.get('MLP', {}).get('test_metrics', {}).get('roc_auc', 0)
        
        print(f"{esm_model:<25} {knn_auc:<12.4f} {mlp_auc:<12.4f}")
    
    print("-"*70)
    print(f"\n⏱️  Tempo total: {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"💾 Resultados salvos em: {results_file}")
    print()
    
    return all_results


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    """Parse argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description='DockTKinase - Baseline Pipeline (Random Split)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
    python baseline.py
    python baseline.py --embeddings-dir results/protein_model_benchmark_non_human_v2
    python baseline.py --output results/baseline_experiment
        """
    )
    
    parser.add_argument(
        '--embeddings-dir',
        type=str,
        default='results/protein_model_benchmark_non_human_v2',
        help='Diretório com embeddings pré-calculados (default: results/protein_model_benchmark_non_human_v2)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='results/baseline_random_split',
        help='Diretório de saída (default: results/baseline_random_split)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Modo silencioso'
    )
    
    return parser.parse_args()


def main():
    """Função principal."""
    args = parse_args()
    
    embeddings_dir = Path(args.embeddings_dir)
    output_dir = Path(args.output)
    
    if not embeddings_dir.exists():
        print(f"❌ Diretório de embeddings não encontrado: {embeddings_dir}")
        print(f"   Execute primeiro o pipeline completo para gerar os embeddings.")
        return 1
    
    results = run_baseline_pipeline(
        embeddings_dir=embeddings_dir,
        output_dir=output_dir,
        verbose=not args.quiet
    )
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
