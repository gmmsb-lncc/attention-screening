#!/usr/bin/env python3
"""
DockTKinase - Stratified Pipeline Multi-Seed
============================================

Executa baseline ESTRATIFICADO com 5 seeds diferentes para avaliação robusta.

Configuração:
- Seeds: [42, 123, 420, 777, 2024]
- Split: 80/10/10 ESTRATIFICADO (clustering adaptativo)
- Modelos ESM-2: 8M, 150M, 3B
- Classificadores: KNN, MLP

Saídas:
- Resultados individuais por seed
- Média ± desvio padrão de cada métrica
- Visualizações comparativas

Uso:
    python stratified_multiseed.py
    python stratified_multiseed.py --n-seeds 5

Autor: DockTKinase Team
Data: Janeiro 2026
"""

import os
import sys
import json
import time
import warnings
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

# Suprimir warnings
warnings.filterwarnings('ignore')

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
    roc_auc_score, matthews_corrcoef, balanced_accuracy_score
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

# Seeds para experimentos reproduzíveis
SEEDS = [42, 123, 420, 777, 2024]

# Modelos ESM-2
ESM_MODELS = [
    'esm2_t6_8M_UR50D',      # 8M parâmetros, 320-dim
    'esm2_t30_150M_UR50D',   # 150M parâmetros, 640-dim
    'esm2_t36_3B_UR50D',     # 3B parâmetros, 2560-dim
]

ESM_DIMS = {
    'esm2_t6_8M_UR50D': 320,
    'esm2_t30_150M_UR50D': 640,
    'esm2_t36_3B_UR50D': 2560,
}

# Proporções do split
TEST_SIZE = 0.1
VAL_SIZE = 0.1


# =============================================================================
# MODELOS
# =============================================================================

def create_knn_model() -> Pipeline:
    """Cria modelo KNN com StandardScaler."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            metric='cosine',
            n_jobs=-1
        ))
    ])


def create_mlp_model(random_state: int) -> Pipeline:
    """Cria modelo MLP com StandardScaler."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size=32,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=random_state,
            verbose=False
        ))
    ])


# =============================================================================
# ESTRATIFICAÇÃO POR CLUSTERING
# =============================================================================

def stratified_cluster_split(
    embeddings: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_state: int = 42,
    target_cluster_ratio: float = 0.01
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Realiza split estratificado baseado em clusters de similaridade.
    
    Args:
        embeddings: Matriz de embeddings [N, D]
        labels: Labels binários
        test_size: Proporção para teste
        val_size: Proporção para validação
        random_state: Seed
        target_cluster_ratio: Proporção alvo de clusters (~1% do dataset)
        
    Returns:
        train_idx, val_idx, test_idx, metrics
    """
    np.random.seed(random_state)
    n_samples = len(embeddings)
    
    # Calcular similaridade de cosseno (amostrar se muito grande)
    if n_samples > 5000:
        sample_idx = np.random.choice(n_samples, 5000, replace=False)
        sample_emb = embeddings[sample_idx]
        sim_matrix_sample = cosine_similarity(sample_emb)
        
        # Estatísticas da amostra
        triu_idx = np.triu_indices(len(sample_emb), k=1)
        similarities = sim_matrix_sample[triu_idx]
    else:
        sim_matrix = cosine_similarity(embeddings)
        triu_idx = np.triu_indices(n_samples, k=1)
        similarities = sim_matrix[triu_idx]
    
    # Estatísticas de similaridade
    sim_stats = {
        'min': float(np.min(similarities)),
        'max': float(np.max(similarities)),
        'mean': float(np.mean(similarities)),
        'std': float(np.std(similarities)),
        'p25': float(np.percentile(similarities, 25)),
        'p50': float(np.percentile(similarities, 50)),
        'p75': float(np.percentile(similarities, 75)),
        'p90': float(np.percentile(similarities, 90)),
        'p95': float(np.percentile(similarities, 95)),
    }
    
    # Definir threshold adaptativo
    target_clusters = max(10, int(n_samples * target_cluster_ratio))
    
    # Calcular matriz completa de distância
    sim_matrix_full = cosine_similarity(embeddings)
    distance_matrix = np.clip(1 - sim_matrix_full, 0, 2)
    
    # Busca binária para encontrar threshold que produz ~target clusters
    low, high = sim_stats['min'], sim_stats['max']
    best_threshold = (low + high) / 2
    best_n_clusters = 0
    
    for iteration in range(20):
        mid = (low + high) / 2
        distance_thresh = 1 - mid
        
        try:
            model = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_thresh,
                metric='precomputed',
                linkage='average'
            )
            cluster_labels = model.fit_predict(distance_matrix)
            n_clusters = len(np.unique(cluster_labels))
            
            if abs(n_clusters - target_clusters) < abs(best_n_clusters - target_clusters):
                best_n_clusters = n_clusters
                best_threshold = mid
            
            if n_clusters < target_clusters:
                low = mid
            elif n_clusters > target_clusters:
                high = mid
            else:
                break
                
            if high - low < 0.001:
                break
        except:
            break
    
    # Clustering final com melhor threshold
    distance_thresh = 1 - best_threshold
    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_thresh,
        metric='precomputed',
        linkage='average'
    )
    cluster_labels = model.fit_predict(distance_matrix)
    n_clusters = len(np.unique(cluster_labels))
    
    # Agrupar amostras por cluster
    clusters = {}
    for idx, cluster_id in enumerate(cluster_labels):
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(idx)
    
    # Dividir cada cluster proporcionalmente
    train_idx, val_idx, test_idx = [], [], []
    
    for cluster_id, cluster_samples in clusters.items():
        n_cluster = len(cluster_samples)
        
        # Garantir pelo menos 1 amostra em cada split se cluster >= 3
        if n_cluster >= 3:
            n_test = max(1, int(n_cluster * test_size))
            n_val = max(1, int(n_cluster * val_size))
            n_train = n_cluster - n_test - n_val
            
            # Ajustar se necessário
            if n_train < 1:
                n_train = 1
                if n_val > 1:
                    n_val -= 1
                elif n_test > 1:
                    n_test -= 1
        elif n_cluster == 2:
            n_train, n_val, n_test = 1, 1, 0
        else:
            n_train, n_val, n_test = 1, 0, 0
        
        # Shuffle e dividir
        np.random.shuffle(cluster_samples)
        train_idx.extend(cluster_samples[:n_train])
        val_idx.extend(cluster_samples[n_train:n_train + n_val])
        test_idx.extend(cluster_samples[n_train + n_val:])
    
    # Shuffle final
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)
    
    # Métricas do clustering
    metrics = {
        'n_clusters': n_clusters,
        'threshold': best_threshold,
        'similarity_stats': sim_stats,
        'cluster_sizes': {
            'min': int(np.min([len(c) for c in clusters.values()])),
            'max': int(np.max([len(c) for c in clusters.values()])),
            'mean': float(np.mean([len(c) for c in clusters.values()])),
        }
    }
    
    return np.array(train_idx), np.array(val_idx), np.array(test_idx), metrics


# =============================================================================
# AVALIAÇÃO
# =============================================================================

def evaluate_model(
    model: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str
) -> Dict[str, Any]:
    """Treina e avalia modelo."""
    
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    
    # Predições
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1]
    
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Métricas
    def compute_metrics(y_true, y_pred, y_proba):
        return {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, zero_division=0)),
            'mcc': float(matthews_corrcoef(y_true, y_pred)),
            'roc_auc': float(roc_auc_score(y_true, y_proba)),
            'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        }
    
    return {
        'model_name': model_name,
        'train_time': train_time,
        'train_metrics': compute_metrics(y_train, y_train_pred, y_train_proba),
        'val_metrics': compute_metrics(y_val, y_val_pred, y_val_proba),
        'test_metrics': compute_metrics(y_test, y_test_pred, y_test_proba),
    }


# =============================================================================
# PIPELINE MULTI-SEED
# =============================================================================

def run_multiseed_pipeline(
    embeddings_dir: Path,
    output_dir: Path,
    n_seeds: int = 5
):
    """Executa pipeline com múltiplas seeds."""
    
    # Seeds
    seeds = SEEDS[:n_seeds]
    
    print("=" * 70)
    print("🎯 STRATIFIED BASELINE MULTI-SEED PIPELINE")
    print("=" * 70)
    print()
    print(f"📋 Configuração:")
    print(f"   Seeds: {seeds}")
    print(f"   Split: 80% treino / 10% val / 10% teste (ESTRATIFICADO)")
    print(f"   Modelos ESM-2: {', '.join(ESM_MODELS)}")
    print(f"   Classificadores: KNN, MLP")
    print()
    
    # Estrutura de resultados
    all_results = {
        'pipeline': 'stratified_multiseed',
        'timestamp': datetime.now().isoformat(),
        'seeds': seeds,
        'n_seeds': n_seeds,
        'models': {}
    }
    
    # Para cada modelo ESM-2
    for esm_model in ESM_MODELS:
        print("=" * 70)
        print(f"🧬 Modelo: {esm_model}")
        print("=" * 70)
        print()
        
        # Carregar embeddings (formato baseline)
        model_dir = embeddings_dir / esm_model / 'build'
        embeddings_file = model_dir / 'embedding_matrix.npy'
        labels_file = model_dir / 'binary_labels.npy'
        
        X = np.load(embeddings_file, allow_pickle=True)
        y = np.load(labels_file, allow_pickle=True)
        
        # Garantir shape correto
        if X.ndim == 1:
            X = np.vstack(X)
        if y.ndim > 1:
            y = y.ravel()
        
        # Filtrar labels inválidos (-1)
        valid_mask = np.isin(y, [0, 1])
        n_invalid = (~valid_mask).sum()
        
        if n_invalid > 0:
            print(f"   ⚠️  Removendo {n_invalid} amostras com labels inválidos")
            X = X[valid_mask]
            y = y[valid_mask].astype(int)
        
        # Resultados por seed
        seed_results = []
        
        # Para cada seed
        for seed in seeds:
            print(f"   🎲 Seed {seed}:")
            
            # Split estratificado
            train_idx, val_idx, test_idx, cluster_metrics = stratified_cluster_split(
                X, y,
                test_size=TEST_SIZE,
                val_size=VAL_SIZE,
                random_state=seed
            )
            
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            
            # Resultados desta seed
            seed_result = {
                'seed': seed,
                'n_train': len(train_idx),
                'n_val': len(val_idx),
                'n_test': len(test_idx),
                'clustering': cluster_metrics,
                'classifiers': {}
            }
            
            # KNN
            knn_model = create_knn_model()
            knn_results = evaluate_model(
                knn_model, X_train, y_train, X_val, y_val, X_test, y_test, 'KNN'
            )
            seed_result['classifiers']['KNN'] = knn_results
            print(f"      KNN: ROC-AUC={knn_results['test_metrics']['roc_auc']:.4f} ({knn_results['train_time']:.1f}s)")
            
            # MLP
            mlp_model = create_mlp_model(random_state=seed)
            mlp_results = evaluate_model(
                mlp_model, X_train, y_train, X_val, y_val, X_test, y_test, 'MLP'
            )
            seed_result['classifiers']['MLP'] = mlp_results
            print(f"      MLP: ROC-AUC={mlp_results['test_metrics']['roc_auc']:.4f} ({mlp_results['train_time']:.1f}s)")
            print()
            
            seed_results.append(seed_result)
        
        # Calcular médias e desvios
        summary = calculate_summary(seed_results, esm_model)
        
        all_results['models'][esm_model] = {
            'seed_results': seed_results,
            'summary': summary
        }
        
        # Print summary
        print(f"   📊 Resumo {esm_model}:")
        for clf_name in ['KNN', 'MLP']:
            mean_auc = summary[clf_name]['test_metrics']['roc_auc']['mean']
            std_auc = summary[clf_name]['test_metrics']['roc_auc']['std']
            print(f"      {clf_name}: ROC-AUC = {mean_auc:.4f} ± {std_auc:.4f}")
    
    # Salvar resultados
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / 'stratified_multiseed_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Visualização
    plot_file = output_dir / 'stratified_multiseed_results.png'
    plot_results(all_results, plot_file)
    
    print(f"📊 Visualização salva em: {plot_file}")
    print()
    
    # Print tabela final
    print_summary_table(all_results)
    
    return all_results


def calculate_summary(seed_results: List[Dict], esm_model: str) -> Dict[str, Any]:
    """Calcula média e desvio padrão das métricas."""
    
    classifiers = ['KNN', 'MLP']
    metrics_names = ['accuracy', 'precision', 'recall', 'f1', 'mcc', 'roc_auc', 'balanced_accuracy']
    
    summary = {}
    
    for clf_name in classifiers:
        summary[clf_name] = {
            'train_metrics': {},
            'val_metrics': {},
            'test_metrics': {},
            'train_time': {}
        }
        
        # Coletar valores de cada seed
        for metric in metrics_names:
            train_values = [r['classifiers'][clf_name]['train_metrics'][metric] for r in seed_results]
            val_values = [r['classifiers'][clf_name]['val_metrics'][metric] for r in seed_results]
            test_values = [r['classifiers'][clf_name]['test_metrics'][metric] for r in seed_results]
            
            summary[clf_name]['train_metrics'][metric] = {
                'mean': float(np.mean(train_values)),
                'std': float(np.std(train_values)),
                'values': train_values
            }
            summary[clf_name]['val_metrics'][metric] = {
                'mean': float(np.mean(val_values)),
                'std': float(np.std(val_values)),
                'values': val_values
            }
            summary[clf_name]['test_metrics'][metric] = {
                'mean': float(np.mean(test_values)),
                'std': float(np.std(test_values)),
                'values': test_values
            }
        
        # Train time
        times = [r['classifiers'][clf_name]['train_time'] for r in seed_results]
        summary[clf_name]['train_time'] = {
            'mean': float(np.mean(times)),
            'std': float(np.std(times)),
            'values': times
        }
    
    return summary


def plot_results(results: Dict[str, Any], output_file: Path):
    """Gera visualização dos resultados."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    models = list(results['models'].keys())
    classifiers = ['KNN', 'MLP']
    seeds = results['seeds']
    n_seeds = len(seeds)
    
    x = np.arange(len(models))
    width = 0.35
    
    # Para cada métrica
    metrics_to_plot = ['roc_auc', 'f1', 'balanced_accuracy']
    metric_labels = ['ROC-AUC', 'F1-Score', 'Balanced Accuracy']
    
    for ax_idx, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
        ax = axes[ax_idx]
        
        for clf_idx, clf_name in enumerate(classifiers):
            means = []
            stds = []
            
            for model in models:
                summary = results['models'][model]['summary']
                mean_val = summary[clf_name]['test_metrics'][metric]['mean']
                std_val = summary[clf_name]['test_metrics'][metric]['std']
                means.append(mean_val)
                stds.append(std_val)
            
            offset = width * (clf_idx - 0.5)
            bars = ax.bar(x + offset, means, width, yerr=stds, 
                         label=clf_name, capsize=5, alpha=0.8)
            
            # Valores em cima das barras
            for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + std,
                       f'{mean:.3f}',
                       ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Modelo ESM-2', fontsize=12, fontweight='bold')
        ax.set_ylabel(label, fontsize=12, fontweight='bold')
        ax.set_title(f'{label} (Test Set)\n{n_seeds} seeds', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('esm2_', '').replace('_UR50D', '') for m in models], 
                          rotation=15, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle(f'Stratified Multi-Seed Results ({n_seeds} seeds)', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def print_summary_table(results: Dict[str, Any]):
    """Imprime tabela resumo formatada."""
    
    seeds = results['seeds']
    models = list(results['models'].keys())
    
    print("=" * 150)
    print(" " * 60 + "📊 RESUMO DOS RESULTADOS (Test ROC-AUC)")
    print("=" * 150)
    
    # Header
    header = f"{'Modelo ESM-2':<25} {'Clf':<5}"
    for seed in seeds:
        header += f"Seed {seed:<7}"
    header += f"{'Média':<10} {'±Std':<10}"
    print(header)
    print("-" * 150)
    
    # Dados
    for model in models:
        for clf_name in ['KNN', 'MLP']:
            summary = results['models'][model]['summary']
            seed_values = summary[clf_name]['test_metrics']['roc_auc']['values']
            mean_val = summary[clf_name]['test_metrics']['roc_auc']['mean']
            std_val = summary[clf_name]['test_metrics']['roc_auc']['std']
            
            model_display = model.replace('_UR50D', '') if clf_name == 'KNN' else ''
            
            row = f"{model_display:<25} {clf_name:<5}"
            for val in seed_values:
                row += f"{val:<8.4f}"
            row += f"{mean_val:<10.4f}±{std_val:<.4f}"
            print(row)
        print()
    
    print("=" * 150)
    print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='DockTKinase Stratified Multi-Seed Pipeline')
    parser.add_argument('--embeddings-dir', type=str, 
                       default='results/protein_model_benchmark_non_human_v2',
                       help='Diretório com embeddings')
    parser.add_argument('--output-dir', type=str,
                       default='results/stratified_multiseed',
                       help='Diretório de saída')
    parser.add_argument('--n-seeds', type=int, default=5,
                       help='Número de seeds (1-5)')
    
    args = parser.parse_args()
    
    embeddings_dir = Path(args.embeddings_dir)
    output_dir = Path(args.output_dir)
    
    if not embeddings_dir.exists():
        print(f"❌ Diretório não encontrado: {embeddings_dir}")
        return 1
    
    if args.n_seeds < 1 or args.n_seeds > 5:
        print(f"❌ n_seeds deve estar entre 1 e 5")
        return 1
    
    start_time = time.time()
    
    try:
        results = run_multiseed_pipeline(embeddings_dir, output_dir, args.n_seeds)
        
        elapsed = time.time() - start_time
        print(f"⏱️  Tempo total: {elapsed:.2f}s ({elapsed/60:.2f} min)")
        print(f"💾 Resultados salvos em: {output_dir / 'stratified_multiseed_results.json'}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
