#!/usr/bin/env python3
"""
DockTKinase - Stratified Pipeline (Comparação com Baseline)
============================================================

Script para comparação de desempenho entre:
- Split ALEATÓRIO (baseline.py) → baseline
- Split ESTRATIFICADO (este script) → otimizado

Objetivo: Verificar a real eficiência da estratificação por clustering
comparando métricas de modelos treinados com split estratificado.

Configuração:
- Seed fixa: 420 (mesma do baseline para comparação justa)
- Split: 80/10/10 ESTRATIFICADO (com clustering adaptativo)
- Modelos ESM-2: 8M, 150M, 3B (embeddings pré-calculados)
- Classificadores: KNN, MLP

Uso:
    python stratified_baseline.py
    python stratified_baseline.py --embeddings-dir results/protein_model_benchmark_non_human_v2

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
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

# Suprimir warnings
warnings.filterwarnings('ignore', message='X does not have valid feature names')
warnings.filterwarnings('ignore', message='.*was fitted with feature names.*')
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
    roc_auc_score, confusion_matrix, classification_report,
    matthews_corrcoef, balanced_accuracy_score
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

RANDOM_SEED = 420

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

TEST_SIZE = 0.1  # 10% para teste
VAL_SIZE = 0.1   # 10% para validação (do total, não do treino)


# =============================================================================
# MODELOS: KNN e MLP
# =============================================================================

def create_knn_model(random_state: int = RANDOM_SEED) -> Pipeline:
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


def create_mlp_model(random_state: int = RANDOM_SEED) -> Pipeline:
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
    random_state: int = RANDOM_SEED,
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
    
    print(f"   🔬 Calculando matriz de similaridade de cosseno...")
    
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
    
    print(f"      Similaridade: min={sim_stats['min']:.3f}, mean={sim_stats['mean']:.3f}, max={sim_stats['max']:.3f}")
    
    # Definir threshold adaptativo (método 'target')
    target_clusters = max(10, int(n_samples * target_cluster_ratio))
    print(f"   🎯 Buscando threshold para ~{target_clusters} clusters...")
    
    # Calcular matriz completa de distância
    print(f"   📊 Calculando matriz de distância...")
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
        except Exception:
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
    
    print(f"      Threshold ótimo: {best_threshold:.4f}")
    print(f"      Clusters gerados: {n_clusters}")
    
    # Agrupar amostras por cluster
    clusters = {}
    for idx, cluster_id in enumerate(cluster_labels):
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(idx)
    
    # Dividir cada cluster proporcionalmente
    train_idx, val_idx, test_idx = [], [], []
    
    for cluster_id, indices in clusters.items():
        n = len(indices)
        np.random.shuffle(indices)
        
        if n == 1:
            train_idx.extend(indices)
        elif n == 2:
            train_idx.append(indices[0])
            test_idx.append(indices[1])
        else:
            n_test = max(1, int(n * test_size))
            n_val = max(1, int(n * val_size))
            
            test_idx.extend(indices[:n_test])
            val_idx.extend(indices[n_test:n_test + n_val])
            train_idx.extend(indices[n_test + n_val:])
    
    # Garantir que val não está vazio
    if len(val_idx) == 0 and len(train_idx) > 10:
        n_move = max(1, int(len(train_idx) * 0.1))
        move_idx = np.random.choice(len(train_idx), n_move, replace=False)
        val_idx = [train_idx[i] for i in move_idx]
        train_idx = [train_idx[i] for i in range(len(train_idx)) if i not in move_idx]
    
    metrics = {
        'threshold': best_threshold,
        'n_clusters': n_clusters,
        'similarity_stats': sim_stats,
    }
    
    return np.array(train_idx), np.array(val_idx), np.array(test_idx), metrics


# =============================================================================
# MÉTRICAS COMPLETAS
# =============================================================================

def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Calcula todas as métricas de classificação."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_prob),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
    }


# =============================================================================
# CARREGAMENTO DE DADOS
# =============================================================================

def load_embeddings(embeddings_dir: Path, model_name: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Carrega embeddings e labels de um modelo."""
    model_dir = embeddings_dir / model_name / "build"
    
    if not model_dir.exists():
        model_dir = embeddings_dir / model_name
    
    embeddings_path = model_dir / "embedding_matrix.npy"
    labels_path = model_dir / "binary_labels.npy"
    
    if not embeddings_path.exists():
        return None, None
    
    embeddings = np.load(embeddings_path)
    
    if labels_path.exists():
        labels = np.load(labels_path)
    else:
        labels_alt = model_dir / "labels.npy"
        if labels_alt.exists():
            labels = np.load(labels_alt)
        else:
            return None, None
    
    return embeddings, labels


# =============================================================================
# VISUALIZAÇÃO COMPARATIVA
# =============================================================================

def load_baseline_results(baseline_path: Path) -> Optional[Dict]:
    """Carrega resultados do baseline."""
    results_file = baseline_path / "baseline_results.json"
    if results_file.exists():
        with open(results_file, 'r') as f:
            return json.load(f)
    return None


def create_comparison_visualization(
    baseline_results: Dict,
    stratified_results: Dict,
    output_dir: Path
):
    """
    Cria visualização comparativa 2x3 entre baseline (random) e estratificado.
    
    Métricas:
    - Accuracy, Precision, Recall (linha 1)
    - F1-Score, MCC, ROC-AUC (linha 2)
    """
    metrics_names = ['accuracy', 'precision', 'recall', 'f1', 'mcc', 'roc_auc']
    metrics_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'MCC', 'ROC-AUC']
    
    models = ['esm2_t6_8M_UR50D', 'esm2_t30_150M_UR50D', 'esm2_t36_3B_UR50D']
    model_labels = ['ESM-2 8M', 'ESM-2 150M', 'ESM-2 3B']
    classifiers = ['KNN', 'MLP']
    
    # Cores
    colors = {
        'Random_KNN': '#e74c3c',      # Vermelho
        'Random_MLP': '#c0392b',      # Vermelho escuro
        'Stratified_KNN': '#3498db',  # Azul
        'Stratified_MLP': '#2980b9',  # Azul escuro
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Comparação: Split Aleatório vs Estratificado\n(Dataset Kinases Não-Humanas)', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    for idx, (metric_name, metric_label) in enumerate(zip(metrics_names, metrics_labels)):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        x = np.arange(len(models))
        width = 0.2
        
        # Extrair valores para cada combinação
        random_knn = []
        random_mlp = []
        strat_knn = []
        strat_mlp = []
        
        for model in models:
            # Baseline (random) - estrutura: results -> esm_model -> models -> KNN/MLP
            baseline_data = baseline_results.get('results', {}).get(model, {}).get('models', {})
            if baseline_data:
                if 'KNN' in baseline_data:
                    val = baseline_data['KNN'].get('test_metrics', {}).get(metric_name, 0)
                    random_knn.append(val)
                else:
                    random_knn.append(0)
                if 'MLP' in baseline_data:
                    val = baseline_data['MLP'].get('test_metrics', {}).get(metric_name, 0)
                    random_mlp.append(val)
                else:
                    random_mlp.append(0)
            else:
                random_knn.append(0)
                random_mlp.append(0)
            
            # Estratificado - estrutura: models -> esm_model -> KNN/MLP
            if model in stratified_results.get('models', {}):
                model_data = stratified_results['models'][model]
                if 'KNN' in model_data:
                    val = model_data['KNN'].get('test_metrics', {}).get(metric_name, 0)
                    strat_knn.append(val)
                else:
                    strat_knn.append(0)
                if 'MLP' in model_data:
                    val = model_data['MLP'].get('test_metrics', {}).get(metric_name, 0)
                    strat_mlp.append(val)
                else:
                    strat_mlp.append(0)
            else:
                strat_knn.append(0)
                strat_mlp.append(0)
        
        # Plotar barras
        bars1 = ax.bar(x - 1.5*width, random_knn, width, label='Random KNN', color=colors['Random_KNN'], alpha=0.8)
        bars2 = ax.bar(x - 0.5*width, random_mlp, width, label='Random MLP', color=colors['Random_MLP'], alpha=0.8)
        bars3 = ax.bar(x + 0.5*width, strat_knn, width, label='Stratified KNN', color=colors['Stratified_KNN'], alpha=0.8)
        bars4 = ax.bar(x + 1.5*width, strat_mlp, width, label='Stratified MLP', color=colors['Stratified_MLP'], alpha=0.8)
        
        # Adicionar valores nas barras
        for bars in [bars1, bars2, bars3, bars4]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(f'{height:.3f}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=7, rotation=90)
        
        ax.set_ylabel(metric_label, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels)
        ax.set_title(metric_label, fontsize=11, fontweight='bold')
        
        # Ajustar limites do eixo Y
        if metric_name == 'mcc':
            ax.set_ylim(-0.1, 1.1)
        else:
            ax.set_ylim(0, 1.15)
        
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Legenda compartilhada
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.98),
               fontsize=10, frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Salvar
    output_path = output_dir / "comparison_random_vs_stratified.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n📊 Visualização salva em: {output_path}")
    
    # Criar também gráfico de diferença (delta)
    create_delta_visualization(baseline_results, stratified_results, output_dir)


def create_delta_visualization(
    baseline_results: Dict,
    stratified_results: Dict,
    output_dir: Path
):
    """
    Cria visualização mostrando a DIFERENÇA (delta) entre random e estratificado.
    Delta negativo = estratificação mostra resultado mais conservador (esperado).
    """
    metrics_names = ['accuracy', 'precision', 'recall', 'f1', 'mcc', 'roc_auc']
    metrics_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'MCC', 'ROC-AUC']
    
    models = ['esm2_t6_8M_UR50D', 'esm2_t30_150M_UR50D', 'esm2_t36_3B_UR50D']
    model_labels = ['ESM-2 8M', 'ESM-2 150M', 'ESM-2 3B']
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Diferença: (Estratificado - Aleatório)\nValores negativos indicam métricas mais conservadoras com estratificação', 
                 fontsize=12, fontweight='bold', y=1.02)
    
    for idx, (metric_name, metric_label) in enumerate(zip(metrics_names, metrics_labels)):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        x = np.arange(len(models))
        width = 0.35
        
        delta_knn = []
        delta_mlp = []
        
        for model in models:
            # Baseline - estrutura: results -> esm_model -> models -> KNN/MLP
            baseline_data = baseline_results.get('results', {}).get(model, {}).get('models', {})
            base_knn = baseline_data.get('KNN', {}).get('test_metrics', {}).get(metric_name, 0)
            base_mlp = baseline_data.get('MLP', {}).get('test_metrics', {}).get(metric_name, 0)
            
            # Estratificado - estrutura: models -> esm_model -> KNN/MLP
            strat_knn = stratified_results.get('models', {}).get(model, {}).get('KNN', {}).get('test_metrics', {}).get(metric_name, 0)
            strat_mlp = stratified_results.get('models', {}).get(model, {}).get('MLP', {}).get('test_metrics', {}).get(metric_name, 0)
            
            delta_knn.append(strat_knn - base_knn)
            delta_mlp.append(strat_mlp - base_mlp)
        
        # Cores baseadas no sinal (negativo = vermelho, positivo = verde)
        colors_knn = ['#e74c3c' if d < 0 else '#27ae60' for d in delta_knn]
        colors_mlp = ['#c0392b' if d < 0 else '#1e8449' for d in delta_mlp]
        
        bars1 = ax.bar(x - width/2, delta_knn, width, label='KNN', color=colors_knn, alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x + width/2, delta_mlp, width, label='MLP', color=colors_mlp, alpha=0.8, edgecolor='black')
        
        # Linha zero
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        # Valores nas barras
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                va = 'bottom' if height >= 0 else 'top'
                offset = 3 if height >= 0 else -3
                ax.annotate(f'{height:+.3f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, offset),
                            textcoords="offset points",
                            ha='center', va=va, fontsize=8, fontweight='bold')
        
        ax.set_ylabel(f'Δ {metric_label}', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels)
        ax.set_title(metric_label, fontsize=11, fontweight='bold')
        
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Legenda
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', edgecolor='black', label='KNN (Δ < 0: mais conservador)'),
        Patch(facecolor='#c0392b', edgecolor='black', label='MLP (Δ < 0: mais conservador)'),
        Patch(facecolor='#27ae60', edgecolor='black', label='KNN (Δ > 0: melhora)'),
        Patch(facecolor='#1e8449', edgecolor='black', label='MLP (Δ > 0: melhora)'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.98),
               fontsize=9, frameon=True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    output_path = output_dir / "comparison_delta.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"📊 Visualização delta salva em: {output_path}")


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def run_stratified_pipeline(
    embeddings_dir: Path,
    output_dir: Path,
    verbose: bool = True
) -> Dict[str, Any]:
    """Executa pipeline com split estratificado."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'pipeline': 'stratified_baseline',
        'timestamp': datetime.now().isoformat(),
        'seed': RANDOM_SEED,
        'split': 'stratified_clustering',
        'test_size': TEST_SIZE,
        'val_size': VAL_SIZE,
        'models': {}
    }
    
    start_time = time.time()
    
    print('=' * 70)
    print('🎯 STRATIFIED PIPELINE - SPLIT ESTRATIFICADO POR CLUSTERS')
    print('=' * 70)
    print()
    print('📋 Configuração:')
    print(f'   Seed: {RANDOM_SEED}')
    print(f'   Split: 80% treino / 10% val / 10% teste (ESTRATIFICADO)')
    print(f'   Modelos ESM-2: {", ".join(ESM_MODELS)}')
    print(f'   Classificadores: KNN, MLP')
    print(f'   Embeddings: {embeddings_dir}')
    print(f'   Output: {output_dir}')
    print()
    
    for model_name in ESM_MODELS:
        model_dim = ESM_DIMS.get(model_name, 0)
        
        print('=' * 70)
        print(f'🧬 Modelo: {model_name} ({model_dim}-dim)')
        print('=' * 70)
        
        # Carregar embeddings
        embeddings, labels = load_embeddings(embeddings_dir, model_name)
        
        if embeddings is None:
            print(f'   ⚠️  Não encontrado, pulando...')
            continue
        
        print(f'   ✅ Carregado: {len(embeddings):,} amostras, {embeddings.shape[1]} dimensões')
        
        # Split ESTRATIFICADO
        print(f'\n   📊 Split ESTRATIFICADO (seed={RANDOM_SEED})')
        
        train_idx, val_idx, test_idx, cluster_metrics = stratified_cluster_split(
            embeddings, labels,
            test_size=TEST_SIZE,
            val_size=VAL_SIZE,
            random_state=RANDOM_SEED
        )
        
        X_train, y_train = embeddings[train_idx], labels[train_idx]
        X_val, y_val = embeddings[val_idx], labels[val_idx]
        X_test, y_test = embeddings[test_idx], labels[test_idx]
        
        print(f'\n   📊 Distribuição:')
        print(f'      Treino:    {len(train_idx):,} amostras ({100*len(train_idx)/len(embeddings):.1f}%)')
        print(f'                 Positivos: {int(y_train.sum()):,} ({100*y_train.mean():.1f}%)')
        print(f'      Validação: {len(val_idx):,} amostras ({100*len(val_idx)/len(embeddings):.1f}%)')
        print(f'                 Positivos: {int(y_val.sum()):,} ({100*y_val.mean():.1f}%)')
        print(f'      Teste:     {len(test_idx):,} amostras ({100*len(test_idx)/len(embeddings):.1f}%)')
        print(f'                 Positivos: {int(y_test.sum()):,} ({100*y_test.mean():.1f}%)')
        
        results['models'][model_name] = {
            'n_samples': len(embeddings),
            'embedding_dim': embeddings.shape[1],
            'n_train': len(train_idx),
            'n_val': len(val_idx),
            'n_test': len(test_idx),
            'cluster_metrics': cluster_metrics,
        }
        
        print(f'\n   🏋️ Treinando modelos (KNN, MLP)...')
        
        # KNN
        print(f'\n   🔧 Treinando KNN...')
        knn_start = time.time()
        knn_model = create_knn_model(RANDOM_SEED)
        knn_model.fit(X_train, y_train)
        knn_time = time.time() - knn_start
        
        knn_val_prob = knn_model.predict_proba(X_val)[:, 1]
        knn_val_pred = knn_model.predict(X_val)
        knn_test_prob = knn_model.predict_proba(X_test)[:, 1]
        knn_test_pred = knn_model.predict(X_test)
        
        knn_val_metrics = calculate_all_metrics(y_val, knn_val_pred, knn_val_prob)
        knn_test_metrics = calculate_all_metrics(y_test, knn_test_pred, knn_test_prob)
        
        print(f'      ✅ Concluído em {knn_time:.2f}s')
        print(f'      📊 Val ROC-AUC: {knn_val_metrics["roc_auc"]:.4f}')
        print(f'      📊 Test ROC-AUC: {knn_test_metrics["roc_auc"]:.4f}')
        
        results['models'][model_name]['KNN'] = {
            'train_time': knn_time,
            'val_metrics': knn_val_metrics,
            'test_metrics': knn_test_metrics,
        }
        
        # MLP
        print(f'\n   🔧 Treinando MLP...')
        mlp_start = time.time()
        mlp_model = create_mlp_model(RANDOM_SEED)
        mlp_model.fit(X_train, y_train)
        mlp_time = time.time() - mlp_start
        
        mlp_val_prob = mlp_model.predict_proba(X_val)[:, 1]
        mlp_val_pred = mlp_model.predict(X_val)
        mlp_test_prob = mlp_model.predict_proba(X_test)[:, 1]
        mlp_test_pred = mlp_model.predict(X_test)
        
        mlp_val_metrics = calculate_all_metrics(y_val, mlp_val_pred, mlp_val_prob)
        mlp_test_metrics = calculate_all_metrics(y_test, mlp_test_pred, mlp_test_prob)
        
        print(f'      ✅ Concluído em {mlp_time:.2f}s')
        print(f'      📊 Val ROC-AUC: {mlp_val_metrics["roc_auc"]:.4f}')
        print(f'      📊 Test ROC-AUC: {mlp_test_metrics["roc_auc"]:.4f}')
        
        results['models'][model_name]['MLP'] = {
            'train_time': mlp_time,
            'val_metrics': mlp_val_metrics,
            'test_metrics': mlp_test_metrics,
        }
        
        # Criar subdiretório
        model_output_dir = output_dir / model_name
        model_output_dir.mkdir(exist_ok=True)
    
    total_time = time.time() - start_time
    results['total_time_seconds'] = total_time
    
    # Salvar resultados
    results_path = output_dir / 'stratified_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Resumo
    print()
    print('=' * 70)
    print('✅ STRATIFIED PIPELINE CONCLUÍDO')
    print('=' * 70)
    print()
    print('📊 RESUMO DOS RESULTADOS (Test ROC-AUC):')
    print('-' * 70)
    print(f'{"Modelo ESM-2":<25} {"KNN":<12} {"MLP":<12}')
    print('-' * 70)
    
    for model_name in ESM_MODELS:
        if model_name in results['models']:
            model_data = results['models'][model_name]
            knn_auc = model_data.get('KNN', {}).get('test_metrics', {}).get('roc_auc', 0)
            mlp_auc = model_data.get('MLP', {}).get('test_metrics', {}).get('roc_auc', 0)
            print(f'{model_name:<25} {knn_auc:<12.4f} {mlp_auc:<12.4f}')
    
    print('-' * 70)
    print()
    print(f'⏱️  Tempo total: {total_time:.2f}s ({total_time/60:.2f} min)')
    print(f'💾 Resultados salvos em: {results_path}')
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='DockTKinase - Stratified Pipeline (Comparação com Baseline)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
    python stratified_baseline.py
    python stratified_baseline.py --embeddings-dir results/protein_model_benchmark_non_human_v2
    python stratified_baseline.py --output results/stratified_experiment
"""
    )
    
    parser.add_argument(
        '--embeddings-dir',
        type=str,
        default='results/protein_model_benchmark_non_human_v2',
        help='Diretório com embeddings pré-calculados'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='results/stratified_baseline',
        help='Diretório de saída'
    )
    
    parser.add_argument(
        '--baseline-dir',
        type=str,
        default='results/baseline_random_split',
        help='Diretório com resultados do baseline (random split)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Modo silencioso'
    )
    
    args = parser.parse_args()
    
    embeddings_dir = Path(args.embeddings_dir)
    output_dir = Path(args.output)
    baseline_dir = Path(args.baseline_dir)
    
    if not embeddings_dir.exists():
        print(f'❌ Diretório de embeddings não encontrado: {embeddings_dir}')
        return 1
    
    # Executar pipeline estratificado
    stratified_results = run_stratified_pipeline(
        embeddings_dir=embeddings_dir,
        output_dir=output_dir,
        verbose=not args.quiet
    )
    
    # Carregar resultados do baseline e criar visualização
    print()
    print('=' * 70)
    print('📊 GERANDO VISUALIZAÇÕES COMPARATIVAS')
    print('=' * 70)
    
    baseline_results = load_baseline_results(baseline_dir)
    
    if baseline_results:
        create_comparison_visualization(
            baseline_results=baseline_results,
            stratified_results=stratified_results,
            output_dir=output_dir
        )
        
        # Tabela comparativa
        print()
        print('📋 TABELA COMPARATIVA (Test ROC-AUC):')
        print('-' * 80)
        print(f'{"Modelo":<20} {"Random KNN":<12} {"Strat KNN":<12} {"Random MLP":<12} {"Strat MLP":<12}')
        print('-' * 80)
        
        for model in ESM_MODELS:
            # Baseline - estrutura: results -> esm_model -> models -> KNN/MLP
            baseline_data = baseline_results.get('results', {}).get(model, {}).get('models', {})
            r_knn = baseline_data.get('KNN', {}).get('test_metrics', {}).get('roc_auc', 0)
            r_mlp = baseline_data.get('MLP', {}).get('test_metrics', {}).get('roc_auc', 0)
            
            # Stratified - estrutura: models -> esm_model -> KNN/MLP
            s_knn = stratified_results.get('models', {}).get(model, {}).get('KNN', {}).get('test_metrics', {}).get('roc_auc', 0)
            s_mlp = stratified_results.get('models', {}).get(model, {}).get('MLP', {}).get('test_metrics', {}).get('roc_auc', 0)
            
            print(f'{model:<20} {r_knn:<12.4f} {s_knn:<12.4f} {r_mlp:<12.4f} {s_mlp:<12.4f}')
        
        print('-' * 80)
    else:
        print(f'⚠️  Resultados do baseline não encontrados em: {baseline_dir}')
        print('   Execute primeiro: python baseline.py')
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
