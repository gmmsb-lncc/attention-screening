#!/usr/bin/env python3
"""
DockTKinase Split Analysis: Avaliação de Modelos com Diferentes Cenários de Split
==================================================================================

Este script avalia o desempenho de modelos usando embeddings pré-computados
(ESM-2 para proteínas + SMI-TED para ligantes) em três cenários de avaliação:

1. Split Aleatório (Original) - com vazamento de dados
2. Split por Composto - nenhum composto repetido entre treino e teste
3. Novo Composto + Nova Quinase - generalização verdadeira

Modelos avaliados:
- KNN (K-Nearest Neighbors) com distância cosseno
- MLP (Multi-Layer Perceptron)
- DeepDTA-style: Rede neural profunda para embeddings concatenados

Embeddings suportados (apenas 8M, 150M e 3B):
- esm2_t6_8M_UR50D
- esm2_t30_150M_UR50D
- esm2_t36_3B_UR50D

Uso:
    # Análise única
    python docktkinase_split_analysis.py --embedding esm2_t30_150M_UR50D --dataset non_human

    # Executar todos os embeddings para um dataset
    python docktkinase_split_analysis.py --run_all --dataset non_human

    # Executar todos os embeddings para todos os datasets
    python docktkinase_split_analysis.py --run_all --dataset all

Author: DockTKinase Team
Date: January 2025
"""

import argparse
import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Configuração dos embeddings suportados (apenas 8M, 150M e 3B)
SUPPORTED_EMBEDDINGS = {
    '8M': 'esm2_t6_8M_UR50D',
    '150M': 'esm2_t30_150M_UR50D',
    '3B': 'esm2_t36_3B_UR50D'
}

# Dataset paths
DATASET_PATHS = {
    'human': '/media/leon/ssd2tb/docktkinase/tests/datasets/kinase_human_compounds.tsv',
    'non_human': '/media/leon/ssd2tb/docktkinase/tests/datasets/kinase_non_human_compounds.tsv',
    'all': '/media/leon/ssd2tb/docktkinase/tests/datasets/kinase_all_compounds.tsv'
}

# Base path for embeddings
EMBEDDING_BASE_PATH = '/media/leon/ssd2tb/docktkinase/results/protein_model_benchmark_{dataset_type}_v2'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Try importing PyTorch for DeepDTA-style model
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. DeepDTA-style model will be skipped.")

# Configuration
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 12


# =============================================================================
# DEEP LEARNING MODELS
# =============================================================================

if TORCH_AVAILABLE:
    class DeepDTAClassifier(nn.Module):
        """
        DeepDTA-style classifier for concatenated protein-ligand embeddings.

        Architecture similar to what DockTKinase would use internally,
        but adapted for pre-pooled embeddings (no cross-attention since
        we only have single-vector representations).
        """

        def __init__(self, input_dim, hidden_dims=[512, 256, 128], dropout=0.3):
            super().__init__()

            layers = []
            prev_dim = input_dim

            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
                prev_dim = hidden_dim

            layers.append(nn.Linear(prev_dim, 1))

            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)


    def train_deep_model(X_train, y_train, X_val, y_val, input_dim,
                         epochs=100, batch_size=64, lr=0.001, patience=15):
        """Train DeepDTA-style model with early stopping."""

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(device)
        y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
        X_val_t = torch.FloatTensor(X_val).to(device)
        y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(device)

        # Create data loaders
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Initialize model
        model = DeepDTAClassifier(input_dim).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        best_val_loss = float('inf')
        best_model_state = None
        no_improve = 0

        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t).item()

            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        return model, device


    def predict_deep_model(model, X, device):
        """Make predictions with DeepDTA model."""
        model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(device)
            outputs = model(X_t)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            preds = (probs >= 0.5).astype(int)
        return preds, probs


# =============================================================================
# DATA LOADING
# =============================================================================

def load_embeddings_and_data(embedding_dir: str, dataset_path: str):
    """
    Load pre-computed embeddings and original dataset.

    Args:
        embedding_dir: Path to embedding directory (e.g., esm2_t30_150M_UR50D)
        dataset_path: Path to original TSV file

    Returns:
        embeddings: numpy array [n_samples, embedding_dim]
        labels: numpy array [n_samples,]
        df: pandas DataFrame with original data
    """
    build_dir = os.path.join(embedding_dir, 'build')

    # Load embeddings
    embedding_path = os.path.join(build_dir, 'embedding_matrix.npy')
    embeddings = np.load(embedding_path)

    # Load binary labels
    labels_path = os.path.join(build_dir, 'binary_labels.npy')
    labels = np.load(labels_path)

    # Load original dataset for compound/kinase info
    df = pd.read_csv(dataset_path, sep='\t')

    # Validate alignment
    assert len(embeddings) == len(labels) == len(df), \
        f"Size mismatch: embeddings={len(embeddings)}, labels={len(labels)}, df={len(df)}"

    print(f"Loaded {len(embeddings)} samples")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Active: {labels.sum()} ({100*labels.mean():.1f}%)")
    print(f"Inactive: {len(labels) - labels.sum()} ({100*(1-labels.mean()):.1f}%)")

    return embeddings, labels, df


def get_embedding_info(embedding_dir: str):
    """Extract embedding model info from directory path."""
    build_dir = os.path.join(embedding_dir, 'build')
    pipeline_path = os.path.join(build_dir, 'pipeline_results.json')

    if os.path.exists(pipeline_path):
        with open(pipeline_path, 'r') as f:
            info = json.load(f)
            protein_model = info.get('embedding_generation', {}).get('protein_embeddings', {}).get('model', 'unknown')
            protein_dim = info.get('matrix_construction', {}).get('matrix_info', {}).get('protein_dim', 0)
            ligand_dim = info.get('matrix_construction', {}).get('matrix_info', {}).get('ligand_dim', 0)
            return {
                'protein_model': protein_model,
                'protein_dim': protein_dim,
                'ligand_dim': ligand_dim,
                'total_dim': protein_dim + ligand_dim
            }

    return {'protein_model': os.path.basename(embedding_dir)}


# =============================================================================
# SPLIT STRATEGIES
# =============================================================================

def split_random(df: pd.DataFrame, labels: np.ndarray, random_state: int = 42):
    """Split aleatório estratificado 80/10/10."""
    indices = np.arange(len(df))

    train_idx, temp_idx = train_test_split(
        indices, test_size=0.2, stratify=labels, random_state=random_state
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=labels[temp_idx], random_state=random_state
    )

    return train_idx, val_idx, test_idx


def split_by_compound(df: pd.DataFrame, random_state: int = 42):
    """
    Split onde nenhum composto do teste aparece no treino.
    Divide primeiro os compostos, depois atribui as linhas.
    """
    compounds = df['chembl_id'].unique()

    train_compounds, temp_compounds = train_test_split(
        compounds, test_size=0.2, random_state=random_state
    )
    val_compounds, test_compounds = train_test_split(
        temp_compounds, test_size=0.5, random_state=random_state
    )

    train_compounds = set(train_compounds)
    val_compounds = set(val_compounds)
    test_compounds = set(test_compounds)

    train_idx = df[df['chembl_id'].isin(train_compounds)].index.tolist()
    val_idx = df[df['chembl_id'].isin(val_compounds)].index.tolist()
    test_idx = df[df['chembl_id'].isin(test_compounds)].index.tolist()

    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def split_new_compound_new_kinase(df: pd.DataFrame, random_state: int = 42,
                                   test_kinase_frac: float = 0.15,
                                   test_compound_frac: float = 0.15):
    """
    Split onde o teste contém APENAS:
    - Compostos nunca vistos no treino
    - Quinases nunca vistas no treino

    Este é o cenário de generalização verdadeira.
    """
    np.random.seed(random_state)

    compounds = df['chembl_id'].unique()
    kinases = df['target_kinase'].unique()

    n_test_kinases = max(1, int(len(kinases) * test_kinase_frac))
    test_kinases = set(np.random.choice(kinases, size=n_test_kinases, replace=False))
    train_kinases = set(kinases) - test_kinases

    n_test_compounds = max(1, int(len(compounds) * test_compound_frac))
    test_compounds = set(np.random.choice(compounds, size=n_test_compounds, replace=False))
    train_compounds = set(compounds) - test_compounds

    train_mask = (df['chembl_id'].isin(train_compounds)) & (df['target_kinase'].isin(train_kinases))
    train_idx = df[train_mask].index.tolist()

    test_mask = (df['chembl_id'].isin(test_compounds)) & (df['target_kinase'].isin(test_kinases))
    test_idx = df[test_mask].index.tolist()

    val_mask = ~train_mask & ~test_mask
    val_idx = df[val_mask].index.tolist()

    if len(test_idx) < 50:
        print(f"  AVISO: Teste muito pequeno ({len(test_idx)}), aumentando frações...")
        return split_new_compound_new_kinase(df, random_state,
                                              test_kinase_frac * 1.5,
                                              test_compound_frac * 1.5)

    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


# =============================================================================
# TRAINING AND EVALUATION
# =============================================================================

def train_and_evaluate(embeddings: np.ndarray, labels: np.ndarray,
                       train_idx: np.ndarray, val_idx: np.ndarray,
                       test_idx: np.ndarray, include_deep: bool = True):
    """
    Train and evaluate models.

    Returns dict with metrics for each model.
    """
    X_train = embeddings[train_idx]
    y_train = labels[train_idx]
    X_val = embeddings[val_idx]
    y_val = labels[val_idx]
    X_test = embeddings[test_idx]
    y_test = labels[test_idx]

    if len(X_test) == 0 or len(X_train) == 0:
        return None

    # Normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # KNN
    print("    Training KNN...")
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='cosine', n_jobs=-1)
    knn.fit(X_train_scaled, y_train)
    knn_pred = knn.predict(X_test_scaled)
    knn_proba = knn.predict_proba(X_test_scaled)[:, 1]

    results['KNN'] = {
        'accuracy': accuracy_score(y_test, knn_pred),
        'mcc': matthews_corrcoef(y_test, knn_pred),
        'f1': f1_score(y_test, knn_pred),
        'auc': roc_auc_score(y_test, knn_proba)
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
        random_state=42
    )
    mlp.fit(X_train_scaled, y_train)
    mlp_pred = mlp.predict(X_test_scaled)
    mlp_proba = mlp.predict_proba(X_test_scaled)[:, 1]

    results['MLP'] = {
        'accuracy': accuracy_score(y_test, mlp_pred),
        'mcc': matthews_corrcoef(y_test, mlp_pred),
        'f1': f1_score(y_test, mlp_pred),
        'auc': roc_auc_score(y_test, mlp_proba)
    }

    # DeepDTA-style
    if include_deep and TORCH_AVAILABLE:
        print("    Training DeepDTA-style...")
        try:
            model, device = train_deep_model(
                X_train_scaled, y_train,
                X_val_scaled, y_val,
                input_dim=X_train_scaled.shape[1],
                epochs=100, batch_size=64, patience=15
            )
            deep_pred, deep_proba = predict_deep_model(model, X_test_scaled, device)

            results['DeepDTA'] = {
                'accuracy': accuracy_score(y_test, deep_pred),
                'mcc': matthews_corrcoef(y_test, deep_pred),
                'f1': f1_score(y_test, deep_pred),
                'auc': roc_auc_score(y_test, deep_proba)
            }
        except Exception as e:
            print(f"      DeepDTA training failed: {e}")

    return results


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_analysis(embedding_dir: str, dataset_path: str, output_dir: str,
                 dataset_type: str = 'non_human'):
    """Run complete analysis with all split scenarios."""

    print("=" * 70)
    print("DOCKTKINASE SPLIT ANALYSIS")
    print("=" * 70)

    # Get embedding info
    emb_info = get_embedding_info(embedding_dir)
    print(f"\nProtein Model: {emb_info.get('protein_model', 'unknown')}")
    print(f"Dataset Type: {dataset_type}")

    # Load data
    print("\nLoading data...")
    embeddings, labels, df = load_embeddings_and_data(embedding_dir, dataset_path)

    all_results = {}
    split_stats = {}

    # SCENARIO 1: Random Split
    print("\n" + "-" * 50)
    print("SCENARIO 1: Random Split (with leakage)")
    print("-" * 50)

    train_idx, val_idx, test_idx = split_random(df, labels)

    train_compounds = set(df.iloc[train_idx]['chembl_id'])
    test_compounds = set(df.iloc[test_idx]['chembl_id'])
    leaked = train_compounds & test_compounds

    split_stats['Random Split\n(Original)'] = {
        'train_size': len(train_idx),
        'val_size': len(val_idx),
        'test_size': len(test_idx),
        'test_compounds': len(test_compounds),
        'leaked_compounds': len(leaked),
        'leak_pct': 100 * len(leaked) / len(test_compounds) if test_compounds else 0
    }

    print(f"  Train: {len(train_idx)} samples")
    print(f"  Val:   {len(val_idx)} samples")
    print(f"  Test:  {len(test_idx)} samples")
    print(f"  Compounds in test: {len(test_compounds)}")
    print(f"  LEAKED compounds: {len(leaked)} ({split_stats['Random Split\n(Original)']['leak_pct']:.1f}%)")

    results = train_and_evaluate(embeddings, labels, train_idx, val_idx, test_idx)
    all_results['Random Split\n(Original)'] = results

    for model_name, metrics in results.items():
        print(f"  {model_name}: Acc={metrics['accuracy']:.4f}, MCC={metrics['mcc']:.4f}, AUC={metrics['auc']:.4f}")

    # SCENARIO 2: Split by Compound
    print("\n" + "-" * 50)
    print("SCENARIO 2: Split by Compound (no leakage)")
    print("-" * 50)

    train_idx, val_idx, test_idx = split_by_compound(df)

    train_compounds = set(df.iloc[train_idx]['chembl_id'])
    test_compounds = set(df.iloc[test_idx]['chembl_id'])
    leaked = train_compounds & test_compounds

    split_stats['Split by\nCompound'] = {
        'train_size': len(train_idx),
        'val_size': len(val_idx),
        'test_size': len(test_idx),
        'test_compounds': len(test_compounds),
        'leaked_compounds': len(leaked),
        'leak_pct': 100 * len(leaked) / len(test_compounds) if test_compounds else 0
    }

    print(f"  Train: {len(train_idx)} samples")
    print(f"  Val:   {len(val_idx)} samples")
    print(f"  Test:  {len(test_idx)} samples")
    print(f"  Compounds in test: {len(test_compounds)}")
    print(f"  LEAKED compounds: {len(leaked)} ({split_stats['Split by\nCompound']['leak_pct']:.1f}%)")

    results = train_and_evaluate(embeddings, labels, train_idx, val_idx, test_idx)
    all_results['Split by\nCompound'] = results

    for model_name, metrics in results.items():
        print(f"  {model_name}: Acc={metrics['accuracy']:.4f}, MCC={metrics['mcc']:.4f}, AUC={metrics['auc']:.4f}")

    # SCENARIO 3: New Compound + New Kinase
    print("\n" + "-" * 50)
    print("SCENARIO 3: New Compound + New Kinase (true generalization)")
    print("-" * 50)

    train_idx, val_idx, test_idx = split_new_compound_new_kinase(df)

    train_compounds = set(df.iloc[train_idx]['chembl_id'])
    test_compounds = set(df.iloc[test_idx]['chembl_id'])
    train_kinases = set(df.iloc[train_idx]['target_kinase'])
    test_kinases = set(df.iloc[test_idx]['target_kinase'])

    leaked_compounds = train_compounds & test_compounds
    leaked_kinases = train_kinases & test_kinases

    split_stats['New Comp.\n+ New Kinase'] = {
        'train_size': len(train_idx),
        'val_size': len(val_idx),
        'test_size': len(test_idx),
        'test_compounds': len(test_compounds),
        'test_kinases': len(test_kinases),
        'leaked_compounds': len(leaked_compounds),
        'leaked_kinases': len(leaked_kinases),
        'leak_pct': 0
    }

    print(f"  Train: {len(train_idx)} samples")
    print(f"  Val:   {len(val_idx)} samples")
    print(f"  Test:  {len(test_idx)} samples")
    print(f"  Compounds in test: {len(test_compounds)} (all new)")
    print(f"  Kinases in test: {len(test_kinases)} (all new)")
    print(f"  Leaked compounds: {len(leaked_compounds)}")
    print(f"  Leaked kinases: {len(leaked_kinases)}")

    results = train_and_evaluate(embeddings, labels, train_idx, val_idx, test_idx)
    all_results['New Comp.\n+ New Kinase'] = results

    for model_name, metrics in results.items():
        print(f"  {model_name}: Acc={metrics['accuracy']:.4f}, MCC={metrics['mcc']:.4f}, AUC={metrics['auc']:.4f}")

    return all_results, split_stats, emb_info


# =============================================================================
# PLOTTING
# =============================================================================

def plot_comparison(all_results: dict, split_stats: dict, emb_info: dict,
                    output_dir: str, prefix: str = ''):
    """Generate comparison plots."""

    scenarios = list(all_results.keys())
    models = list(all_results[scenarios[0]].keys())

    # Color palette
    colors = {
        'KNN': '#3498db',
        'MLP': '#e74c3c',
        'DeepDTA': '#2ecc71'
    }

    test_sizes = [split_stats[s]['test_size'] for s in scenarios]
    test_compounds = [split_stats[s]['test_compounds'] for s in scenarios]

    # Figure with 2 subplots: Accuracy and MCC
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    x = np.arange(len(scenarios))
    n_models = len(models)
    width = 0.8 / n_models

    # Plot 1: Accuracy
    ax1 = axes[0]
    for i, model in enumerate(models):
        accs = [all_results[s][model]['accuracy'] for s in scenarios]
        offset = (i - (n_models-1)/2) * width
        bars = ax1.bar(x + offset, accs, width, label=model,
                       color=colors.get(model, '#95a5a6'), edgecolor='black')

        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title(f'Accuracy by Evaluation Scenario\n({emb_info.get("protein_model", "Unknown")})',
                  fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, fontsize=10)
    ax1.legend(loc='upper right', fontsize=11)
    ax1.set_ylim(0, 1.15)

    for i, (ts, tc) in enumerate(zip(test_sizes, test_compounds)):
        ax1.annotate(f'n={ts}\n({tc} comp.)',
                    xy=(i, 0), xytext=(0, -25),
                    textcoords="offset points",
                    ha='center', va='top', fontsize=9, color='gray')

    # Plot 2: MCC
    ax2 = axes[1]
    for i, model in enumerate(models):
        mccs = [all_results[s][model]['mcc'] for s in scenarios]
        offset = (i - (n_models-1)/2) * width
        bars = ax2.bar(x + offset, mccs, width, label=model,
                       color=colors.get(model, '#95a5a6'), edgecolor='black')

        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax2.set_ylabel('MCC', fontsize=12)
    ax2.set_title(f'MCC by Evaluation Scenario\n({emb_info.get("protein_model", "Unknown")})',
                  fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios, fontsize=10)
    ax2.legend(loc='upper right', fontsize=11)
    ax2.set_ylim(0, 1.0)

    for i, (ts, tc) in enumerate(zip(test_sizes, test_compounds)):
        ax2.annotate(f'n={ts}\n({tc} comp.)',
                    xy=(i, 0), xytext=(0, -25),
                    textcoords="offset points",
                    ha='center', va='top', fontsize=9, color='gray')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    filename = f'{prefix}08_embedding_split_comparison.png' if prefix else '08_embedding_split_comparison.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved: {filepath}")

    # Plot: Inflated vs Real Performance
    plot_inflated_vs_real(all_results, emb_info, output_dir, prefix)

    return filepath


def plot_inflated_vs_real(all_results: dict, emb_info: dict, output_dir: str, prefix: str = ''):
    """Plot comparing inflated (random split) vs real (generalization) performance."""

    inflated_key = 'Random Split\n(Original)'
    real_key = 'New Comp.\n+ New Kinase'

    models = list(all_results[inflated_key].keys())

    # Color palette
    colors = {
        'KNN': '#3498db',
        'MLP': '#e74c3c',
        'DeepDTA': '#2ecc71'
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    x = np.arange(len(models))
    width = 0.35

    # MCC comparison
    ax1 = axes[0]
    inflated_mcc = [all_results[inflated_key][m]['mcc'] for m in models]
    real_mcc = [all_results[real_key][m]['mcc'] for m in models]

    bars1 = ax1.bar(x - width/2, inflated_mcc, width, label='Reported MCC (Random Split)',
                    color='#3498db', edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x + width/2, real_mcc, width, label='Real MCC (Generalization)',
                    color='#e74c3c', edgecolor='black', linewidth=1.5)

    ax1.set_ylabel('MCC', fontsize=12)
    ax1.set_title(f'Inflated vs Real Performance (MCC)\n{emb_info.get("protein_model", "Unknown")}',
                  fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_ylim(0, 1.0)

    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add drop arrows
    for i, (inf, real) in enumerate(zip(inflated_mcc, real_mcc)):
        if inf > 0:
            drop_pct = 100 * (inf - real) / inf
            mid_y = (inf + real) / 2

            ax1.annotate('',
                        xy=(i + width/2, real + 0.02),
                        xytext=(i - width/2, inf - 0.02),
                        arrowprops=dict(arrowstyle='->', color='#8e44ad', lw=2.5))

            ax1.text(i, mid_y + 0.08, f'{drop_pct:.0f}%',
                    ha='center', va='center', fontsize=13, fontweight='bold',
                    color='#8e44ad',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                             edgecolor='#8e44ad', alpha=0.9))

    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    # Accuracy comparison
    ax2 = axes[1]
    inflated_acc = [all_results[inflated_key][m]['accuracy'] for m in models]
    real_acc = [all_results[real_key][m]['accuracy'] for m in models]

    bars3 = ax2.bar(x - width/2, inflated_acc, width, label='Reported Accuracy (Random Split)',
                    color='#3498db', edgecolor='black', linewidth=1.5)
    bars4 = ax2.bar(x + width/2, real_acc, width, label='Real Accuracy (Generalization)',
                    color='#e74c3c', edgecolor='black', linewidth=1.5)

    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title(f'Inflated vs Real Performance (Accuracy)\n{emb_info.get("protein_model", "Unknown")}',
                  fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.set_ylim(0, 1.0)

    for bar in bars3:
        height = bar.get_height()
        ax2.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    for bar in bars4:
        height = bar.get_height()
        ax2.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    for i, (inf, real) in enumerate(zip(inflated_acc, real_acc)):
        if inf > 0:
            drop_pct = 100 * (inf - real) / inf
            mid_y = (inf + real) / 2

            ax2.annotate('',
                        xy=(i + width/2, real + 0.02),
                        xytext=(i - width/2, inf - 0.02),
                        arrowprops=dict(arrowstyle='->', color='#8e44ad', lw=2.5))

            ax2.text(i, mid_y + 0.06, f'{drop_pct:.0f}%',
                    ha='center', va='center', fontsize=13, fontweight='bold',
                    color='#8e44ad',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                             edgecolor='#8e44ad', alpha=0.9))

    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()

    filename = f'{prefix}09_embedding_inflated_vs_real.png' if prefix else '09_embedding_inflated_vs_real.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {filepath}")

    return filepath


def save_results_json(all_results: dict, split_stats: dict, emb_info: dict,
                      output_dir: str, dataset_type: str, prefix: str = ''):
    """Save results to JSON file."""

    results_dict = {
        'metadata': {
            'analysis_date': datetime.now().isoformat(),
            'dataset_type': dataset_type,
            'embedding_info': emb_info
        },
        'split_statistics': {k.replace('\n', ' '): v for k, v in split_stats.items()},
        'model_results': {
            scenario.replace('\n', ' '): {
                model: metrics for model, metrics in models.items()
            } for scenario, models in all_results.items()
        }
    }

    filename = f'{prefix}embedding_analysis_results.json' if prefix else 'embedding_analysis_results.json'
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"Results saved: {filepath}")
    return filepath


# =============================================================================
# SINGLE ANALYSIS RUNNER
# =============================================================================

def run_single_analysis(embedding_name: str, dataset_type: str, output_dir: str, force: bool = False):
    """
    Run analysis for a single embedding model and dataset type.

    Args:
        embedding_name: Embedding model name (e.g., 'esm2_t30_150M_UR50D' or '150M')
        dataset_type: Dataset type ('human', 'non_human', or 'all')
        output_dir: Output directory
        force: Force recalculation even if results exist
    """
    # Resolve embedding name if short form provided
    if embedding_name in SUPPORTED_EMBEDDINGS:
        embedding_name = SUPPORTED_EMBEDDINGS[embedding_name]

    # Get dataset path
    dataset_path = DATASET_PATHS.get(dataset_type)
    if not dataset_path:
        print(f"Error: Unknown dataset type '{dataset_type}'")
        return None

    # Build embedding directory path
    embedding_dir = os.path.join(
        EMBEDDING_BASE_PATH.format(dataset_type=dataset_type),
        embedding_name
    )

    # Generate prefix from embedding and dataset
    short_name = embedding_name.replace('esm2_', '').replace('_UR50D', '')
    prefix = f"{dataset_type}_{short_name}_"

    # Check if results already exist
    json_file = os.path.join(output_dir, f'{prefix}embedding_analysis_results.json')
    plot_file = os.path.join(output_dir, f'{prefix}08_embedding_split_comparison.png')

    if os.path.exists(json_file) and os.path.exists(plot_file) and not force:
        print(f"\n[CACHE] Resultados já existem para {embedding_name} + {dataset_type}")
        print(f"        JSON: {json_file}")
        print(f"        Plot: {plot_file}")
        print(f"        Use --force para recalcular.")
        return None

    # Check if embedding exists
    if not os.path.exists(embedding_dir):
        print(f"Warning: Embedding directory not found: {embedding_dir}")
        return None

    if not os.path.exists(dataset_path):
        print(f"Warning: Dataset not found: {dataset_path}")
        return None

    print(f"\n{'='*70}")
    print(f"ANALYSIS: {embedding_name} + {dataset_type}")
    print(f"{'='*70}")

    # Run analysis
    all_results, split_stats, emb_info = run_analysis(
        embedding_dir, dataset_path, output_dir, dataset_type
    )

    if all_results is None:
        return None

    # Generate plots
    plot_comparison(all_results, split_stats, emb_info, output_dir, prefix)

    # Save results
    save_results_json(all_results, split_stats, emb_info, output_dir,
                      dataset_type, prefix)

    # Print summary
    print_summary(all_results, embedding_name, dataset_type)

    return all_results


def print_summary(all_results: dict, embedding_name: str, dataset_type: str):
    """Print summary of results."""
    print(f"\n{'-'*50}")
    print(f"SUMMARY: {embedding_name} + {dataset_type}")
    print(f"{'-'*50}")

    inflated_key = 'Random Split\n(Original)'
    real_key = 'New Comp.\n+ New Kinase'

    for model in all_results[inflated_key].keys():
        orig_mcc = all_results[inflated_key][model]['mcc']
        real_mcc = all_results[real_key][model]['mcc']
        drop = orig_mcc - real_mcc
        drop_pct = 100 * drop / orig_mcc if orig_mcc > 0 else 0

        print(f"  {model}: MCC {orig_mcc:.3f} -> {real_mcc:.3f} ({drop_pct:.0f}% drop)")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='DockTKinase Split Analysis with ESM-2 Embeddings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single analysis
    python docktkinase_split_analysis.py --embedding 150M --dataset non_human

    # Run all embeddings (8M, 150M, 3B) for non_human dataset
    python docktkinase_split_analysis.py --run_all --dataset non_human

    # Run all embeddings for all datasets (human + non_human)
    python docktkinase_split_analysis.py --run_all --dataset all
        """
    )

    parser.add_argument('--embedding', type=str, default='150M',
                        choices=['8M', '150M', '3B',
                                 'esm2_t6_8M_UR50D', 'esm2_t30_150M_UR50D', 'esm2_t36_3B_UR50D'],
                        help='Embedding model (8M, 150M, or 3B)')
    parser.add_argument('--dataset', type=str, default='non_human',
                        choices=['human', 'non_human', 'all'],
                        help='Dataset type (human, non_human, or all for both)')
    parser.add_argument('--run_all', action='store_true',
                        help='Run all embeddings (8M, 150M, 3B) for the specified dataset(s)')
    parser.add_argument('--force', action='store_true',
                        help='Force recalculation even if results already exist')
    parser.add_argument('--output_dir', type=str,
                        default='/media/leon/ssd2tb/docktkinase/leakage_analysis_results',
                        help='Output directory for plots and results')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.run_all:
        # Determine which datasets to run
        if args.dataset == 'all':
            datasets_to_run = ['non_human', 'human']
        else:
            datasets_to_run = [args.dataset]

        print("\n" + "=" * 70)
        print("RUNNING ALL EMBEDDING ANALYSES")
        print(f"Embeddings: 8M, 150M, 3B")
        print(f"Datasets: {', '.join(datasets_to_run)}")
        if args.force:
            print("Mode: FORCE (recalculating all)")
        else:
            print("Mode: CACHE (skipping existing results)")
        print("=" * 70)

        # Run all combinations
        for dataset_type in datasets_to_run:
            for emb_short, emb_full in SUPPORTED_EMBEDDINGS.items():
                run_single_analysis(emb_full, dataset_type, args.output_dir, args.force)

        print("\n" + "=" * 70)
        print("ALL ANALYSES COMPLETED!")
        print("=" * 70)
    else:
        # Run single analysis
        run_single_analysis(args.embedding, args.dataset, args.output_dir, args.force)

    print(f"\nResults saved in: {args.output_dir}")


if __name__ == '__main__':
    main()
