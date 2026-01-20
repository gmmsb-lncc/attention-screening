#!/usr/bin/env python3
"""
CrossAttention Split Analysis: Avaliação do Modelo CNN + Cross-Attention
=========================================================================

Este script avalia o modelo DockTKinase (CNN + Cross-Attention) usando
embeddings per-token em três cenários de avaliação:

1. Split Aleatório (Original) - com vazamento de dados
2. Split por Composto - nenhum composto repetido entre treino e teste
3. Novo Composto + Nova Quinase - generalização verdadeira

Diferença dos outros scripts:
- Usa embeddings PER-TOKEN (matrizes), não embeddings poolados
- Usa CNN encoders + Cross-Attention bidirectional
- Treina por 500 epochs com early stopping

Uso:
    python crossattention_split_analysis.py --embedding 150M --dataset non_human
    python crossattention_split_analysis.py --run_all --dataset non_human
    python crossattention_split_analysis.py --run_all --dataset non_human --force

Author: DockTKinase Team
Date: January 2025
"""

import argparse
import os
import sys
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.classifier.models.cross_attention_model import (
    CrossAttentionAffinityModel,
    MultiTaskLoss,
    create_cross_attention_model
)
from src.classifier.utils.matrix_dataloader import (
    MatrixEmbeddingDataset,
    collate_matrix_batch,
    create_matrix_dataloader
)

# Configuration
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 12

# Embeddings suportados (apenas 8M, 150M e 3B)
SUPPORTED_EMBEDDINGS = {
    '8M': 'esm2_t6_8M_UR50D',
    '150M': 'esm2_t30_150M_UR50D',
    '3B': 'esm2_t36_3B_UR50D'
}

# Dimensões dos embeddings
PROTEIN_DIMS = {
    'esm2_t6_8M_UR50D': 320,
    'esm2_t30_150M_UR50D': 640,
    'esm2_t36_3B_UR50D': 2560
}

LIGAND_DIM = 768  # SMI-TED

# Dataset paths
DATASET_PATHS = {
    'human': './tests/datasets/kinase_human_compounds.tsv',
    'non_human': './tests/datasets/kinase_non_human_compounds.tsv',
    'all': './tests/datasets/kinase_all_compounds.tsv'
}

# Base path for embeddings
EMBEDDING_BASE_PATH = './results/protein_model_benchmark_{dataset_type}_v2'


@dataclass
class TrainingConfig:
    """Training configuration for CrossAttention model."""
    protein_dim: int = 640
    ligand_dim: int = 768
    hidden_dim: int = 256
    num_cnn_layers: int = 3
    num_cross_attn_layers: int = 2
    num_heads: int = 8
    ff_dim: int = 1024
    dropout: float = 0.1

    # Training
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 500
    patience: int = 30

    # Device
    device: str = 'auto'


def get_device(config: TrainingConfig) -> torch.device:
    """Get the appropriate device."""
    if config.device == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
    return torch.device('cpu')


# =============================================================================
# CHECKPOINT MANAGEMENT
# =============================================================================

def get_checkpoint_path(output_dir: str, prefix: str, scenario_name: str) -> str:
    """Get checkpoint file path for a scenario."""
    safe_scenario = scenario_name.replace('\n', '_').replace(' ', '_').replace('+', 'plus')
    return os.path.join(output_dir, f'{prefix}checkpoint_{safe_scenario}.pt')


def save_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    epoch: int,
    best_val_mcc: float,
    best_model_state: dict,
    patience_counter: int,
    history: Dict,
    scenario_completed: bool = False
):
    """Save training checkpoint to disk."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_mcc': best_val_mcc,
        'best_model_state': best_model_state,
        'patience_counter': patience_counter,
        'history': history,
        'scenario_completed': scenario_completed,
        'timestamp': datetime.now().isoformat()
    }

    # Save to temp file first, then rename (atomic operation)
    temp_path = checkpoint_path + '.tmp'
    torch.save(checkpoint, temp_path)
    os.replace(temp_path, checkpoint_path)

    return checkpoint_path


def load_checkpoint(checkpoint_path: str, device: torch.device) -> Optional[Dict]:
    """Load checkpoint from disk if it exists."""
    if not os.path.exists(checkpoint_path):
        return None

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        print(f"  [CHECKPOINT] Loaded checkpoint from epoch {checkpoint['epoch']+1}")
        print(f"               Best val_mcc so far: {checkpoint['best_val_mcc']:.4f}")
        print(f"               Saved at: {checkpoint['timestamp']}")
        return checkpoint
    except Exception as e:
        print(f"  [WARNING] Failed to load checkpoint: {e}")
        return None


def is_scenario_completed(checkpoint_path: str, device: torch.device) -> bool:
    """Check if a scenario has been completed (for skipping)."""
    checkpoint = load_checkpoint(checkpoint_path, device)
    if checkpoint is None:
        return False
    return checkpoint.get('scenario_completed', False)


# =============================================================================
# SPLIT STRATEGIES
# =============================================================================

def split_random(df: pd.DataFrame, random_state: int = 42):
    """Split aleatório estratificado 80/10/10."""
    indices = np.arange(len(df))
    labels = df['label'].values

    train_idx, temp_idx = train_test_split(
        indices, test_size=0.2, stratify=labels, random_state=random_state
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=labels[temp_idx], random_state=random_state
    )

    return train_idx, val_idx, test_idx


def split_by_compound(df: pd.DataFrame, random_state: int = 42):
    """Split onde nenhum composto do teste aparece no treino."""
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
    """Split com compostos E quinases nunca vistos no treino."""
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
        print(f"  WARNING: Test set too small ({len(test_idx)}), increasing fractions...")
        return split_new_compound_new_kinase(df, random_state,
                                              test_kinase_frac * 1.5,
                                              test_compound_frac * 1.5)

    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(model, train_loader, optimizer, loss_fn, device, epoch, num_epochs):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_cls_loss = 0
    total_reg_loss = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)

    for batch in pbar:
        protein_matrix = batch['protein_matrix'].to(device)
        ligand_matrix = batch['ligand_matrix'].to(device)
        protein_mask = batch['protein_mask'].to(device)
        ligand_mask = batch['ligand_mask'].to(device)
        labels = batch['labels'].to(device)
        reg_targets = batch['regression_targets'].to(device)
        reg_mask = batch['regression_mask'].to(device)

        optimizer.zero_grad()

        output = model(protein_matrix, ligand_matrix, protein_mask, ligand_mask)

        losses = loss_fn(
            output['classification'],
            output['regression'],
            labels,
            reg_targets,
            reg_mask
        )

        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += losses['total'].item()
        total_cls_loss += losses['classification'].item()
        total_reg_loss += losses['regression'].item()
        num_batches += 1

        pbar.set_postfix({
            'loss': f"{losses['total'].item():.4f}",
            'cls': f"{losses['classification'].item():.4f}"
        })

    return {
        'total': total_loss / num_batches,
        'classification': total_cls_loss / num_batches,
        'regression': total_reg_loss / num_batches
    }


def evaluate(model, data_loader, device):
    """Evaluate model and return predictions."""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            protein_matrix = batch['protein_matrix'].to(device)
            ligand_matrix = batch['ligand_matrix'].to(device)
            protein_mask = batch['protein_mask'].to(device)
            ligand_mask = batch['ligand_mask'].to(device)
            labels = batch['labels']

            output = model(protein_matrix, ligand_matrix, protein_mask, ligand_mask)
            probs = torch.sigmoid(output['classification']).cpu().numpy()
            preds = (probs >= 0.5).astype(int)

            all_probs.extend(probs.flatten())
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.numpy().flatten())

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'mcc': matthews_corrcoef(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds),
        'auc': roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5
    }

    return metrics


def train_model(
    model: CrossAttentionAffinityModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    device: torch.device,
    checkpoint_path: Optional[str] = None,
    checkpoint_interval: int = 10
) -> Tuple[CrossAttentionAffinityModel, Dict]:
    """Train the model with early stopping and checkpoint support.

    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        device: Device to train on
        checkpoint_path: Path to save/load checkpoints (optional)
        checkpoint_interval: Save checkpoint every N epochs (default: 10)

    Returns:
        Trained model and training history
    """

    model = model.to(device)

    loss_fn = MultiTaskLoss(
        classification_weight=1.0,
        regression_weight=0.5  # Less weight on regression
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_epochs, eta_min=1e-6
    )

    best_val_mcc = -1
    best_model_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_mcc': [], 'val_acc': []}
    start_epoch = 0

    # Try to load checkpoint if it exists
    if checkpoint_path is not None:
        checkpoint = load_checkpoint(checkpoint_path, device)
        if checkpoint is not None and not checkpoint.get('scenario_completed', False):
            # Resume from checkpoint
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            best_val_mcc = checkpoint['best_val_mcc']
            best_model_state = checkpoint['best_model_state']
            patience_counter = checkpoint['patience_counter']
            history = checkpoint['history']
            start_epoch = checkpoint['epoch'] + 1
            print(f"  [CHECKPOINT] Resuming from epoch {start_epoch}")
        elif checkpoint is not None and checkpoint.get('scenario_completed', False):
            # Scenario already completed, load best model and return
            print(f"  [CHECKPOINT] Scenario already completed, loading best model...")
            if checkpoint['best_model_state'] is not None:
                model.load_state_dict(checkpoint['best_model_state'])
            return model, checkpoint['history']

    print(f"  Training for up to {config.num_epochs} epochs (patience={config.patience})...")
    if start_epoch > 0:
        print(f"  Continuing from epoch {start_epoch + 1}...")

    for epoch in range(start_epoch, config.num_epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch, config.num_epochs
        )

        # Validate
        val_metrics = evaluate(model, val_loader, device)

        scheduler.step()

        history['train_loss'].append(train_metrics['total'])
        history['val_mcc'].append(val_metrics['mcc'])
        history['val_acc'].append(val_metrics['accuracy'])

        # Early stopping check
        improved = False
        if val_metrics['mcc'] > best_val_mcc:
            best_val_mcc = val_metrics['mcc']
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            improved = True
        else:
            patience_counter += 1

        # Print progress every 50 epochs
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}: loss={train_metrics['total']:.4f}, "
                  f"val_mcc={val_metrics['mcc']:.4f}, val_acc={val_metrics['accuracy']:.4f}")

        # Save checkpoint periodically or when improved
        if checkpoint_path is not None:
            should_save = (
                (epoch + 1) % checkpoint_interval == 0 or  # Periodic save
                improved or  # Save when model improved
                patience_counter >= config.patience  # Save before early stopping
            )
            if should_save:
                save_checkpoint(
                    checkpoint_path=checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_val_mcc=best_val_mcc,
                    best_model_state=best_model_state,
                    patience_counter=patience_counter,
                    history=history,
                    scenario_completed=False
                )

        if patience_counter >= config.patience:
            print(f"    Early stopping at epoch {epoch+1} (best val_mcc={best_val_mcc:.4f})")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Save final checkpoint marking scenario as completed
    if checkpoint_path is not None:
        save_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_val_mcc=best_val_mcc,
            best_model_state=best_model_state,
            patience_counter=patience_counter,
            history=history,
            scenario_completed=True
        )
        print(f"  [CHECKPOINT] Saved final checkpoint (scenario completed)")

    return model, history


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_scenario(
    scenario_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    protein_matrix_dir: str,
    ligand_matrix_dir: str,
    config: TrainingConfig,
    device: torch.device,
    checkpoint_path: Optional[str] = None
) -> Dict:
    """Run training and evaluation for one scenario.

    Args:
        scenario_name: Name of the scenario being run
        train_df: Training dataframe
        val_df: Validation dataframe
        test_df: Test dataframe
        protein_matrix_dir: Path to protein matrix embeddings
        ligand_matrix_dir: Path to ligand matrix embeddings
        config: Training configuration
        device: Device to train on
        checkpoint_path: Path to save/load checkpoints (optional)

    Returns:
        Dictionary with test metrics
    """

    print(f"\n  Creating data loaders...")

    # Create data loaders
    train_loader = create_matrix_dataloader(
        train_df, protein_matrix_dir, ligand_matrix_dir,
        batch_size=config.batch_size, shuffle=True,
        label_column='label', protein_id_column='seq_id', ligand_id_column='chembl_id'
    )

    val_loader = create_matrix_dataloader(
        val_df, protein_matrix_dir, ligand_matrix_dir,
        batch_size=config.batch_size, shuffle=False,
        label_column='label', protein_id_column='seq_id', ligand_id_column='chembl_id'
    )

    test_loader = create_matrix_dataloader(
        test_df, protein_matrix_dir, ligand_matrix_dir,
        batch_size=config.batch_size, shuffle=False,
        label_column='label', protein_id_column='seq_id', ligand_id_column='chembl_id'
    )

    print(f"  Creating model (protein_dim={config.protein_dim}, ligand_dim={config.ligand_dim})...")

    # Create model
    model = CrossAttentionAffinityModel(
        protein_dim=config.protein_dim,
        ligand_dim=config.ligand_dim,
        hidden_dim=config.hidden_dim,
        num_cnn_layers=config.num_cnn_layers,
        num_cross_attn_layers=config.num_cross_attn_layers,
        num_heads=config.num_heads,
        ff_dim=config.ff_dim,
        dropout=config.dropout
    )

    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train with checkpoint support
    model, history = train_model(
        model, train_loader, val_loader, config, device,
        checkpoint_path=checkpoint_path,
        checkpoint_interval=10  # Save every 10 epochs
    )

    # Evaluate on test set
    print(f"  Evaluating on test set...")
    test_metrics = evaluate(model, test_loader, device)

    print(f"  Results: Acc={test_metrics['accuracy']:.4f}, MCC={test_metrics['mcc']:.4f}, "
          f"AUC={test_metrics['auc']:.4f}")

    return test_metrics


def run_analysis(
    embedding_name: str,
    dataset_type: str,
    output_dir: str,
    config: TrainingConfig,
    prefix: str = ""
) -> Tuple[Dict, Dict]:
    """Run complete analysis for one embedding + dataset combination.

    Args:
        embedding_name: Name of the embedding model
        dataset_type: Type of dataset (human, non_human, all)
        output_dir: Directory to save results and checkpoints
        config: Training configuration
        prefix: Prefix for checkpoint files

    Returns:
        Tuple of (all_results, split_stats) dictionaries
    """

    # Get paths
    dataset_path = DATASET_PATHS.get(dataset_type)
    embedding_dir = os.path.join(
        EMBEDDING_BASE_PATH.format(dataset_type=dataset_type),
        embedding_name,
        'build'
    )

    protein_matrix_dir = os.path.join(embedding_dir, 'protein_matrices')
    ligand_matrix_dir = os.path.join(embedding_dir, 'ligand_matrices')

    # Check paths
    if not os.path.exists(protein_matrix_dir):
        print(f"ERROR: Protein matrices not found: {protein_matrix_dir}")
        return None, None

    if not os.path.exists(ligand_matrix_dir):
        print(f"ERROR: Ligand matrices not found: {ligand_matrix_dir}")
        return None, None

    # Load dataset
    print(f"\nLoading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path, sep='\t')
    df['label'] = (df['standard_value'] <= 1000).astype(int)

    # Convert seq_id to string for matching
    df['seq_id'] = df['seq_id'].astype(str)

    print(f"  Total: {len(df)} samples, {df['chembl_id'].nunique()} compounds, "
          f"{df['target_kinase'].nunique()} kinases")

    # Get device
    device = get_device(config)
    print(f"  Device: {device}")

    all_results = {}
    split_stats = {}

    # SCENARIO 03: New Compound + New Kinase (true generalization) - FIRST
    print("\n" + "=" * 60)
    print("SCENARIO 03: New Compound + New Kinase (true generalization)")
    print("=" * 60)

    scenario_name = 'New Compound + New Kinase'
    checkpoint_path = get_checkpoint_path(output_dir, prefix, scenario_name)
    print(f"  Checkpoint: {checkpoint_path}")

    train_idx, val_idx, test_idx = split_new_compound_new_kinase(df)

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    split_stats['New Comp.\n+ New Kinase'] = {
        'train_size': len(train_idx),
        'val_size': len(val_idx),
        'test_size': len(test_idx),
        'test_compounds': test_df['chembl_id'].nunique(),
        'test_kinases': test_df['target_kinase'].nunique()
    }

    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    results = run_scenario(
        scenario_name, train_df, val_df, test_df,
        protein_matrix_dir, ligand_matrix_dir, config, device,
        checkpoint_path=checkpoint_path
    )
    all_results['New Comp.\n+ New Kinase'] = {'CNN+CrossAttn': results}

    # SCENARIO 02: Split by Compound (no leakage)
    print("\n" + "=" * 60)
    print("SCENARIO 02: Split by Compound (no leakage)")
    print("=" * 60)

    scenario_name = 'Split by Compound'
    checkpoint_path = get_checkpoint_path(output_dir, prefix, scenario_name)
    print(f"  Checkpoint: {checkpoint_path}")

    train_idx, val_idx, test_idx = split_by_compound(df)

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    split_stats['Split by\nCompound'] = {
        'train_size': len(train_idx),
        'val_size': len(val_idx),
        'test_size': len(test_idx),
        'test_compounds': test_df['chembl_id'].nunique()
    }

    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    results = run_scenario(
        scenario_name, train_df, val_df, test_df,
        protein_matrix_dir, ligand_matrix_dir, config, device,
        checkpoint_path=checkpoint_path
    )
    all_results['Split by\nCompound'] = {'CNN+CrossAttn': results}

    # SCENARIO 01: Random Split (with leakage) - LAST
    print("\n" + "=" * 60)
    print("SCENARIO 01: Random Split (with leakage)")
    print("=" * 60)

    scenario_name = 'Random Split'
    checkpoint_path = get_checkpoint_path(output_dir, prefix, scenario_name)
    print(f"  Checkpoint: {checkpoint_path}")

    train_idx, val_idx, test_idx = split_random(df)

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    split_stats['Random Split\n(Original)'] = {
        'train_size': len(train_idx),
        'val_size': len(val_idx),
        'test_size': len(test_idx),
        'test_compounds': test_df['chembl_id'].nunique()
    }

    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    results = run_scenario(
        scenario_name, train_df, val_df, test_df,
        protein_matrix_dir, ligand_matrix_dir, config, device,
        checkpoint_path=checkpoint_path
    )
    all_results['Random Split\n(Original)'] = {'CNN+CrossAttn': results}

    return all_results, split_stats


# =============================================================================
# PLOTTING
# =============================================================================

def plot_results(all_results: Dict, split_stats: Dict, embedding_name: str,
                 output_dir: str, prefix: str):
    """Generate comparison plots."""

    scenarios = list(all_results.keys())

    # Extract metrics
    accs = [all_results[s]['CNN+CrossAttn']['accuracy'] for s in scenarios]
    mccs = [all_results[s]['CNN+CrossAttn']['mcc'] for s in scenarios]
    aucs = [all_results[s]['CNN+CrossAttn']['auc'] for s in scenarios]

    test_sizes = [split_stats[s]['test_size'] for s in scenarios]

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    x = np.arange(len(scenarios))
    width = 0.6

    # Plot 1: Accuracy
    ax1 = axes[0]
    bars1 = ax1.bar(x, accs, width, color='#2ecc71', edgecolor='black', linewidth=1.5)

    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title(f'CNN+CrossAttention Accuracy\n({embedding_name})', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, fontsize=10)
    ax1.set_ylim(0, 1.1)

    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

    for i, ts in enumerate(test_sizes):
        ax1.annotate(f'n={ts}',
                    xy=(i, 0), xytext=(0, -20),
                    textcoords="offset points",
                    ha='center', va='top', fontsize=9, color='gray')

    # Plot 2: MCC
    ax2 = axes[1]
    bars2 = ax2.bar(x, mccs, width, color='#9b59b6', edgecolor='black', linewidth=1.5)

    ax2.set_ylabel('MCC', fontsize=12)
    ax2.set_title(f'CNN+CrossAttention MCC\n({embedding_name})', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios, fontsize=10)
    ax2.set_ylim(0, 1.0)

    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Add drop percentage
    if len(mccs) >= 2:
        drop_pct = 100 * (mccs[0] - mccs[-1]) / mccs[0] if mccs[0] > 0 else 0
        ax2.annotate(f'↓ {drop_pct:.0f}%',
                    xy=(len(scenarios)-1, mccs[-1]),
                    xytext=(30, 50), textcoords="offset points",
                    ha='center', fontsize=14, fontweight='bold', color='#e74c3c',
                    arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2))

    plt.tight_layout()

    filename = f'{prefix}10_crossattention_split_comparison.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nPlot saved: {filepath}")
    return filepath


def save_results_json(all_results: Dict, split_stats: Dict, embedding_name: str,
                      dataset_type: str, output_dir: str, prefix: str):
    """Save results to JSON."""

    results_dict = {
        'metadata': {
            'analysis_date': datetime.now().isoformat(),
            'model': 'CNN+CrossAttention',
            'embedding': embedding_name,
            'dataset_type': dataset_type,
            'epochs': 500,
            'patience': 30
        },
        'split_statistics': {k.replace('\n', ' '): v for k, v in split_stats.items()},
        'model_results': {
            scenario.replace('\n', ' '): metrics
            for scenario, models in all_results.items()
            for model_name, metrics in models.items()
        }
    }

    filename = f'{prefix}crossattention_analysis_results.json'
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"Results saved: {filepath}")
    return filepath


# =============================================================================
# MAIN
# =============================================================================

def run_single_analysis(embedding_name: str, dataset_type: str, output_dir: str,
                        force: bool = False):
    """Run analysis for a single embedding + dataset combination."""

    # Resolve embedding name
    if embedding_name in SUPPORTED_EMBEDDINGS:
        embedding_name = SUPPORTED_EMBEDDINGS[embedding_name]

    # Generate prefix
    short_name = embedding_name.replace('esm2_', '').replace('_UR50D', '')
    prefix = f"{dataset_type}_{short_name}_"

    # Check cache
    json_file = os.path.join(output_dir, f'{prefix}crossattention_analysis_results.json')

    if os.path.exists(json_file) and not force:
        print(f"\n[CACHE] Results already exist: {json_file}")
        print(f"        Use --force to recalculate.")
        return None

    print("\n" + "=" * 70)
    print(f"CNN+CROSSATTENTION ANALYSIS: {embedding_name} + {dataset_type}")
    print("=" * 70)

    # Create config
    protein_dim = PROTEIN_DIMS.get(embedding_name, 640)
    config = TrainingConfig(
        protein_dim=protein_dim,
        ligand_dim=LIGAND_DIM,
        num_epochs=500,
        patience=30,
        batch_size=32
    )

    # Run analysis with checkpoint support
    all_results, split_stats = run_analysis(
        embedding_name, dataset_type, output_dir, config, prefix=prefix
    )

    if all_results is None:
        return None

    # Generate plots
    plot_results(all_results, split_stats, embedding_name, output_dir, prefix)

    # Save results
    save_results_json(all_results, split_stats, embedding_name, dataset_type, output_dir, prefix)

    # Print summary
    print("\n" + "-" * 50)
    print("SUMMARY")
    print("-" * 50)

    for scenario in all_results:
        metrics = all_results[scenario]['CNN+CrossAttn']
        print(f"  {scenario.replace(chr(10), ' ')}: "
              f"Acc={metrics['accuracy']:.4f}, MCC={metrics['mcc']:.4f}")

    # Calculate drop
    scenarios = list(all_results.keys())
    orig_mcc = all_results[scenarios[0]]['CNN+CrossAttn']['mcc']
    real_mcc = all_results[scenarios[-1]]['CNN+CrossAttn']['mcc']
    drop_pct = 100 * (orig_mcc - real_mcc) / orig_mcc if orig_mcc > 0 else 0

    print(f"\n  MCC drop: {orig_mcc:.3f} -> {real_mcc:.3f} ({drop_pct:.0f}% drop)")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='CrossAttention (CNN + Cross-Attention) Split Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--embedding', type=str, default='150M',
                        choices=['8M', '150M', '3B'],
                        help='Embedding model (8M, 150M, or 3B)')
    parser.add_argument('--dataset', type=str, default='non_human',
                        choices=['human', 'non_human'],
                        help='Dataset type')
    parser.add_argument('--run_all', action='store_true',
                        help='Run all embeddings (8M, 150M, 3B)')
    parser.add_argument('--force', action='store_true',
                        help='Force recalculation even if results exist')
    parser.add_argument('--output_dir', type=str,
                        default='./results/leakage_analysis_results',
                        help='Output directory')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.run_all:
        print("\n" + "=" * 70)
        print("RUNNING ALL CROSSATTENTION ANALYSES")
        print(f"Embeddings: 8M, 150M, 3B")
        print(f"Dataset: {args.dataset}")
        print(f"Epochs: 500, Patience: 30")
        print("=" * 70)

        for emb_short, emb_full in SUPPORTED_EMBEDDINGS.items():
            run_single_analysis(emb_full, args.dataset, args.output_dir, args.force)
    else:
        run_single_analysis(args.embedding, args.dataset, args.output_dir, args.force)

    print(f"\nResults saved in: {args.output_dir}")


if __name__ == '__main__':
    main()
