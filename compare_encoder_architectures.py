#!/usr/bin/env python3
"""
Compare Encoder Architectures for Protein-Ligand Affinity Prediction.

This script compares three encoder variants:
    1. Linear: Simple projection baseline
    2. CNN: Convolutional encoder (current default)
    3. CNN+Attention: Hybrid with local self-attention

Evaluates on three data split scenarios to measure generalization:
    - Random Split (with leakage)
    - Split by Compound (no compound leakage)
    - New Compound + New Kinase (true generalization)

Usage:
    python compare_encoder_architectures.py --embedding 150M --dataset non_human
    python compare_encoder_architectures.py --run_all --dataset non_human

Author: DockTKinase Team
Date: January 2025
"""

import os
import sys
import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project to path
sys.path.insert(0, '/media/leon/ssd2tb/docktkinase')
sys.path.insert(0, '/media/leon/ssd2tb/docktkinase/src')

from src.classifier.models.encoder_variants import (
    LinearEncoder, CNNEncoder, CNNAttentionEncoder, create_encoder
)
from src.classifier.models.cross_attention_model import (
    CrossAttention, CrossAttentionAffinityModel
)
from src.classifier.utils.matrix_dataloader import (
    MatrixEmbeddingDataset, collate_matrix_batch, create_matrix_dataloader
)

# Configuration
plt.style.use('seaborn-v0_8-whitegrid')

# Embeddings
SUPPORTED_EMBEDDINGS = {
    '8M': 'esm2_t6_8M_UR50D',
    '150M': 'esm2_t30_150M_UR50D',
    '650M': 'esm2_t33_650M_UR50D'
}

PROTEIN_DIMS = {
    'esm2_t6_8M_UR50D': 320,
    'esm2_t30_150M_UR50D': 640,
    'esm2_t33_650M_UR50D': 1280
}

LIGAND_DIM = 768

# Paths
DATASET_PATHS = {
    'human': '/media/leon/ssd2tb/docktkinase/tests/datasets/kinase_human_compounds.tsv',
    'non_human': '/media/leon/ssd2tb/docktkinase/tests/datasets/kinase_non_human_compounds.tsv',
}

EMBEDDING_BASE_PATH = '/media/leon/ssd2tb/docktkinase/results/protein_model_benchmark_{dataset_type}_v2'

# Encoder types to compare
ENCODER_TYPES = ['linear', 'cnn', 'cnn_attention']


@dataclass
class TrainingConfig:
    """Training configuration."""
    protein_dim: int = 640
    ligand_dim: int = 768
    hidden_dim: int = 256
    num_heads: int = 8
    num_cross_attn_layers: int = 2
    ff_dim: int = 512
    dropout: float = 0.1
    num_epochs: int = 200  # Reduced for faster comparison
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01


# =============================================================================
# FLEXIBLE MODEL
# =============================================================================

class FlexibleCrossAttentionModel(nn.Module):
    """
    Cross-attention model with configurable encoder type.
    """

    def __init__(
        self,
        protein_dim: int,
        ligand_dim: int,
        hidden_dim: int = 256,
        encoder_type: str = 'cnn',
        num_cross_attn_layers: int = 2,
        num_heads: int = 8,
        ff_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.encoder_type = encoder_type

        # Create encoders based on type
        self.protein_encoder = create_encoder(
            encoder_type, protein_dim, hidden_dim, dropout=dropout,
            num_layers=3, num_cnn_layers=2, num_attention_layers=1
        )
        self.ligand_encoder = create_encoder(
            encoder_type, ligand_dim, hidden_dim, dropout=dropout,
            num_layers=3, num_cnn_layers=2, num_attention_layers=1
        )

        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList()
        for _ in range(num_cross_attn_layers):
            self.cross_attention_layers.append(
                nn.ModuleDict({
                    'protein_to_ligand': nn.MultiheadAttention(
                        hidden_dim, num_heads, dropout=dropout, batch_first=True
                    ),
                    'ligand_to_protein': nn.MultiheadAttention(
                        hidden_dim, num_heads, dropout=dropout, batch_first=True
                    ),
                    'protein_norm': nn.LayerNorm(hidden_dim),
                    'ligand_norm': nn.LayerNorm(hidden_dim),
                    'protein_ff': nn.Sequential(
                        nn.Linear(hidden_dim, ff_dim),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(ff_dim, hidden_dim),
                        nn.Dropout(dropout)
                    ),
                    'ligand_ff': nn.Sequential(
                        nn.Linear(hidden_dim, ff_dim),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(ff_dim, hidden_dim),
                        nn.Dropout(dropout)
                    ),
                    'protein_ff_norm': nn.LayerNorm(hidden_dim),
                    'ligand_ff_norm': nn.LayerNorm(hidden_dim)
                })
            )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(
        self,
        protein_matrix: torch.Tensor,
        ligand_matrix: torch.Tensor,
        protein_mask: Optional[torch.Tensor] = None,
        ligand_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Encode
        protein_encoded = self.protein_encoder(protein_matrix, protein_mask)
        ligand_encoded = self.ligand_encoder(ligand_matrix, ligand_mask)

        # Create attention masks
        protein_key_mask = (protein_mask == 0) if protein_mask is not None else None
        ligand_key_mask = (ligand_mask == 0) if ligand_mask is not None else None

        # Cross-attention
        for layer in self.cross_attention_layers:
            # Protein attends to ligand
            residual = protein_encoded
            protein_encoded = layer['protein_norm'](protein_encoded)
            attn_out, _ = layer['protein_to_ligand'](
                protein_encoded, ligand_encoded, ligand_encoded,
                key_padding_mask=ligand_key_mask
            )
            protein_encoded = residual + attn_out

            # Feed-forward
            residual = protein_encoded
            protein_encoded = layer['protein_ff_norm'](protein_encoded)
            protein_encoded = residual + layer['protein_ff'](protein_encoded)

            # Ligand attends to protein
            residual = ligand_encoded
            ligand_encoded = layer['ligand_norm'](ligand_encoded)
            attn_out, _ = layer['ligand_to_protein'](
                ligand_encoded, protein_encoded, protein_encoded,
                key_padding_mask=protein_key_mask
            )
            ligand_encoded = residual + attn_out

            # Feed-forward
            residual = ligand_encoded
            ligand_encoded = layer['ligand_ff_norm'](ligand_encoded)
            ligand_encoded = residual + layer['ligand_ff'](ligand_encoded)

        # Pool
        if protein_mask is not None:
            protein_mask_expanded = protein_mask.unsqueeze(-1)
            protein_pooled = (protein_encoded * protein_mask_expanded).sum(1) / protein_mask_expanded.sum(1).clamp(min=1)
        else:
            protein_pooled = protein_encoded.mean(1)

        if ligand_mask is not None:
            ligand_mask_expanded = ligand_mask.unsqueeze(-1)
            ligand_pooled = (ligand_encoded * ligand_mask_expanded).sum(1) / ligand_mask_expanded.sum(1).clamp(min=1)
        else:
            ligand_pooled = ligand_encoded.mean(1)

        # Classify
        combined = torch.cat([protein_pooled, ligand_pooled], dim=-1)
        return self.classifier(combined)


# =============================================================================
# SPLITS
# =============================================================================

def split_random(df: pd.DataFrame, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random split (has leakage)."""
    np.random.seed(seed)
    indices = np.random.permutation(len(df))
    n = len(df)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)
    return indices[:train_end], indices[train_end:val_end], indices[val_end:]


def split_by_compound(df: pd.DataFrame, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split by compound (no compound leakage)."""
    np.random.seed(seed)
    compounds = df['chembl_id'].unique()
    np.random.shuffle(compounds)
    n = len(compounds)
    train_compounds = set(compounds[:int(0.8 * n)])
    val_compounds = set(compounds[int(0.8 * n):int(0.9 * n)])
    test_compounds = set(compounds[int(0.9 * n):])

    train_idx = df[df['chembl_id'].isin(train_compounds)].index.values
    val_idx = df[df['chembl_id'].isin(val_compounds)].index.values
    test_idx = df[df['chembl_id'].isin(test_compounds)].index.values

    return train_idx, val_idx, test_idx


def split_new_compound_new_kinase(df: pd.DataFrame, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split ensuring test has new compounds AND new kinases."""
    np.random.seed(seed)

    compounds = df['chembl_id'].unique()
    kinases = df['target_kinase'].unique()
    np.random.shuffle(compounds)
    np.random.shuffle(kinases)

    test_kinases = set(kinases[:int(0.15 * len(kinases))])
    val_kinases = set(kinases[int(0.15 * len(kinases)):int(0.25 * len(kinases))])
    train_kinases = set(kinases[int(0.25 * len(kinases)):])

    test_compounds = set(compounds[:int(0.15 * len(compounds))])
    val_compounds = set(compounds[int(0.15 * len(compounds)):int(0.25 * len(compounds))])
    train_compounds = set(compounds[int(0.25 * len(compounds)):])

    train_mask = df['chembl_id'].isin(train_compounds) & df['target_kinase'].isin(train_kinases)
    val_mask = (df['chembl_id'].isin(val_compounds) | df['target_kinase'].isin(val_kinases)) & ~train_mask
    test_mask = (df['chembl_id'].isin(test_compounds) | df['target_kinase'].isin(test_kinases)) & ~train_mask & ~val_mask

    train_idx = df[train_mask].index.values
    val_idx = df[val_mask].index.values
    test_idx = df[test_mask].index.values

    return train_idx, val_idx, test_idx


# =============================================================================
# TRAINING
# =============================================================================

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in loader:
        protein = batch['protein_matrix'].to(device)
        ligand = batch['ligand_matrix'].to(device)
        protein_mask = batch['protein_mask'].to(device)
        ligand_mask = batch['ligand_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        output = model(protein, ligand, protein_mask, ligand_mask)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device) -> Dict:
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            protein = batch['protein_matrix'].to(device)
            ligand = batch['ligand_matrix'].to(device)
            protein_mask = batch['protein_mask'].to(device)
            ligand_mask = batch['ligand_mask'].to(device)
            labels = batch['labels'].to(device)

            output = model(protein, ligand, protein_mask, ligand_mask)
            probs = torch.sigmoid(output)

            all_probs.extend(probs.cpu().numpy().flatten())
            all_preds.extend((probs > 0.5).cpu().numpy().flatten().astype(int))
            all_labels.extend(labels.cpu().numpy().flatten().astype(int))

    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'mcc': matthews_corrcoef(all_labels, all_preds),
        'auc': roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    device: torch.device
) -> Tuple[nn.Module, Dict]:
    """Train model and return best model + history."""
    model = model.to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_epochs, eta_min=1e-6
    )
    criterion = nn.BCEWithLogitsLoss()

    best_val_mcc = -1
    best_model_state = None
    best_epoch = 0
    history = {'train_loss': [], 'val_mcc': []}

    for epoch in range(config.num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_mcc'].append(val_metrics['mcc'])

        if val_metrics['mcc'] > best_val_mcc:
            best_val_mcc = val_metrics['mcc']
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1

        if (epoch + 1) % 50 == 0:
            print(f"      Epoch {epoch+1}: loss={train_loss:.4f}, val_mcc={val_metrics['mcc']:.4f}")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"      Best model from epoch {best_epoch} (val_mcc={best_val_mcc:.4f})")

    return model, history


# =============================================================================
# COMPARISON
# =============================================================================

def run_encoder_comparison(
    embedding_name: str,
    dataset_type: str,
    output_dir: str,
    config: TrainingConfig
) -> Dict:
    """Run comparison of all encoder types."""

    # Paths
    embedding_dir = os.path.join(
        EMBEDDING_BASE_PATH.format(dataset_type=dataset_type),
        embedding_name, 'build'
    )
    protein_matrix_dir = os.path.join(embedding_dir, 'protein_matrices')
    ligand_matrix_dir = os.path.join(embedding_dir, 'ligand_matrices')
    dataset_path = DATASET_PATHS[dataset_type]

    # Check paths
    if not os.path.exists(protein_matrix_dir):
        print(f"ERROR: {protein_matrix_dir} not found")
        return None

    # Load data
    print(f"\nLoading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path, sep='\t')
    df['label'] = (df['standard_value'] <= 1000).astype(int)
    df['seq_id'] = df['seq_id'].astype(str)
    print(f"  Total: {len(df)} samples")

    device = get_device()
    print(f"  Device: {device}")

    # Results storage
    all_results = {}

    # Define scenarios
    scenarios = [
        ('Random Split', split_random),
        ('Split by Compound', split_by_compound),
        ('New Comp + New Kinase', split_new_compound_new_kinase)
    ]

    # Test each encoder type
    for encoder_type in ENCODER_TYPES:
        print(f"\n{'='*60}")
        print(f"ENCODER: {encoder_type.upper()}")
        print(f"{'='*60}")

        encoder_results = {}

        for scenario_name, split_fn in scenarios:
            print(f"\n  Scenario: {scenario_name}")

            # Split data
            train_idx, val_idx, test_idx = split_fn(df)
            train_df = df.iloc[train_idx].reset_index(drop=True)
            val_df = df.iloc[val_idx].reset_index(drop=True)
            test_df = df.iloc[test_idx].reset_index(drop=True)

            print(f"    Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

            # Create loaders
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

            # Create model
            model = FlexibleCrossAttentionModel(
                protein_dim=config.protein_dim,
                ligand_dim=config.ligand_dim,
                hidden_dim=config.hidden_dim,
                encoder_type=encoder_type,
                num_cross_attn_layers=config.num_cross_attn_layers,
                num_heads=config.num_heads,
                ff_dim=config.ff_dim,
                dropout=config.dropout
            )

            n_params = sum(p.numel() for p in model.parameters())
            print(f"    Parameters: {n_params:,}")

            # Train
            print(f"    Training for {config.num_epochs} epochs...")
            model, history = train_model(model, train_loader, val_loader, config, device)

            # Evaluate on test
            test_metrics = evaluate(model, test_loader, device)
            print(f"    Test: Acc={test_metrics['accuracy']:.4f}, MCC={test_metrics['mcc']:.4f}, AUC={test_metrics['auc']:.4f}")

            encoder_results[scenario_name] = {
                'accuracy': test_metrics['accuracy'],
                'mcc': test_metrics['mcc'],
                'auc': test_metrics['auc'],
                'f1': test_metrics['f1'],
                'n_params': n_params
            }

        all_results[encoder_type] = encoder_results

    return all_results


def save_results(results: Dict, embedding_name: str, dataset_type: str, output_dir: str):
    """Save results to JSON."""
    output = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'embedding': embedding_name,
            'dataset': dataset_type,
            'encoders': ENCODER_TYPES
        },
        'results': results
    }

    os.makedirs(output_dir, exist_ok=True)
    short_name = embedding_name.replace('esm2_', '').replace('_UR50D', '')
    filepath = os.path.join(output_dir, f'{dataset_type}_{short_name}_encoder_comparison.json')

    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved: {filepath}")
    return filepath


def plot_comparison(results: Dict, embedding_name: str, dataset_type: str, output_dir: str):
    """Plot comparison of encoders."""
    scenarios = list(list(results.values())[0].keys())
    encoders = list(results.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # MCC comparison
    ax1 = axes[0]
    x = np.arange(len(scenarios))
    width = 0.25

    for i, encoder in enumerate(encoders):
        mccs = [results[encoder][s]['mcc'] for s in scenarios]
        bars = ax1.bar(x + i*width, mccs, width, label=encoder.upper())
        for bar, mcc in zip(bars, mccs):
            ax1.annotate(f'{mcc:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)

    ax1.set_xlabel('Scenario')
    ax1.set_ylabel('MCC')
    ax1.set_title(f'MCC Comparison - {embedding_name}')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels([s.replace(' ', '\n') for s in scenarios])
    ax1.legend()
    ax1.set_ylim(0, 1)

    # Generalization drop
    ax2 = axes[1]
    drops = []
    for encoder in encoders:
        random_mcc = results[encoder]['Random Split']['mcc']
        hard_mcc = results[encoder]['New Comp + New Kinase']['mcc']
        drop = 100 * (random_mcc - hard_mcc) / random_mcc if random_mcc > 0 else 0
        drops.append(drop)

    bars = ax2.bar(encoders, drops, color=['#3498db', '#2ecc71', '#e74c3c'])
    for bar, drop in zip(bars, drops):
        ax2.annotate(f'{drop:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10, fontweight='bold')

    ax2.set_xlabel('Encoder Type')
    ax2.set_ylabel('MCC Drop (%)')
    ax2.set_title('Generalization Drop (Random → New Comp+Kinase)')
    ax2.set_xticklabels([e.upper() for e in encoders])

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    short_name = embedding_name.replace('esm2_', '').replace('_UR50D', '')
    filepath = os.path.join(output_dir, f'{dataset_type}_{short_name}_encoder_comparison.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Plot saved: {filepath}")


def print_summary(results: Dict):
    """Print summary table."""
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    scenarios = list(list(results.values())[0].keys())
    encoders = list(results.keys())

    # Header
    print(f"\n{'Encoder':<15}", end='')
    for s in scenarios:
        print(f"{s:<20}", end='')
    print("Drop%")
    print("-"*70)

    # Data
    for encoder in encoders:
        print(f"{encoder.upper():<15}", end='')
        for s in scenarios:
            mcc = results[encoder][s]['mcc']
            print(f"{mcc:.4f}{'':<14}", end='')

        random_mcc = results[encoder]['Random Split']['mcc']
        hard_mcc = results[encoder]['New Comp + New Kinase']['mcc']
        drop = 100 * (random_mcc - hard_mcc) / random_mcc if random_mcc > 0 else 0
        print(f"{drop:.1f}%")

    # Best encoder
    print("\n" + "-"*70)
    best_encoder = min(encoders, key=lambda e: 100 * (results[e]['Random Split']['mcc'] - results[e]['New Comp + New Kinase']['mcc']) / max(results[e]['Random Split']['mcc'], 0.001))
    best_hard_mcc = max(encoders, key=lambda e: results[e]['New Comp + New Kinase']['mcc'])

    print(f"Best generalization: {best_encoder.upper()}")
    print(f"Best hard MCC: {best_hard_mcc.upper()} ({results[best_hard_mcc]['New Comp + New Kinase']['mcc']:.4f})")


def main():
    parser = argparse.ArgumentParser(description='Compare encoder architectures')

    parser.add_argument('--embedding', type=str, default='150M',
                       choices=['8M', '150M', '650M'])
    parser.add_argument('--dataset', type=str, default='non_human',
                       choices=['human', 'non_human'])
    parser.add_argument('--run_all', action='store_true',
                       help='Run all embeddings')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--output_dir', type=str,
                       default='/media/leon/ssd2tb/docktkinase/encoder_comparison_results')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("ENCODER ARCHITECTURE COMPARISON")
    print("="*70)
    print(f"Encoders: {', '.join(e.upper() for e in ENCODER_TYPES)}")
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}")

    embeddings_to_run = list(SUPPORTED_EMBEDDINGS.values()) if args.run_all else [SUPPORTED_EMBEDDINGS.get(args.embedding, args.embedding)]

    for embedding_name in embeddings_to_run:
        print(f"\n{'#'*70}")
        print(f"EMBEDDING: {embedding_name}")
        print(f"{'#'*70}")

        protein_dim = PROTEIN_DIMS.get(embedding_name, 640)
        config = TrainingConfig(
            protein_dim=protein_dim,
            ligand_dim=LIGAND_DIM,
            num_epochs=args.epochs
        )

        results = run_encoder_comparison(
            embedding_name, args.dataset, args.output_dir, config
        )

        if results:
            save_results(results, embedding_name, args.dataset, args.output_dir)
            plot_comparison(results, embedding_name, args.dataset, args.output_dir)
            print_summary(results)


if __name__ == '__main__':
    main()
