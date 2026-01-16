#!/usr/bin/env python3
"""
DtKinase Multi-Seed Execution
=============================

Executa DT-Kinase (CNN + Cross-Attention) com múltiplas seeds para análise
estatística robusta, compatível com os resultados do baseline_multiseed.py.

Suporta:
- 5 seeds diferentes: [42, 123, 420, 777, 2024]
- Split Random e Stratified
- 3 modelos ESM-2: 8M, 150M, 3B

Autor: DockTKinase Project
Data: 2025
"""

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Adicionar src ao path para imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef
)
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')

# ==============================================================================
# CONSTANTES GLOBAIS
# ==============================================================================

# Seeds para reproducibilidade (mesmas do baseline_multiseed.py)
SEEDS = [42, 123, 420, 777, 2024]

# Configurações de split
TEST_SIZE = 0.1
VAL_SIZE = 0.1

# Modelos ESM-2 e suas dimensões
ESM_MODELS = [
    'esm2_t6_8M_UR50D',
    'esm2_t30_150M_UR50D',
    'esm2_t36_3B_UR50D',
]

ESM_DIMS = {
    'esm2_t6_8M_UR50D': 320,
    'esm2_t30_150M_UR50D': 640,
    'esm2_t36_3B_UR50D': 2560,
}

# Dimensão dos embeddings de ligantes (SMI-TED)
LIGAND_DIM = 768

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==============================================================================
# DATASET
# ==============================================================================

class ProteinLigandMatrixDataset(Dataset):
    """Dataset para carregar matrizes per-token de proteínas e ligantes."""
    
    def __init__(self, df: pd.DataFrame, protein_dir: Path, ligand_dir: Path):
        self.df = df.reset_index(drop=True)
        self.protein_dir = protein_dir
        self.ligand_dir = ligand_dir
        
        # Pré-verificar arquivos existentes
        self.valid_indices = []
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            prot_file = self.protein_dir / f"{row['seq_id']}_matrix.npy"
            lig_file = self.ligand_dir / f"{row['chembl_id']}_matrix.npy"
            if prot_file.exists() and lig_file.exists():
                self.valid_indices.append(idx)
        
        if len(self.valid_indices) < len(self.df):
            print(f"      ⚠️ Dataset: {len(self.valid_indices)}/{len(self.df)} amostras válidas")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        row = self.df.iloc[real_idx]
        
        # Carregar matrizes
        prot_matrix = np.load(self.protein_dir / f"{row['seq_id']}_matrix.npy")
        lig_matrix = np.load(self.ligand_dir / f"{row['chembl_id']}_matrix.npy")
        
        # Ligantes são vetores (1, 768) - squeeze para (768,) e unsqueeze para (1, 768)
        if lig_matrix.ndim == 2 and lig_matrix.shape[0] == 1:
            lig_matrix = lig_matrix  # Manter (1, 768) para compatibilidade com collate
        
        # Label binário
        label = 1.0 if row['pchembl_value'] >= 6.0 else 0.0
        
        return (
            torch.tensor(prot_matrix, dtype=torch.float32),
            torch.tensor(lig_matrix, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32)
        )


def collate_matrices(batch):
    """Collate function para matrizes de tamanhos variáveis."""
    proteins, ligands, labels = zip(*batch)
    
    # Pad proteínas (seq_len varia)
    max_prot_len = max(p.shape[0] for p in proteins)
    prot_dim = proteins[0].shape[1]
    padded_proteins = torch.zeros(len(proteins), max_prot_len, prot_dim)
    prot_masks = torch.zeros(len(proteins), max_prot_len, dtype=torch.bool)
    
    for i, p in enumerate(proteins):
        padded_proteins[i, :p.shape[0], :] = p
        prot_masks[i, :p.shape[0]] = True
    
    # Pad ligantes (token_len varia)
    max_lig_len = max(l.shape[0] for l in ligands)
    lig_dim = ligands[0].shape[1]
    padded_ligands = torch.zeros(len(ligands), max_lig_len, lig_dim)
    lig_masks = torch.zeros(len(ligands), max_lig_len, dtype=torch.bool)
    
    for i, l in enumerate(ligands):
        padded_ligands[i, :l.shape[0], :] = l
        lig_masks[i, :l.shape[0]] = True
    
    labels = torch.stack(labels)
    
    return padded_proteins, prot_masks, padded_ligands, lig_masks, labels


# ==============================================================================
# TRAINING FUNCTIONS
# ==============================================================================

def set_seed(seed: int):
    """Define seed para reproducibilidade."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict:
    """Calcula métricas de classificação."""
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_true, y_prob)),
        'pr_auc': float(average_precision_score(y_true, y_prob)),
        'mcc': float(matthews_corrcoef(y_true, y_pred)),
    }


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Treina o modelo por uma época."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    for proteins, prot_masks, ligands, lig_masks, labels in dataloader:
        proteins = proteins.to(device)
        prot_masks = prot_masks.to(device)
        ligands = ligands.to(device)
        lig_masks = lig_masks.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(proteins, ligands, prot_masks, lig_masks)
        # O modelo retorna dict com 'classification' e 'regression'
        logits = outputs['classification'].squeeze()
        loss = criterion(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * len(labels)
        
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        
        all_probs.extend(probs.tolist() if hasattr(probs, 'tolist') else [probs])
        all_preds.extend(preds.tolist() if hasattr(preds, 'tolist') else [preds])
        all_labels.extend(labels.cpu().numpy().tolist())
    
    avg_loss = total_loss / len(dataloader.dataset)
    metrics = calculate_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs)
    )
    metrics['loss'] = avg_loss
    
    return metrics


def evaluate(model, dataloader, criterion, device):
    """Avalia o modelo."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for proteins, prot_masks, ligands, lig_masks, labels in dataloader:
            proteins = proteins.to(device)
            prot_masks = prot_masks.to(device)
            ligands = ligands.to(device)
            lig_masks = lig_masks.to(device)
            labels = labels.to(device)
            
            outputs = model(proteins, ligands, prot_masks, lig_masks)
            # O modelo retorna dict com 'classification' e 'regression'
            logits = outputs['classification'].squeeze()
            loss = criterion(logits, labels)
            
            total_loss += loss.item() * len(labels)
            
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            
            all_probs.extend(probs.tolist() if hasattr(probs, 'tolist') else [probs])
            all_preds.extend(preds.tolist() if hasattr(preds, 'tolist') else [preds])
            all_labels.extend(labels.cpu().numpy().tolist())
    
    avg_loss = total_loss / len(dataloader.dataset)
    metrics = calculate_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs)
    )
    metrics['loss'] = avg_loss
    
    return metrics


# ==============================================================================
# SPLIT FUNCTIONS
# ==============================================================================

def split_random(df: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split aleatório estratificado."""
    df = df.copy()
    df['binary_label'] = (df['pchembl_value'] >= 6.0).astype(int)
    
    train_df, temp_df = train_test_split(
        df, test_size=TEST_SIZE + VAL_SIZE,
        random_state=seed, stratify=df['binary_label']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=TEST_SIZE / (TEST_SIZE + VAL_SIZE),
        random_state=seed, stratify=temp_df['binary_label']
    )
    
    return train_df, val_df, test_df


def split_stratified_by_protein(df: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split estratificado por proteína (sem data leakage)."""
    df = df.copy()
    df['binary_label'] = (df['pchembl_value'] >= 6.0).astype(int)
    
    # Agregação por proteína
    protein_stats = df.groupby('seq_id').agg({
        'binary_label': ['mean', 'count']
    }).reset_index()
    protein_stats.columns = ['seq_id', 'positive_ratio', 'n_samples']
    
    # Criar strata baseados na razão de positivos
    protein_stats['strata'] = pd.cut(
        protein_stats['positive_ratio'],
        bins=[0, 0.3, 0.7, 1.0],
        labels=['low', 'medium', 'high'],
        include_lowest=True
    )
    
    # Proteínas únicas
    proteins = protein_stats['seq_id'].values
    strata = protein_stats['strata'].values
    
    # Tentar split estratificado, fallback para random se necessário
    try:
        train_prot, temp_prot, _, temp_strata = train_test_split(
            proteins, strata,
            test_size=TEST_SIZE + VAL_SIZE,
            random_state=seed,
            stratify=strata
        )
        val_prot, test_prot = train_test_split(
            temp_prot,
            test_size=TEST_SIZE / (TEST_SIZE + VAL_SIZE),
            random_state=seed,
            stratify=temp_strata
        )
    except ValueError:
        # Fallback se estratificação falhar
        train_prot, temp_prot = train_test_split(
            proteins,
            test_size=TEST_SIZE + VAL_SIZE,
            random_state=seed
        )
        val_prot, test_prot = train_test_split(
            temp_prot,
            test_size=TEST_SIZE / (TEST_SIZE + VAL_SIZE),
            random_state=seed
        )
    
    # Filtrar DataFrame
    train_df = df[df['seq_id'].isin(train_prot)]
    val_df = df[df['seq_id'].isin(val_prot)]
    test_df = df[df['seq_id'].isin(test_prot)]
    
    return train_df, val_df, test_df


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def run_single_seed(
    esm_model: str,
    data_df: pd.DataFrame,
    embeddings_dir: Path,
    seed: int,
    split_type: str,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    hidden_dim: int = 256,
    verbose: bool = True
) -> Dict:
    """Executa DtKinase para uma única seed."""
    
    set_seed(seed)
    
    # Paths
    build_dir = embeddings_dir / esm_model / 'build'
    protein_dim = ESM_DIMS[esm_model]
    
    # Split
    if split_type == 'random':
        train_df, val_df, test_df = split_random(data_df, seed)
    else:  # stratified
        train_df, val_df, test_df = split_stratified_by_protein(data_df, seed)
    
    if verbose:
        print(f"         Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    
    # Datasets
    protein_matrix_dir = build_dir / 'protein_matrices'
    ligand_matrix_dir = build_dir / 'ligand_matrices'
    
    train_dataset = ProteinLigandMatrixDataset(train_df, protein_matrix_dir, ligand_matrix_dir)
    val_dataset = ProteinLigandMatrixDataset(val_df, protein_matrix_dir, ligand_matrix_dir)
    test_dataset = ProteinLigandMatrixDataset(test_df, protein_matrix_dir, ligand_matrix_dir)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_matrices, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_matrices, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_matrices, num_workers=4, pin_memory=True
    )
    
    # Import modelo
    from classifier.models.cross_attention_model import CrossAttentionAffinityModel
    
    # Criar modelo
    model = CrossAttentionAffinityModel(
        protein_dim=protein_dim,
        ligand_dim=LIGAND_DIM,
        hidden_dim=hidden_dim,
        num_cnn_layers=3,
        num_cross_attn_layers=2,
        num_heads=8,
        ff_dim=512,
        dropout=0.1,
        use_positional_encoding=True,
        positional_encoding_type='sinusoidal'
    ).to(DEVICE)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()
    
    # Training
    best_val_auc = 0.0
    best_model_state = None
    patience_counter = 0
    patience = 10
    
    train_start = time.time()
    
    for epoch in range(epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_metrics = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        
        if val_metrics['roc_auc'] > best_val_auc:
            best_val_auc = val_metrics['roc_auc']
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            if verbose:
                print(f"         Early stopping at epoch {epoch+1}")
            break
    
    train_time = time.time() - train_start
    
    # Carregar melhor modelo
    if best_model_state:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_model_state.items()})
    
    # Avaliar em todos os conjuntos
    final_train_metrics = evaluate(model, train_loader, criterion, DEVICE)
    final_val_metrics = evaluate(model, val_loader, criterion, DEVICE)
    final_test_metrics = evaluate(model, test_loader, criterion, DEVICE)
    
    return {
        'seed': seed,
        'split_type': split_type,
        'n_train': len(train_df),
        'n_val': len(val_df),
        'n_test': len(test_df),
        'train_time': train_time,
        'epochs_trained': epoch + 1,
        'best_val_auc': best_val_auc,
        'train_metrics': final_train_metrics,
        'val_metrics': final_val_metrics,
        'test_metrics': final_test_metrics,
    }


def run_multiseed_for_model(
    esm_model: str,
    data_df: pd.DataFrame,
    embeddings_dir: Path,
    split_type: str,
    seeds: List[int],
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    hidden_dim: int = 256,
    verbose: bool = True
) -> Dict:
    """Executa DtKinase para múltiplas seeds."""
    
    results = {
        'esm_model': esm_model,
        'protein_dim': ESM_DIMS[esm_model],
        'split_type': split_type,
        'seeds': seeds,
        'per_seed_results': [],
    }
    
    test_aucs = []
    
    for i, seed in enumerate(seeds):
        if verbose:
            print(f"\n      🎲 Seed {seed} ({i+1}/{len(seeds)})")
        
        seed_result = run_single_seed(
            esm_model=esm_model,
            data_df=data_df,
            embeddings_dir=embeddings_dir,
            seed=seed,
            split_type=split_type,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            hidden_dim=hidden_dim,
            verbose=verbose
        )
        
        results['per_seed_results'].append(seed_result)
        test_aucs.append(seed_result['test_metrics']['roc_auc'])
        
        if verbose:
            print(f"         Test AUC: {seed_result['test_metrics']['roc_auc']:.4f} | "
                  f"Train AUC: {seed_result['train_metrics']['roc_auc']:.4f}")
    
    # Estatísticas agregadas
    results['statistics'] = {
        'test_auc_mean': float(np.mean(test_aucs)),
        'test_auc_std': float(np.std(test_aucs)),
        'test_auc_min': float(np.min(test_aucs)),
        'test_auc_max': float(np.max(test_aucs)),
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='DtKinase Multi-Seed Execution',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--data-path', type=str,
                        default='tests/datasets/kinase_non_human_compounds.tsv',
                        help='Caminho para o arquivo TSV de dados')
    parser.add_argument('--embeddings-dir', type=str,
                        default='results/protein_model_benchmark_non_human_v2',
                        help='Diretório com embeddings pré-calculados')
    parser.add_argument('--output', type=str,
                        default='results/dtkinase_multiseed',
                        help='Diretório de saída')
    parser.add_argument('--split', type=str, choices=['random', 'stratified', 'both'],
                        default='both',
                        help='Tipo de split (random, stratified ou both)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Número de épocas')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Tamanho do batch')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='Dimensão oculta do modelo')
    parser.add_argument('--esm-models', type=str, nargs='+',
                        default=None,
                        help='Modelos ESM-2 específicos (default: todos)')
    
    args = parser.parse_args()
    
    # Paths
    data_path = Path(args.data_path)
    embeddings_dir = Path(args.embeddings_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Header
    print('=' * 70)
    print('🧠 DtKinase Multi-Seed Execution (CNN + Cross-Attention)')
    print('=' * 70)
    print()
    print('📋 Configuração:')
    print(f'   Device: {DEVICE}')
    print(f'   Seeds: {SEEDS}')
    print(f'   Split: {args.split}')
    print(f'   Epochs: {args.epochs}')
    print(f'   Batch Size: {args.batch_size}')
    print(f'   Learning Rate: {args.learning_rate}')
    print()
    
    # Carregar dados
    print(f'📂 Carregando dados de: {data_path}')
    df = pd.read_csv(data_path, sep='\t')
    print(f'   Total amostras: {len(df):,}')
    print(f'   Proteínas únicas: {df["seq_id"].nunique()}')
    print()
    
    # Modelos a processar
    models_to_run = args.esm_models if args.esm_models else ESM_MODELS
    
    # Splits a executar
    splits_to_run = ['random', 'stratified'] if args.split == 'both' else [args.split]
    
    # Resultados globais
    all_results = {
        'pipeline': 'dtkinase_multiseed',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'seeds': SEEDS,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'hidden_dim': args.hidden_dim,
            'device': str(DEVICE),
        },
        'results': {}
    }
    
    total_start = time.time()
    
    for split_type in splits_to_run:
        print('=' * 70)
        print(f'📊 Split: {split_type.upper()}')
        print('=' * 70)
        
        split_results = {}
        
        for esm_model in models_to_run:
            print(f'\n   🧬 Modelo: {esm_model} ({ESM_DIMS.get(esm_model, "?")}D)')
            
            try:
                result = run_multiseed_for_model(
                    esm_model=esm_model,
                    data_df=df,
                    embeddings_dir=embeddings_dir,
                    split_type=split_type,
                    seeds=SEEDS,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    hidden_dim=args.hidden_dim,
                    verbose=True
                )
                split_results[esm_model] = result
                
                stats = result['statistics']
                print(f"\n      📈 Resultado: {stats['test_auc_mean']:.4f} ± {stats['test_auc_std']:.4f}")
                
            except Exception as e:
                print(f'   ❌ Erro: {e}')
                import traceback
                traceback.print_exc()
                split_results[esm_model] = {'error': str(e)}
        
        all_results['results'][split_type] = split_results
        
        # Salvar resultados parciais
        results_file = output_dir / f'dtkinase_{split_type}_multiseed_results.json'
        with open(results_file, 'w') as f:
            json.dump({'results': split_results}, f, indent=2, default=str)
        print(f'\n   💾 Salvo: {results_file}')
    
    total_time = time.time() - total_start
    all_results['total_time_seconds'] = total_time
    
    # Salvar resultados completos
    final_file = output_dir / 'dtkinase_multiseed_complete.json'
    with open(final_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Resumo final
    print()
    print('=' * 70)
    print('✅ EXECUÇÃO CONCLUÍDA')
    print('=' * 70)
    print()
    print('📊 RESUMO (Test ROC-AUC):')
    print('-' * 70)
    print(f'{"Modelo ESM-2":<25} {"Random":<20} {"Stratified":<20}')
    print('-' * 70)
    
    for esm_model in models_to_run:
        random_res = all_results['results'].get('random', {}).get(esm_model, {})
        strat_res = all_results['results'].get('stratified', {}).get(esm_model, {})
        
        if 'error' in random_res:
            random_str = "ERROR"
        else:
            s = random_res.get('statistics', {})
            random_str = f"{s.get('test_auc_mean', 0):.4f} ± {s.get('test_auc_std', 0):.4f}"
        
        if 'error' in strat_res:
            strat_str = "ERROR"
        else:
            s = strat_res.get('statistics', {})
            strat_str = f"{s.get('test_auc_mean', 0):.4f} ± {s.get('test_auc_std', 0):.4f}"
        
        print(f'{esm_model:<25} {random_str:<20} {strat_str:<20}')
    
    print('-' * 70)
    print()
    print(f'⏱️  Tempo total: {total_time:.2f}s ({total_time/60:.2f} min)')
    print(f'💾 Resultados completos: {final_file}')
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
