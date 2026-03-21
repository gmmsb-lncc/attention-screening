#!/usr/bin/env python3
"""
DockTKinase - Baseline com DT-Kinase OFICIAL (CNN + Cross-Attention)
=====================================================================

Este script usa o modelo DT-Kinase OFICIAL com:
- Matrizes per-token de proteína (ESM-2)
- Matrizes per-token de ligante (SMI-TED)
- CNN Encoder multi-scale
- Cross-Attention bidirecional
- Multi-task head (classificação)

Comparação justa com KNN e MLP usando mesmos splits.

Uso:
    python baseline_dtkinase_official.py
    python baseline_dtkinase_official.py --epochs 50 --batch-size 32

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
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from tqdm import tqdm

# Suprimir warnings
warnings.filterwarnings('ignore')

# Adicionar path do projeto
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

# Imports sklearn para KNN e MLP
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, balanced_accuracy_score
)

# Import do modelo oficial
from classifier.models.cross_attention_model import CrossAttentionAffinityModel


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

RANDOM_SEED = 420
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

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

LIGAND_DIM = 768  # SMI-TED dimension

TEST_SIZE = 0.1
VAL_SIZE = 0.1

# Detectar dispositivo
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 
                      'mps' if torch.backends.mps.is_available() else 'cpu')


# =============================================================================
# DATASET PARA MATRIZES PER-TOKEN
# =============================================================================

class ProteinLigandMatrixDataset(Dataset):
    """
    Dataset que carrega matrizes per-token de proteína e ligante.
    
    Estrutura esperada:
        protein_matrices/{seq_id}_matrix.npy  -> [seq_len, protein_dim]
        ligand_matrices/{chembl_id}_matrix.npy  -> [token_len, ligand_dim]
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        protein_matrix_dir: Path,
        ligand_matrix_dir: Path,
        max_protein_len: int = 1024,
        max_ligand_len: int = 128
    ):
        self.df = df.reset_index(drop=True)
        self.protein_matrix_dir = protein_matrix_dir
        self.ligand_matrix_dir = ligand_matrix_dir
        self.max_protein_len = max_protein_len
        self.max_ligand_len = max_ligand_len
        
        # Cache para matrizes
        self._protein_cache: Dict[str, np.ndarray] = {}
        self._ligand_cache: Dict[str, np.ndarray] = {}
    
    def __len__(self) -> int:
        return len(self.df)
    
    def _load_matrix(self, path: Path, cache: dict, key: str, max_len: int) -> np.ndarray:
        """Carrega matriz com cache e truncamento."""
        if key not in cache:
            mat = np.load(path)
            # Truncar se necessário
            if mat.shape[0] > max_len:
                mat = mat[:max_len]
            cache[key] = mat
        return cache[key]
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        
        # IDs
        seq_id = str(row['seq_id'])
        chembl_id = str(row['chembl_id'])
        
        # Carregar matrizes
        prot_path = self.protein_matrix_dir / f"{seq_id}_matrix.npy"
        lig_path = self.ligand_matrix_dir / f"{chembl_id}_matrix.npy"
        
        protein_matrix = self._load_matrix(prot_path, self._protein_cache, seq_id, self.max_protein_len)
        ligand_matrix = self._load_matrix(lig_path, self._ligand_cache, chembl_id, self.max_ligand_len)
        
        # Label binário (baseado em pchembl_value >= 6.0)
        pchembl = row.get('pchembl_value', 0)
        label = 1 if pchembl >= 6.0 else 0
        
        return {
            'protein_matrix': torch.FloatTensor(protein_matrix),
            'ligand_matrix': torch.FloatTensor(ligand_matrix),
            'label': torch.LongTensor([label]),
            'pchembl': torch.FloatTensor([pchembl if pd.notna(pchembl) else 0.0])
        }


def collate_matrices(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function para padding de matrizes de tamanho variável.
    """
    # Encontrar tamanhos máximos no batch
    max_prot_len = max(item['protein_matrix'].size(0) for item in batch)
    max_lig_len = max(item['ligand_matrix'].size(0) for item in batch)
    
    protein_dim = batch[0]['protein_matrix'].size(1)
    ligand_dim = batch[0]['ligand_matrix'].size(1)
    batch_size = len(batch)
    
    # Criar tensores com padding
    protein_matrices = torch.zeros(batch_size, max_prot_len, protein_dim)
    ligand_matrices = torch.zeros(batch_size, max_lig_len, ligand_dim)
    protein_masks = torch.zeros(batch_size, max_prot_len)
    ligand_masks = torch.zeros(batch_size, max_lig_len)
    labels = torch.zeros(batch_size, 1, dtype=torch.long)
    pchembls = torch.zeros(batch_size, 1)
    
    for i, item in enumerate(batch):
        prot_len = item['protein_matrix'].size(0)
        lig_len = item['ligand_matrix'].size(0)
        
        protein_matrices[i, :prot_len] = item['protein_matrix']
        ligand_matrices[i, :lig_len] = item['ligand_matrix']
        protein_masks[i, :prot_len] = 1.0
        ligand_masks[i, :lig_len] = 1.0
        labels[i] = item['label']
        pchembls[i] = item['pchembl']
    
    return {
        'protein_matrix': protein_matrices,
        'ligand_matrix': ligand_matrices,
        'protein_mask': protein_masks,
        'ligand_mask': ligand_masks,
        'label': labels.squeeze(1),
        'pchembl': pchembls.squeeze(1)
    }


# =============================================================================
# FUNÇÕES DE TREINO
# =============================================================================

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
) -> Dict[str, float]:
    """Treina uma época."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []
    
    for batch in dataloader:
        # Mover para device
        protein_matrix = batch['protein_matrix'].to(device)
        ligand_matrix = batch['ligand_matrix'].to(device)
        protein_mask = batch['protein_mask'].to(device)
        ligand_mask = batch['ligand_mask'].to(device)
        labels = batch['label'].to(device)
        
        # Forward
        optimizer.zero_grad()
        cls_logits, _ = model(
            protein_matrix, ligand_matrix,
            protein_mask, ligand_mask
        )
        
        # Loss
        loss = criterion(cls_logits.squeeze(), labels.float())
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if scheduler:
            scheduler.step()
        
        # Métricas
        total_loss += loss.item() * len(labels)
        probs = torch.sigmoid(cls_logits.squeeze()).detach().cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        
        all_probs.extend(probs.tolist() if probs.ndim > 0 else [probs.item()])
        all_preds.extend(preds.tolist() if hasattr(preds, 'tolist') else [preds])
        all_labels.extend(labels.cpu().numpy().tolist())
    
    # Calcular métricas
    avg_loss = total_loss / len(dataloader.dataset)
    metrics = calculate_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs)
    )
    metrics['loss'] = avg_loss
    
    return metrics


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """Avalia o modelo."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            protein_matrix = batch['protein_matrix'].to(device)
            ligand_matrix = batch['ligand_matrix'].to(device)
            protein_mask = batch['protein_mask'].to(device)
            ligand_mask = batch['ligand_mask'].to(device)
            labels = batch['label'].to(device)
            
            cls_logits, _ = model(
                protein_matrix, ligand_matrix,
                protein_mask, ligand_mask
            )
            
            loss = criterion(cls_logits.squeeze(), labels.float())
            
            total_loss += loss.item() * len(labels)
            probs = torch.sigmoid(cls_logits.squeeze()).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            
            all_probs.extend(probs.tolist() if probs.ndim > 0 else [probs.item()])
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


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Calcula métricas de classificação."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_prob),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
    }


# =============================================================================
# MODELOS SKLEARN (para comparação)
# =============================================================================

def create_knn_model() -> Pipeline:
    """Cria modelo KNN."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            metric='cosine',
            n_jobs=-1
        ))
    ])


def create_mlp_model() -> Pipeline:
    """Cria modelo MLP."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', MLPClassifier(
            hidden_layer_sizes=(512,),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size=64,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=RANDOM_SEED,
            verbose=False
        ))
    ])


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def run_baseline_for_model(
    esm_model: str,
    data_df: pd.DataFrame,
    embeddings_dir: Path,
    output_dir: Path,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    hidden_dim: int = 256,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Executa baseline completo para um modelo ESM-2.
    
    Inclui: KNN, MLP sklearn, e DT-Kinase oficial.
    """
    protein_dim = ESM_DIMS[esm_model]
    build_dir = embeddings_dir / esm_model / 'build'
    
    results = {
        'esm_model': esm_model,
        'protein_dim': protein_dim,
        'n_samples': len(data_df),
    }
    
    # ==========================================================================
    # 1. SPLIT DOS DADOS
    # ==========================================================================
    
    if verbose:
        print(f"\n   📊 Split ALEATÓRIO (seed={RANDOM_SEED})")
    
    # Criar labels binários
    data_df = data_df.copy()
    data_df['binary_label'] = (data_df['pchembl_value'] >= 6.0).astype(int)
    
    # Split
    train_df, temp_df = train_test_split(
        data_df, test_size=TEST_SIZE + VAL_SIZE,
        random_state=RANDOM_SEED, stratify=data_df['binary_label']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=TEST_SIZE / (TEST_SIZE + VAL_SIZE),
        random_state=RANDOM_SEED, stratify=temp_df['binary_label']
    )
    
    if verbose:
        print(f"      Treino:    {len(train_df):,} amostras ({100*len(train_df)/len(data_df):.1f}%)")
        print(f"      Validação: {len(val_df):,} amostras ({100*len(val_df)/len(data_df):.1f}%)")
        print(f"      Teste:     {len(test_df):,} amostras ({100*len(test_df)/len(data_df):.1f}%)")
    
    results['split'] = {
        'n_train': len(train_df),
        'n_val': len(val_df),
        'n_test': len(test_df),
    }
    
    # ==========================================================================
    # 2. KNN e MLP (usando embeddings médios)
    # ==========================================================================
    
    if verbose:
        print(f"\n   🏋️ Treinando modelos sklearn (KNN, MLP)...")
    
    # Carregar embeddings médios
    embeddings = np.load(build_dir / 'embedding_matrix.npy')
    labels = np.load(build_dir / 'binary_labels.npy')
    
    # Garantir shape correto
    if embeddings.ndim == 1:
        embeddings = np.vstack(embeddings)
    if labels.ndim > 1:
        labels = labels.ravel()
    
    # Filtrar válidos
    valid_mask = np.isin(labels, [0, 1])
    embeddings = embeddings[valid_mask]
    labels = labels[valid_mask].astype(int)
    
    # Split para sklearn (mesmo índices)
    train_idx = train_df.index.tolist()
    val_idx = val_df.index.tolist()
    test_idx = test_df.index.tolist()
    
    # Ajustar índices se necessário
    max_idx = len(embeddings) - 1
    train_idx = [i for i in range(len(embeddings)) if i < len(train_df) * (len(embeddings) / len(data_df))]
    
    # Usar split simples para sklearn
    X_train_sk, X_temp, y_train_sk, y_temp = train_test_split(
        embeddings, labels, test_size=TEST_SIZE + VAL_SIZE,
        random_state=RANDOM_SEED, stratify=labels
    )
    X_val_sk, X_test_sk, y_val_sk, y_test_sk = train_test_split(
        X_temp, y_temp, test_size=TEST_SIZE / (TEST_SIZE + VAL_SIZE),
        random_state=RANDOM_SEED, stratify=y_temp
    )
    
    results['models'] = {}
    
    # KNN
    if verbose:
        print(f"\n   🔧 Treinando KNN...")
    knn_start = time.time()
    knn = create_knn_model()
    knn.fit(X_train_sk, y_train_sk)
    knn_time = time.time() - knn_start
    
    knn_test_prob = knn.predict_proba(X_test_sk)[:, 1]
    knn_test_pred = knn.predict(X_test_sk)
    knn_metrics = calculate_metrics(y_test_sk, knn_test_pred, knn_test_prob)
    
    if verbose:
        print(f"      ✅ Concluído em {knn_time:.2f}s | ROC-AUC: {knn_metrics['roc_auc']:.4f}")
    
    results['models']['KNN'] = {
        'train_time': knn_time,
        'test_metrics': knn_metrics
    }
    
    # MLP
    if verbose:
        print(f"\n   🔧 Treinando MLP sklearn...")
    mlp_start = time.time()
    mlp = create_mlp_model()
    mlp.fit(X_train_sk, y_train_sk)
    mlp_time = time.time() - mlp_start
    
    mlp_test_prob = mlp.predict_proba(X_test_sk)[:, 1]
    mlp_test_pred = mlp.predict(X_test_sk)
    mlp_metrics = calculate_metrics(y_test_sk, mlp_test_pred, mlp_test_prob)
    
    if verbose:
        print(f"      ✅ Concluído em {mlp_time:.2f}s | ROC-AUC: {mlp_metrics['roc_auc']:.4f}")
    
    results['models']['MLP'] = {
        'train_time': mlp_time,
        'test_metrics': mlp_metrics
    }
    
    # ==========================================================================
    # 3. DT-KINASE OFICIAL (CNN + Cross-Attention)
    # ==========================================================================
    
    if verbose:
        print(f"\n   🧠 Treinando DT-Kinase OFICIAL (CNN + Cross-Attention)...")
        print(f"      Device: {DEVICE}")
        print(f"      Epochs: {epochs}, Batch: {batch_size}, LR: {learning_rate}")
    
    # Criar datasets
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
    
    if verbose:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"      Parâmetros: {n_params:,}")
    
    # Optimizer e scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    dtkinase_start = time.time()
    best_val_auc = 0.0
    best_model_state = None
    patience_counter = 0
    patience = 10
    
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
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"      Epoch {epoch+1}/{epochs}: Train AUC={train_metrics['roc_auc']:.4f}, Val AUC={val_metrics['roc_auc']:.4f}")
        
        if patience_counter >= patience:
            if verbose:
                print(f"      Early stopping at epoch {epoch+1}")
            break
    
    # Carregar melhor modelo
    if best_model_state:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_model_state.items()})
    
    # Avaliar no teste
    test_metrics = evaluate(model, test_loader, criterion, DEVICE)
    dtkinase_time = time.time() - dtkinase_start
    
    if verbose:
        print(f"      ✅ Concluído em {dtkinase_time:.2f}s | ROC-AUC: {test_metrics['roc_auc']:.4f}")
    
    results['models']['DT-Kinase'] = {
        'train_time': dtkinase_time,
        'best_val_auc': best_val_auc,
        'test_metrics': test_metrics,
        'epochs_trained': epoch + 1,
        'n_parameters': sum(p.numel() for p in model.parameters())
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='DockTKinase - Baseline com DT-Kinase OFICIAL',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--data-path', type=str,
                        default='tests/datasets/kinase_non_human_compounds.tsv',
                        help='Caminho para o arquivo TSV de dados')
    parser.add_argument('--embeddings-dir', type=str,
                        default='results/protein_model_benchmark_non_human_v2',
                        help='Diretório com embeddings pré-calculados')
    parser.add_argument('--output', type=str,
                        default='results/baseline_dtkinase_official',
                        help='Diretório de saída')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Número de épocas para DT-Kinase')
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
    
    # Carregar dados
    print('=' * 70)
    print('🎯 BASELINE COM DT-KINASE OFICIAL (CNN + Cross-Attention)')
    print('=' * 70)
    print()
    print(f'📋 Configuração:')
    print(f'   Device: {DEVICE}')
    print(f'   Seed: {RANDOM_SEED}')
    print(f'   Epochs: {args.epochs}')
    print(f'   Batch Size: {args.batch_size}')
    print(f'   Learning Rate: {args.learning_rate}')
    print(f'   Hidden Dim: {args.hidden_dim}')
    print()
    
    # Carregar DataFrame
    print(f'📂 Carregando dados de: {data_path}')
    df = pd.read_csv(data_path, sep='\t')
    print(f'   Total amostras: {len(df):,}')
    print(f'   Proteínas únicas: {df["seq_id"].nunique()}')
    print(f'   Ligantes únicos: {df["chembl_id"].nunique()}')
    print()
    
    # Modelos a processar
    models_to_run = args.esm_models if args.esm_models else ESM_MODELS
    
    all_results = {
        'pipeline': 'baseline_dtkinase_official',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'random_seed': RANDOM_SEED,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'hidden_dim': args.hidden_dim,
            'device': str(DEVICE),
        },
        'results': {}
    }
    
    start_time = time.time()
    
    for esm_model in models_to_run:
        print('=' * 70)
        print(f'🧬 Modelo: {esm_model} ({ESM_DIMS.get(esm_model, "?")}D)')
        print('=' * 70)
        
        try:
            result = run_baseline_for_model(
                esm_model=esm_model,
                data_df=df,
                embeddings_dir=embeddings_dir,
                output_dir=output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                hidden_dim=args.hidden_dim,
                verbose=True
            )
            all_results['results'][esm_model] = result
            
        except Exception as e:
            print(f'   ❌ Erro: {e}')
            import traceback
            traceback.print_exc()
            all_results['results'][esm_model] = {'error': str(e)}
    
    total_time = time.time() - start_time
    all_results['total_time_seconds'] = total_time
    
    # Salvar resultados
    results_file = output_dir / 'baseline_dtkinase_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Resumo
    print()
    print('=' * 70)
    print('✅ BASELINE CONCLUÍDO')
    print('=' * 70)
    print()
    print('📊 RESUMO DOS RESULTADOS (Test ROC-AUC):')
    print('-' * 70)
    print(f'{"Modelo ESM-2":<25} {"KNN":<12} {"MLP":<12} {"DT-Kinase":<12}')
    print('-' * 70)
    
    for esm_model in models_to_run:
        result = all_results['results'].get(esm_model, {})
        if 'error' in result:
            print(f'{esm_model:<25} {"ERROR":<12} {"ERROR":<12} {"ERROR":<12}')
        else:
            models = result.get('models', {})
            knn_auc = models.get('KNN', {}).get('test_metrics', {}).get('roc_auc', 0)
            mlp_auc = models.get('MLP', {}).get('test_metrics', {}).get('roc_auc', 0)
            dtk_auc = models.get('DT-Kinase', {}).get('test_metrics', {}).get('roc_auc', 0)
            print(f'{esm_model:<25} {knn_auc:<12.4f} {mlp_auc:<12.4f} {dtk_auc:<12.4f}')
    
    print('-' * 70)
    print()
    print(f'⏱️  Tempo total: {total_time:.2f}s ({total_time/60:.2f} min)')
    print(f'💾 Resultados salvos em: {results_file}')
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
