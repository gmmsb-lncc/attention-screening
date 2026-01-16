#!/usr/bin/env python3
"""
DockTKinase - Baseline com DT-Kinase (Deep Learning)
=====================================================

Comparação entre classificadores tradicionais e modelo neural:
- KNN (k=5, cosine, distance-weighted)
- MLP sklearn (256, 128, 64)
- DT-Kinase: Deep Neural Network com PyTorch

O modelo DT-Kinase aqui é uma versão simplificada que usa embeddings
médios concatenados (mesma entrada de KNN/MLP), permitindo comparação justa.

A versão completa com Cross-Attention requer matrizes per-token.

Uso:
    python baseline_with_dtkinase.py
    python baseline_with_dtkinase.py --epochs 50 --batch-size 64

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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

RANDOM_SEED = 420
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

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

TEST_SIZE = 0.1
VAL_SIZE = 0.1

# Detectar dispositivo
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 
                      'mps' if torch.backends.mps.is_available() else 'cpu')


# =============================================================================
# DT-KINASE MODEL (Versão simplificada para embeddings médios)
# =============================================================================

class DTKinaseClassifier(nn.Module):
    """
    DT-Kinase: Deep Neural Network para classificação binária.
    
    Versão simplificada que usa embeddings médios concatenados
    (proteína + ligante) como entrada.
    
    Arquitetura:
        - Input: [batch, protein_dim + ligand_dim]
        - Encoder separado para proteína e ligante
        - Fusion layer com cross-attention simplificada
        - Classification head
    
    Esta é uma versão adaptada do CrossAttentionAffinityModel para
    trabalhar com embeddings médios em vez de matrizes per-token.
    """
    
    def __init__(
        self,
        protein_dim: int,
        ligand_dim: int = 768,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.2,
        use_batch_norm: bool = True
    ):
        super().__init__()
        
        self.protein_dim = protein_dim
        self.ligand_dim = ligand_dim
        self.hidden_dim = hidden_dim
        
        # Protein encoder
        self.protein_encoder = self._build_encoder(
            protein_dim, hidden_dim, num_layers, dropout, use_batch_norm
        )
        
        # Ligand encoder
        self.ligand_encoder = self._build_encoder(
            ligand_dim, hidden_dim, num_layers, dropout, use_batch_norm
        )
        
        # Cross-attention simplificada (bilinear interaction)
        self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, hidden_dim)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),  # concat + interaction
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Inicialização de pesos
        self.apply(self._init_weights)
    
    def _build_encoder(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        use_batch_norm: bool
    ) -> nn.Sequential:
        """Constrói encoder MLP com residual connections."""
        layers = []
        
        # Input projection
        layers.append(nn.Linear(input_dim, hidden_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self, module):
        """Inicialização Xavier/Glorot."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: [batch, protein_dim + ligand_dim] embeddings concatenados
        
        Returns:
            logits: [batch, 1] classificação
        """
        # Separar proteína e ligante
        protein_emb = x[:, :self.protein_dim]
        ligand_emb = x[:, self.protein_dim:]
        
        # Encode
        protein_hidden = self.protein_encoder(protein_emb)
        ligand_hidden = self.ligand_encoder(ligand_emb)
        
        # Interação bilinear (simula cross-attention)
        interaction = self.bilinear(protein_hidden, ligand_hidden)
        
        # Fusion
        fused = torch.cat([protein_hidden, ligand_hidden, interaction], dim=1)
        fused = self.fusion(fused)
        
        # Classificação
        logits = self.classifier(fused)
        
        return logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Retorna probabilidades."""
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            return torch.cat([1 - probs, probs], dim=1)


class DTKinaseWrapper:
    """
    Wrapper para DT-Kinase compatível com interface sklearn.
    
    Permite usar DT-Kinase com a mesma API de KNN/MLP.
    """
    
    def __init__(
        self,
        protein_dim: int,
        ligand_dim: int = 768,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 50,
        batch_size: int = 64,
        patience: int = 10,
        device: str = None,
        verbose: bool = True
    ):
        self.protein_dim = protein_dim
        self.ligand_dim = ligand_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.device = torch.device(device) if device else DEVICE
        self.verbose = verbose
        
        self.model = None
        self.scaler = StandardScaler()
        self.best_val_loss = float('inf')
        self.training_history = []
    
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None):
        """Treina o modelo."""
        # Normalizar dados
        X_scaled = self.scaler.fit_transform(X)
        
        # Criar modelo
        self.model = DTKinaseClassifier(
            protein_dim=self.protein_dim,
            ligand_dim=self.ligand_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        # Preparar dados
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y).unsqueeze(1).to(self.device)
        
        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Validação
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            X_val_tensor = torch.FloatTensor(X_val_scaled).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)
        else:
            # Split interno para validação
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                X_scaled, y, test_size=0.1, random_state=RANDOM_SEED, stratify=y
            )
            X_tensor = torch.FloatTensor(X_train_split).to(self.device)
            y_tensor = torch.FloatTensor(y_train_split).unsqueeze(1).to(self.device)
            X_val_tensor = torch.FloatTensor(X_val_split).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val_split).unsqueeze(1).to(self.device)
            
            train_dataset = TensorDataset(X_tensor, y_tensor)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Optimizer e scheduler
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Loss com class weights
        pos_weight = torch.tensor([(y == 0).sum() / (y == 1).sum()]).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Training loop
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(self.epochs):
            # Train
            self.model.train()
            train_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(X_val_tensor)
                val_loss = criterion(val_logits, y_val_tensor).item()
                
                # Métricas
                val_probs = torch.sigmoid(val_logits).cpu().numpy()
                val_preds = (val_probs > 0.5).astype(int)
                val_auc = roc_auc_score(y_val_tensor.cpu().numpy(), val_probs)
            
            scheduler.step(val_loss)
            
            self.training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_auc': val_auc
            })
            
            if self.verbose and (epoch + 1) % 10 == 0:
                print(f"        Epoch {epoch+1}/{self.epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_auc={val_auc:.4f}")
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    if self.verbose:
                        print(f"        Early stopping at epoch {epoch+1}")
                    break
        
        # Restaurar melhor modelo
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predição de classes."""
        probs = self.predict_proba(X)[:, 1]
        return (probs > 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predição de probabilidades."""
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
        
        return np.column_stack([1 - probs, probs])


# =============================================================================
# MODELOS SKLEARN (KNN e MLP)
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
            batch_size=64,
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
# MÉTRICAS
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
        return None, None
    
    # Filtrar labels inválidos
    valid_mask = np.isin(labels, [0, 1])
    if (~valid_mask).sum() > 0:
        embeddings = embeddings[valid_mask]
        labels = labels[valid_mask].astype(int)
    
    return embeddings, labels


def random_split(X, y, test_size=0.1, val_size=0.1, random_state=RANDOM_SEED):
    """Divide dados em treino/validação/teste."""
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def run_baseline_with_dtkinase(
    embeddings_dir: Path,
    output_dir: Path,
    epochs: int = 50,
    batch_size: int = 64,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Executa baseline com KNN, MLP e DT-Kinase.
    """
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'pipeline': 'baseline_with_dtkinase',
        'timestamp': datetime.now().isoformat(),
        'seed': RANDOM_SEED,
        'split': 'random',
        'device': str(DEVICE),
        'config': {
            'epochs': epochs,
            'batch_size': batch_size,
        },
        'models': {}
    }
    
    start_time = time.time()
    
    print('=' * 70)
    print('🎯 BASELINE + DT-KINASE')
    print('=' * 70)
    print()
    print(f'📋 Configuração:')
    print(f'   Device: {DEVICE}')
    print(f'   Seed: {RANDOM_SEED}')
    print(f'   DT-Kinase epochs: {epochs}')
    print(f'   Batch size: {batch_size}')
    print()
    
    for model_name in ESM_MODELS:
        protein_dim = ESM_DIMS.get(model_name, 320)
        ligand_dim = 768  # SMI-TED
        total_dim = protein_dim + ligand_dim
        
        print('=' * 70)
        print(f'🧬 Modelo: {model_name} ({protein_dim}+{ligand_dim}={total_dim} dims)')
        print('=' * 70)
        
        # Carregar embeddings
        embeddings, labels = load_embeddings(embeddings_dir, model_name)
        
        if embeddings is None:
            print(f'   ⚠️  Não encontrado, pulando...')
            continue
        
        print(f'   ✅ Carregado: {len(embeddings):,} amostras')
        
        # Split
        X_train, X_val, X_test, y_train, y_val, y_test = random_split(
            embeddings, labels, TEST_SIZE, VAL_SIZE, RANDOM_SEED
        )
        
        print(f'   📊 Split: {len(X_train):,} train / {len(X_val):,} val / {len(X_test):,} test')
        
        results['models'][model_name] = {
            'protein_dim': protein_dim,
            'n_samples': len(embeddings),
            'n_train': len(X_train),
            'n_val': len(X_val),
            'n_test': len(X_test),
        }
        
        # === KNN ===
        print(f'\n   🔧 Treinando KNN...')
        knn_start = time.time()
        knn = create_knn_model(RANDOM_SEED)
        knn.fit(X_train, y_train)
        knn_time = time.time() - knn_start
        
        knn_test_prob = knn.predict_proba(X_test)[:, 1]
        knn_test_pred = knn.predict(X_test)
        knn_test_metrics = calculate_all_metrics(y_test, knn_test_pred, knn_test_prob)
        
        print(f'      ✅ Concluído em {knn_time:.2f}s')
        print(f'      📊 Test ROC-AUC: {knn_test_metrics["roc_auc"]:.4f}')
        
        results['models'][model_name]['KNN'] = {
            'train_time': knn_time,
            'test_metrics': knn_test_metrics,
        }
        
        # === MLP sklearn ===
        print(f'\n   🔧 Treinando MLP (sklearn)...')
        mlp_start = time.time()
        mlp = create_mlp_model(RANDOM_SEED)
        mlp.fit(X_train, y_train)
        mlp_time = time.time() - mlp_start
        
        mlp_test_prob = mlp.predict_proba(X_test)[:, 1]
        mlp_test_pred = mlp.predict(X_test)
        mlp_test_metrics = calculate_all_metrics(y_test, mlp_test_pred, mlp_test_prob)
        
        print(f'      ✅ Concluído em {mlp_time:.2f}s')
        print(f'      📊 Test ROC-AUC: {mlp_test_metrics["roc_auc"]:.4f}')
        
        results['models'][model_name]['MLP'] = {
            'train_time': mlp_time,
            'test_metrics': mlp_test_metrics,
        }
        
        # === DT-Kinase ===
        print(f'\n   🔧 Treinando DT-Kinase (PyTorch)...')
        dtk_start = time.time()
        
        dtkinase = DTKinaseWrapper(
            protein_dim=protein_dim,
            ligand_dim=ligand_dim,
            hidden_dim=256,
            num_layers=3,
            dropout=0.2,
            lr=1e-3,
            weight_decay=1e-4,
            epochs=epochs,
            batch_size=batch_size,
            patience=15,
            device=str(DEVICE),
            verbose=verbose
        )
        
        dtkinase.fit(X_train, y_train, X_val, y_val)
        dtk_time = time.time() - dtk_start
        
        dtk_test_prob = dtkinase.predict_proba(X_test)[:, 1]
        dtk_test_pred = dtkinase.predict(X_test)
        dtk_test_metrics = calculate_all_metrics(y_test, dtk_test_pred, dtk_test_prob)
        
        print(f'      ✅ Concluído em {dtk_time:.2f}s')
        print(f'      📊 Test ROC-AUC: {dtk_test_metrics["roc_auc"]:.4f}')
        
        results['models'][model_name]['DT-Kinase'] = {
            'train_time': dtk_time,
            'test_metrics': dtk_test_metrics,
            'training_history': dtkinase.training_history,
        }
    
    total_time = time.time() - start_time
    results['total_time_seconds'] = total_time
    
    # Salvar resultados
    results_path = output_dir / 'baseline_dtkinase_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Resumo
    print()
    print('=' * 70)
    print('✅ BASELINE + DT-KINASE CONCLUÍDO')
    print('=' * 70)
    print()
    print('📊 RESUMO DOS RESULTADOS (Test ROC-AUC):')
    print('-' * 70)
    print(f'{"Modelo ESM-2":<25} {"KNN":<12} {"MLP":<12} {"DT-Kinase":<12}')
    print('-' * 70)
    
    for model_name in ESM_MODELS:
        if model_name in results['models']:
            model_data = results['models'][model_name]
            knn_auc = model_data.get('KNN', {}).get('test_metrics', {}).get('roc_auc', 0)
            mlp_auc = model_data.get('MLP', {}).get('test_metrics', {}).get('roc_auc', 0)
            dtk_auc = model_data.get('DT-Kinase', {}).get('test_metrics', {}).get('roc_auc', 0)
            print(f'{model_name:<25} {knn_auc:<12.4f} {mlp_auc:<12.4f} {dtk_auc:<12.4f}')
    
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
        description='DockTKinase - Baseline com KNN, MLP e DT-Kinase',
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        default='results/baseline_dtkinase',
        help='Diretório de saída'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Número de epochs para DT-Kinase'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size para DT-Kinase'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Modo silencioso'
    )
    
    args = parser.parse_args()
    
    embeddings_dir = Path(args.embeddings_dir)
    output_dir = Path(args.output)
    
    if not embeddings_dir.exists():
        print(f'❌ Diretório de embeddings não encontrado: {embeddings_dir}')
        return 1
    
    run_baseline_with_dtkinase(
        embeddings_dir=embeddings_dir,
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=not args.quiet
    )
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
