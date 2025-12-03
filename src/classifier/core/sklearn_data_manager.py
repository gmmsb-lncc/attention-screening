#!/usr/bin/env python3
"""
Data Manager para Sklearn - Classificação
=========================================

Gerencia carregamento e divisão de dados para modelos sklearn.
"""

import warnings
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from sklearn.model_selection import train_test_split


class SklearnDataManager:
    """
    Gerenciador de dados para modelos sklearn de classificação.
    Equivalente ao DataManager de regressão.
    """
    
    def __init__(self, embeddings_path: str, labels_path: str):
        """
        Inicializar gerenciador de dados.
        
        Args:
            embeddings_path: Caminho para embeddings (.npy)
            labels_path: Caminho para labels (.npy)
        """
        self.embeddings_path = Path(embeddings_path)
        self.labels_path = Path(labels_path)
        
        # Carregar dados
        self.embeddings = np.load(self.embeddings_path, allow_pickle=True)
        self.labels = np.load(self.labels_path, allow_pickle=True)
        
        # Garantir shape correto
        if self.embeddings.ndim == 1:
            # Se for (N,), pode ser array de arrays - tentar converter
            self.embeddings = np.vstack(self.embeddings)
        
        if self.labels.ndim > 1:
            self.labels = self.labels.ravel()
        
        # Validar tamanhos
        if len(self.embeddings) != len(self.labels):
            raise ValueError(
                f"Número de embeddings ({len(self.embeddings)}) não "
                f"corresponde ao número de labels ({len(self.labels)})"
            )
        
        # CRITICAL FIX: Filter out invalid labels (-1)
        # Binary labels may contain -1 for invalid entries
        self._filter_invalid_labels()
    
    def _filter_invalid_labels(self) -> None:
        """
        Filter out invalid labels (-1) from embeddings and labels.
        
        Binary labels from BinaryLabels class use -1 for invalid entries
        (e.g., missing or non-numeric standard_value). We must remove
        these before training to ensure proper binary classification.
        """
        original_size = len(self.labels)
        
        # Find valid indices (labels that are 0 or 1, not -1 or other values)
        valid_mask = np.isin(self.labels, [0, 1])
        
        if not valid_mask.all():
            n_invalid = (~valid_mask).sum()
            unique_invalid = np.unique(self.labels[~valid_mask])
            
            warnings.warn(
                f"Removed {n_invalid} samples with invalid labels {unique_invalid.tolist()}. "
                f"Only binary labels (0, 1) are kept for classification."
            )
            
            # Filter embeddings and labels
            self.embeddings = self.embeddings[valid_mask]
            self.labels = self.labels[valid_mask]
            
            print(f"   ⚠️  Filtered: {original_size} → {len(self.labels)} samples "
                  f"(removed {n_invalid} invalid labels)")
        
        # Ensure labels are integers
        self.labels = self.labels.astype(int)
    
    def split_data(
        self,
        test_size: float = 0.2,
        val_size: float = 0.1,
        stratify: bool = True,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Dividir dados em treino/validação/teste.
        
        Args:
            test_size: Proporção do conjunto de teste
            val_size: Proporção do conjunto de validação
            stratify: Usar stratified split (manter proporção de classes)
            random_state: Seed para reprodutibilidade
            
        Returns:
            Tuple (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Primeiro split: separar teste
        stratify_labels = self.labels if stratify else None
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.embeddings,
            self.labels,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_labels
        )
        
        # Segundo split: separar validação do treino
        # Ajustar val_size para ser relativo ao tamanho restante
        val_size_adjusted = val_size / (1 - test_size)
        
        stratify_temp = y_temp if stratify else None
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=stratify_temp
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Retorna estatísticas dos dados.
        
        Returns:
            Dict com estatísticas
        """
        n_samples, embedding_dim = self.embeddings.shape
        
        # Distribuição de classes
        unique, counts = np.unique(self.labels, return_counts=True)
        class_distribution = dict(zip(unique.tolist(), counts.tolist()))
        
        return {
            'n_samples': n_samples,
            'embedding_dim': embedding_dim,
            'n_classes': len(unique),
            'class_distribution': class_distribution,
            'positive_ratio': np.mean(self.labels == 1),
            'negative_ratio': np.mean(self.labels == 0)
        }
    
    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retorna embeddings e labels.
        
        Returns:
            Tuple (embeddings, labels)
        """
        return self.embeddings, self.labels


if __name__ == '__main__':
    print('SklearnDataManager - Gerenciador de Dados para Sklearn')
    print('=' * 70)
    print('\nGerencia carregamento e divisão de dados para classificação.')
