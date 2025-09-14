"""
Data Manager Simplificado - REPLICANDO EXATAMENTE O ORIGINAL classifier.py
"""
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from typing import Optional, Tuple
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


class SimpleDataManager:
    """Data manager replicando exatamente o comportamento do classifier.py original."""
    
    def __init__(self):
        logger.info("DataManager simplificado inicializado (modo compatibilidade classifier.py)")
    
    def create_dataset_from_arrays(self, X, y) -> TensorDataset:
        """Cria TensorDataset exatamente como no classifier.py original."""
        # REPLICAR EXATAMENTE O COMPORTAMENTO ORIGINAL:
        # Aceitar tanto numpy arrays quanto tensors
        if isinstance(X, torch.Tensor):
            X_tensor = X.to(torch.float32)
        else:
            X_tensor = torch.from_numpy(X.copy()).to(torch.float32)
            
        if isinstance(y, torch.Tensor):
            y_tensor = y.to(torch.float32)
            if len(y_tensor.shape) == 1:
                y_tensor = y_tensor.unsqueeze(1)
        else:
            y_tensor = torch.as_tensor(y, dtype=torch.float32).unsqueeze(1)  # ← CHAVE: (N,1)
        
        logger.info(f"TensorDataset criado: X={X_tensor.shape}, y={y_tensor.shape}")
        return TensorDataset(X_tensor, y_tensor)
    
    def create_dataloader(self, dataset: TensorDataset, 
                         batch_size: int = 32, 
                         shuffle: bool = True) -> DataLoader:
        """Cria DataLoader exatamente como no original."""
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            pin_memory=False,  # Simplificado
            num_workers=0,     # Sem workers para evitar problemas
        )
    
    def train_test_split(self, X: np.ndarray, y: np.ndarray, 
                        test_size: float = 0.2, 
                        random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Divide dados em treino e teste."""
        return train_test_split(X, y, test_size=test_size, 
                               random_state=random_state, stratify=y)
    
    def load_from_arrays(self, X: np.ndarray, y: np.ndarray, 
                        batch_size: int = 32, 
                        shuffle: bool = True) -> Tuple[TensorDataset, DataLoader]:
        """Carrega dados de arrays e cria dataset + dataloader compatível."""
        dataset = self.create_dataset_from_arrays(X, y)
        dataloader = self.create_dataloader(dataset, batch_size, shuffle)
        return dataset, dataloader


# Manter aliases para compatibilidade
DataManager = SimpleDataManager
ScalableDataset = TensorDataset  # Usar TensorDataset nativo
