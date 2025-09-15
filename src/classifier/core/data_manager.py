"""
Data Manager Simplificado - Apenas o essencial para MLP
"""
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)

# Definir exportações explicitamente
__all__ = ['Dataset', 'SimpleDataManager', 'DataManager', 'ScalableDataset', 'SimpleDataset']


class Dataset(torch.utils.data.Dataset):
    """Dataset PyTorch simples para dados de classificação."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: Features (N, D)
            y: Labels (N,)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.n_samples, self.n_features = X.shape
        
        logger.info(f"Dataset criado: {self.n_samples} amostras, {self.n_features} features")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SimpleDataManager:
    """Data manager minimalista - foco apenas na funcionalidade essencial."""
    
    def __init__(self):
        logger.info("DataManager simplificado inicializado")
    
    def create_dataset(self, X: np.ndarray, y: np.ndarray) -> Dataset:
        """Cria dataset a partir de arrays numpy."""
        return Dataset(X, y)
    
    def create_dataloader(self, dataset: Dataset, 
                         batch_size: int = 32, 
                         shuffle: bool = True) -> DataLoader:
        """Cria DataLoader para treinamento."""
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def train_test_split(self, X: np.ndarray, y: np.ndarray, 
                        test_size: float = 0.2, 
                        random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Divide dados em treino e teste."""
        return train_test_split(X, y, test_size=test_size, 
                               random_state=random_state, stratify=y)


# Para compatibilidade com código existente
DataManager = SimpleDataManager
ScalableDataset = Dataset
SimpleDataset = Dataset  # Alias para compatibilidade
