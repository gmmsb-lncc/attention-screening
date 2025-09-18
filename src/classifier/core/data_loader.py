"""
Gerenciamento de dados para o classificador MLP.

Implementa exatamente a mesma lógica de carregamento e preparação
de dados do classifier.py original.
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from typing import Optional, Tuple, List, Union
import logging

logger = logging.getLogger(__name__)


class DataManager:
    """
    Gerenciador de dados que implementa exatamente a mesma lógica
    do classifier.py original para carregamento e preparação de dados.
    """
    
    def __init__(self, embeddings_path: str, labels_path: str, device: torch.device):
        """
        Args:
            embeddings_path: Caminho para arquivo .npy com embeddings
            labels_path: Caminho para arquivo .npy com labels
            device: Device para colocar os tensores
        """
        self.embeddings_path = embeddings_path
        self.labels_path = labels_path
        self.device = device
        
        # Cache dos dados carregados
        self._embeddings: Optional[np.ndarray] = None
        self._labels: Optional[np.ndarray] = None
        self._dataset: Optional[TensorDataset] = None
    
    def get_embedding_dim(self) -> int:
        """
        Obtém automaticamente a dimensão do embedding carregando a primeira amostra.
        
        IMPLEMENTAÇÃO IDÊNTICA ao classifier.py original.
        
        Returns:
            Dimensão dos embeddings
        """
        embeddings = np.load(self.embeddings_path, allow_pickle=True)
        return embeddings.shape[1]
    
    def load_data(self, 
                  train_idx: Optional[np.ndarray] = None,
                  val_idx: Optional[np.ndarray] = None, 
                  test_idx: Optional[np.ndarray] = None) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
        """
        Carrega embeddings e rótulos binários, realizando a divisão estratificada em 80% treino, 
        10% validação e 10% teste.
        
        IMPLEMENTAÇÃO IDÊNTICA ao classifier.py original.
        
        Se os índices já forem fornecidos, eles serão usados; caso contrário, a divisão padrão é aplicada.
        
        Args:
            train_idx: Índices para conjunto de treino
            val_idx: Índices para conjunto de validação  
            test_idx: Índices para conjunto de teste
            
        Returns:
            Tupla com (train_loader, val_loader, test_loader)
            test_loader pode ser None se test_idx for None
        """
        # Carregar dados se não estiverem em cache
        if self._embeddings is None or self._labels is None:
            embeddings = np.load(self.embeddings_path, allow_pickle=True)
            labels = np.load(self.labels_path, allow_pickle=True)
            labels = labels.flatten()
            
            # Cache dos dados
            self._embeddings = embeddings
            self._labels = labels
            
            # Criar dataset
            X = torch.tensor(embeddings, dtype=torch.float32)
            y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
            self._dataset = TensorDataset(X, y)
        
        # Usar dados do cache
        dataset = self._dataset
        labels = self._labels
        
        # Se índices não fornecidos, fazer divisão padrão EXATAMENTE como no original
        if train_idx is None or val_idx is None or test_idx is None:
            indices = np.arange(len(dataset))
            train_idx, temp_idx = train_test_split(
                indices, test_size=0.2, stratify=labels, random_state=42
            )
            val_idx, test_idx = train_test_split(
                temp_idx, test_size=0.5, stratify=labels[temp_idx], random_state=42
            )
        
        # Criar data loaders EXATAMENTE como no original
        train_loader = DataLoader(
            Subset(dataset, train_idx), 
            batch_size=64,  # Valor padrão, será sobrescrito
            shuffle=True
        )
        
        val_loader = DataLoader(
            Subset(dataset, val_idx),
            batch_size=64,  # Valor padrão, será sobrescrito  
            shuffle=False
        )
        
        test_loader = None
        if test_idx is not None:
            test_loader = DataLoader(
                Subset(dataset, test_idx),
                batch_size=64,  # Valor padrão, será sobrescrito
                shuffle=False
            )
        
        return train_loader, val_loader, test_loader
    
    def create_data_loaders(self,
                           train_idx: Optional[np.ndarray] = None,
                           val_idx: Optional[np.ndarray] = None,
                           test_idx: Optional[np.ndarray] = None,
                           batch_size: int = 64) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
        """
        Cria data loaders com batch_size customizado.
        
        Args:
            train_idx: Índices para treino
            val_idx: Índices para validação
            test_idx: Índices para teste  
            batch_size: Tamanho do batch
            
        Returns:
            Tupla com (train_loader, val_loader, test_loader)
        """
        train_loader, val_loader, test_loader = self.load_data(train_idx, val_idx, test_idx)
        
        # Recriar com batch_size correto
        dataset = self._dataset
        
        train_loader = DataLoader(
            Subset(dataset, train_idx if train_idx is not None else train_loader.dataset.indices),
            batch_size=batch_size,
            shuffle=True
        )
        
        val_loader = DataLoader(
            Subset(dataset, val_idx if val_idx is not None else val_loader.dataset.indices),
            batch_size=batch_size,
            shuffle=False
        )
        
        if test_loader is not None:
            test_loader = DataLoader(
                Subset(dataset, test_idx if test_idx is not None else test_loader.dataset.indices),
                batch_size=batch_size,
                shuffle=False
            )
        
        return train_loader, val_loader, test_loader
    
    def get_data_for_cv(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retorna embeddings e labels para cross-validation.
        
        Returns:
            Tupla com (embeddings, labels) como numpy arrays
        """
        if self._embeddings is None or self._labels is None:
            # Forçar carregamento
            self.load_data()
        
        return self._embeddings, self._labels
    
    def get_dataset_info(self) -> dict:
        """
        Retorna informações sobre o dataset.
        
        Returns:
            Dicionário com informações do dataset
        """
        if self._embeddings is None or self._labels is None:
            embeddings = np.load(self.embeddings_path, allow_pickle=True)
            labels = np.load(self.labels_path, allow_pickle=True).flatten()
        else:
            embeddings = self._embeddings
            labels = self._labels
        
        unique, counts = np.unique(labels, return_counts=True)
        
        return {
            'n_samples': len(embeddings),
            'embedding_dim': embeddings.shape[1],
            'n_classes': len(unique),
            'class_distribution': dict(zip(unique.astype(int), counts.astype(int))),
            'class_balance_ratio': max(counts) / min(counts) if len(counts) > 1 else 1.0
        }


# Função de conveniência para compatibilidade
def create_data_manager(embeddings_path: str, 
                       labels_path: str,
                       device: Optional[torch.device] = None) -> DataManager:
    """
    Factory function para criar gerenciador de dados.
    
    Args:
        embeddings_path: Caminho para embeddings
        labels_path: Caminho para labels
        device: Device para tensores
        
    Returns:
        DataManager configurado
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return DataManager(embeddings_path, labels_path, device)


if __name__ == "__main__":
    # Teste básico do gerenciador de dados
    print("🧪 Testando DataManager...")
    
    # Criar dados sintéticos para teste
    import tempfile
    import os
    
    # Dados sintéticos
    n_samples = 200
    n_features = 128
    
    embeddings = np.random.randn(n_samples, n_features).astype(np.float32)
    labels = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
    
    # Salvar em arquivos temporários
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f_emb:
        np.save(f_emb.name, embeddings)
        emb_path = f_emb.name
    
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f_lab:
        np.save(f_lab.name, labels)
        lab_path = f_lab.name
    
    try:
        # Testar gerenciador
        device = torch.device("cpu")
        data_manager = DataManager(emb_path, lab_path, device)
        
        # Testar info do dataset
        info = data_manager.get_dataset_info()
        print(f"✅ Dataset info: {info}")
        
        # Testar dimensão
        dim = data_manager.get_embedding_dim()
        print(f"✅ Embedding dim: {dim}")
        
        # Testar carregamento de dados
        train_loader, val_loader, test_loader = data_manager.load_data()
        print(f"✅ Data loaders criados:")
        print(f"   Train: {len(train_loader.dataset)} samples")
        print(f"   Val: {len(val_loader.dataset)} samples") 
        print(f"   Test: {len(test_loader.dataset) if test_loader else 0} samples")
        
        print("🎯 DataManager funcionando perfeitamente!")
        
    finally:
        # Limpar arquivos temporários
        if os.path.exists(emb_path):
            os.unlink(emb_path)
        if os.path.exists(lab_path):
            os.unlink(lab_path)
