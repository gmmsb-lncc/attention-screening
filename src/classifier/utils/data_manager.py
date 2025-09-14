"""
Sistema de gerenciamento de dados com carregamento inteligente em lotes.
Previne OutOfMemory errors em datasets grandes.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union, Dict, Any
import logging
from pathlib import Path
import psutil
import gc

logger = logging.getLogger(__name__)


class ScalableDataset(Dataset):
    """
    Dataset que carrega dados sob demanda para evitar OOM.
    
    Características:
    - Lazy loading: dados ficam em CPU/disco
    - Transfere para GPU apenas no momento do uso
    - Suporte a datasets grandes que não cabem na GPU
    """
    
    def __init__(self, 
                 X: Union[np.ndarray, torch.Tensor, str, Path],
                 y: Union[np.ndarray, torch.Tensor, str, Path],
                 device: torch.device = torch.device('cpu'),
                 dtype: torch.dtype = torch.float32):
        """
        Args:
            X: Features (array, tensor, ou caminho para arquivo)
            y: Labels (array, tensor, ou caminho para arquivo)  
            device: Device para onde transferir dados quando solicitados
            dtype: Tipo de dados para features
        """
        self.device = device
        self.dtype = dtype
        
        # Carregar dados na CPU (memory-efficient)
        self.X_cpu = self._load_data(X, "features")
        self.y_cpu = self._load_data(y, "labels")
        
        # Validações
        if len(self.X_cpu) != len(self.y_cpu):
            raise ValueError(f"Incompatibilidade: X tem {len(self.X_cpu)} samples, y tem {len(self.y_cpu)}")
        
        self.n_samples = len(self.X_cpu)
        self.n_features = self.X_cpu.shape[1] if len(self.X_cpu.shape) > 1 else 1
        
        logger.info(f"📊 ScalableDataset criado: {self.n_samples} samples, {self.n_features} features")
        logger.info(f"🖥️  Dados mantidos em: CPU (transfere para {device} sob demanda)")
        
    def _load_data(self, data: Union[np.ndarray, torch.Tensor, str, Path], data_type: str):
        """Carrega dados de forma inteligente."""
        if isinstance(data, (str, Path)):
            # Carregar de arquivo
            path = Path(data)
            if not path.exists():
                raise FileNotFoundError(f"Arquivo {data_type} não encontrado: {path}")
            
            if path.suffix == '.npy':
                return np.load(path)
            elif path.suffix == '.csv':
                return pd.read_csv(path).values
            else:
                raise ValueError(f"Formato não suportado para {data_type}: {path.suffix}")
        
        elif isinstance(data, torch.Tensor):
            # Mover tensor para CPU se estiver na GPU
            return data.detach().cpu().numpy()
        
        elif isinstance(data, np.ndarray):
            # Já está em numpy (CPU)
            return data
        
        else:
            raise TypeError(f"Tipo não suportado para {data_type}: {type(data)}")
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retorna sample transferindo para device apenas no momento do uso.
        Isso evita carregar todo dataset na GPU de uma vez.
        """
        # Obter dados da CPU
        x = self.X_cpu[idx]
        y = self.y_cpu[idx]
        
        # Converter para tensors e transferir para device
        x_tensor = torch.from_numpy(x if isinstance(x, np.ndarray) else np.array(x))
        y_tensor = torch.from_numpy(y if isinstance(y, np.ndarray) else np.array(y))
        
        # Aplicar dtype e device
        x_tensor = x_tensor.to(dtype=self.dtype, device=self.device)
        y_tensor = y_tensor.to(device=self.device)
        
        return x_tensor, y_tensor
    
    def get_full_data(self, max_samples: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retorna dados completos (use com cuidado em datasets grandes).
        
        Args:
            max_samples: Limita número de samples para evitar OOM
        """
        if max_samples and max_samples < self.n_samples:
            logger.warning(f"⚠️  Limitando a {max_samples} samples (total: {self.n_samples})")
            X = self.X_cpu[:max_samples]
            y = self.y_cpu[:max_samples]
        else:
            X = self.X_cpu
            y = self.y_cpu
        
        # Converter para tensors
        X_tensor = torch.from_numpy(X).to(dtype=self.dtype, device=self.device)
        y_tensor = torch.from_numpy(y).to(device=self.device)
        
        return X_tensor, y_tensor
    
    def get_subset(self, indices: Union[list, np.ndarray, torch.Tensor]) -> 'ScalableDataset':
        """Cria subset do dataset."""
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy()
        elif isinstance(indices, list):
            indices = np.array(indices)
        
        X_subset = self.X_cpu[indices]
        y_subset = self.y_cpu[indices]
        
        # Criar novo dataset com subset
        new_dataset = ScalableDataset.__new__(ScalableDataset)
        new_dataset.device = self.device
        new_dataset.dtype = self.dtype
        new_dataset.X_cpu = X_subset
        new_dataset.y_cpu = y_subset
        new_dataset.n_samples = len(X_subset)
        new_dataset.n_features = self.n_features
        
        return new_dataset


class MemoryManager:
    """Gerenciador de memória para monitoramento e otimização."""
    
    @staticmethod
    def get_gpu_memory_info() -> Dict[str, float]:
        """Retorna informações de memória GPU."""
        if not torch.cuda.is_available():
            return {"available": 0, "used": 0, "total": 0}
        
        try:
            gpu_mem = torch.cuda.get_device_properties(0)
            allocated = torch.cuda.memory_allocated(0)
            cached = torch.cuda.memory_reserved(0)
            total = gpu_mem.total_memory
            
            return {
                "total": total / 1024**3,  # GB
                "allocated": allocated / 1024**3,  # GB  
                "cached": cached / 1024**3,  # GB
                "available": (total - cached) / 1024**3  # GB
            }
        except Exception as e:
            logger.warning(f"Erro ao obter info GPU: {e}")
            return {"available": 0, "used": 0, "total": 0}
    
    @staticmethod
    def get_ram_info() -> Dict[str, float]:
        """Retorna informações de memória RAM."""
        mem = psutil.virtual_memory()
        return {
            "total": mem.total / 1024**3,  # GB
            "available": mem.available / 1024**3,  # GB
            "used": mem.used / 1024**3,  # GB
            "percent": mem.percent
        }
    
    @staticmethod
    def estimate_dataset_memory(n_samples: int, n_features: int, 
                              dtype: torch.dtype = torch.float32) -> Dict[str, float]:
        """Estima uso de memória do dataset."""
        bytes_per_element = 4 if dtype == torch.float32 else 8 if dtype == torch.float64 else 2
        
        # Memória para features
        features_bytes = n_samples * n_features * bytes_per_element
        # Memória para labels (assumindo float32)  
        labels_bytes = n_samples * 4
        
        total_bytes = features_bytes + labels_bytes
        total_gb = total_bytes / 1024**3
        
        return {
            "features_gb": features_bytes / 1024**3,
            "labels_gb": labels_bytes / 1024**3, 
            "total_gb": total_gb,
            "total_mb": total_bytes / 1024**2
        }
    
    @staticmethod
    def recommend_batch_size(n_samples: int, n_features: int, 
                           available_gpu_gb: float = None) -> int:
        """Recomenda batch size baseado na memória disponível."""
        if available_gpu_gb is None:
            gpu_info = MemoryManager.get_gpu_memory_info()
            available_gpu_gb = gpu_info["available"]
        
        if available_gpu_gb <= 0:
            return min(32, n_samples)  # Fallback para CPU
        
        # Estimar quantos samples cabem em 50% da memória disponível  
        safety_factor = 0.5
        usable_memory_gb = available_gpu_gb * safety_factor
        
        # Bytes por sample (features + gradients + ativações)
        bytes_per_sample = n_features * 4 * 3  # features, gradients, ativações
        samples_per_gb = (1024**3) / bytes_per_sample
        
        recommended_batch = int(usable_memory_gb * samples_per_gb)
        
        # Limites práticos
        recommended_batch = max(8, min(recommended_batch, 512, n_samples))
        
        return recommended_batch
    
    @staticmethod
    def clear_cache():
        """Limpa cache de memória."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.debug("🧹 Cache de memória limpo")


class DataManager:
    """
    Gerenciador principal de dados com carregamento inteligente.
    
    Substitui o carregamento direto na GPU por sistema escalável.
    """
    
    def __init__(self, device: Optional[torch.device] = None, 
                 auto_batch_size: bool = True):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.auto_batch_size = auto_batch_size
        self.memory_manager = MemoryManager()
        
        logger.info(f"🚀 DataManager inicializado - Device: {self.device}")
        
        # Log de memória inicial
        if self.device.type == "cuda":
            gpu_info = self.memory_manager.get_gpu_memory_info()
            logger.info(f"💾 GPU Memory: {gpu_info['available']:.1f}GB disponível")
        
        ram_info = self.memory_manager.get_ram_info() 
        logger.info(f"💽 RAM: {ram_info['available']:.1f}GB disponível")
    
    def create_dataset(self, 
                      X: Union[np.ndarray, torch.Tensor, str, Path],
                      y: Union[np.ndarray, torch.Tensor, str, Path],
                      **kwargs) -> ScalableDataset:
        """Cria dataset escalável."""
        return ScalableDataset(X, y, device=self.device, **kwargs)
    
    def create_dataloader(self, 
                         dataset: ScalableDataset,
                         batch_size: Optional[int] = None,
                         shuffle: bool = True,
                         num_workers: int = 0,
                         **kwargs) -> DataLoader:
        """
        Cria DataLoader com batch size inteligente.
        
        Args:
            dataset: ScalableDataset
            batch_size: Tamanho do batch (None = automático)
            shuffle: Se deve embaralhar dados
            num_workers: Número de workers para carregamento
        """
        if batch_size is None and self.auto_batch_size:
            batch_size = self.memory_manager.recommend_batch_size(
                dataset.n_samples, dataset.n_features
            )
            logger.info(f"📦 Batch size automático: {batch_size}")
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size or 32,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=self.device.type == "cuda",
            **kwargs
        )
        
        logger.info(f"🔄 DataLoader criado: {len(dataloader)} batches")
        return dataloader
    
    def load_from_arrays(self, 
                        X: np.ndarray, 
                        y: np.ndarray,
                        batch_size: Optional[int] = None,
                        shuffle: bool = True) -> Tuple[ScalableDataset, DataLoader]:
        """
        Conveniência para carregar de arrays NumPy.
        
        Returns:
            (dataset, dataloader)
        """
        # Análise de memória
        memory_est = self.memory_manager.estimate_dataset_memory(
            X.shape[0], X.shape[1]
        )
        logger.info(f"📊 Estimativa de memória: {memory_est['total_gb']:.2f}GB")
        
        # Criar dataset e dataloader
        dataset = self.create_dataset(X, y)
        dataloader = self.create_dataloader(dataset, batch_size, shuffle)
        
        return dataset, dataloader
    
    def load_from_dataframe(self,
                           df: pd.DataFrame,
                           target_column: str,
                           feature_columns: Optional[list] = None,
                           **kwargs) -> Tuple[ScalableDataset, DataLoader]:
        """Carrega dados de DataFrame pandas."""
        if feature_columns is None:
            feature_columns = [col for col in df.columns if col != target_column]
        
        X = df[feature_columns].values.astype(np.float32)
        y = df[target_column].values.astype(np.float32)
        
        return self.load_from_arrays(X, y, **kwargs)
