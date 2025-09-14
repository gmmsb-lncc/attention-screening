"""
Gerenciamento de dados para o DockTKinase Classifier.
Fornece funcionalidades para carregamento, pré-processamento e validação de dados.
"""

import logging
import gc
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List, Union
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.utils import shuffle
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


@dataclass
class DatasetInfo:
    """Informações sobre um dataset carregado."""
    name: str
    shape: Tuple[int, int]
    n_samples: int
    n_features: int
    n_classes: Optional[int] = None
    has_missing: bool = False
    memory_usage_mb: float = 0.0
    feature_types: Dict[str, str] = field(default_factory=dict)


class ScalableDataset(Dataset):
    """Dataset PyTorch que suporta carregamento lazy e gestão de memória."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, 
                 lazy_loading: bool = False,
                 chunk_size: int = 1000):
        """
        Inicializa dataset escalável.
        
        Args:
            X: Features
            y: Labels
            lazy_loading: Se True, carrega dados sob demanda
            chunk_size: Tamanho do chunk para lazy loading
        """
        self.lazy_loading = lazy_loading
        self.chunk_size = chunk_size
        
        if lazy_loading:
            # Salvar referências para lazy loading
            self._X_ref = X
            self._y_ref = y
            self._cache = {}
        else:
            # Carregar tudo na memória
            self.X = torch.FloatTensor(X)
            self.y = torch.FloatTensor(y) if y.ndim == 1 else torch.LongTensor(y)
        
        self.length = len(X)
        
        logger.info(f"🗂️  Dataset criado: {self.length} amostras")
        logger.info(f"   Lazy loading: {lazy_loading}")
        if lazy_loading:
            logger.info(f"   Chunk size: {chunk_size}")
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.lazy_loading:
            return self._get_lazy(idx)
        else:
            return self.X[idx], self.y[idx]
    
    def _get_lazy(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Carregamento lazy de uma amostra."""
        chunk_id = idx // self.chunk_size
        
        if chunk_id not in self._cache:
            # Carregar chunk
            start_idx = chunk_id * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, self.length)
            
            X_chunk = self._X_ref[start_idx:end_idx]
            y_chunk = self._y_ref[start_idx:end_idx]
            
            self._cache[chunk_id] = (
                torch.FloatTensor(X_chunk),
                torch.FloatTensor(y_chunk) if y_chunk.ndim == 1 else torch.LongTensor(y_chunk)
            )
            
            # Limitar cache size
            if len(self._cache) > 10:  # Manter máximo 10 chunks
                oldest_chunk = min(self._cache.keys())
                del self._cache[oldest_chunk]
        
        X_chunk, y_chunk = self._cache[chunk_id]
        local_idx = idx % self.chunk_size
        
        return X_chunk[local_idx], y_chunk[local_idx]
    
    def clear_cache(self):
        """Limpa cache de lazy loading."""
        if hasattr(self, '_cache'):
            self._cache.clear()
            gc.collect()


class DataManager:
    """
    Gerenciador principal de dados para o DockTKinase.
    Responsável por carregamento, pré-processamento e validação.
    """
    
    def __init__(self, 
                 lazy_loading: bool = True,
                 memory_efficient: bool = True,
                 max_samples_in_memory: int = 100000):
        """
        Inicializa o gerenciador de dados.
        
        Args:
            lazy_loading: Ativar carregamento lazy
            memory_efficient: Usar técnicas de economia de memória
            max_samples_in_memory: Máximo de amostras na memória
        """
        self.lazy_loading = lazy_loading
        self.memory_efficient = memory_efficient
        self.max_samples_in_memory = max_samples_in_memory
        
        self.datasets: Dict[str, DatasetInfo] = {}
        self.scalers: Dict[str, Any] = {}
        
        logger.info("🗂️  DataManager inicializado")
        logger.info(f"   Lazy loading: {lazy_loading}")
        logger.info(f"   Memory efficient: {memory_efficient}")
        logger.info(f"   Max samples in memory: {max_samples_in_memory:,}")
    
    def load_csv_data(self, 
                      file_path: Union[str, Path],
                      target_column: str,
                      feature_columns: Optional[List[str]] = None,
                      name: str = "dataset") -> DatasetInfo:
        """
        Carrega dados de arquivo CSV.
        
        Args:
            file_path: Caminho para o arquivo CSV
            target_column: Nome da coluna target
            feature_columns: Colunas de features (None = todas exceto target)
            name: Nome identificador do dataset
            
        Returns:
            Informações do dataset carregado
        """
        logger.info(f"📂 Carregando dados de: {file_path}")
        
        # Carregar dados
        df = pd.read_csv(file_path)
        
        # Separar features e target
        if feature_columns is None:
            feature_columns = [col for col in df.columns if col != target_column]
        
        X = df[feature_columns].values
        y = df[target_column].values
        
        # Criar info do dataset
        info = DatasetInfo(
            name=name,
            shape=(len(df), len(feature_columns)),
            n_samples=len(df),
            n_features=len(feature_columns),
            n_classes=len(np.unique(y)) if y.dtype.kind in 'biufc' else None,
            has_missing=df.isnull().any().any(),
            memory_usage_mb=df.memory_usage(deep=True).sum() / 1024 / 1024,
            feature_types={col: str(df[col].dtype) for col in feature_columns}
        )
        
        self.datasets[name] = info
        
        # Armazenar dados (pode ser lazy)
        setattr(self, f'_{name}_X', X)
        setattr(self, f'_{name}_y', y)
        
        logger.info(f"✅ Dataset '{name}' carregado:")
        logger.info(f"   Forma: {info.shape}")
        logger.info(f"   Classes: {info.n_classes}")
        logger.info(f"   Valores faltantes: {info.has_missing}")
        logger.info(f"   Uso de memória: {info.memory_usage_mb:.1f}MB")
        
        return info
    
    def load_numpy_data(self,
                        X: np.ndarray,
                        y: np.ndarray,
                        name: str = "dataset") -> DatasetInfo:
        """
        Carrega dados de arrays NumPy.
        
        Args:
            X: Array de features
            y: Array de targets
            name: Nome identificador do dataset
            
        Returns:
            Informações do dataset carregado
        """
        logger.info(f"🔢 Carregando dados NumPy: {name}")
        
        # Validar dimensões
        if len(X) != len(y):
            raise ValueError(f"X e y devem ter o mesmo número de amostras: {len(X)} vs {len(y)}")
        
        # Criar info do dataset
        info = DatasetInfo(
            name=name,
            shape=X.shape,
            n_samples=len(X),
            n_features=X.shape[1] if X.ndim > 1 else 1,
            n_classes=len(np.unique(y)) if y.dtype.kind in 'biufc' else None,
            has_missing=np.isnan(X).any() or np.isnan(y).any(),
            memory_usage_mb=(X.nbytes + y.nbytes) / 1024 / 1024
        )
        
        self.datasets[name] = info
        
        # Armazenar dados
        setattr(self, f'_{name}_X', X)
        setattr(self, f'_{name}_y', y)
        
        logger.info(f"✅ Dataset '{name}' carregado:")
        logger.info(f"   Forma: {info.shape}")
        logger.info(f"   Classes: {info.n_classes}")
        logger.info(f"   Uso de memória: {info.memory_usage_mb:.1f}MB")
        
        return info
    
    def preprocess_data(self,
                        dataset_name: str,
                        scale_method: str = "standard",
                        handle_missing: str = "drop",
                        outlier_method: str = "none") -> None:
        """
        Pré-processa dados de um dataset.
        
        Args:
            dataset_name: Nome do dataset
            scale_method: Método de escalonamento ('standard', 'minmax', 'robust', 'none')
            handle_missing: Como lidar com valores faltantes ('drop', 'mean', 'median')
            outlier_method: Método para outliers ('none', 'iqr', 'zscore')
        """
        logger.info(f"🔧 Pré-processando dataset '{dataset_name}'")
        
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' não encontrado")
        
        X = getattr(self, f'_{dataset_name}_X')
        y = getattr(self, f'_{dataset_name}_y')
        
        # 1. Lidar com valores faltantes
        if handle_missing != "none" and np.isnan(X).any():
            logger.info(f"   Tratando valores faltantes: {handle_missing}")
            X = self._handle_missing_values(X, method=handle_missing)
        
        # 2. Remover outliers
        if outlier_method != "none":
            logger.info(f"   Removendo outliers: {outlier_method}")
            X, y = self._remove_outliers(X, y, method=outlier_method)
        
        # 3. Escalonamento
        if scale_method != "none":
            logger.info(f"   Escalonando features: {scale_method}")
            X, scaler = self._scale_features(X, method=scale_method)
            self.scalers[dataset_name] = scaler
        
        # Atualizar dados
        setattr(self, f'_{dataset_name}_X', X)
        setattr(self, f'_{dataset_name}_y', y)
        
        # Atualizar info
        info = self.datasets[dataset_name]
        info.shape = X.shape
        info.n_samples = len(X)
        info.has_missing = np.isnan(X).any()
        
        logger.info("✅ Pré-processamento concluído")
    
    def split_data(self,
                   dataset_name: str,
                   train_ratio: float = 0.7,
                   val_ratio: float = 0.15,
                   test_ratio: float = 0.15,
                   random_seed: int = 42,
                   stratify: bool = True) -> Tuple[ScalableDataset, ScalableDataset, ScalableDataset]:
        """
        Divide dados em conjuntos de treino, validação e teste.
        
        Args:
            dataset_name: Nome do dataset
            train_ratio: Proporção para treino
            val_ratio: Proporção para validação  
            test_ratio: Proporção para teste
            random_seed: Seed para reprodutibilidade
            stratify: Estratificar divisão por classes
            
        Returns:
            Tupla com datasets de treino, validação e teste
        """
        logger.info(f"✂️  Dividindo dataset '{dataset_name}'")
        logger.info(f"   Proporções: {train_ratio:.1%} treino, {val_ratio:.1%} val, {test_ratio:.1%} teste")
        
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Proporções devem somar 1.0")
        
        X = getattr(self, f'_{dataset_name}_X')
        y = getattr(self, f'_{dataset_name}_y')
        
        # Estratificação
        stratify_y = y if stratify and len(np.unique(y)) > 1 else None
        
        # Primeira divisão: treino + val vs teste
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_ratio,
            random_state=random_seed,
            stratify=stratify_y
        )
        
        # Segunda divisão: treino vs val
        val_size_adjusted = val_ratio / (train_ratio + val_ratio)
        stratify_temp = y_temp if stratify and len(np.unique(y_temp)) > 1 else None
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=random_seed,
            stratify=stratify_temp
        )
        
        # Criar datasets escaláveis
        train_dataset = ScalableDataset(
            X_train, y_train,
            lazy_loading=self.lazy_loading and len(X_train) > self.max_samples_in_memory
        )
        
        val_dataset = ScalableDataset(
            X_val, y_val,
            lazy_loading=self.lazy_loading and len(X_val) > self.max_samples_in_memory
        )
        
        test_dataset = ScalableDataset(
            X_test, y_test,
            lazy_loading=self.lazy_loading and len(X_test) > self.max_samples_in_memory
        )
        
        logger.info(f"✅ Divisão concluída:")
        logger.info(f"   Treino: {len(train_dataset)} amostras")
        logger.info(f"   Validação: {len(val_dataset)} amostras")
        logger.info(f"   Teste: {len(test_dataset)} amostras")
        
        return train_dataset, val_dataset, test_dataset
    
    def create_dataloader(self,
                          dataset: ScalableDataset,
                          batch_size: int = 32,
                          shuffle: bool = True,
                          num_workers: int = 0,
                          pin_memory: bool = True) -> DataLoader:
        """
        Cria DataLoader para um dataset.
        
        Args:
            dataset: Dataset escalável
            batch_size: Tamanho do batch
            shuffle: Embaralhar dados
            num_workers: Número de workers para carregamento
            pin_memory: Usar pin memory para GPU
            
        Returns:
            DataLoader configurado
        """
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )
        
        logger.info(f"🔄 DataLoader criado:")
        logger.info(f"   Batch size: {batch_size}")
        logger.info(f"   Batches: {len(dataloader)}")
        logger.info(f"   Workers: {num_workers}")
        
        return dataloader
    
    def get_dataset_info(self, dataset_name: str) -> DatasetInfo:
        """Retorna informações sobre um dataset."""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' não encontrado")
        return self.datasets[dataset_name]
    
    def list_datasets(self) -> List[str]:
        """Lista todos os datasets carregados."""
        return list(self.datasets.keys())
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Retorna uso de memória por dataset."""
        return {name: info.memory_usage_mb for name, info in self.datasets.items()}
    
    def clear_dataset(self, dataset_name: str) -> None:
        """Remove um dataset da memória."""
        if dataset_name in self.datasets:
            # Remover referências
            if hasattr(self, f'_{dataset_name}_X'):
                delattr(self, f'_{dataset_name}_X')
            if hasattr(self, f'_{dataset_name}_y'):
                delattr(self, f'_{dataset_name}_y')
            
            # Remover da lista
            del self.datasets[dataset_name]
            
            # Remover scaler se existir
            if dataset_name in self.scalers:
                del self.scalers[dataset_name]
            
            logger.info(f"🗑️  Dataset '{dataset_name}' removido da memória")
            gc.collect()
    
    def clear_all(self) -> None:
        """Remove todos os datasets da memória."""
        dataset_names = list(self.datasets.keys())
        for name in dataset_names:
            self.clear_dataset(name)
        
        logger.info("🗑️  Todos os datasets removidos da memória")
    
    def _handle_missing_values(self, X: np.ndarray, method: str) -> np.ndarray:
        """Trata valores faltantes."""
        if method == "drop":
            # Remover linhas com NaN
            mask = ~np.isnan(X).any(axis=1)
            return X[mask]
        elif method == "mean":
            # Substituir por média
            return np.where(np.isnan(X), np.nanmean(X, axis=0), X)
        elif method == "median":
            # Substituir por mediana
            return np.where(np.isnan(X), np.nanmedian(X, axis=0), X)
        else:
            return X
    
    def _remove_outliers(self, X: np.ndarray, y: np.ndarray, method: str) -> Tuple[np.ndarray, np.ndarray]:
        """Remove outliers dos dados."""
        if method == "iqr":
            # Método IQR
            Q1 = np.percentile(X, 25, axis=0)
            Q3 = np.percentile(X, 75, axis=0)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            mask = np.all((X >= lower_bound) & (X <= upper_bound), axis=1)
            
        elif method == "zscore":
            # Método Z-score
            z_scores = np.abs((X - np.mean(X, axis=0)) / np.std(X, axis=0))
            mask = np.all(z_scores < 3, axis=1)
        
        else:
            mask = np.ones(len(X), dtype=bool)
        
        return X[mask], y[mask]
    
    def _scale_features(self, X: np.ndarray, method: str) -> Tuple[np.ndarray, Any]:
        """Escalona features."""
        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        elif method == "robust":
            scaler = RobustScaler()
        else:
            raise ValueError(f"Método de escalonamento inválido: {method}")
        
        X_scaled = scaler.fit_transform(X)
        return X_scaled, scaler


def create_sample_data(n_samples: int = 1000, 
                       n_features: int = 20,
                       n_classes: int = 2,
                       random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cria dados sintéticos para testes.
    
    Args:
        n_samples: Número de amostras
        n_features: Número de features
        n_classes: Número de classes
        random_seed: Seed para reprodutibilidade
        
    Returns:
        Tupla com features e targets
    """
    np.random.seed(random_seed)
    
    # Gerar features aleatórias
    X = np.random.randn(n_samples, n_features)
    
    # Gerar targets
    if n_classes == 2:
        # Classificação binária
        weights = np.random.randn(n_features)
        logits = X @ weights
        y = (logits > 0).astype(int)
    else:
        # Classificação multi-classe
        y = np.random.randint(0, n_classes, n_samples)
    
    return X, y


# Exemplo de uso
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    
    # Criar gerenciador
    data_mgr = DataManager()
    
    # Criar dados sintéticos
    X, y = create_sample_data(1000, 50, 2)
    
    # Carregar dados
    info = data_mgr.load_numpy_data(X, y, "sample")
    
    # Pré-processar
    data_mgr.preprocess_data("sample", scale_method="standard")
    
    # Dividir dados
    train_ds, val_ds, test_ds = data_mgr.split_data("sample")
    
    # Criar dataloaders
    train_loader = data_mgr.create_dataloader(train_ds, batch_size=32)
    
    print(f"✅ Exemplo concluído: {len(train_loader)} batches de treino")
