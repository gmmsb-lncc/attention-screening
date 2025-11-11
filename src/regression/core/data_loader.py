#!/usr/bin/env python3
"""
Gerenciador de Dados para Regressão - DockTKinase
==================================================

Implementa carregamento e preparação de dados para regressão,
seguindo o mesmo padrão do classificador modular.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from sklearn.model_selection import train_test_split


class DataManager:
    """
    Gerenciador de dados para regressão.
    
    Implementa cache em memória e divisão estratificada de dados,
    mantendo compatibilidade total com o pipeline original.
    """
    
    def __init__(self, embeddings_path: str = None, targets_path: str = None):
        """
        Inicializar gerenciador de dados.
        
        Args:
            embeddings_path: Caminho para embeddings (.npy ou .npz)
            targets_path: Caminho para targets (.npy)
        """
        self.embeddings_path = embeddings_path
        self.targets_path = targets_path
        
        # Cache em memória
        self._embeddings = None
        self._targets = None
        self._data_loaded = False
    
    def load_embeddings(self, embeddings_path: Optional[str] = None) -> np.ndarray:
        """
        Carregar embeddings de arquivo.
        
        Args:
            embeddings_path: Caminho alternativo para embeddings
            
        Returns:
            Array de embeddings (n_samples, embedding_dim)
        """
        if embeddings_path is not None:
            self.embeddings_path = embeddings_path
        
        if self.embeddings_path is None:
            raise ValueError("embeddings_path não fornecido")
        
        path = Path(self.embeddings_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")
        
        # Carregar baseado na extensão
        if path.suffix == '.npz':
            data = np.load(path, allow_pickle=True)
            # Tentar diferentes chaves comuns
            if 'embeddings' in data:
                embeddings = data['embeddings']
            elif 'features' in data:
                embeddings = data['features']
            elif 'X' in data:
                embeddings = data['X']
            else:
                # Pegar primeiro array
                embeddings = data[data.files[0]]
        else:
            embeddings = np.load(path, allow_pickle=True)
        
        self._embeddings = embeddings
        return embeddings
    
    def load_targets(self, targets_path: Optional[str] = None, target_column: int = 3) -> np.ndarray:
        """
        Carregar targets de arquivo.
        
        Args:
            targets_path: Caminho alternativo para targets
            target_column: Índice da coluna a usar se targets for 2D (default: 3 para standard_value)
            
        Returns:
            Array de targets (n_samples,)
        """
        if targets_path is not None:
            self.targets_path = targets_path
        
        if self.targets_path is None:
            raise ValueError("targets_path não fornecido")
        
        path = Path(self.targets_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")
        
        targets = np.load(path, allow_pickle=True)
        
        # Se array 2D (ex: interaction_labels com múltiplas colunas)
        # Extrair coluna específica ao invés de flatten
        if len(targets.shape) > 1:
            if targets.shape[1] > target_column:
                # Extrair coluna target_column e converter para float
                targets = targets[:, target_column]
            else:
                # Se não tiver coluna suficiente, usar primeira coluna
                targets = targets[:, 0]
        
        # Converter para float e garantir array 1D
        try:
            targets = np.array([float(x) for x in targets])
        except (ValueError, TypeError) as e:
            raise ValueError(f"Erro ao converter targets para float: {e}")
        
        self._targets = targets
        return targets
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Carregar embeddings e targets juntos.
        
        Returns:
            Tuple (embeddings, targets)
        """
        if self._embeddings is None:
            self.load_embeddings()
        
        if self._targets is None:
            self.load_targets()
        
        # Validar compatibilidade
        if len(self._embeddings) != len(self._targets):
            raise ValueError(
                f"Incompatibilidade: {len(self._embeddings)} embeddings vs "
                f"{len(self._targets)} targets"
            )
        
        self._data_loaded = True
        return self._embeddings, self._targets
    
    def split_data(
        self,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        stratify_bins: int = 5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Dividir dados em treino, validação e teste.
        
        Para regressão, usa stratification baseada em bins quantílicos
        para manter distribuição similar nos splits.
        
        Args:
            test_size: Proporção do conjunto de teste (0.2 = 20%)
            val_size: Proporção do conjunto de validação (0.1 = 10%)
            random_state: Seed para reprodutibilidade
            stratify_bins: Número de bins para stratification (default: 5)
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        if not self._data_loaded:
            self.load_data()
        
        X = self._embeddings
        y = self._targets
        
        # Criar bins para stratification
        # Isso garante distribuição similar de valores em todos os splits
        y_bins = pd.qcut(y, q=stratify_bins, labels=False, duplicates='drop')
        
        # Primeiro split: treino+val vs teste
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y_bins
        )
        
        # Segundo split: treino vs validação
        # Ajustar val_size relativamente ao tamanho temporário
        val_size_adjusted = val_size / (1 - test_size)
        
        # Criar bins para o split restante
        y_temp_bins = pd.qcut(y_temp, q=stratify_bins, labels=False, duplicates='drop')
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=y_temp_bins
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_embedding_dim(self) -> int:
        """
        Obter dimensão dos embeddings.
        
        Returns:
            Dimensão dos embeddings
        """
        if self._embeddings is None:
            self.load_embeddings()
        
        return self._embeddings.shape[1]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obter estatísticas dos dados carregados.
        
        Returns:
            Dict com estatísticas
        """
        if not self._data_loaded:
            self.load_data()
        
        return {
            'n_samples': len(self._embeddings),
            'embedding_dim': self._embeddings.shape[1],
            'target_mean': float(np.mean(self._targets)),
            'target_std': float(np.std(self._targets)),
            'target_min': float(np.min(self._targets)),
            'target_max': float(np.max(self._targets)),
            'target_median': float(np.median(self._targets)),
            'embeddings_memory_mb': self._embeddings.nbytes / 1024 / 1024,
            'targets_memory_mb': self._targets.nbytes / 1024 / 1024
        }


# Funções de conveniência
def load_regression_data(
    embeddings_path: str,
    targets_path: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
) -> Dict[str, np.ndarray]:
    """
    Função de conveniência para carregar e dividir dados.
    
    Args:
        embeddings_path: Caminho para embeddings
        targets_path: Caminho para targets
        test_size: Proporção do teste
        val_size: Proporção da validação
        random_state: Seed
        
    Returns:
        Dict com 'X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test'
    """
    manager = DataManager(embeddings_path, targets_path)
    X_train, X_val, X_test, y_train, y_val, y_test = manager.split_data(
        test_size=test_size,
        val_size=val_size,
        random_state=random_state
    )
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test
    }


if __name__ == '__main__':
    # Teste básico
    print("DataManager para Regressão - DockTKinase")
    print("=" * 60)
    
    # Exemplo de uso
    example = """
    # Uso básico:
    manager = DataManager('embeddings.npy', 'targets.npy')
    X_train, X_val, X_test, y_train, y_val, y_test = manager.split_data()
    
    # Estatísticas:
    stats = manager.get_stats()
    print(stats)
    """
    print(example)
