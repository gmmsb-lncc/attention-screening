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
    
    Suporta transformação pChEMBL para melhor performance de regressão.
    """
    
    def __init__(self, embeddings_path: str = None, targets_path: str = None, use_pchembl: bool = True):
        """
        Inicializar gerenciador de dados.
        
        Args:
            embeddings_path: Caminho para embeddings (.npy ou .npz)
            targets_path: Caminho para targets (.npy)
            use_pchembl: Se True, converte valores nM para pChEMBL (-log10(M)).
                        Isso é RECOMENDADO para regressão pois valores Ki/IC50
                        variam em várias ordens de magnitude (1 nM a 100,000 nM).
                        pChEMBL = 9 - log10(nM)
        """
        self.embeddings_path = embeddings_path
        self.targets_path = targets_path
        self.use_pchembl = use_pchembl
        
        # Cache em memória
        self._embeddings = None
        self._targets = None
        self._targets_original_nm = None  # Guardar valores originais em nM
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
    
    def load_targets(self, targets_path: Optional[str] = None, target_column: int = 3, pchembl_column: int = 4) -> np.ndarray:
        """
        Carregar targets de arquivo.
        
        Para regressão, preferimos usar pchembl_value (escala logarítmica) quando disponível.
        Se não disponível, usamos standard_value e convertemos para pChEMBL.
        
        Args:
            targets_path: Caminho alternativo para targets
            target_column: Índice da coluna standard_value (default: 3)
            pchembl_column: Índice da coluna pchembl_value (default: 4)
            
        Returns:
            Array de targets (n_samples,) em pChEMBL se use_pchembl=True, senão em nM
        """
        if targets_path is not None:
            self.targets_path = targets_path
        
        if self.targets_path is None:
            raise ValueError("targets_path não fornecido")
        
        path = Path(self.targets_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")
        
        raw_data = np.load(path, allow_pickle=True)
        
        # Se array 2D (ex: interaction_labels com múltiplas colunas)
        if len(raw_data.shape) > 1:
            n_cols = raw_data.shape[1]
            has_pchembl_col = n_cols > pchembl_column
            
            # Extrair standard_value (sempre disponível)
            if n_cols > target_column:
                standard_values = raw_data[:, target_column]
            else:
                standard_values = raw_data[:, 0]
            
            # Converter standard_value para float
            try:
                standard_values = np.array([float(x) if x is not None else np.nan for x in standard_values])
            except (ValueError, TypeError) as e:
                raise ValueError(f"Erro ao converter standard_value para float: {e}")
            
            # Guardar valores originais em nM
            self._targets_original_nm = standard_values.copy()
            
            if self.use_pchembl:
                if has_pchembl_col:
                    # Tentar usar pchembl_value da coluna 4
                    pchembl_values = raw_data[:, pchembl_column]
                    
                    # Converter para float, tratando None/vazio como NaN
                    pchembl_arr = []
                    for val in pchembl_values:
                        try:
                            if val is None or val == '' or val == 'None':
                                pchembl_arr.append(np.nan)
                            else:
                                pchembl_arr.append(float(val))
                        except (ValueError, TypeError):
                            pchembl_arr.append(np.nan)
                    pchembl_values = np.array(pchembl_arr)
                    
                    # Calcular pChEMBL para valores faltantes a partir de standard_value
                    missing_mask = np.isnan(pchembl_values) & (standard_values > 0) & np.isfinite(standard_values)
                    n_missing = np.sum(missing_mask)
                    if n_missing > 0:
                        print(f"   ℹ️  Calculando pChEMBL para {n_missing} amostras sem valor original")
                        pchembl_values[missing_mask] = 9 - np.log10(standard_values[missing_mask])
                    
                    targets = pchembl_values
                else:
                    # Não tem coluna pchembl_value, calcular a partir de standard_value
                    print(f"   ℹ️  Coluna pchembl_value não encontrada, calculando a partir de standard_value")
                    valid_mask = (standard_values > 0) & np.isfinite(standard_values)
                    targets = np.where(valid_mask, 9 - np.log10(standard_values), np.nan)
            else:
                # Usar standard_value diretamente (nM)
                targets = standard_values
        else:
            # Array 1D - assumir que são valores em nM
            try:
                targets = np.array([float(x) for x in raw_data])
            except (ValueError, TypeError) as e:
                raise ValueError(f"Erro ao converter targets para float: {e}")
            
            self._targets_original_nm = targets.copy()
            
            if self.use_pchembl:
                valid_mask = (targets > 0) & np.isfinite(targets)
                targets = np.where(valid_mask, 9 - np.log10(targets), np.nan)
        
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
        
        stats = {
            'n_samples': len(self._embeddings),
            'embedding_dim': self._embeddings.shape[1],
            'target_mean': float(np.nanmean(self._targets)),
            'target_std': float(np.nanstd(self._targets)),
            'target_min': float(np.nanmin(self._targets)),
            'target_max': float(np.nanmax(self._targets)),
            'target_median': float(np.nanmedian(self._targets)),
            'embeddings_memory_mb': self._embeddings.nbytes / 1024 / 1024,
            'targets_memory_mb': self._targets.nbytes / 1024 / 1024,
            'use_pchembl': self.use_pchembl,
            'target_unit': 'pChEMBL' if self.use_pchembl else 'nM'
        }
        
        # Adicionar estatísticas originais em nM se pChEMBL estiver ativo
        if self.use_pchembl and self._targets_original_nm is not None:
            stats['target_original_nm'] = {
                'mean': float(np.nanmean(self._targets_original_nm)),
                'min': float(np.nanmin(self._targets_original_nm)),
                'max': float(np.nanmax(self._targets_original_nm)),
            }
        
        return stats


# Funções de conveniência
def load_regression_data(
    embeddings_path: str,
    targets_path: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    use_pchembl: bool = True
) -> Dict[str, np.ndarray]:
    """
    Função de conveniência para carregar e dividir dados.
    
    Args:
        embeddings_path: Caminho para embeddings
        targets_path: Caminho para targets
        test_size: Proporção do teste
        val_size: Proporção da validação
        random_state: Seed
        use_pchembl: Se True, converte valores nM para pChEMBL
        
    Returns:
        Dict com 'X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test'
    """
    manager = DataManager(embeddings_path, targets_path, use_pchembl=use_pchembl)
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
