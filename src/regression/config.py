#!/usr/bin/env python3
"""
Configuração - Regressão DockTKinase
=====================================

Configurações centralizadas para o pipeline de regressão.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import json


@dataclass
class RegressionConfig:
    """
    Configuração completa para pipeline de regressão.
    
    Todas as configurações em um só lugar para facilitar:
    - Reprodutibilidade
    - Testes com diferentes parâmetros
    - Serialização/desserialização
    """
    
    # Dados
    dataset_name: str = 'human'
    measure_priority: List[str] = field(default_factory=lambda: ['Ki', 'Kd', 'IC50'])
    min_samples_per_class: int = 5
    test_size: float = 0.2
    val_size: float = 0.1
    
    # Modelos
    models_to_use: Optional[List[str]] = None  # None = todos
    random_state: int = 42
    n_jobs: int = -1
    
    # Hiperparâmetros Random Forest
    rf_n_estimators: int = 100
    rf_max_depth: int = 20
    rf_min_samples_split: int = 5
    
    # Hiperparâmetros Gradient Boosting
    gb_n_estimators: int = 100
    gb_max_depth: int = 5
    gb_learning_rate: float = 0.1
    
    # Hiperparâmetros Ridge/Lasso
    ridge_alpha: float = 1.0
    lasso_alpha: float = 0.1
    
    # Hiperparâmetros XGBoost
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1
    
    # Treinamento
    use_early_stopping: bool = True
    early_stopping_rounds: int = 10
    cv_folds: int = 5
    
    # Avaliação
    metrics_to_compute: List[str] = field(default_factory=lambda: [
        'mse', 'rmse', 'mae', 'r2', 'mape', 'explained_variance'
    ])
    primary_metric: str = 'rmse'  # Métrica para seleção do melhor modelo
    
    # Visualização
    generate_plots: bool = True
    plot_formats: List[str] = field(default_factory=lambda: ['png', 'pdf'])
    plot_dpi: int = 300
    plot_style: str = 'seaborn-v0_8-darkgrid'
    
    # Saída
    output_dir: Path = field(default_factory=lambda: Path('results/regression'))
    save_predictions: bool = True
    save_models: bool = True
    save_best_only: bool = False
    
    # Reprodutibilidade
    save_config: bool = True
    save_environment: bool = True
    
    # Logging
    verbose: bool = True
    log_level: str = 'INFO'
    save_logs: bool = True
    
    # Performance
    use_gpu: bool = False
    batch_size: int = 32
    num_workers: int = 4
    
    # Cache
    use_embeddings_cache: bool = True
    cache_dir: Path = field(default_factory=lambda: Path('models_cache/embeddings'))
    
    def __post_init__(self):
        """Validação e conversões pós-inicialização."""
        # Converter paths para Path objects
        if not isinstance(self.output_dir, Path):
            self.output_dir = Path(self.output_dir)
        
        if not isinstance(self.cache_dir, Path):
            self.cache_dir = Path(self.cache_dir)
        
        # Validações básicas
        if not 0 < self.test_size < 1:
            raise ValueError(f'test_size deve estar entre 0 e 1, recebido {self.test_size}')
        
        if not 0 < self.val_size < 1:
            raise ValueError(f'val_size deve estar entre 0 e 1, recebido {self.val_size}')
        
        if self.test_size + self.val_size >= 1:
            raise ValueError(
                f'test_size + val_size deve ser < 1, '
                f'recebido {self.test_size} + {self.val_size} = {self.test_size + self.val_size}'
            )
        
        if self.random_state < 0:
            raise ValueError(f'random_state deve ser >= 0, recebido {self.random_state}')
        
        if self.primary_metric not in self.metrics_to_compute:
            raise ValueError(
                f'primary_metric "{self.primary_metric}" deve estar em metrics_to_compute'
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converte config para dicionário.
        
        Returns:
            Dict com todas as configurações
        """
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
            elif isinstance(value, (list, dict, str, int, float, bool, type(None))):
                config_dict[key] = value
            else:
                config_dict[key] = str(value)
        return config_dict
    
    def save(self, filepath: Path):
        """
        Salva configuração em arquivo JSON.
        
        Args:
            filepath: Path para salvar
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: Path) -> 'RegressionConfig':
        """
        Carrega configuração de arquivo JSON.
        
        Args:
            filepath: Path do arquivo
            
        Returns:
            RegressionConfig carregada
        """
        filepath = Path(filepath)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        # Converter paths de volta
        if 'output_dir' in config_dict:
            config_dict['output_dir'] = Path(config_dict['output_dir'])
        if 'cache_dir' in config_dict:
            config_dict['cache_dir'] = Path(config_dict['cache_dir'])
        
        return cls(**config_dict)
    
    def update(self, **kwargs):
        """
        Atualiza configuração com novos valores.
        
        Args:
            **kwargs: Parâmetros a atualizar
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f'Parâmetro desconhecido: {key}')
        
        # Re-validar
        self.__post_init__()
    
    def get_model_params(self, model_name: str) -> Dict[str, Any]:
        """
        Retorna parâmetros específicos de um modelo.
        
        Args:
            model_name: Nome do modelo
            
        Returns:
            Dict com parâmetros
        """
        params = {'random_state': self.random_state}
        
        if model_name == 'RandomForest':
            params.update({
                'n_estimators': self.rf_n_estimators,
                'max_depth': self.rf_max_depth,
                'min_samples_split': self.rf_min_samples_split,
                'n_jobs': self.n_jobs
            })
        
        elif model_name == 'GradientBoosting':
            params.update({
                'n_estimators': self.gb_n_estimators,
                'max_depth': self.gb_max_depth,
                'learning_rate': self.gb_learning_rate
            })
        
        elif model_name == 'Ridge':
            params.update({
                'alpha': self.ridge_alpha
            })
        
        elif model_name == 'Lasso':
            params.update({
                'alpha': self.lasso_alpha
            })
        
        elif model_name in ['XGBoost', 'XGB']:
            params.update({
                'n_estimators': self.xgb_n_estimators,
                'max_depth': self.xgb_max_depth,
                'learning_rate': self.xgb_learning_rate,
                'n_jobs': self.n_jobs
            })
        
        return params
    
    def __repr__(self) -> str:
        """Representação legível."""
        lines = ['RegressionConfig(']
        for key, value in self.to_dict().items():
            lines.append(f'  {key}={value!r},')
        lines.append(')')
        return '\n'.join(lines)


# Configurações pré-definidas
def get_fast_config() -> RegressionConfig:
    """Config rápida para testes (menos estimadores, sem plots)."""
    return RegressionConfig(
        rf_n_estimators=50,
        gb_n_estimators=50,
        xgb_n_estimators=50,
        generate_plots=False,
        save_models=False,
        verbose=False
    )


def get_production_config() -> RegressionConfig:
    """Config para produção (mais estimadores, todos os outputs)."""
    return RegressionConfig(
        rf_n_estimators=200,
        gb_n_estimators=200,
        xgb_n_estimators=200,
        rf_max_depth=30,
        generate_plots=True,
        save_models=True,
        save_best_only=False,
        plot_dpi=600
    )


def get_debug_config() -> RegressionConfig:
    """Config para debug (mínimo de dados, verbose)."""
    return RegressionConfig(
        rf_n_estimators=10,
        gb_n_estimators=10,
        xgb_n_estimators=10,
        generate_plots=False,
        save_models=False,
        verbose=True,
        log_level='DEBUG'
    )
