"""
Sistema de otimização de hiperparâmetros usando Optuna.

CORREÇÕES IMPLEMENTADAS:
- Objective function retorna métrica única e consistente
- Integração correta com cross-validator (sem data leakage)  
- Configuração flexível de espaços de busca
- Logging e checkpointing de estudos
"""

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable, Any, Union, Tuple
import logging
from dataclasses import dataclass, field
from pathlib import Path
import json
import pickle
import numpy as np
import sys
import os

# Imports relativos
from ..models.base_model import BaseClassifier  
from .cross_validator import CrossValidator, CrossValidationConfig
from .trainer import TrainingConfig
from ..config.mlp_config import MLPConfig
from ..utils.metrics import MetricsCalculator

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuração para otimização de hiperparâmetros."""
    
    # Parâmetros do estudo
    study_name: str = "mlp_optimization"
    direction: str = "maximize"  # "maximize" ou "minimize"
    n_trials: int = 100
    timeout: Optional[float] = None  # Timeout em segundos
    
    # Sampler e pruner
    sampler_type: str = "TPE"  # "TPE", "Random", "CmaEs"
    pruner_type: str = "Median"  # "Median", "Successive", "Hyperband"
    
    # Early stopping para trials
    enable_pruning: bool = True
    pruning_warmup_steps: int = 5
    
    # Métrica de otimização
    optimization_metric: str = "roc_auc"
    
    # Paralelização
    n_jobs: int = 1
    
    # Persistência
    storage_url: Optional[str] = None  # e.g., "sqlite:///optuna.db"
    load_if_exists: bool = True
    
    def __post_init__(self):
        """Validação dos parâmetros."""
        if self.direction not in ["maximize", "minimize"]:
            raise ValueError("direction deve ser 'maximize' ou 'minimize'")
        
        if self.n_trials <= 0:
            raise ValueError("n_trials deve ser positivo")
        
        if self.sampler_type not in ["TPE", "Random", "CmaEs"]:
            raise ValueError("sampler_type não suportado")
            
        if self.pruner_type not in ["Median", "Successive", "Hyperband"]:
            raise ValueError("pruner_type não suportado")


@dataclass 
class HyperparameterSpace:
    """Definição do espaço de busca para hiperparâmetros."""
    
    # Arquitetura
    hidden_layers: Dict[str, Any] = field(default_factory=lambda: {
        "type": "categorical",
        "choices": [[64], [128], [64, 32], [128, 64], [256, 128], [128, 64, 32]]
    })
    
    dropout_rate: Dict[str, Any] = field(default_factory=lambda: {
        "type": "float", 
        "low": 0.0, 
        "high": 0.7
    })
    
    activation: Dict[str, Any] = field(default_factory=lambda: {
        "type": "categorical",
        "choices": ["ReLU", "GELU", "LeakyReLU", "ELU"]
    })
    
    use_batch_norm: Dict[str, Any] = field(default_factory=lambda: {
        "type": "categorical",
        "choices": [True, False]
    })
    
    # Otimização
    learning_rate: Dict[str, Any] = field(default_factory=lambda: {
        "type": "float",
        "low": 1e-5,
        "high": 1e-2,
        "log": True
    })
    
    weight_decay: Dict[str, Any] = field(default_factory=lambda: {
        "type": "float",
        "low": 1e-6,
        "high": 1e-2,
        "log": True
    })
    
    batch_size: Dict[str, Any] = field(default_factory=lambda: {
        "type": "categorical",
        "choices": [16, 32, 64, 128]
    })
    
    # Treinamento
    max_epochs: Dict[str, Any] = field(default_factory=lambda: {
        "type": "int",
        "low": 50,
        "high": 300
    })
    
    patience: Dict[str, Any] = field(default_factory=lambda: {
        "type": "int", 
        "low": 5,
        "high": 25
    })


class HyperparameterOptimizer:
    """Otimizador de hiperparâmetros usando Optuna."""
    
    def __init__(
        self,
        optimization_config: OptimizationConfig,
        cv_config: CrossValidationConfig,
        hyperparameter_space: Optional[HyperparameterSpace] = None,
        device: Optional[torch.device] = None
    ):
        self.opt_config = optimization_config
        self.cv_config = cv_config  
        self.hp_space = hyperparameter_space or HyperparameterSpace()
        self.device = device or torch.device("cpu")
        
        # Componentes Optuna
        self.study: Optional[optuna.Study] = None
        self.sampler = self._create_sampler()
        self.pruner = self._create_pruner()
        
        # Dados e factories para objective
        self.X: Optional[torch.Tensor] = None
        self.y: Optional[torch.Tensor] = None
        self.base_model_config: Optional[MLPConfig] = None
        
        logger.info(f"HyperparameterOptimizer inicializado - {self.opt_config.n_trials} trials")
    
    def _create_sampler(self) -> optuna.samplers.BaseSampler:
        """Cria sampler baseado na configuração."""
        if self.opt_config.sampler_type == "TPE":
            return TPESampler(seed=42)
        elif self.opt_config.sampler_type == "Random":
            return optuna.samplers.RandomSampler(seed=42)
        elif self.opt_config.sampler_type == "CmaEs":
            return optuna.samplers.CmaEsSampler(seed=42)
        else:
            raise ValueError(f"Sampler {self.opt_config.sampler_type} não suportado")
    
    def _create_pruner(self) -> Optional[optuna.pruners.BasePruner]:
        """Cria pruner baseado na configuração."""
        if not self.opt_config.enable_pruning:
            return None
            
        if self.opt_config.pruner_type == "Median":
            return MedianPruner(
                n_startup_trials=self.opt_config.pruning_warmup_steps,
                n_warmup_steps=5
            )
        elif self.opt_config.pruner_type == "Successive":
            return optuna.pruners.SuccessiveHalvingPruner()
        elif self.opt_config.pruner_type == "Hyperband":
            return optuna.pruners.HyperbandPruner()
        else:
            raise ValueError(f"Pruner {self.opt_config.pruner_type} não suportado")
    
    def _suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sugere hiperparâmetros baseado no espaço de busca."""
        suggested = {}
        
        for param_name, param_config in self.hp_space.__dict__.items():
            if param_config["type"] == "categorical":
                suggested[param_name] = trial.suggest_categorical(
                    param_name, param_config["choices"]
                )
            elif param_config["type"] == "float":
                if param_config.get("log", False):
                    suggested[param_name] = trial.suggest_float(
                        param_name, 
                        param_config["low"], 
                        param_config["high"],
                        log=True
                    )
                else:
                    suggested[param_name] = trial.suggest_float(
                        param_name,
                        param_config["low"],
                        param_config["high"]
                    )
            elif param_config["type"] == "int":
                suggested[param_name] = trial.suggest_int(
                    param_name,
                    param_config["low"], 
                    param_config["high"]
                )
        
        return suggested
    
    def _create_model_config(self, suggested_params: Dict[str, Any]) -> MLPConfig:
        """Cria configuração do modelo a partir dos hiperparâmetros sugeridos."""
        # Começa com configuração base
        config_dict = self.base_model_config.__dict__.copy()
        
        # Atualiza com parâmetros sugeridos
        config_dict.update({
            "hidden_layers": suggested_params.get("hidden_layers", config_dict["hidden_layers"]),
            "dropout_rate": suggested_params.get("dropout_rate", config_dict["dropout_rate"]),
            "activation": suggested_params.get("activation", config_dict["activation"]),
            "use_batch_norm": suggested_params.get("use_batch_norm", config_dict["use_batch_norm"]),
            "learning_rate": suggested_params.get("learning_rate", config_dict["learning_rate"]),
            "weight_decay": suggested_params.get("weight_decay", config_dict["weight_decay"])
        })
        
        return MLPConfig(**config_dict)
    
    def _create_training_config(self, suggested_params: Dict[str, Any]) -> TrainingConfig:
        """Cria configuração de treinamento a partir dos hiperparâmetros."""
        return TrainingConfig(
            max_epochs=suggested_params.get("max_epochs", 100),
            patience=suggested_params.get("patience", 10),
            monitor_metric=self.opt_config.optimization_metric,
            monitor_mode="max" if self.opt_config.direction == "maximize" else "min",
            amp_enabled=False,  # Simplificar para otimização
            log_interval=50  # Menos verbose durante otimização
        )
    
    def _objective_function(self, trial: optuna.Trial) -> float:
        """
        Função objetivo para otimização.
        
        CORREÇÃO CRÍTICA: Retorna valor único e consistente do CV.
        """
        if self.X is None or self.y is None:
            raise ValueError("Dados não configurados. Use set_data() primeiro.")
        
        # Sugerir hiperparâmetros
        suggested_params = self._suggest_hyperparameters(trial)
        
        # Configurar modelo e treinamento
        model_config = self._create_model_config(suggested_params)
        training_config = self._create_training_config(suggested_params)
        
        # Atualizar batch_size no CV config
        cv_config = CrossValidationConfig(
            n_splits=self.cv_config.n_splits,
            shuffle=self.cv_config.shuffle,
            random_state=self.cv_config.random_state,
            batch_size=suggested_params.get("batch_size", self.cv_config.batch_size)
        )
        
        try:
            # Executar cross-validation  
            from ..models.mlp import MLPEmbeddingClassifier
            
            def model_factory():
                return MLPEmbeddingClassifier(model_config)
            
            def optimizer_factory(model):
                return torch.optim.Adam(
                    model.parameters(),
                    lr=model_config.learning_rate,
                    weight_decay=model_config.weight_decay
                )
            
            cross_validator = CrossValidator(cv_config, training_config, self.device)
            cv_results = cross_validator.cross_validate(
                model_factory, optimizer_factory, self.X, self.y
            )
            
            # Extrair métrica de otimização (média dos folds)
            summary_stats = cv_results["summary_statistics"]
            metric_stats = summary_stats.get(self.opt_config.optimization_metric, {})
            metric_value = metric_stats.get("mean", 0.0)
            
            # Log do progresso
            logger.info(
                f"Trial {trial.number}: {self.opt_config.optimization_metric}="
                f"{metric_value:.4f}, params={suggested_params}"
            )
            
            # CRÍTICO: Retornar valor único para Optuna
            return metric_value
            
        except Exception as e:
            logger.error(f"Erro no trial {trial.number}: {e}")
            # Retornar valor ruim para penalizar configuração problemática
            return -1.0 if self.opt_config.direction == "maximize" else 1.0
    
    def set_data(self, X: torch.Tensor, y: torch.Tensor, base_config: MLPConfig):
        """Configura dados e configuração base para otimização."""
        self.X = X
        self.y = y  
        self.base_model_config = base_config
        
        logger.info(f"Dados configurados: {X.shape[0]} samples, {X.shape[1]} features")
    
    def optimize(self) -> optuna.Study:
        """
        Executa otimização de hiperparâmetros.
        
        Returns:
            Estudo Optuna completo
        """
        if self.X is None or self.y is None:
            raise ValueError("Dados não configurados. Use set_data() primeiro.")
        
        # Criar ou carregar estudo
        self.study = optuna.create_study(
            study_name=self.opt_config.study_name,
            direction=self.opt_config.direction,
            sampler=self.sampler,
            pruner=self.pruner,
            storage=self.opt_config.storage_url,
            load_if_exists=self.opt_config.load_if_exists
        )
        
        logger.info(f"Iniciando otimização: {self.opt_config.n_trials} trials")
        logger.info(f"Métrica objetivo: {self.opt_config.optimization_metric} ({self.opt_config.direction})")
        
        # Executar otimização
        self.study.optimize(
            self._objective_function,
            n_trials=self.opt_config.n_trials,
            timeout=self.opt_config.timeout,
            n_jobs=self.opt_config.n_jobs
        )
        
        # Logging dos resultados
        logger.info(f"Otimização concluída!")
        logger.info(f"Número de trials: {len(self.study.trials)}")
        logger.info(f"Melhor valor: {self.study.best_value:.4f}")
        logger.info(f"Melhores parâmetros: {self.study.best_params}")
        
        return self.study
    
    def get_best_config(self) -> Tuple[MLPConfig, TrainingConfig]:
        """
        Retorna configurações ótimas encontradas.
        
        Returns:
            (melhor_modelo_config, melhor_training_config)
        """
        if not self.study:
            raise ValueError("Otimização não executada. Use optimize() primeiro.")
        
        best_params = self.study.best_params
        best_model_config = self._create_model_config(best_params)
        best_training_config = self._create_training_config(best_params)
        
        return best_model_config, best_training_config
    
    def save_study(self, filepath: Path) -> None:
        """Salva estudo para análise posterior."""
        if not self.study:
            logger.warning("Nenhum estudo para salvar")
            return
        
        study_data = {
            "study": self.study,
            "config": self.opt_config,
            "hyperparameter_space": self.hp_space,
            "best_params": self.study.best_params,
            "best_value": self.study.best_value,
            "n_trials": len(self.study.trials)
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(study_data, f)
        
        logger.info(f"Estudo salvo em: {filepath}")
    
    def load_study(self, filepath: Path) -> None:
        """Carrega estudo previamente salvo."""
        with open(filepath, 'rb') as f:
            study_data = pickle.load(f)
        
        self.study = study_data["study"]
        self.opt_config = study_data["config"]
        self.hp_space = study_data["hyperparameter_space"]
        
        logger.info(f"Estudo carregado: {len(self.study.trials)} trials")


def quick_hyperparameter_search(
    X: torch.Tensor,
    y: torch.Tensor,
    base_config: MLPConfig,
    n_trials: int = 50,
    cv_folds: int = 3,
    device: Optional[torch.device] = None
) -> Tuple[MLPConfig, TrainingConfig, optuna.Study]:
    """
    Função de conveniência para busca rápida de hiperparâmetros.
    
    Returns:
        (melhor_modelo_config, melhor_training_config, estudo_completo)
    """
    # Configurações para busca rápida
    opt_config = OptimizationConfig(
        n_trials=n_trials,
        optimization_metric="roc_auc",
        direction="maximize"
    )
    
    cv_config = CrossValidationConfig(
        n_splits=cv_folds,
        batch_size=64
    )
    
    # Espaço reduzido para busca rápida
    hp_space = HyperparameterSpace()
    hp_space.hidden_layers = {
        "type": "categorical", 
        "choices": [[64], [128], [128, 64]]
    }
    hp_space.max_epochs = {"type": "int", "low": 30, "high": 100}
    
    # Executar otimização
    optimizer = HyperparameterOptimizer(opt_config, cv_config, hp_space, device)
    optimizer.set_data(X, y, base_config)
    study = optimizer.optimize()
    
    # Retornar configurações ótimas
    best_model_config, best_training_config = optimizer.get_best_config()
    
    return best_model_config, best_training_config, study
