"""
Cross-validator corrigido para classificação binária.

CORREÇÕES CRÍTICAS:
- Elimina data leakage usando índices corretos do StratifiedKFold
- Validação adequada de dados para cada fold
- Estratificação mantida mesmo com dados desbalanceados
- Métricas consistentes e reproduzíveis
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import StratifiedKFold
from typing import Dict, List, Optional, Tuple, Callable, Any
import numpy as np
import logging
from dataclasses import dataclass
from pathlib import Path
import sys
import os

# Imports relativos com fallbacks para execução direta
try:
    from ..models.base_model import BaseClassifier
    from .trainer import ModelTrainer, TrainingConfig
    from ..utils.metrics import MetricsCalculator, ClassificationMetrics, MetricsAggregator
    from ..utils.data_validation import DataValidator
    from ..config.mlp_config import MLPConfig
except ImportError:
    # Fallback para execução direta - ajustar sys.path se necessário
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    classifier_dir = os.path.dirname(current_dir)
    if classifier_dir not in sys.path:
        sys.path.insert(0, classifier_dir)
    
    from models.base_model import BaseClassifier
    from core.trainer import ModelTrainer, TrainingConfig
    from utils.metrics import MetricsCalculator, ClassificationMetrics, MetricsAggregator
    from utils.data_validation import DataValidator
    from config.mlp_config import MLPConfig

logger = logging.getLogger(__name__)


@dataclass
class CrossValidationConfig:
    """Configuração para cross-validation."""
    
    # Parâmetros de CV
    n_splits: int = 5
    shuffle: bool = True
    random_state: Optional[int] = 42
    
    # Validação de dados
    validate_splits: bool = True
    min_samples_per_class: int = 2
    
    # Batch size para cada fold
    batch_size: int = 32
    
    # Paralelização (futuro)
    n_jobs: int = 1
    
    def __post_init__(self):
        """Validação dos parâmetros."""
        if self.n_splits < 2:
            raise ValueError("n_splits deve ser >= 2")
        if self.min_samples_per_class < 1:
            raise ValueError("min_samples_per_class deve ser >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size deve ser >= 1")


@dataclass
class FoldResult:
    """Resultado de um fold específico."""
    
    fold_index: int
    train_indices: np.ndarray
    val_indices: np.ndarray
    
    # Métricas
    train_metrics: ClassificationMetrics
    val_metrics: ClassificationMetrics
    
    # Informações de treinamento
    training_history: Any  # TrainingHistory
    final_epoch: int
    training_time: float
    
    # Validação de dados
    fold_data_quality: Optional[Dict[str, Any]] = None


class CrossValidator:
    """
    Cross-validator corrigido que elimina data leakage.
    
    CORREÇÕES IMPLEMENTADAS:
    1. Usa índices corretos do StratifiedKFold (não ignora test_indices)
    2. Validação de integridade de dados para cada fold
    3. Estratificação robusta mesmo com classes desbalanceadas
    4. Logging detalhado para debug
    """
    
    def __init__(
        self,
        cv_config: CrossValidationConfig,
        training_config: TrainingConfig,
        device: Optional[torch.device] = None
    ):
        self.cv_config = cv_config
        self.training_config = training_config
        self.device = device or torch.device("cpu")
        
        # Componentes
        self.data_validator = DataValidator()
        self.metrics_calculator = MetricsCalculator(device=self.device)
        self.metrics_aggregator = MetricsAggregator()
        
        # Resultados
        self.fold_results: List[FoldResult] = []
        self.cv_summary: Optional[Dict[str, Any]] = None
        
        logger.info(f"CrossValidator inicializado - {self.cv_config.n_splits} folds, device: {self.device}")
    
    def _validate_input_data(self, X: torch.Tensor, y: torch.Tensor) -> Dict[str, Any]:
        """Valida dados de entrada antes do CV."""
        logger.info("Validando dados de entrada...")
        
        # Conversão para numpy para validação
        X_np = X.cpu().numpy() if isinstance(X, torch.Tensor) else X
        y_np = y.cpu().numpy() if isinstance(y, torch.Tensor) else y
        
        # Validações básicas
        validation_report = self.data_validator.validate_arrays(X_np, y_np)
        
        # Verificação específica para CV
        unique_labels, label_counts = np.unique(y_np, return_counts=True)
        min_class_samples = np.min(label_counts)
        
        if len(unique_labels) < 2:
            raise ValueError("Dados devem ter pelo menos 2 classes para classificação")
        
        if min_class_samples < self.cv_config.n_splits:
            logger.warning(
                f"Classe minoritária tem {min_class_samples} samples, "
                f"menor que n_splits={self.cv_config.n_splits}. "
                "Reduzindo n_splits automaticamente."
            )
            self.cv_config.n_splits = min_class_samples
        
        validation_summary = {
            "total_samples": len(y_np),
            "n_features": X_np.shape[1] if len(X_np.shape) > 1 else 1,
            "class_distribution": {str(label): count for label, count in zip(unique_labels, label_counts)},
            "min_class_samples": min_class_samples,
            "adjusted_n_splits": self.cv_config.n_splits,
            "validation_report": validation_report
        }
        
        logger.info(f"Validação concluída - {validation_summary['total_samples']} samples, "
                   f"{len(unique_labels)} classes")
        
        return validation_summary
    
    def _create_fold_datasets(
        self, 
        X: torch.Tensor, 
        y: torch.Tensor, 
        train_indices: np.ndarray, 
        val_indices: np.ndarray
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Cria DataLoaders para train e validação usando índices corretos.
        
        CRÍTICO: Esta função garante que não há vazamento de dados
        usando exatamente os índices fornecidos pelo StratifiedKFold.
        """
        # Criar datasets usando índices específicos
        full_dataset = TensorDataset(X, y)
        
        train_dataset = Subset(full_dataset, train_indices.tolist())
        val_dataset = Subset(full_dataset, val_indices.tolist())
        
        # Criar DataLoaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.cv_config.batch_size,
            shuffle=True,  # Shuffle apenas no treino
            drop_last=False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.cv_config.batch_size,
            shuffle=False,  # NUNCA shuffle na validação
            drop_last=False
        )
        
        return train_loader, val_loader
    
    def _validate_fold_split(
        self, 
        y: torch.Tensor, 
        train_indices: np.ndarray, 
        val_indices: np.ndarray,
        fold_idx: int
    ) -> Dict[str, Any]:
        """Valida integridade da divisão de um fold."""
        y_np = y.cpu().numpy() if isinstance(y, torch.Tensor) else y
        
        # Verificar sobreposição
        overlap = set(train_indices) & set(val_indices)
        if overlap:
            raise ValueError(f"Fold {fold_idx}: Sobreposição entre treino e validação: {overlap}")
        
        # Verificar cobertura total
        total_indices = set(range(len(y_np)))
        fold_indices = set(train_indices) | set(val_indices)
        missing = total_indices - fold_indices
        extra = fold_indices - total_indices
        
        if fold_indices != total_indices:
            raise ValueError(f"Fold {fold_idx}: Índices faltando: {missing}, extras: {extra}")
        
        # Distribuição de classes
        train_labels = y_np[train_indices]
        val_labels = y_np[val_indices]
        
        train_unique, train_counts = np.unique(train_labels, return_counts=True)
        val_unique, val_counts = np.unique(val_labels, return_counts=True)
        
        fold_validation = {
            "fold_index": fold_idx,
            "train_size": len(train_indices),
            "val_size": len(val_indices),
            "train_class_dist": {str(label): count for label, count in zip(train_unique, train_counts)},
            "val_class_dist": {str(label): count for label, count in zip(val_unique, val_counts)},
            "overlap_check": len(overlap) == 0,
            "coverage_check": len(missing) == 0 and len(extra) == 0
        }
        
        logger.debug(f"Fold {fold_idx} validado: Train={len(train_indices)}, Val={len(val_indices)}")
        
        return fold_validation
    
    def cross_validate(
        self,
        model_factory: Callable[[], BaseClassifier],
        optimizer_factory: Callable[[BaseClassifier], torch.optim.Optimizer],
        X: torch.Tensor,
        y: torch.Tensor,
        criterion: Optional[nn.Module] = None
    ) -> Dict[str, Any]:
        """
        Executa cross-validation corrigido.
        
        Args:
            model_factory: Função que cria nova instância do modelo
            optimizer_factory: Função que cria otimizador para o modelo
            X: Features (tensor)
            y: Labels (tensor)
            criterion: Função de loss
            
        Returns:
            Dicionário com resultados completos do CV
        """
        logger.info("Iniciando cross-validation...")
        
        # Validação inicial
        input_validation = self._validate_input_data(X, y)
        
        # Conversão para numpy para StratifiedKFold
        y_np = y.cpu().numpy() if isinstance(y, torch.Tensor) else y
        
        # Configurar StratifiedKFold
        skf = StratifiedKFold(
            n_splits=self.cv_config.n_splits,
            shuffle=self.cv_config.shuffle,
            random_state=self.cv_config.random_state
        )
        
        # Reset de resultados
        self.fold_results = []
        self.metrics_aggregator = MetricsAggregator()
        
        # Executar cada fold
        for fold_idx, (train_indices, val_indices) in enumerate(skf.split(X, y_np)):
            logger.info(f"=== FOLD {fold_idx + 1}/{self.cv_config.n_splits} ===")
            
            # Validar divisão do fold
            if self.cv_config.validate_splits:
                fold_validation = self._validate_fold_split(y, train_indices, val_indices, fold_idx)
            else:
                fold_validation = None
            
            # Criar novo modelo para este fold
            model = model_factory()
            model.to(self.device)
            
            # Criar otimizador
            optimizer = optimizer_factory(model)
            
            # Criar datasets
            train_loader, val_loader = self._create_fold_datasets(X, y, train_indices, val_indices)
            
            # Configurar trainer
            trainer = ModelTrainer(
                model=model,
                config=self.training_config,
                device=self.device
            )
            trainer.setup_training(optimizer, criterion)
            
            # Treinar modelo
            import time
            fold_start = time.time()
            
            training_history = trainer.train(train_loader, val_loader)
            
            fold_time = time.time() - fold_start
            
            # Avaliar métricas finais
            train_metrics = self.metrics_calculator.evaluate_model(
                model, train_loader, criterion,
                self.training_config.amp_enabled,
                self.training_config.amp_dtype
            )
            
            val_metrics = self.metrics_calculator.evaluate_model(
                model, val_loader, criterion,
                self.training_config.amp_enabled,
                self.training_config.amp_dtype
            )
            
            # Salvar resultado do fold
            fold_result = FoldResult(
                fold_index=fold_idx,
                train_indices=train_indices,
                val_indices=val_indices,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                training_history=training_history,
                final_epoch=training_history.total_epochs,
                training_time=fold_time,
                fold_data_quality=fold_validation
            )
            
            self.fold_results.append(fold_result)
            self.metrics_aggregator.add_fold_metrics(val_metrics)
            
            logger.info(
                f"Fold {fold_idx + 1} concluído - "
                f"Val {self.training_config.monitor_metric}: {getattr(val_metrics, self.training_config.monitor_metric):.4f}, "
                f"Tempo: {fold_time:.2f}s"
            )
        
        # Calcular estatísticas finais
        summary_stats = self.metrics_aggregator.get_summary_statistics()
        best_fold_idx, best_fold_metrics = self.metrics_aggregator.get_best_fold(
            metric_name=self.training_config.monitor_metric,
            maximize=(self.training_config.monitor_mode == "max")
        )
        
        # Compilar resultados finais
        self.cv_summary = {
            "input_validation": input_validation,
            "cv_config": self.cv_config,
            "training_config": self.training_config,
            "n_folds": len(self.fold_results),
            "summary_statistics": summary_stats,
            "best_fold": {
                "fold_index": best_fold_idx,
                "metrics": best_fold_metrics.to_dict()
            },
            "fold_details": [
                {
                    "fold_index": result.fold_index,
                    "val_metrics": result.val_metrics.to_dict(),
                    "training_epochs": result.final_epoch,
                    "training_time": result.training_time,
                    "data_quality": result.fold_data_quality
                }
                for result in self.fold_results
            ],
            "total_cv_time": sum(result.training_time for result in self.fold_results)
        }
        
        logger.info(f"Cross-validation concluído!")
        logger.info(f"Melhor fold: {best_fold_idx + 1}")
        logger.info(f"Métrica média: {summary_stats.get(self.training_config.monitor_metric, {}).get('mean', 'N/A'):.4f}")
        
        return self.cv_summary
    
    def get_fold_predictions(self, fold_idx: int) -> Optional[Dict[str, np.ndarray]]:
        """Retorna predições de um fold específico (para análise detalhada)."""
        if fold_idx >= len(self.fold_results):
            return None
        
        # Esta funcionalidade requereria armazenar predições durante o CV
        # Por enquanto, retornamos None - pode ser implementado se necessário
        logger.warning("get_fold_predictions não implementado - requer modificação do CV loop")
        return None
    
    def save_results(self, save_path: Path) -> None:
        """Salva resultados completos do CV."""
        if not self.cv_summary:
            logger.warning("Nenhum resultado de CV para salvar")
            return
        
        import pickle
        
        save_data = {
            "cv_summary": self.cv_summary,
            "fold_results": self.fold_results,
            "config": {
                "cv_config": self.cv_config,
                "training_config": self.training_config
            }
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"Resultados de CV salvos em: {save_path}")


def quick_cross_validate(
    model_config: MLPConfig,
    X: torch.Tensor,
    y: torch.Tensor,
    n_splits: int = 5,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Função de conveniência para CV rápido com configurações padrão.
    """
    try:
        from ..models.mlp import MLPEmbeddingClassifier
    except ImportError:
        from models.mlp import MLPEmbeddingClassifier
    
    # Configurações padrão
    cv_config = CrossValidationConfig(n_splits=n_splits)
    training_config = TrainingConfig(max_epochs=50, patience=10)
    
    # Factories
    def model_factory():
        return MLPEmbeddingClassifier(model_config)
    
    def optimizer_factory(model):
        return torch.optim.Adam(model.parameters(), lr=model_config.learning_rate)
    
    # Executar CV
    cv = CrossValidator(cv_config, training_config, device)
    return cv.cross_validate(model_factory, optimizer_factory, X, y)
