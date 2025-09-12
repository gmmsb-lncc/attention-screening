"""
Core trainer para modelos de classificação com early stopping e logging avançado.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import sys
import os

# Adiciona o diretório src/classifier ao path para imports absolutos
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.classifier.utils.metrics import MetricsCalculator, ClassificationMetrics
from src.classifier.models.base_model import BaseClassifier

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuração específica para treinamento."""
    
    # Parâmetros de treinamento
    max_epochs: int = 100
    patience: int = 10
    min_delta: float = 1e-4
    
    # Controle de learning rate
    use_scheduler: bool = True
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    scheduler_min_lr: float = 1e-6
    
    # Mixed precision
    amp_enabled: bool = False
    amp_dtype: Optional[torch.dtype] = torch.float16  # torch.float16 ou torch.bfloat16
    
    # Gradient clipping
    gradient_clip_value: Optional[float] = None
    gradient_clip_norm: Optional[float] = None
    
    # Logging e checkpoints
    log_interval: int = 10
    save_best_model: bool = True
    save_checkpoint_interval: Optional[int] = None
    
    # Métrica para early stopping
    monitor_metric: str = "roc_auc"
    monitor_mode: str = "max"  # "max" ou "min"
    
    # Validação
    validate_every: int = 1  # Validar a cada N epochs
    
    def __post_init__(self):
        """Validação dos parâmetros."""
        if self.monitor_mode not in ["max", "min"]:
            raise ValueError("monitor_mode deve ser 'max' ou 'min'")
        
        if self.amp_dtype and self.amp_dtype not in [torch.float16, torch.bfloat16]:
            raise ValueError("amp_dtype deve ser torch.float16 ou torch.bfloat16")
        
        if self.patience <= 0:
            raise ValueError("patience deve ser positivo")
        
        if self.max_epochs <= 0:
            raise ValueError("max_epochs deve ser positivo")
    
    def get_amp_dtype(self) -> Optional[torch.dtype]:
        """Retorna o dtype para AMP."""
        if not self.amp_enabled or not self.amp_dtype:
            return None
        return self.amp_dtype


@dataclass
class TrainingHistory:
    """Histórico de treinamento."""
    
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    train_metrics: List[ClassificationMetrics] = field(default_factory=list)
    val_metrics: List[ClassificationMetrics] = field(default_factory=list)
    
    learning_rates: List[float] = field(default_factory=list)
    epoch_times: List[float] = field(default_factory=list)
    
    best_epoch: int = -1
    best_metric_value: float = float('-inf')
    early_stopped: bool = False
    total_epochs: int = 0
    
    def get_best_metrics(self) -> Optional[ClassificationMetrics]:
        """Retorna as métricas de validação da melhor época."""
        if self.best_epoch >= 0 and self.best_epoch < len(self.val_metrics):
            return self.val_metrics[self.best_epoch]
        return None
    
    def get_summary(self) -> Dict[str, Any]:
        """Retorna resumo do treinamento."""
        best_metrics = self.get_best_metrics()
        
        return {
            "total_epochs": self.total_epochs,
            "best_epoch": self.best_epoch + 1 if self.best_epoch >= 0 else None,
            "best_metric_value": self.best_metric_value if self.best_epoch >= 0 else None,
            "early_stopped": self.early_stopped,
            "final_train_loss": self.train_losses[-1] if self.train_losses else None,
            "final_val_loss": self.val_losses[-1] if self.val_losses else None,
            "best_val_metrics": best_metrics.to_dict() if best_metrics else None,
            "total_training_time": sum(self.epoch_times),
            "avg_epoch_time": np.mean(self.epoch_times) if self.epoch_times else None
        }


class ModelTrainer:
    """Trainer principal para modelos de classificação."""
    
    def __init__(
        self, 
        model: BaseClassifier,
        config: TrainingConfig,
        device: Optional[torch.device] = None,
        checkpoint_dir: Optional[Path] = None
    ):
        self.model = model
        self.config = config
        self.device = device or torch.device("cpu")
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        
        # Componentes de treinamento
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.criterion: Optional[nn.Module] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        self.scaler: Optional[torch.cuda.amp.GradScaler] = None
        
        # Calculador de métricas
        self.metrics_calculator = MetricsCalculator(device=self.device)
        
        # História de treinamento
        self.history = TrainingHistory()
        
        # Configuração de checkpoints
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuração de AMP
        if self.config.amp_enabled:
            self.scaler = torch.cuda.amp.GradScaler()
        
        logger.info(f"Trainer inicializado - Device: {self.device}, AMP: {self.config.amp_enabled}")
    
    def setup_training(
        self,
        optimizer: torch.optim.Optimizer,
        criterion: Optional[nn.Module] = None
    ) -> "ModelTrainer":
        """
        Configura componentes de treinamento.
        
        Args:
            optimizer: Otimizador PyTorch
            criterion: Função de loss (padrão: BCEWithLogitsLoss)
        
        Returns:
            Self para method chaining
        """
        self.optimizer = optimizer
        self.criterion = criterion or nn.BCEWithLogitsLoss()
        
        # Scheduler opcional
        if self.config.use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=self.config.monitor_mode,
                factor=self.config.scheduler_factor,
                patience=self.config.scheduler_patience,
                min_lr=self.config.scheduler_min_lr,
                verbose=True
            )
        
        return self
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, ClassificationMetrics]:
        """
        Executa uma época de treinamento.
        
        Returns:
            (loss médio, métricas de treinamento)
        """
        self.model.train()
        epoch_losses = []
        
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass com AMP
            with torch.cuda.amp.autocast(
                enabled=self.config.amp_enabled, 
                dtype=self.config.get_amp_dtype()
            ):
                logits = self.model(batch_x)
                # Garante dimensões corretas para BCE loss
                if logits.dim() > 1 and logits.size(1) == 1:
                    logits = logits.squeeze(1)
                loss = self.criterion(logits, batch_y)
            
            # Backward pass com gradient scaling
            if self.scaler:
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.gradient_clip_norm:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.gradient_clip_norm
                    )
                elif self.config.gradient_clip_value:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_value_(
                        self.model.parameters(), 
                        self.config.gradient_clip_value
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                
                # Gradient clipping sem AMP
                if self.config.gradient_clip_norm:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.gradient_clip_norm
                    )
                elif self.config.gradient_clip_value:
                    torch.nn.utils.clip_grad_value_(
                        self.model.parameters(), 
                        self.config.gradient_clip_value
                    )
                
                self.optimizer.step()
            
            epoch_losses.append(loss.item())
            
            # Log intermitente
            if batch_idx % self.config.log_interval == 0:
                logger.debug(
                    f"Batch {batch_idx}/{len(train_loader)}, "
                    f"Loss: {loss.item():.6f}"
                )
        
        # Calcular métricas de treinamento
        train_metrics = self.metrics_calculator.evaluate_model(
            self.model, 
            train_loader,
            self.criterion,
            self.config.amp_enabled,
            self.config.get_amp_dtype()
        )
        
        return np.mean(epoch_losses), train_metrics
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, ClassificationMetrics]:
        """
        Executa uma época de validação.
        
        Returns:
            (loss médio, métricas de validação)
        """
        val_metrics = self.metrics_calculator.evaluate_model(
            self.model,
            val_loader,
            self.criterion,
            self.config.amp_enabled,
            self.config.get_amp_dtype()
        )
        
        return val_metrics.loss, val_metrics
    
    def check_early_stopping(self, val_metrics: ClassificationMetrics) -> bool:
        """
        Verifica critérios de early stopping.
        
        Returns:
            True se deve parar o treinamento
        """
        metric_value = getattr(val_metrics, self.config.monitor_metric)
        
        # Verifica se é a melhor métrica até agora
        is_best = False
        if self.config.monitor_mode == "max":
            is_best = metric_value > (self.history.best_metric_value + self.config.min_delta)
        else:
            is_best = metric_value < (self.history.best_metric_value - self.config.min_delta)
        
        if is_best:
            self.history.best_metric_value = metric_value
            self.history.best_epoch = len(self.history.val_metrics)
            return False
        
        # Verifica paciência
        epochs_without_improvement = len(self.history.val_metrics) - self.history.best_epoch
        return epochs_without_improvement >= self.config.patience
    
    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Salva checkpoint do modelo."""
        if not self.checkpoint_dir:
            return
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'history': self.history,
            'config': self.config
        }
        
        # Checkpoint regular
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Melhor modelo
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Novo melhor modelo salvo: {best_path}")
    
    def train(
        self, 
        train_loader: DataLoader, 
        val_loader: Optional[DataLoader] = None
    ) -> TrainingHistory:
        """
        Executa o treinamento completo.
        
        Args:
            train_loader: DataLoader de treinamento
            val_loader: DataLoader de validação (opcional)
            
        Returns:
            Histórico completo de treinamento
        """
        if not self.optimizer:
            raise ValueError("Optimizer não configurado. Use setup_training() primeiro.")
        
        logger.info(f"Iniciando treinamento - {self.config.max_epochs} epochs máximos")
        training_start = time.time()
        
        for epoch in range(self.config.max_epochs):
            epoch_start = time.time()
            
            # Treinamento
            train_loss, train_metrics = self.train_epoch(train_loader)
            self.history.train_losses.append(train_loss)
            self.history.train_metrics.append(train_metrics)
            
            # Validação
            if val_loader and (epoch + 1) % self.config.validate_every == 0:
                val_loss, val_metrics = self.validate_epoch(val_loader)
                self.history.val_losses.append(val_loss)
                self.history.val_metrics.append(val_metrics)
                
                # Learning rate scheduling
                if self.scheduler:
                    if self.config.monitor_mode == "max":
                        self.scheduler.step(-getattr(val_metrics, self.config.monitor_metric))
                    else:
                        self.scheduler.step(getattr(val_metrics, self.config.monitor_metric))
                
                # Early stopping
                should_stop = self.check_early_stopping(val_metrics)
                
                # Checkpoint
                is_best = (len(self.history.val_metrics) - 1 == self.history.best_epoch)
                if self.config.save_best_model and is_best:
                    self.save_checkpoint(epoch, is_best=True)
                
                if should_stop:
                    logger.info(f"Early stopping na época {epoch + 1}")
                    self.history.early_stopped = True
                    break
            
            # Salvar checkpoint regular
            if (self.config.save_checkpoint_interval and 
                (epoch + 1) % self.config.save_checkpoint_interval == 0):
                self.save_checkpoint(epoch, is_best=False)
            
            # Timing e logging
            epoch_time = time.time() - epoch_start
            self.history.epoch_times.append(epoch_time)
            self.history.learning_rates.append(self.optimizer.param_groups[0]['lr'])
            
            # Log do progresso
            if val_loader:
                val_metrics = self.history.val_metrics[-1] if self.history.val_metrics else None
                logger.info(
                    f"Epoch {epoch + 1}/{self.config.max_epochs} - "
                    f"Train Loss: {train_loss:.6f}, "
                    f"Val Loss: {val_metrics.loss if val_metrics else 'N/A':.6f}, "
                    f"Val {self.config.monitor_metric}: {getattr(val_metrics, self.config.monitor_metric) if val_metrics else 'N/A':.4f}, "
                    f"Time: {epoch_time:.2f}s"
                )
            else:
                logger.info(
                    f"Epoch {epoch + 1}/{self.config.max_epochs} - "
                    f"Train Loss: {train_loss:.6f}, "
                    f"Time: {epoch_time:.2f}s"
                )
        
        # Finalização
        self.history.total_epochs = epoch + 1
        total_time = time.time() - training_start
        
        logger.info(f"Treinamento concluído em {total_time:.2f}s")
        logger.info(f"Melhor época: {self.history.best_epoch + 1}" if self.history.best_epoch >= 0 else "Nenhuma validação")
        logger.info(f"Melhor {self.config.monitor_metric}: {self.history.best_metric_value:.4f}" if self.history.best_epoch >= 0 else "")
        
        return self.history
    
    def load_checkpoint(self, checkpoint_path: Path, load_optimizer: bool = True) -> int:
        """
        Carrega checkpoint para continuar treinamento.
        
        Returns:
            Época do checkpoint carregado
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if load_optimizer and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint.get('scheduler_state_dict') and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if checkpoint.get('scaler_state_dict') and self.scaler:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.history = checkpoint.get('history', TrainingHistory())
        
        epoch = checkpoint['epoch']
        logger.info(f"Checkpoint carregado da época {epoch + 1}")
        
        return epoch
