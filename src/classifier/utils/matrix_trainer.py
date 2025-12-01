"""
Trainer for CrossAttentionAffinityModel.

This module provides training utilities for the CNN + Cross-Attention model,
including multi-task training, validation, and evaluation.

Author: DockTKinase Team
Date: November 2025
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Callable
import logging
from dataclasses import dataclass, field
from tqdm import tqdm
import json
from datetime import datetime

try:
    from ..models.cross_attention_model import CrossAttentionAffinityModel, MultiTaskLoss
except ImportError:
    from src.classifier.models.cross_attention_model import CrossAttentionAffinityModel, MultiTaskLoss

# Import matrix metrics for comprehensive evaluation
try:
    from .matrix_metrics import MatrixMetricsCalculator, ClassificationMetrics, RegressionMetrics
except ImportError:
    MatrixMetricsCalculator = None
    ClassificationMetrics = None
    RegressionMetrics = None

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training CrossAttentionAffinityModel."""
    
    # Model configuration
    protein_dim: int = 2560
    ligand_dim: int = 768
    hidden_dim: int = 256
    num_cnn_layers: int = 3
    num_cross_attn_layers: int = 2
    num_heads: int = 8
    ff_dim: int = 1024
    model_dropout: float = 0.1
    
    # Training configuration
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 100
    warmup_epochs: int = 5
    
    # Loss configuration
    classification_weight: float = 1.0
    regression_weight: float = 1.0
    use_uncertainty_weighting: bool = False
    
    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4
    
    # Checkpointing
    save_dir: str = 'checkpoints'
    save_best_only: bool = True
    
    # Device
    device: str = 'auto'
    
    # Logging
    log_interval: int = 10
    eval_interval: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TrainingConfig':
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class MetricTracker:
    """Track training metrics."""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
    
    def update(self, name: str, value: float):
        """Update metric."""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
    
    def get_last(self, name: str) -> Optional[float]:
        """Get last value."""
        if name in self.metrics and self.metrics[name]:
            return self.metrics[name][-1]
        return None
    
    def get_best(self, name: str, mode: str = 'min') -> Optional[float]:
        """Get best value."""
        if name in self.metrics and self.metrics[name]:
            if mode == 'min':
                return min(self.metrics[name])
            return max(self.metrics[name])
        return None
    
    def to_dict(self) -> Dict[str, List[float]]:
        """Convert to dictionary."""
        return self.metrics.copy()


class EarlyStopping:
    """Early stopping handler."""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        """Check if should stop."""
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


class CrossAttentionTrainer:
    """
    Trainer for CrossAttentionAffinityModel.
    
    Handles training loop, validation, checkpointing, and metrics.
    """
    
    def __init__(
        self,
        model: CrossAttentionAffinityModel,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None
    ):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Device
        if config.device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(config.device)
        
        logger.info(f"Using device: {self.device}")
        
        # Model
        self.model = model.to(self.device)
        
        # Loss function
        self.loss_fn = MultiTaskLoss(
            classification_weight=config.classification_weight,
            regression_weight=config.regression_weight,
            use_uncertainty_weighting=config.use_uncertainty_weighting
        )
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Scheduler
        total_steps = len(train_loader) * config.num_epochs
        # Ensure pct_start is between 0 and 1
        pct_start = min(0.3, config.warmup_epochs / max(config.num_epochs, 1))
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            total_steps=total_steps,
            pct_start=pct_start
        )
        
        # Metrics
        self.metrics = MetricTracker()
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.patience,
            min_delta=config.min_delta,
            mode='min'
        )
        
        # Checkpointing
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_loss = float('inf')
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        total_cls_loss = 0
        total_reg_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch in pbar:
            # Move to device
            protein_matrix = batch['protein_matrix'].to(self.device)
            ligand_matrix = batch['ligand_matrix'].to(self.device)
            protein_mask = batch['protein_mask'].to(self.device)
            ligand_mask = batch['ligand_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            reg_targets = batch['regression_targets'].to(self.device)
            reg_mask = batch['regression_mask'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(
                protein_matrix, ligand_matrix,
                protein_mask, ligand_mask
            )
            
            # Compute loss
            losses = self.loss_fn(
                output['classification'],
                output['regression'],
                labels,
                reg_targets,
                reg_mask
            )
            
            # Backward pass
            losses['total'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Track metrics
            total_loss += losses['total'].item()
            total_cls_loss += losses['classification'].item()
            total_reg_loss += losses['regression'].item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': losses['total'].item(),
                'cls': losses['classification'].item(),
                'reg': losses['regression'].item()
            })
        
        # Average metrics
        avg_loss = total_loss / num_batches
        avg_cls_loss = total_cls_loss / num_batches
        avg_reg_loss = total_reg_loss / num_batches
        
        self.metrics.update('train_loss', avg_loss)
        self.metrics.update('train_cls_loss', avg_cls_loss)
        self.metrics.update('train_reg_loss', avg_reg_loss)
        
        return {
            'loss': avg_loss,
            'cls_loss': avg_cls_loss,
            'reg_loss': avg_reg_loss
        }
    
    @torch.no_grad()
    def validate(self, loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """Validate model."""
        loader = loader or self.val_loader
        if loader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0
        total_cls_loss = 0
        total_reg_loss = 0
        all_preds = []
        all_labels = []
        all_reg_preds = []
        all_reg_targets = []
        all_reg_mask = []
        num_batches = 0
        
        for batch in loader:
            # Move to device
            protein_matrix = batch['protein_matrix'].to(self.device)
            ligand_matrix = batch['ligand_matrix'].to(self.device)
            protein_mask = batch['protein_mask'].to(self.device)
            ligand_mask = batch['ligand_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            reg_targets = batch['regression_targets'].to(self.device)
            reg_mask = batch['regression_mask'].to(self.device)
            
            # Forward pass
            output = self.model(
                protein_matrix, ligand_matrix,
                protein_mask, ligand_mask
            )
            
            # Compute loss
            losses = self.loss_fn(
                output['classification'],
                output['regression'],
                labels,
                reg_targets,
                reg_mask
            )
            
            # Track metrics
            total_loss += losses['total'].item()
            total_cls_loss += losses['classification'].item()
            total_reg_loss += losses['regression'].item()
            num_batches += 1
            
            # Collect predictions
            preds = torch.sigmoid(output['classification'])
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_reg_preds.append(output['regression'].cpu())
            all_reg_targets.append(reg_targets.cpu())
            all_reg_mask.append(reg_mask.cpu())
        
        # Concatenate
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_reg_preds = torch.cat(all_reg_preds, dim=0)
        all_reg_targets = torch.cat(all_reg_targets, dim=0)
        all_reg_mask = torch.cat(all_reg_mask, dim=0)
        
        # Compute metrics
        avg_loss = total_loss / num_batches
        avg_cls_loss = total_cls_loss / num_batches
        avg_reg_loss = total_reg_loss / num_batches
        
        # Classification metrics
        binary_preds = (all_preds >= 0.5).float()
        accuracy = (binary_preds == all_labels).float().mean().item()
        
        # ROC-AUC
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(all_labels.numpy(), all_preds.numpy())
        except:
            auc = 0.0
        
        # Regression metrics (only for samples with labels)
        reg_mask_bool = all_reg_mask.bool().squeeze()
        if reg_mask_bool.sum() > 0:
            reg_preds_masked = all_reg_preds[reg_mask_bool]
            reg_targets_masked = all_reg_targets[reg_mask_bool]
            mse = ((reg_preds_masked - reg_targets_masked) ** 2).mean().item()
            rmse = np.sqrt(mse)
            
            try:
                from scipy.stats import pearsonr
                r, _ = pearsonr(
                    reg_preds_masked.numpy().flatten(),
                    reg_targets_masked.numpy().flatten()
                )
            except:
                r = 0.0
        else:
            rmse = 0.0
            r = 0.0
        
        # Update metrics
        self.metrics.update('val_loss', avg_loss)
        self.metrics.update('val_cls_loss', avg_cls_loss)
        self.metrics.update('val_reg_loss', avg_reg_loss)
        self.metrics.update('val_accuracy', accuracy)
        self.metrics.update('val_auc', auc)
        self.metrics.update('val_rmse', rmse)
        self.metrics.update('val_pearson', r)
        
        return {
            'loss': avg_loss,
            'cls_loss': avg_cls_loss,
            'reg_loss': avg_reg_loss,
            'accuracy': accuracy,
            'auc': auc,
            'rmse': rmse,
            'pearson': r
        }
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config.to_dict(),
            'metrics': self.metrics.to_dict(),
            'best_val_loss': self.best_val_loss
        }
        
        path = self.save_dir / filename
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")
        
        if is_best:
            best_path = self.save_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model: {best_path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self) -> Dict[str, Any]:
        """
        Full training loop.
        
        Returns:
            Training history and final metrics
        """
        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        logger.info(f"Model parameters: {self.model.count_parameters():,}")
        
        start_time = datetime.now()
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch + 1
            
            # Train
            train_metrics = self.train_epoch()
            logger.info(
                f"Epoch {self.current_epoch}/{self.config.num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"CLS: {train_metrics['cls_loss']:.4f}, "
                f"REG: {train_metrics['reg_loss']:.4f}"
            )
            
            # Validate
            if self.val_loader and (epoch + 1) % self.config.eval_interval == 0:
                val_metrics = self.validate()
                logger.info(
                    f"Validation - "
                    f"Loss: {val_metrics['loss']:.4f}, "
                    f"Acc: {val_metrics['accuracy']:.4f}, "
                    f"AUC: {val_metrics['auc']:.4f}, "
                    f"RMSE: {val_metrics['rmse']:.4f}"
                )
                
                # Checkpointing
                is_best = val_metrics['loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics['loss']
                
                if self.config.save_best_only:
                    if is_best:
                        self.save_checkpoint('best_model.pt', is_best=True)
                else:
                    self.save_checkpoint(f'checkpoint_epoch_{self.current_epoch}.pt', is_best=is_best)
                
                # Early stopping
                if self.early_stopping(val_metrics['loss']):
                    logger.info(f"Early stopping at epoch {self.current_epoch}")
                    break
        
        # Final evaluation
        final_metrics = {}
        if self.test_loader:
            logger.info("Running final evaluation on test set...")
            final_metrics = self.validate(self.test_loader)
            logger.info(
                f"Test Results - "
                f"Acc: {final_metrics['accuracy']:.4f}, "
                f"AUC: {final_metrics['auc']:.4f}, "
                f"RMSE: {final_metrics['rmse']:.4f}"
            )
        
        training_time = datetime.now() - start_time
        
        return {
            'history': self.metrics.to_dict(),
            'final_metrics': final_metrics,
            'best_val_loss': self.best_val_loss,
            'training_time': str(training_time),
            'epochs_trained': self.current_epoch
        }

    @torch.no_grad()
    def evaluate_comprehensive(
        self,
        loader: Optional[DataLoader] = None,
        threshold: float = 0.5,
        save_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation with full metrics (matching vector pipeline).
        
        This method provides the same metrics as the vector-based pipeline:
        - Classification: Accuracy, Precision, Recall, F1, ROC-AUC, Average Precision,
                         Brier Score, MCC, Specificity, Sensitivity, Fbeta variants
        - Regression: MAE, MSE, RMSE, R², MedianAE, MAPE, Pearson, Spearman,
                     Explained Variance, Max Error, residual statistics
        
        Args:
            loader: DataLoader to evaluate (defaults to test_loader)
            threshold: Classification threshold
            save_dir: Directory to save detailed results
            
        Returns:
            Dictionary with comprehensive metrics
        """
        if MatrixMetricsCalculator is None:
            logger.warning("MatrixMetricsCalculator not available, falling back to basic metrics")
            return self.validate(loader)
        
        loader = loader or self.test_loader or self.val_loader
        if loader is None:
            logger.warning("No data loader available for evaluation")
            return {}
        
        self.model.eval()
        
        # Collect all predictions and targets
        all_cls_probs = []
        all_cls_labels = []
        all_reg_preds = []
        all_reg_targets = []
        all_reg_mask = []
        
        logger.info("Running comprehensive evaluation...")
        
        for batch in tqdm(loader, desc="Evaluating"):
            # Move to device
            protein_matrix = batch['protein_matrix'].to(self.device)
            ligand_matrix = batch['ligand_matrix'].to(self.device)
            protein_mask = batch['protein_mask'].to(self.device)
            ligand_mask = batch['ligand_mask'].to(self.device)
            labels = batch['labels']
            reg_targets = batch['regression_targets']
            reg_mask = batch['regression_mask']
            
            # Forward pass
            output = self.model(
                protein_matrix, ligand_matrix,
                protein_mask, ligand_mask
            )
            
            # Collect predictions
            cls_probs = torch.sigmoid(output['classification']).cpu()
            reg_preds = output['regression'].cpu()
            
            all_cls_probs.append(cls_probs)
            all_cls_labels.append(labels)
            all_reg_preds.append(reg_preds)
            all_reg_targets.append(reg_targets)
            all_reg_mask.append(reg_mask)
        
        # Concatenate all
        all_cls_probs = torch.cat(all_cls_probs, dim=0).numpy().flatten()
        all_cls_labels = torch.cat(all_cls_labels, dim=0).numpy().flatten()
        all_reg_preds = torch.cat(all_reg_preds, dim=0).numpy().flatten()
        all_reg_targets = torch.cat(all_reg_targets, dim=0).numpy().flatten()
        all_reg_mask = torch.cat(all_reg_mask, dim=0).numpy().flatten().astype(bool)
        
        # Use comprehensive metrics calculator
        metrics_calculator = MatrixMetricsCalculator(threshold=threshold)
        
        # Compute classification metrics
        cls_metrics = metrics_calculator.calculate_classification_metrics(
            y_true=all_cls_labels,
            y_prob=all_cls_probs,
            threshold=threshold
        )
        
        # Compute regression metrics if targets available
        reg_metrics = None
        if all_reg_mask.any():
            reg_metrics = metrics_calculator.calculate_regression_metrics(
                y_true=all_reg_targets[all_reg_mask],
                y_pred=all_reg_preds[all_reg_mask]
            )
        
        # Log summary
        logger.info("\n" + "="*60)
        logger.info("COMPREHENSIVE EVALUATION RESULTS")
        logger.info("="*60)
        
        # Classification summary
        logger.info("\n--- Classification Metrics ---")
        logger.info(f"  Accuracy:         {cls_metrics.accuracy:.4f}")
        logger.info(f"  Precision:        {cls_metrics.precision:.4f}")
        logger.info(f"  Recall:           {cls_metrics.recall:.4f}")
        logger.info(f"  F1 Score:         {cls_metrics.f1:.4f}")
        logger.info(f"  ROC-AUC:          {cls_metrics.roc_auc:.4f}")
        logger.info(f"  Average Precision:{cls_metrics.average_precision:.4f}")
        logger.info(f"  MCC:              {cls_metrics.mcc:.4f}")
        logger.info(f"  Specificity:      {cls_metrics.specificity:.4f}")
        logger.info(f"  Sensitivity:      {cls_metrics.sensitivity:.4f}")
        logger.info(f"  Brier Score:      {cls_metrics.brier_score:.4f}")
        
        # Regression summary
        if reg_metrics is not None:
            logger.info("\n--- Regression Metrics ---")
            logger.info(f"  MAE:              {reg_metrics.mae:.4f}")
            logger.info(f"  MSE:              {reg_metrics.mse:.4f}")
            logger.info(f"  RMSE:             {reg_metrics.rmse:.4f}")
            logger.info(f"  R²:               {reg_metrics.r2:.4f}")
            logger.info(f"  Pearson r:        {reg_metrics.pearson_r:.4f}")
            logger.info(f"  Spearman ρ:       {reg_metrics.spearman_r:.4f}")
            logger.info(f"  Kendall τ:        {reg_metrics.kendall_tau:.4f}")
            logger.info(f"  CCC (Lin):        {reg_metrics.ccc:.4f}")
            logger.info(f"  Median AE:        {reg_metrics.median_ae:.4f}")
            logger.info(f"  Explained Var:    {reg_metrics.explained_variance:.4f}")
        
        logger.info("="*60 + "\n")
        
        # Optionally save results
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save as JSON
            json_results = {
                'classification': cls_metrics.to_dict(),
                'regression': reg_metrics.to_dict() if reg_metrics else {},
                'samples': {
                    'total': len(all_cls_labels),
                    'with_regression_target': int(all_reg_mask.sum())
                }
            }
            
            json_path = save_path / 'comprehensive_metrics.json'
            with open(json_path, 'w') as f:
                json.dump(json_results, f, indent=2)
            logger.info(f"Saved metrics to: {json_path}")
            
            # Save formatted report
            report = metrics_calculator.format_classification_report(cls_metrics)
            if reg_metrics:
                report += "\n\n" + metrics_calculator.format_regression_report(reg_metrics)
            
            report_path = save_path / 'metrics_report.txt'
            with open(report_path, 'w') as f:
                f.write(report)
            logger.info(f"Saved report to: {report_path}")
        
        # Return results in dictionary format for compatibility
        return {
            'classification': cls_metrics,
            'regression': reg_metrics,
            'classification_dict': cls_metrics.to_dict(),
            'regression_dict': reg_metrics.to_dict() if reg_metrics else {},
            # Legacy format for compatibility
            'accuracy': cls_metrics.accuracy,
            'auc': cls_metrics.roc_auc,
            'f1': cls_metrics.f1,
            'rmse': reg_metrics.rmse if reg_metrics else 0.0,
            'pearson': reg_metrics.pearson_r if reg_metrics else 0.0
        }


def train_cross_attention_model(
    train_df,
    val_df,
    protein_matrix_dir: str,
    ligand_matrix_dir: str,
    config: Optional[TrainingConfig] = None,
    **kwargs
) -> Tuple[CrossAttentionAffinityModel, Dict[str, Any]]:
    """
    Convenience function for training CrossAttentionAffinityModel.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        protein_matrix_dir: Directory with protein matrices
        ligand_matrix_dir: Directory with ligand matrices
        config: Training configuration
        **kwargs: Additional config overrides
    
    Returns:
        Trained model and training history
    """
    from .matrix_dataloader import create_matrix_dataloader
    
    # Config
    if config is None:
        config = TrainingConfig(**kwargs)
    
    # Create data loaders
    train_loader = create_matrix_dataloader(
        train_df,
        protein_matrix_dir,
        ligand_matrix_dir,
        batch_size=config.batch_size,
        shuffle=True
    )
    
    val_loader = create_matrix_dataloader(
        val_df,
        protein_matrix_dir,
        ligand_matrix_dir,
        batch_size=config.batch_size,
        shuffle=False
    )
    
    # Create model
    model = CrossAttentionAffinityModel(
        protein_dim=config.protein_dim,
        ligand_dim=config.ligand_dim,
        hidden_dim=config.hidden_dim,
        num_cnn_layers=config.num_cnn_layers,
        num_cross_attn_layers=config.num_cross_attn_layers,
        num_heads=config.num_heads,
        ff_dim=config.ff_dim,
        dropout=config.model_dropout
    )
    
    # Create trainer
    trainer = CrossAttentionTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader
    )
    
    # Train
    history = trainer.train()
    
    return model, history


if __name__ == "__main__":
    # Quick test
    print("Testing CrossAttentionTrainer...")
    
    import tempfile
    import pandas as pd
    from .matrix_dataloader import create_matrix_dataloader
    
    # Create dummy data
    train_df = pd.DataFrame({
        'seq_id': ['P1'] * 10 + ['P2'] * 10,
        'chembl_id': ['C1', 'C2'] * 10,
        'label': [1, 0] * 10,
        'pchembl_value': [7.5, 6.2] * 10
    })
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        prot_dir = tmpdir / 'protein_matrix_embeddings'
        lig_dir = tmpdir / 'ligand_matrix_embeddings'
        prot_dir.mkdir()
        lig_dir.mkdir()
        
        # Create dummy matrices (small for test)
        np.save(prot_dir / 'P1_matrix.npy', np.random.randn(20, 320))
        np.save(prot_dir / 'P2_matrix.npy', np.random.randn(30, 320))
        np.save(lig_dir / 'C1_matrix.npy', np.random.randn(15, 768))
        np.save(lig_dir / 'C2_matrix.npy', np.random.randn(10, 768))
        
        # Create data loader
        train_loader = create_matrix_dataloader(
            train_df, prot_dir, lig_dir, batch_size=4
        )
        
        # Create model
        model = CrossAttentionAffinityModel(
            protein_dim=320,
            ligand_dim=768,
            hidden_dim=64,
            num_cnn_layers=2,
            num_cross_attn_layers=1
        )
        
        # Create config
        config = TrainingConfig(
            protein_dim=320,
            ligand_dim=768,
            hidden_dim=64,
            num_cnn_layers=2,
            num_cross_attn_layers=1,
            num_epochs=2,
            batch_size=4,
            save_dir=str(tmpdir / 'checkpoints')
        )
        
        # Create trainer
        trainer = CrossAttentionTrainer(
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=train_loader  # Use same for test
        )
        
        # Train
        history = trainer.train()
        
        print(f"\nTraining completed!")
        print(f"Epochs trained: {history['epochs_trained']}")
        print(f"Best val loss: {history['best_val_loss']:.4f}")
    
    print("\n✅ All tests passed!")
