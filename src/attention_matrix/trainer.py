"""
Trainer for Attention Matrix Models.

Single Responsibility: Training loop management only.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging
import time
import json

from .config import AttentionMatrixConfig


logger = logging.getLogger(__name__)


class AttentionTrainer:
    """
    Trainer for Cross-Attention models.
    
    Handles:
    - Training loop with validation
    - Loss computation (combined regression + classification)
    - Early stopping
    - Checkpointing
    - Gradient clipping
    
    Args:
        model: CrossAttentionModel or ImprovedCrossAttentionModel
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        config: AttentionMatrixConfig with training parameters
        device: Torch device
        output_dir: Directory to save checkpoints
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: AttentionMatrixConfig,
        device: Optional[torch.device] = None,
        output_dir: Optional[str] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Device
        if device is None:
            device = torch.device(config.get_device())
        self.device = device
        
        # Output directory
        self.output_dir = Path(output_dir) if output_dir else Path('models')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Move model to device
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs
        )
        
        # Loss functions
        self.regression_loss = nn.HuberLoss(delta=1.0)
        self.classification_loss = nn.BCEWithLogitsLoss()
        
        # Training state
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.training_history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'val_mae': [],
            'learning_rate': []
        }
    
    def _compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        classification_weight: float,
        regression_weight: float
    ) -> torch.Tensor:
        """Compute combined loss."""
        # Regression loss
        pred_reg = outputs['regression']
        true_reg = batch['activity'].to(self.device)
        loss_reg = self.regression_loss(pred_reg, true_reg)
        
        # Classification loss
        pred_cls = outputs['classification']
        true_cls = batch['is_active'].float().to(self.device)
        loss_cls = self.classification_loss(pred_cls, true_cls)
        
        return regression_weight * loss_reg + classification_weight * loss_cls
    
    def train_epoch(
        self,
        classification_weight: float,
        regression_weight: float
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch in self.train_loader:
            # Move to device
            protein_emb = batch['protein_embedding'].to(self.device)
            ligand_emb = batch['ligand_embedding'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(protein_emb, ligand_emb)
            
            # Compute loss
            loss = self._compute_loss(
                outputs, batch, classification_weight, regression_weight
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                protein_emb = batch['protein_embedding'].to(self.device)
                ligand_emb = batch['ligand_embedding'].to(self.device)
                
                outputs = self.model(protein_emb, ligand_emb)
                
                # Regression loss only for validation metric
                pred_reg = outputs['regression']
                true_reg = batch['activity'].to(self.device)
                loss = self.regression_loss(pred_reg, true_reg)
                total_loss += loss.item()
                
                all_preds.extend(pred_reg.cpu().numpy())
                all_targets.extend(true_reg.cpu().numpy())
        
        # Compute MAE
        import numpy as np
        preds = np.array(all_preds)
        targets = np.array(all_targets)
        mae = np.mean(np.abs(preds - targets))
        
        return {
            'loss': total_loss / len(self.val_loader),
            'mae': mae
        }
    
    def train(
        self,
        epochs: Optional[int] = None,
        patience: Optional[int] = None,
        classification_weight: float = 0.3,
        regression_weight: float = 0.7
    ) -> Dict[str, Any]:
        """
        Full training loop with early stopping.
        
        Args:
            epochs: Number of epochs (uses config if None)
            patience: Early stopping patience (uses config if None)
            classification_weight: Weight for classification loss
            regression_weight: Weight for regression loss
            
        Returns:
            Dictionary with training history
        """
        epochs = epochs or self.config.epochs
        patience = patience or self.config.early_stopping_patience
        
        logger.info(f"Starting training on {self.device}")
        logger.info(f"  Epochs: {epochs}")
        logger.info(f"  Patience: {patience}")
        logger.info(f"  Learning rate: {self.config.learning_rate}")
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(classification_weight, regression_weight)
            
            # Validate
            val_metrics = self.validate()
            val_loss = val_metrics['loss']
            val_mae = val_metrics['mae']
            
            # Update scheduler
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            # Record history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_mae'].append(val_mae)
            self.training_history['learning_rate'].append(current_lr)
            
            # Check for improvement
            improved = val_loss < self.best_val_loss
            if improved:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                
                # Save best model
                self.save_checkpoint(self.output_dir / 'best_model.pt')
            else:
                self.epochs_without_improvement += 1
            
            # Logging
            if (epoch + 1) % 5 == 0 or improved:
                marker = ' *' if improved else ''
                logger.info(
                    f"Epoch {epoch+1:3d}: "
                    f"train_loss={train_loss:.4f}, "
                    f"val_loss={val_loss:.4f}, "
                    f"val_mae={val_mae:.3f}{marker}"
                )
            
            # Early stopping
            if self.epochs_without_improvement >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
        
        total_time = time.time() - start_time
        
        logger.info(f"Training complete in {total_time:.1f}s")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        
        return {
            'best_val_loss': float(self.best_val_loss),
            'epochs_trained': epoch + 1,
            'training_time': total_time,
            'train_loss': self.training_history['train_loss'],
            'val_loss': self.training_history['val_loss'],
            'val_mae': self.training_history['val_mae']
        }
    
    def save_checkpoint(self, path: Path):
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history
        }, path)
        
        logger.debug(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_history = checkpoint['training_history']
        
        logger.info(f"Checkpoint loaded: {path}")
