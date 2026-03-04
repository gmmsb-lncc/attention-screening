"""Training utilities for Level 5-Lite."""

import os
import json
import time
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    roc_auc_score,
)
from tqdm import tqdm


@dataclass
class Level5LiteConfig:
    """Configuration for Level 5-Lite training."""
    
    # Model architecture
    protein_input_dim: int = 320
    ligand_input_dim: int = 768
    hidden_dim: int = 512
    num_encoder_layers: int = 2
    num_cross_attn_layers: int = 1
    num_heads: int = 8
    dropout: float = 0.1
    classifier_dropout: float = 0.3
    
    # Data
    max_protein_len: int = 1024
    max_ligand_len: int = 256
    pchembl_threshold: float = 6.0
    
    # Training
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 200
    patience: Optional[int] = 15
    max_grad_norm: float = 1.0
    
    # Optimization
    use_focal_loss: bool = False
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    optimize_threshold: bool = False  # Use fixed 0.5 to avoid validation overfitting
    
    # DataLoader
    num_workers: int = 0
    pin_memory: bool = True
    cache_in_memory: bool = False
    
    def to_dict(self) -> Dict:
        return asdict(self)


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        loss = alpha_t * focal_weight * bce
        return loss.mean()


def optimize_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    metric: str = 'mcc',
) -> Tuple[float, float]:
    """Find optimal classification threshold.
    
    Args:
        probs: Predicted probabilities
        labels: True binary labels
        metric: Metric to optimize ('mcc', 'f1', 'balanced_accuracy')
    
    Returns:
        (best_threshold, best_score)
    """
    thresholds = np.linspace(0.1, 0.9, 81)
    best_threshold = 0.5
    best_score = -np.inf
    
    for t in thresholds:
        preds = (probs >= t).astype(int)
        
        if metric == 'mcc':
            score = matthews_corrcoef(labels, preds)
        elif metric == 'f1':
            score = f1_score(labels, preds, zero_division=0)
        elif metric == 'balanced_accuracy':
            from sklearn.metrics import balanced_accuracy_score
            score = balanced_accuracy_score(labels, preds)
        else:
            score = matthews_corrcoef(labels, preds)
        
        if score > best_score:
            best_score = score
            best_threshold = t
    
    return best_threshold, best_score


def compute_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute classification metrics.
    
    Args:
        probs: Predicted probabilities
        labels: True binary labels
        threshold: Classification threshold
    
    Returns:
        Dict with accuracy, mcc, f1, precision, recall, auc
    """
    preds = (probs >= threshold).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(labels, preds),
        'mcc': matthews_corrcoef(labels, preds),
        'f1': f1_score(labels, preds, zero_division=0),
        'precision': precision_score(labels, preds, zero_division=0),
        'recall': recall_score(labels, preds, zero_division=0),
    }
    
    # AUC requires both classes present
    if len(np.unique(labels)) > 1:
        metrics['auc'] = roc_auc_score(labels, probs)
    else:
        metrics['auc'] = 0.0
    
    metrics['threshold'] = threshold
    
    return metrics


class Level5LiteTrainer:
    """Trainer for Level 5-Lite model."""
    
    def __init__(
        self,
        model: nn.Module,
        config: Level5LiteConfig,
        device: torch.device = None,
    ):
        """Initialize trainer.
        
        Args:
            model: Level5LiteModel instance
            config: Training configuration
            device: Device to train on
        """
        self.model = model
        self.config = config
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        self.model = self.model.to(self.device)
        
        # Loss function
        if config.use_focal_loss:
            self.criterion = FocalLoss(
                alpha=config.focal_alpha,
                gamma=config.focal_gamma,
            )
        else:
            # BCEWithLogitsLoss with label smoothing for better generalization
            self.criterion = nn.BCEWithLogitsLoss()

        # Optimizer with weight decay for regularization
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Scheduler (will be set in train())
        self.scheduler = None
        
        # Tracking
        self.best_val_mcc = -np.inf
        self.best_threshold = 0.5
        self.patience_counter = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_mcc': [],
            'val_auc': [],
            'learning_rate': [],
        }
        
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch.
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            # Move to device
            protein = batch['protein_matrix'].to(self.device)
            ligand = batch['ligand_matrix'].to(self.device)
            protein_mask = batch['protein_mask'].to(self.device)
            ligand_mask = batch['ligand_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            logits = self.model(protein, ligand, protein_mask, ligand_mask)
            loss = self.criterion(logits.squeeze(-1), labels)
            
            # Backward
            loss.backward()
            
            # Gradient clipping
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        optimize_thresh: bool = False,
    ) -> Tuple[float, Dict[str, float], float]:
        """Evaluate model.
        
        Args:
            loader: DataLoader for evaluation
            optimize_thresh: If True, find optimal threshold
        
        Returns:
            (loss, metrics_dict, threshold)
        """
        self.model.eval()
        
        all_probs = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0
        
        for batch in loader:
            protein = batch['protein_matrix'].to(self.device)
            ligand = batch['ligand_matrix'].to(self.device)
            protein_mask = batch['protein_mask'].to(self.device)
            ligand_mask = batch['ligand_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            logits = self.model(protein, ligand, protein_mask, ligand_mask)
            loss = self.criterion(logits.squeeze(-1), labels)
            
            probs = torch.sigmoid(logits).squeeze(-1)
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            total_loss += loss.item()
            num_batches += 1
        
        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
        avg_loss = total_loss / num_batches
        
        # Optimize threshold if requested
        if optimize_thresh:
            threshold, _ = optimize_threshold(all_probs, all_labels, 'mcc')
        else:
            threshold = self.best_threshold
        
        metrics = compute_metrics(all_probs, all_labels, threshold)
        
        return avg_loss, metrics, threshold
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_dir: Optional[str] = None,
    ) -> Dict:
        """Full training loop.
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            checkpoint_dir: Directory to save checkpoints
        
        Returns:
            Training history dict
        """
        # Warmup + Cosine Annealing for better convergence
        warmup_epochs = 5
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.num_epochs - warmup_epochs,
            eta_min=1e-6,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )
        
        # Create checkpoint dir
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Training loop
        start_time = time.time()
        
        pbar = tqdm(range(1, self.config.num_epochs + 1), desc="Training")
        
        for epoch in pbar:
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Evaluate
            val_loss, val_metrics, threshold = self.evaluate(
                val_loader,
                optimize_thresh=self.config.optimize_threshold,
            )
            
            # Update scheduler
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_mcc'].append(val_metrics['mcc'])
            self.history['val_auc'].append(val_metrics['auc'])
            self.history['learning_rate'].append(current_lr)
            
            # Check for improvement
            if val_metrics['mcc'] > self.best_val_mcc:
                self.best_val_mcc = val_metrics['mcc']
                self.best_threshold = threshold
                self.patience_counter = 0
                
                # Save best model
                if checkpoint_dir:
                    self._save_checkpoint(
                        os.path.join(checkpoint_dir, 'best_model.pt'),
                        epoch,
                        val_metrics,
                    )
            else:
                self.patience_counter += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{train_loss:.4f}',
                'val_mcc': f'{val_metrics["mcc"]:.4f}',
                'best': f'{self.best_val_mcc:.4f}',
                'pat': self.patience_counter,
            })
            
            # Early stopping
            if self.config.patience and self.patience_counter >= self.config.patience:
                tqdm.write(f"Early stopping at epoch {epoch}")
                break
        
        elapsed = time.time() - start_time
        
        # Load best model
        if checkpoint_dir:
            best_path = os.path.join(checkpoint_dir, 'best_model.pt')
            if os.path.exists(best_path):
                self._load_checkpoint(best_path)
        
        return {
            'history': self.history,
            'best_val_mcc': self.best_val_mcc,
            'best_threshold': self.best_threshold,
            'elapsed_seconds': elapsed,
            'final_epoch': epoch,
        }
    
    def _save_checkpoint(
        self,
        path: str,
        epoch: int,
        metrics: Dict,
    ):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_mcc': self.best_val_mcc,
            'best_threshold': self.best_threshold,
            'metrics': metrics,
            'config': self.config.to_dict(),
        }
        torch.save(checkpoint, path)
    
    def _load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_val_mcc = checkpoint.get('best_val_mcc', -np.inf)
        self.best_threshold = checkpoint.get('best_threshold', 0.5)


def train_level5_lite(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: Optional[DataLoader],
    config: Level5LiteConfig,
    checkpoint_dir: str,
    device: torch.device = None,
) -> Dict:
    """Train Level 5-Lite model and evaluate on test set.
    
    Args:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        test_loader: Test DataLoader (optional)
        config: Training configuration
        checkpoint_dir: Directory to save checkpoints
        device: Device to train on
    
    Returns:
        Results dict with metrics and history
    """
    from .model import Level5LiteModel
    
    # Create model
    model = Level5LiteModel(
        protein_input_dim=config.protein_input_dim,
        ligand_input_dim=config.ligand_input_dim,
        hidden_dim=config.hidden_dim,
        num_encoder_layers=config.num_encoder_layers,
        num_cross_attn_layers=config.num_cross_attn_layers,
        num_heads=config.num_heads,
        dropout=config.dropout,
        classifier_dropout=config.classifier_dropout,
    )
    
    # Create trainer
    trainer = Level5LiteTrainer(model, config, device)
    
    # Train
    train_results = trainer.train(train_loader, val_loader, checkpoint_dir)
    
    # Evaluate on test set
    test_metrics = None
    if test_loader is not None:
        _, test_metrics, _ = trainer.evaluate(test_loader, optimize_thresh=False)
    
    # Compile results
    results = {
        'config': config.to_dict(),
        'train_results': train_results,
        'val_metrics': {
            'mcc': train_results['best_val_mcc'],
            'threshold': train_results['best_threshold'],
        },
        'test_metrics': test_metrics,
        'parameter_count': model.count_parameters(),
    }
    
    # Save results
    results_path = os.path.join(checkpoint_dir, 'results.json')
    with open(results_path, 'w') as f:
        # Convert non-serializable items
        save_results = {
            'config': results['config'],
            'best_val_mcc': results['train_results']['best_val_mcc'],
            'best_threshold': results['train_results']['best_threshold'],
            'elapsed_seconds': results['train_results']['elapsed_seconds'],
            'final_epoch': results['train_results']['final_epoch'],
            'test_metrics': results['test_metrics'],
            'parameter_count': results['parameter_count'],
        }
        json.dump(save_results, f, indent=2)
    
    return results
