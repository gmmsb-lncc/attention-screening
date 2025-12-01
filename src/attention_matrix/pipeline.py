"""
Pipeline Orchestrator for Attention Matrix Module.

Single Responsibility: Coordinate training and evaluation workflow.
Implements the main entry point for attention-based analysis.
"""

import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np
import torch
from datetime import datetime

from .config import AttentionMatrixConfig
from .model import CrossAttentionModel, ImprovedCrossAttentionModel
from .dataset import ProteinLigandDataset, create_dataloaders
from .trainer import AttentionTrainer
from .evaluator import AttentionEvaluator
from .splitter import LeakageAwareSplitter


logger = logging.getLogger(__name__)


class AttentionMatrixPipeline:
    """
    Main pipeline orchestrator for Cross-Attention training and analysis.
    
    Coordinates:
    1. Data loading and preprocessing
    2. Leakage-aware data splitting
    3. Model training with early stopping
    4. Evaluation and metrics
    5. Results persistence
    
    Args:
        config: Configuration object or path to JSON config
        output_dir: Directory for results
    """
    
    def __init__(
        self,
        config: Optional[AttentionMatrixConfig] = None,
        output_dir: Optional[str] = None
    ):
        self.config = config or AttentionMatrixConfig()
        
        # Setup output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.output_dir = Path(f'results/attention_matrix_{timestamp}')
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize components
        self.model: Optional[torch.nn.Module] = None
        self.trainer: Optional[AttentionTrainer] = None
        self.evaluator = AttentionEvaluator()
        
        # Store data references
        self.df: Optional[pd.DataFrame] = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        logger.info(f"Pipeline initialized. Output: {self.output_dir}")
    
    def _setup_logging(self):
        """Configure logging for the pipeline."""
        log_file = self.output_dir / 'pipeline.log'
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        
        # Add to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        root_logger.setLevel(logging.INFO)
    
    def _detect_device(self) -> torch.device:
        """Auto-detect best available device."""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            logger.info("Using Apple MPS")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU")
        
        return device
    
    def load_data(
        self,
        data_path: str,
        protein_dir: str,
        ligand_dir: str,
        activity_col: str = 'pchembl_value',
        activity_threshold: Optional[float] = None
    ) -> 'AttentionMatrixPipeline':
        """
        Load and prepare data for training.
        
        Args:
            data_path: Path to TSV/CSV file with compound data
            protein_dir: Directory with protein embeddings (.npy)
            ligand_dir: Directory with ligand embeddings (.npy)
            activity_col: Column with activity values
            activity_threshold: Threshold for active/inactive classification
            
        Returns:
            Self for method chaining
        """
        logger.info("=" * 60)
        logger.info("Loading data...")
        
        # Load DataFrame
        data_path = Path(data_path)
        if data_path.suffix == '.tsv':
            self.df = pd.read_csv(data_path, sep='\t')
        else:
            self.df = pd.read_csv(data_path)
        
        logger.info(f"  Total rows: {len(self.df)}")
        
        # Store paths
        self.protein_dir = Path(protein_dir)
        self.ligand_dir = Path(ligand_dir)
        
        # Apply threshold
        threshold = activity_threshold or self.config.activity_threshold
        self.df['is_active'] = (self.df[activity_col] >= threshold).astype(int)
        
        # Filter valid samples (with embeddings)
        valid_mask = self._filter_valid_samples()
        self.df = self.df[valid_mask].reset_index(drop=True)
        
        logger.info(f"  Valid samples: {len(self.df)}")
        logger.info(f"  Active: {self.df['is_active'].sum()} ({self.df['is_active'].mean()*100:.1f}%)")
        
        # Store activity column name
        self.activity_col = activity_col
        
        return self
    
    def _filter_valid_samples(self) -> pd.Series:
        """Filter samples with existing embeddings."""
        valid = []
        
        for _, row in self.df.iterrows():
            prot_path = self.protein_dir / f"{row['seq_id']}.npy"
            lig_path = self.ligand_dir / f"{row['molecule_chembl_id']}.npy"
            
            is_valid = prot_path.exists() and lig_path.exists()
            is_valid = is_valid and pd.notna(row.get(self.activity_col, None))
            valid.append(is_valid)
        
        return pd.Series(valid)
    
    def split_data(
        self,
        split_type: str = 'leakage_aware',
        n_clusters: Optional[int] = None,
        split_dir: Optional[str] = None
    ) -> 'AttentionMatrixPipeline':
        """
        Split data into train/val/test sets.
        
        Args:
            split_type: 'leakage_aware' or 'simple'
            n_clusters: Number of protein clusters (auto if None)
            split_dir: Load pre-computed split from directory
            
        Returns:
            Self for method chaining
        """
        logger.info("=" * 60)
        logger.info(f"Splitting data ({split_type})...")
        
        if split_dir and Path(split_dir).exists():
            # Load existing split
            train_idx, val_idx, test_idx = LeakageAwareSplitter.load_split(
                Path(split_dir)
            )
            logger.info(f"  Loaded split from: {split_dir}")
        else:
            # Perform new split
            if split_type == 'leakage_aware':
                splitter = LeakageAwareSplitter(
                    n_clusters=n_clusters,
                    test_size=0.1,
                    val_size=0.1,
                    random_state=42
                )
                train_idx, val_idx, test_idx = splitter.split(
                    self.df, self.protein_dir
                )
                
                # Save split
                splitter.save_split(
                    self.output_dir / 'splits',
                    train_idx, val_idx, test_idx
                )
            else:
                from .splitter import SimpleSplitter
                splitter = SimpleSplitter(test_size=0.1, val_size=0.1)
                train_idx, val_idx, test_idx = splitter.split(self.df)
        
        # Create datasets
        train_df = self.df.iloc[train_idx].reset_index(drop=True)
        val_df = self.df.iloc[val_idx].reset_index(drop=True)
        test_df = self.df.iloc[test_idx].reset_index(drop=True)
        
        # Create data loaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            protein_dir=str(self.protein_dir),
            ligand_dir=str(self.ligand_dir),
            activity_col=self.activity_col,
            batch_size=self.config.batch_size,
            max_protein_len=self.config.max_protein_len,
            max_ligand_len=self.config.max_ligand_len,
            num_workers=4
        )
        
        logger.info(f"  Train batches: {len(self.train_loader)}")
        logger.info(f"  Val batches: {len(self.val_loader)}")
        logger.info(f"  Test batches: {len(self.test_loader)}")
        
        return self
    
    def build_model(
        self,
        model_type: str = 'improved',
        checkpoint_path: Optional[str] = None
    ) -> 'AttentionMatrixPipeline':
        """
        Build or load the model.
        
        Args:
            model_type: 'basic' or 'improved'
            checkpoint_path: Path to load pre-trained weights
            
        Returns:
            Self for method chaining
        """
        logger.info("=" * 60)
        logger.info(f"Building model ({model_type})...")
        
        # Select model class
        if model_type == 'improved':
            ModelClass = ImprovedCrossAttentionModel
        else:
            ModelClass = CrossAttentionModel
        
        # Create model
        self.model = ModelClass(
            protein_dim=self.config.protein_dim,
            ligand_dim=self.config.ligand_dim,
            hidden_dim=self.config.hidden_dim,
            num_heads=self.config.num_heads,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout
        )
        
        # Load checkpoint if provided
        if checkpoint_path and Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"  Loaded checkpoint: {checkpoint_path}")
        
        # Count parameters
        n_params = sum(p.numel() for p in self.model.parameters())
        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"  Total parameters: {n_params:,}")
        logger.info(f"  Trainable: {n_trainable:,}")
        
        return self
    
    def train(
        self,
        epochs: Optional[int] = None,
        patience: Optional[int] = None,
        classification_weight: float = 0.3,
        regression_weight: float = 0.7
    ) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            epochs: Number of epochs (uses config if None)
            patience: Early stopping patience (uses config if None)
            classification_weight: Weight for classification loss
            regression_weight: Weight for regression loss
            
        Returns:
            Training history dictionary
        """
        logger.info("=" * 60)
        logger.info("Starting training...")
        
        epochs = epochs or self.config.epochs
        patience = patience or self.config.early_stopping_patience
        
        device = self._detect_device()
        self.model = self.model.to(device)
        
        # Create trainer
        self.trainer = AttentionTrainer(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            config=self.config,
            device=device,
            output_dir=self.output_dir / 'models'
        )
        
        # Train
        history = self.trainer.train(
            epochs=epochs,
            patience=patience,
            classification_weight=classification_weight,
            regression_weight=regression_weight
        )
        
        # Save training history
        with open(self.output_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        return history
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate model on test set.
        
        Returns:
            Evaluation metrics dictionary
        """
        logger.info("=" * 60)
        logger.info("Evaluating on test set...")
        
        device = next(self.model.parameters()).device
        
        # Get predictions
        self.model.eval()
        all_preds_cls = []
        all_preds_reg = []
        all_labels_cls = []
        all_labels_reg = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                protein_emb = batch['protein_embedding'].to(device)
                ligand_emb = batch['ligand_embedding'].to(device)
                
                # Forward pass
                outputs = self.model(protein_emb, ligand_emb)
                
                all_preds_cls.append(outputs['classification'].cpu().numpy())
                all_preds_reg.append(outputs['regression'].cpu().numpy())
                all_labels_cls.append(batch['is_active'].numpy())
                all_labels_reg.append(batch['activity'].numpy())
        
        # Concatenate
        cls_preds = np.concatenate(all_preds_cls)
        reg_preds = np.concatenate(all_preds_reg)
        cls_labels = np.concatenate(all_labels_cls)
        reg_labels = np.concatenate(all_labels_reg)
        
        # Compute metrics
        metrics = self.evaluator.evaluate(
            cls_preds=cls_preds,
            cls_labels=cls_labels,
            reg_preds=reg_preds,
            reg_labels=reg_labels
        )
        
        # Save metrics
        with open(self.output_dir / 'test_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Print summary
        logger.info("Test Results:")
        logger.info("  Classification:")
        logger.info(f"    Accuracy: {metrics['classification']['accuracy']:.4f}")
        logger.info(f"    Precision: {metrics['classification']['precision']:.4f}")
        logger.info(f"    Recall: {metrics['classification']['recall']:.4f}")
        logger.info(f"    F1-Score: {metrics['classification']['f1_score']:.4f}")
        logger.info(f"    ROC-AUC: {metrics['classification']['roc_auc']:.4f}")
        logger.info("  Regression:")
        logger.info(f"    MAE: {metrics['regression']['mae']:.4f}")
        logger.info(f"    RMSE: {metrics['regression']['rmse']:.4f}")
        logger.info(f"    R²: {metrics['regression']['r2']:.4f}")
        logger.info(f"    Pearson: {metrics['regression']['pearson']:.4f}")
        
        return metrics
    
    def run(
        self,
        data_path: str,
        protein_dir: str,
        ligand_dir: str,
        split_type: str = 'leakage_aware',
        model_type: str = 'improved',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run complete pipeline: load, split, build, train, evaluate.
        
        Args:
            data_path: Path to data file
            protein_dir: Directory with protein embeddings
            ligand_dir: Directory with ligand embeddings
            split_type: 'leakage_aware' or 'simple'
            model_type: 'basic' or 'improved'
            **kwargs: Additional arguments
            
        Returns:
            Results dictionary with all metrics
        """
        logger.info("=" * 60)
        logger.info("ATTENTION MATRIX PIPELINE")
        logger.info("=" * 60)
        
        # Save config
        self.config.save(self.output_dir / 'config.json')
        
        # Run pipeline steps
        self.load_data(
            data_path=data_path,
            protein_dir=protein_dir,
            ligand_dir=ligand_dir,
            activity_col=kwargs.get('activity_col', 'pchembl_value'),
            activity_threshold=kwargs.get('activity_threshold', None)
        )
        
        self.split_data(
            split_type=split_type,
            n_clusters=kwargs.get('n_clusters', None),
            split_dir=kwargs.get('split_dir', None)
        )
        
        self.build_model(
            model_type=model_type,
            checkpoint_path=kwargs.get('checkpoint_path', None)
        )
        
        history = self.train(
            epochs=kwargs.get('epochs', None),
            patience=kwargs.get('patience', None)
        )
        
        metrics = self.evaluate()
        
        # Compile results
        results = {
            'config': self.config.to_dict(),
            'training': history,
            'test_metrics': metrics,
            'output_dir': str(self.output_dir)
        }
        
        # Save final results
        with open(self.output_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info("=" * 60)
        logger.info(f"Pipeline complete. Results saved to: {self.output_dir}")
        logger.info("=" * 60)
        
        return results
    
    def run_with_precomputed_embeddings(
        self,
        protein_embeddings,  # np.ndarray or List[np.ndarray]
        ligand_embeddings,   # np.ndarray or List[np.ndarray]
        binary_labels: np.ndarray,
        regression_targets: np.ndarray,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        test_idx: np.ndarray,
        model_type: str = 'improved'
    ) -> Dict[str, Any]:
        """
        Run pipeline with pre-computed embeddings and splits.
        
        Args:
            protein_embeddings: Array/List of protein embeddings
                - Vector mode: (N, dim) array
                - Matrix mode: List of (seq_len, dim) arrays
            ligand_embeddings: Array/List of ligand embeddings
                - Vector mode: (N, dim) array
                - Matrix mode: List of (tokens, dim) arrays
            binary_labels: Binary classification labels (N,)
            regression_targets: Regression targets (N,)
            train_idx: Training indices
            val_idx: Validation indices
            test_idx: Test indices
            model_type: 'basic' or 'improved'
            
        Returns:
            Results dictionary with metrics
        """
        from torch.utils.data import DataLoader
        from torch.nn.utils.rnn import pad_sequence
        import time
        
        logger.info("=" * 60)
        logger.info("ATTENTION MATRIX PIPELINE (Pre-computed)")
        logger.info("=" * 60)
        
        # Detect if we're in matrix mode (list of arrays) or vector mode (single array)
        is_matrix_mode = isinstance(protein_embeddings, list)
        logger.info(f"  Mode: {'matrix' if is_matrix_mode else 'vector'}")
        
        # Save config
        self.config.save(self.output_dir / 'config.json')
        
        # Detect device
        if self.config.device == 'auto':
            device = self._detect_device()
        else:
            device = torch.device(self.config.device)
        
        if is_matrix_mode:
            # Matrix mode: use custom collate with padding
            def collate_fn(batch):
                """Custom collate function for variable-length sequences."""
                prot_list, lig_list, cls_list, reg_list = [], [], [], []
                for prot, lig, cls_label, reg_target in batch:
                    prot_list.append(torch.from_numpy(prot).float())
                    lig_list.append(torch.from_numpy(lig).float())
                    cls_list.append(cls_label)
                    reg_list.append(reg_target)
                
                # Pad sequences to max length in batch
                prot_padded = pad_sequence(prot_list, batch_first=True)
                lig_padded = pad_sequence(lig_list, batch_first=True)
                
                return (
                    prot_padded,
                    lig_padded,
                    torch.tensor(cls_list, dtype=torch.float),
                    torch.tensor(reg_list, dtype=torch.float)
                )
            
            class EmbeddingDataset:
                def __init__(self, prot_embs, lig_embs, cls_labels, reg_targets, indices):
                    self.prot_embs = prot_embs
                    self.lig_embs = lig_embs
                    self.cls_labels = cls_labels
                    self.reg_targets = reg_targets
                    self.indices = indices
                
                def __len__(self):
                    return len(self.indices)
                
                def __getitem__(self, idx):
                    i = self.indices[idx]
                    return (
                        self.prot_embs[i],
                        self.lig_embs[i],
                        self.cls_labels[i],
                        self.reg_targets[i]
                    )
            
            train_dataset = EmbeddingDataset(
                protein_embeddings, ligand_embeddings, 
                binary_labels, regression_targets, train_idx
            )
            val_dataset = EmbeddingDataset(
                protein_embeddings, ligand_embeddings,
                binary_labels, regression_targets, val_idx
            )
            test_dataset = EmbeddingDataset(
                protein_embeddings, ligand_embeddings,
                binary_labels, regression_targets, test_idx
            )
            
            train_loader = DataLoader(
                train_dataset, batch_size=self.config.batch_size, 
                shuffle=True, collate_fn=collate_fn
            )
            val_loader = DataLoader(
                val_dataset, batch_size=self.config.batch_size, 
                shuffle=False, collate_fn=collate_fn
            )
            test_loader = DataLoader(
                test_dataset, batch_size=self.config.batch_size, 
                shuffle=False, collate_fn=collate_fn
            )
        else:
            # Vector mode: use TensorDataset
            from torch.utils.data import TensorDataset
            
            def make_loader(idx, shuffle=False):
                prot = torch.from_numpy(protein_embeddings[idx]).float().unsqueeze(1)
                lig = torch.from_numpy(ligand_embeddings[idx]).float().unsqueeze(1)
                cls = torch.from_numpy(binary_labels[idx]).float()
                reg = torch.from_numpy(regression_targets[idx]).float()
                dataset = TensorDataset(prot, lig, cls, reg)
                return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=shuffle)
            
            train_loader = make_loader(train_idx, shuffle=True)
            val_loader = make_loader(val_idx)
            test_loader = make_loader(test_idx)
        
        logger.info(f"  Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")
        
        # Build model
        if model_type == 'improved':
            self.model = ImprovedCrossAttentionModel(
                protein_dim=self.config.protein_dim,
                ligand_dim=self.config.ligand_dim,
                hidden_dim=self.config.hidden_dim,
                num_heads=self.config.num_heads,
                num_layers=self.config.num_layers,
                dropout=self.config.dropout
            )
        else:
            self.model = CrossAttentionModel(
                protein_dim=self.config.protein_dim,
                ligand_dim=self.config.ligand_dim,
                hidden_dim=self.config.hidden_dim,
                num_heads=self.config.num_heads,
                dropout=self.config.dropout
            )
        
        self.model = self.model.to(device)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"  Model: {model_type} ({total_params:,} params)")
        
        # Training setup
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.epochs)
        cls_criterion = torch.nn.BCEWithLogitsLoss()
        reg_criterion = torch.nn.HuberLoss(delta=1.0)
        
        cls_weight = self.config.classification_weight
        reg_weight = self.config.regression_weight
        
        # Training loop
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience_counter = 0
        
        logger.info(f"\n  Training for up to {self.config.epochs} epochs...")
        
        for epoch in range(self.config.epochs):
            # Train
            self.model.train()
            train_losses = []
            for prot, lig, cls_lbl, reg_lbl in train_loader:
                prot, lig = prot.to(device), lig.to(device)
                cls_lbl, reg_lbl = cls_lbl.to(device), reg_lbl.to(device)
                
                optimizer.zero_grad()
                outputs = self.model(prot, lig)
                
                loss = cls_weight * cls_criterion(outputs['classification'], cls_lbl) + \
                       reg_weight * reg_criterion(outputs['regression'], reg_lbl)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_losses.append(loss.item())
            
            # Validate
            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for prot, lig, cls_lbl, reg_lbl in val_loader:
                    prot, lig = prot.to(device), lig.to(device)
                    cls_lbl, reg_lbl = cls_lbl.to(device), reg_lbl.to(device)
                    outputs = self.model(prot, lig)
                    loss = cls_weight * cls_criterion(outputs['classification'], cls_lbl) + \
                           reg_weight * reg_criterion(outputs['regression'], reg_lbl)
                    val_losses.append(loss.item())
            
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            scheduler.step()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.output_dir / 'best_model.pt')
            else:
                patience_counter += 1
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(f"    Epoch {epoch+1}: train={train_loss:.4f}, val={val_loss:.4f}")
            
            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"  Early stopping at epoch {epoch+1}")
                break
        
        # Load best model and evaluate
        self.model.load_state_dict(torch.load(self.output_dir / 'best_model.pt'))
        self.model.eval()
        
        # Test evaluation
        all_cls_preds, all_cls_probs, all_cls_labels = [], [], []
        all_reg_preds, all_reg_targets = [], []
        
        with torch.no_grad():
            for prot, lig, cls_lbl, reg_lbl in test_loader:
                prot, lig = prot.to(device), lig.to(device)
                outputs = self.model(prot, lig)
                
                probs = torch.sigmoid(outputs['classification']).cpu().numpy()
                preds = (probs >= 0.5).astype(int)
                
                all_cls_probs.extend(probs.tolist())
                all_cls_preds.extend(preds.tolist())
                all_cls_labels.extend(cls_lbl.numpy().tolist())
                all_reg_preds.extend(outputs['regression'].cpu().numpy().tolist())
                all_reg_targets.extend(reg_lbl.numpy().tolist())
        
        # Compute metrics
        from sklearn.metrics import (
            accuracy_score, roc_auc_score, f1_score, matthews_corrcoef,
            r2_score, mean_squared_error, mean_absolute_error, explained_variance_score
        )
        from scipy.stats import pearsonr, spearmanr, kendalltau
        
        # Calculate regression metrics
        reg_targets = np.array(all_reg_targets)
        reg_preds = np.array(all_reg_preds)
        
        # Concordance Correlation Coefficient (Lin's CCC)
        mean_true = np.mean(reg_targets)
        mean_pred = np.mean(reg_preds)
        var_true = np.var(reg_targets)
        var_pred = np.var(reg_preds)
        covariance = np.mean((reg_targets - mean_true) * (reg_preds - mean_pred))
        ccc = (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred) ** 2)
        
        metrics = {
            'classification': {
                'accuracy': accuracy_score(all_cls_labels, all_cls_preds),
                'roc_auc': roc_auc_score(all_cls_labels, all_cls_probs),
                'f1': f1_score(all_cls_labels, all_cls_preds),
                'mcc': matthews_corrcoef(all_cls_labels, all_cls_preds)
            },
            'regression': {
                'r2': r2_score(reg_targets, reg_preds),
                'pearson_r': pearsonr(reg_targets, reg_preds)[0],
                'pearson_p': pearsonr(reg_targets, reg_preds)[1],
                'spearman_r': spearmanr(reg_targets, reg_preds)[0],
                'spearman_p': spearmanr(reg_targets, reg_preds)[1],
                'kendall_tau': kendalltau(reg_targets, reg_preds)[0],
                'kendall_p': kendalltau(reg_targets, reg_preds)[1],
                'ccc': float(ccc),
                'rmse': np.sqrt(mean_squared_error(reg_targets, reg_preds)),
                'mae': mean_absolute_error(reg_targets, reg_preds),
                'explained_variance': explained_variance_score(reg_targets, reg_preds)
            }
        }
        
        # Compile results
        results = {
            'config': self.config.to_dict(),
            'training': history,
            'metrics': metrics,
            'output_dir': str(self.output_dir)
        }
        
        # Save results
        with open(self.output_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"\n  Results saved to: {self.output_dir}")
        
        return results
    
    @classmethod
    def from_config_file(cls, config_path: str) -> 'AttentionMatrixPipeline':
        """Create pipeline from JSON config file."""
        config = AttentionMatrixConfig.load(config_path)
        return cls(config=config)
