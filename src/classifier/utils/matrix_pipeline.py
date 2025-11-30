"""
Pipeline Integration for Matrix-Based Affinity Prediction.

This module provides high-level functions to:
1. Generate matrix embeddings from proteins and ligands
2. Train the CrossAttentionAffinityModel
3. Make predictions on new data
4. Comprehensive evaluation with metrics matching vector pipeline

Author: DockTKinase Team
Date: November 2025
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
import pandas as pd
import numpy as np
import torch
from dataclasses import dataclass

# Import metrics
from .matrix_metrics import MatrixMetricsCalculator, ClassificationMetrics, RegressionMetrics

logger = logging.getLogger(__name__)


@dataclass
class MatrixPipelineConfig:
    """Configuration for the matrix-based affinity prediction pipeline."""
    
    # Data paths
    data_path: str = None
    output_dir: str = 'results/matrix_pipeline'
    
    # Embedding configuration
    protein_model: str = 'esm2_t33'  # Options: esm2_t33, esm2_t36, esmc, boltz
    ligand_model: str = 'smited'
    
    # Matrix directories (auto-generated if not specified)
    protein_matrix_dir: Optional[str] = None
    ligand_matrix_dir: Optional[str] = None
    
    # Vector embedding directories (for stratification)
    protein_vector_dir: Optional[str] = None
    ligand_vector_dir: Optional[str] = None
    
    # Model configuration
    hidden_dim: int = 256
    num_cnn_layers: int = 3
    num_cross_attn_layers: int = 2
    num_heads: int = 8
    ff_dim: int = 1024
    dropout: float = 0.1
    
    # Training configuration
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 100
    patience: int = 10
    
    # Loss weights
    classification_weight: float = 1.0
    regression_weight: float = 1.0
    
    # Stratification configuration
    use_stratification: bool = True  # Use embedding-based stratification
    stratification_method: str = 'adaptive'  # 'adaptive', 'dbscan', 'hierarchical', 'simple'
    adaptive_method: str = 'target'  # 'silhouette', 'elbow', 'target', 'leakage_aware'
    similarity_threshold: Optional[float] = None  # Manual threshold (None = auto)
    stratify_by: str = 'both'  # 'protein', 'ligand', 'both'
    protein_weight: float = 0.6  # Weight for protein in multi-view
    ligand_weight: float = 0.4  # Weight for ligand in multi-view
    
    # Device
    device: str = 'auto'
    
    def get_embedding_dims(self) -> Tuple[int, int]:
        """Get embedding dimensions based on model choices."""
        protein_dims = {
            'esm2_t6': 320,
            'esm2_t12': 480,
            'esm2_t30': 640,
            'esm2_t33': 1280,
            'esm2_t36': 2560,
            'esm2_t48': 5120,
            'esmc_300m': 960,
            'esmc_600m': 1152,
            'esmc': 1152,
            'boltz': 384,
        }
        
        ligand_dims = {
            'smited': 768,
            'smi-ted': 768,
        }
        
        protein_dim = protein_dims.get(self.protein_model, 2560)
        ligand_dim = ligand_dims.get(self.ligand_model, 768)
        
        return protein_dim, ligand_dim


class MatrixAffinityPipeline:
    """
    End-to-end pipeline for matrix-based protein-ligand affinity prediction.
    
    This pipeline uses:
    - Matrix embeddings [seq_len, dim] instead of vectors [dim]
    - CNN encoders to process variable-length sequences
    - Cross-attention to model protein-ligand interactions
    - Multi-task learning for classification and regression
    """
    
    def __init__(self, config: MatrixPipelineConfig):
        """Initialize pipeline with configuration."""
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set default matrix directories
        if config.protein_matrix_dir is None:
            config.protein_matrix_dir = str(self.output_dir / 'protein_matrix_embeddings')
        if config.ligand_matrix_dir is None:
            config.ligand_matrix_dir = str(self.output_dir / 'ligand_matrix_embeddings')
        
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
        
        self.model = None
        self.trainer = None
        
        logger.info(f"MatrixAffinityPipeline initialized")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Device: {self.device}")
    
    def generate_embeddings(
        self,
        df: pd.DataFrame,
        protein_col: str = 'sequence',
        protein_id_col: str = 'seq_id',
        smiles_col: str = 'smiles',
        ligand_id_col: str = 'chembl_id',
        force_regenerate: bool = False
    ) -> Tuple[Path, Path]:
        """
        Generate matrix embeddings for proteins and ligands.
        
        Args:
            df: DataFrame with protein sequences and SMILES
            protein_col: Column name for protein sequences
            protein_id_col: Column name for protein IDs
            smiles_col: Column name for SMILES strings
            ligand_id_col: Column name for ligand IDs
            force_regenerate: If True, regenerate even if exists
        
        Returns:
            Paths to protein and ligand matrix directories
        """
        from src.build.embeddings import ProteinEmbedding, LigandEmbedding
        
        protein_dir = Path(self.config.protein_matrix_dir)
        ligand_dir = Path(self.config.ligand_matrix_dir)
        
        protein_dir.mkdir(parents=True, exist_ok=True)
        ligand_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare protein data
        unique_proteins = df[[protein_id_col, protein_col]].drop_duplicates()
        protein_df = unique_proteins.rename(columns={
            protein_id_col: 'seq_id',
            protein_col: 'sequence'
        })
        
        # Prepare ligand data
        unique_ligands = df[[ligand_id_col, smiles_col]].drop_duplicates()
        ligand_df = unique_ligands.rename(columns={
            ligand_id_col: 'chembl_id',
            smiles_col: 'smiles'
        })
        
        # Check if embeddings already exist
        existing_proteins = set(f.stem.replace('_matrix', '') for f in protein_dir.glob('*_matrix.npy'))
        existing_ligands = set(f.stem.replace('_matrix', '') for f in ligand_dir.glob('*_matrix.npy'))
        
        proteins_to_generate = set(protein_df['seq_id']) - existing_proteins if not force_regenerate else set(protein_df['seq_id'])
        ligands_to_generate = set(ligand_df['chembl_id']) - existing_ligands if not force_regenerate else set(ligand_df['chembl_id'])
        
        logger.info(f"Proteins to generate: {len(proteins_to_generate)}")
        logger.info(f"Ligands to generate: {len(ligands_to_generate)}")
        
        # Generate protein embeddings
        if proteins_to_generate:
            protein_subset = protein_df[protein_df['seq_id'].isin(proteins_to_generate)]
            
            logger.info(f"Generating {len(protein_subset)} protein matrix embeddings...")
            embedder = ProteinEmbedding(strategy_name=self.config.protein_model)
            embedder.generate_embeddings(
                protein_subset,
                save_matrix=True,
                matrix_output_dir=str(protein_dir)
            )
        
        # Generate ligand embeddings
        if ligands_to_generate:
            ligand_subset = ligand_df[ligand_df['chembl_id'].isin(ligands_to_generate)]
            
            logger.info(f"Generating {len(ligand_subset)} ligand matrix embeddings...")
            embedder = LigandEmbedding(strategy_name=self.config.ligand_model)
            embedder.generate_embeddings(
                ligand_subset,
                save_matrix=True,
                matrix_output_dir=str(ligand_dir)
            )
        
        return protein_dir, ligand_dir
    
    def _load_vectors_from_matrices(
        self,
        df: pd.DataFrame,
        protein_id_col: str = 'seq_id',
        ligand_id_col: str = 'chembl_id'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load vector embeddings by mean-pooling matrices.
        
        This enables stratification using the same embeddings as training.
        
        Args:
            df: DataFrame with IDs
            protein_id_col: Column for protein IDs
            ligand_id_col: Column for ligand IDs
            
        Returns:
            protein_vectors [N, protein_dim], ligand_vectors [N, ligand_dim]
        """
        protein_dir = Path(self.config.protein_matrix_dir)
        ligand_dir = Path(self.config.ligand_matrix_dir)
        
        protein_vectors = []
        ligand_vectors = []
        
        for _, row in df.iterrows():
            # Load protein matrix and mean-pool
            prot_file = protein_dir / f"{row[protein_id_col]}_matrix.npy"
            if prot_file.exists():
                prot_matrix = np.load(prot_file)
                prot_vector = prot_matrix.mean(axis=0)  # Mean over seq_len
            else:
                raise FileNotFoundError(f"Protein matrix not found: {prot_file}")
            
            # Load ligand matrix and mean-pool
            lig_file = ligand_dir / f"{row[ligand_id_col]}_matrix.npy"
            if lig_file.exists():
                lig_matrix = np.load(lig_file)
                lig_vector = lig_matrix.mean(axis=0)  # Mean over tokens
            else:
                raise FileNotFoundError(f"Ligand matrix not found: {lig_file}")
            
            protein_vectors.append(prot_vector)
            ligand_vectors.append(lig_vector)
        
        return np.array(protein_vectors), np.array(ligand_vectors)
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        label_col: str = 'label',
        regression_col: Optional[str] = 'pchembl_value',
        val_split: float = 0.1,
        test_split: float = 0.1,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare and split data for training.
        
        Uses embedding-based stratification if enabled in config.
        This ensures train/val/test have minimal data leakage based on
        protein-ligand similarity.
        
        Args:
            df: DataFrame with embeddings
            label_col: Column for classification labels
            regression_col: Column for regression targets (optional)
            val_split: Validation set fraction
            test_split: Test set fraction
            random_state: Random seed
        
        Returns:
            train_df, val_df, test_df
        """
        # Ensure required columns
        required_cols = ['seq_id', 'chembl_id', label_col]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Use stratification system if enabled
        if self.config.use_stratification and self.config.stratification_method != 'simple':
            return self._stratified_split(df, label_col, val_split, test_split, random_state)
        else:
            return self._simple_split(df, label_col, val_split, test_split, random_state)
    
    def _simple_split(
        self,
        df: pd.DataFrame,
        label_col: str,
        val_split: float,
        test_split: float,
        random_state: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Simple random split with label stratification."""
        from sklearn.model_selection import train_test_split
        
        train_val_df, test_df = train_test_split(
            df, test_size=test_split, random_state=random_state, stratify=df[label_col]
        )
        
        val_size = val_split / (1 - test_split)
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_size, random_state=random_state, stratify=train_val_df[label_col]
        )
        
        logger.info(f"Simple split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        
        return train_df, val_df, test_df
    
    def _stratified_split(
        self,
        df: pd.DataFrame,
        label_col: str,
        val_split: float,
        test_split: float,
        random_state: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Embedding-based stratified split using the Stratifier system.
        
        This prevents data leakage by grouping similar protein-ligand pairs
        and ensuring they end up in the same split.
        """
        from src.build.stratification import Stratifier
        
        logger.info(f"Using embedding-based stratification: method={self.config.stratification_method}")
        logger.info(f"  Adaptive method: {self.config.adaptive_method}")
        logger.info(f"  Stratify by: {self.config.stratify_by}")
        
        # Load vector embeddings (mean-pooled from matrices)
        try:
            protein_vectors, ligand_vectors = self._load_vectors_from_matrices(df)
            logger.info(f"Loaded vectors: protein={protein_vectors.shape}, ligand={ligand_vectors.shape}")
        except FileNotFoundError as e:
            logger.warning(f"Could not load matrices for stratification: {e}")
            logger.warning("Falling back to simple split")
            return self._simple_split(df, label_col, val_split, test_split, random_state)
        
        # Create stratifier
        stratifier = Stratifier(
            clustering_algorithm=self.config.stratification_method,
            similarity_threshold=self.config.similarity_threshold,
            adaptive_method=self.config.adaptive_method,
            test_size=test_split,
            val_size=val_split,
            output_dir=str(self.output_dir / 'stratification')
        )
        
        # Get labels
        labels = df[label_col].values
        
        # Perform stratified split based on config
        if self.config.stratify_by == 'protein':
            train_idx, val_idx, test_idx = stratifier.stratified_split(
                protein_vectors, labels,
                test_size=test_split,
                val_size=val_split,
                output_dir=str(self.output_dir / 'stratification')
            )
        elif self.config.stratify_by == 'ligand':
            train_idx, val_idx, test_idx = stratifier.stratified_split(
                ligand_vectors, labels,
                test_size=test_split,
                val_size=val_split,
                output_dir=str(self.output_dir / 'stratification')
            )
        else:  # 'both' - multi-view
            train_idx, val_idx, test_idx = stratifier.multi_view_stratified_split(
                protein_vectors, ligand_vectors, labels,
                test_size=test_split,
                val_size=val_split,
                protein_weight=self.config.protein_weight,
                ligand_weight=self.config.ligand_weight,
                output_dir=str(self.output_dir / 'stratification')
            )
        
        # Create DataFrames from indices
        df_indexed = df.reset_index(drop=True)
        train_df = df_indexed.iloc[train_idx].copy()
        val_df = df_indexed.iloc[val_idx].copy()
        test_df = df_indexed.iloc[test_idx].copy()
        
        logger.info(f"Stratified split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        
        # Log cluster info
        cluster_info = stratifier.get_cluster_info()
        if cluster_info:
            logger.info(f"  Clusters: {cluster_info.get('n_clusters', 'N/A')}")
            logger.info(f"  Noise points: {cluster_info.get('n_noise_points', 0)}")
        
        return train_df, val_df, test_df
    
    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Train the CrossAttentionAffinityModel.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame (optional)
        
        Returns:
            Training history and metrics
        """
        from .matrix_dataloader import create_matrix_dataloader
        from .matrix_trainer import CrossAttentionTrainer, TrainingConfig
        from ..models.cross_attention_model import CrossAttentionAffinityModel
        
        protein_dim, ligand_dim = self.config.get_embedding_dims()
        
        # Create data loaders
        train_loader = create_matrix_dataloader(
            train_df,
            self.config.protein_matrix_dir,
            self.config.ligand_matrix_dir,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        val_loader = create_matrix_dataloader(
            val_df,
            self.config.protein_matrix_dir,
            self.config.ligand_matrix_dir,
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
        test_loader = None
        if test_df is not None:
            test_loader = create_matrix_dataloader(
                test_df,
                self.config.protein_matrix_dir,
                self.config.ligand_matrix_dir,
                batch_size=self.config.batch_size,
                shuffle=False
            )
        
        # Create model
        self.model = CrossAttentionAffinityModel(
            protein_dim=protein_dim,
            ligand_dim=ligand_dim,
            hidden_dim=self.config.hidden_dim,
            num_cnn_layers=self.config.num_cnn_layers,
            num_cross_attn_layers=self.config.num_cross_attn_layers,
            num_heads=self.config.num_heads,
            ff_dim=self.config.ff_dim,
            dropout=self.config.dropout
        )
        
        logger.info(f"Model parameters: {self.model.count_parameters():,}")
        
        # Create training config
        training_config = TrainingConfig(
            protein_dim=protein_dim,
            ligand_dim=ligand_dim,
            hidden_dim=self.config.hidden_dim,
            num_cnn_layers=self.config.num_cnn_layers,
            num_cross_attn_layers=self.config.num_cross_attn_layers,
            num_heads=self.config.num_heads,
            ff_dim=self.config.ff_dim,
            model_dropout=self.config.dropout,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            num_epochs=self.config.num_epochs,
            patience=self.config.patience,
            classification_weight=self.config.classification_weight,
            regression_weight=self.config.regression_weight,
            save_dir=str(self.output_dir / 'checkpoints'),
            device=str(self.device)
        )
        
        # Create trainer
        self.trainer = CrossAttentionTrainer(
            model=self.model,
            config=training_config,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader
        )
        
        # Train
        history = self.trainer.train()
        
        # Save config
        import json
        config_path = self.output_dir / 'pipeline_config.json'
        with open(config_path, 'w') as f:
            json.dump({
                'protein_model': self.config.protein_model,
                'ligand_model': self.config.ligand_model,
                'protein_dim': protein_dim,
                'ligand_dim': ligand_dim,
                'hidden_dim': self.config.hidden_dim,
                'num_cnn_layers': self.config.num_cnn_layers,
                'num_cross_attn_layers': self.config.num_cross_attn_layers,
                'num_heads': self.config.num_heads,
                'ff_dim': self.config.ff_dim,
                'dropout': self.config.dropout,
            }, f, indent=2)
        
        return history
    
    @torch.no_grad()
    def predict(
        self,
        df: pd.DataFrame,
        return_embeddings: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Make predictions on new data.
        
        Args:
            df: DataFrame with seq_id and chembl_id
            return_embeddings: If True, return intermediate embeddings
        
        Returns:
            Dictionary with predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first or load a checkpoint.")
        
        from .matrix_dataloader import create_matrix_dataloader
        
        loader = create_matrix_dataloader(
            df,
            self.config.protein_matrix_dir,
            self.config.ligand_matrix_dir,
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
        self.model.eval()
        
        all_cls_preds = []
        all_reg_preds = []
        all_embeddings = []
        
        for batch in loader:
            protein_matrix = batch['protein_matrix'].to(self.device)
            ligand_matrix = batch['ligand_matrix'].to(self.device)
            protein_mask = batch['protein_mask'].to(self.device)
            ligand_mask = batch['ligand_mask'].to(self.device)
            
            output = self.model(
                protein_matrix, ligand_matrix,
                protein_mask, ligand_mask,
                return_embeddings=return_embeddings
            )
            
            cls_probs = torch.sigmoid(output['classification'])
            all_cls_preds.append(cls_probs.cpu().numpy())
            all_reg_preds.append(output['regression'].cpu().numpy())
            
            if return_embeddings and 'embeddings' in output:
                all_embeddings.append(output['embeddings'].cpu().numpy())
        
        result = {
            'classification_prob': np.concatenate(all_cls_preds),
            'classification_pred': (np.concatenate(all_cls_preds) >= 0.5).astype(int),
            'regression_pred': np.concatenate(all_reg_preds)
        }
        
        if return_embeddings and all_embeddings:
            result['embeddings'] = np.concatenate(all_embeddings)
        
        return result
    
    def load_model(self, checkpoint_path: str):
        """Load model from checkpoint."""
        from ..models.cross_attention_model import CrossAttentionAffinityModel
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = checkpoint.get('config', {})
        
        protein_dim = config.get('protein_dim', 2560)
        ligand_dim = config.get('ligand_dim', 768)
        
        self.model = CrossAttentionAffinityModel(
            protein_dim=protein_dim,
            ligand_dim=ligand_dim,
            hidden_dim=config.get('hidden_dim', 256),
            num_cnn_layers=config.get('num_cnn_layers', 3),
            num_cross_attn_layers=config.get('num_cross_attn_layers', 2),
            num_heads=config.get('num_heads', 8),
            ff_dim=config.get('ff_dim', 1024),
            dropout=config.get('model_dropout', 0.1)
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Loaded model from {checkpoint_path}")
    
    def save_predictions(
        self,
        df: pd.DataFrame,
        predictions: Dict[str, np.ndarray],
        output_path: str
    ):
        """Save predictions to CSV."""
        result_df = df.copy()
        result_df['pred_binding_prob'] = predictions['classification_prob'].flatten()
        result_df['pred_binding'] = predictions['classification_pred'].flatten()
        result_df['pred_affinity'] = predictions['regression_pred'].flatten()
        
        result_df.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")

    @torch.no_grad()
    def evaluate_comprehensive(
        self,
        df: pd.DataFrame,
        threshold: float = 0.5,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation with full metrics matching vector pipeline.
        
        This method provides the same metrics as the vector-based pipeline:
        
        Classification Metrics:
        - Accuracy, Precision, Recall, F1 Score
        - ROC-AUC, Average Precision
        - Brier Score, MCC (Matthews Correlation Coefficient)
        - Specificity, Sensitivity
        - Fbeta variants (β=0.5, β=2.0)
        
        Regression Metrics:
        - MAE, MSE, RMSE, R²
        - Median Absolute Error
        - MAPE (Mean Absolute Percentage Error)
        - Pearson and Spearman correlations
        - Explained Variance, Max Error
        - Residual statistics (mean, std, percentiles)
        
        Args:
            df: DataFrame with seq_id, chembl_id, label, and optionally affinity_uM
            threshold: Classification threshold (default 0.5)
            save_results: If True, save detailed results to output directory
            
        Returns:
            Dictionary with comprehensive metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first or load a checkpoint.")
        
        # Get predictions
        predictions = self.predict(df)
        
        # Extract labels
        cls_labels = df['label'].values if 'label' in df.columns else None
        reg_targets = df['affinity_uM'].values if 'affinity_uM' in df.columns else None
        
        if cls_labels is None:
            raise ValueError("DataFrame must have 'label' column for evaluation")
        
        # Prepare data
        cls_probs = predictions['classification_prob'].flatten()
        cls_preds = (cls_probs >= threshold).astype(int)
        reg_preds = predictions['regression_pred'].flatten()
        
        # Determine which samples have regression targets
        if reg_targets is not None:
            # Handle NaN or missing values
            reg_mask = ~np.isnan(reg_targets)
            reg_targets_valid = reg_targets[reg_mask]
            reg_preds_valid = reg_preds[reg_mask]
        else:
            reg_mask = np.zeros(len(df), dtype=bool)
            reg_targets_valid = np.array([])
            reg_preds_valid = np.array([])
        
        # Initialize metrics calculator
        metrics_calculator = MatrixMetricsCalculator(threshold=threshold)
        
        # Compute classification metrics
        cls_metrics = metrics_calculator.calculate_classification_metrics(
            y_true=cls_labels,
            y_prob=cls_probs,
            threshold=threshold
        )
        
        # Compute regression metrics if targets available
        reg_metrics = None
        if len(reg_preds_valid) > 0:
            reg_metrics = metrics_calculator.calculate_regression_metrics(
                y_true=reg_targets_valid,
                y_pred=reg_preds_valid
            )
        
        # Log comprehensive results
        logger.info("\n" + "="*70)
        logger.info("COMPREHENSIVE EVALUATION RESULTS (Matrix Pipeline)")
        logger.info("="*70)
        
        # Classification summary
        logger.info("\n" + "-"*35)
        logger.info("CLASSIFICATION METRICS")
        logger.info("-"*35)
        logger.info(f"  Total Samples:          {len(cls_labels)}")
        logger.info(f"  Positive Class:         {int(cls_labels.sum())} ({100*cls_labels.mean():.1f}%)")
        logger.info(f"  Negative Class:         {int(len(cls_labels) - cls_labels.sum())} ({100*(1-cls_labels.mean()):.1f}%)")
        logger.info("")
        logger.info(f"  Accuracy:               {cls_metrics.accuracy:.4f}")
        logger.info(f"  Precision:              {cls_metrics.precision:.4f}")
        logger.info(f"  Recall (Sensitivity):   {cls_metrics.recall:.4f}")
        logger.info(f"  Specificity:            {cls_metrics.specificity:.4f}")
        logger.info(f"  F1 Score:               {cls_metrics.f1:.4f}")
        logger.info(f"  F0.5 Score:             {cls_metrics.fbeta_05:.4f}")
        logger.info(f"  F2 Score:               {cls_metrics.fbeta_2:.4f}")
        logger.info(f"  ROC-AUC:                {cls_metrics.roc_auc:.4f}")
        logger.info(f"  Average Precision:      {cls_metrics.average_precision:.4f}")
        logger.info(f"  Brier Score:            {cls_metrics.brier_score:.4f}")
        logger.info(f"  MCC:                    {cls_metrics.mcc:.4f}")
        
        # Regression summary
        if reg_metrics is not None:
            logger.info("\n" + "-"*35)
            logger.info("REGRESSION METRICS")
            logger.info("-"*35)
            logger.info(f"  Samples with Target:    {len(reg_targets_valid)}")
            logger.info("")
            logger.info(f"  MAE:                    {reg_metrics.mae:.4f}")
            logger.info(f"  MSE:                    {reg_metrics.mse:.4f}")
            logger.info(f"  RMSE:                   {reg_metrics.rmse:.4f}")
            logger.info(f"  R²:                     {reg_metrics.r2:.4f}")
            logger.info(f"  Median AE:              {reg_metrics.median_ae:.4f}")
            logger.info(f"  MAPE:                   {reg_metrics.mape if reg_metrics.mape else 0:.4f}")
            logger.info(f"  Pearson r:              {reg_metrics.pearson_r:.4f}")
            logger.info(f"  Spearman ρ:             {reg_metrics.spearman_r:.4f}")
            logger.info(f"  Explained Variance:     {reg_metrics.explained_variance:.4f}")
            logger.info(f"  Max Error:              {reg_metrics.max_error:.4f}")
            
            if reg_metrics.mean_residual is not None:
                logger.info("")
                logger.info(f"  Residual Mean:          {reg_metrics.mean_residual:.4f}")
                logger.info(f"  Residual Std:           {reg_metrics.std_residual:.4f}")
        
        logger.info("\n" + "="*70 + "\n")
        
        # Save results if requested
        if save_results:
            save_path = self.output_dir / 'evaluation'
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save metrics as JSON
            json_results = {
                'classification': cls_metrics.to_dict(),
                'regression': reg_metrics.to_dict() if reg_metrics else {},
                'metadata': {
                    'total_samples': len(cls_labels),
                    'positive_samples': int(cls_labels.sum()),
                    'negative_samples': int(len(cls_labels) - cls_labels.sum()),
                    'samples_with_regression_target': len(reg_targets_valid),
                    'threshold': threshold
                }
            }
            
            json_path = save_path / 'comprehensive_metrics.json'
            with open(json_path, 'w') as f:
                json.dump(json_results, f, indent=2, default=float)
            logger.info(f"Saved metrics JSON: {json_path}")
            
            # Save formatted report
            report = metrics_calculator.format_classification_report(cls_metrics)
            if reg_metrics:
                report += "\n\n" + metrics_calculator.format_regression_report(reg_metrics)
            
            report_path = save_path / 'metrics_report.txt'
            with open(report_path, 'w') as f:
                f.write(report)
            logger.info(f"Saved metrics report: {report_path}")
            
            # Save predictions with labels for analysis
            eval_df = df.copy()
            eval_df['pred_prob'] = cls_probs
            eval_df['pred_label'] = cls_preds
            eval_df['pred_affinity'] = reg_preds
            eval_df['correct'] = (cls_preds == cls_labels).astype(int)
            
            eval_path = save_path / 'evaluation_predictions.csv'
            eval_df.to_csv(eval_path, index=False)
            logger.info(f"Saved evaluation predictions: {eval_path}")
        
        # Return results
        return {
            'classification': cls_metrics,
            'regression': reg_metrics,
            'classification_dict': cls_metrics.to_dict(),
            'regression_dict': reg_metrics.to_dict() if reg_metrics else {},
            # Legacy format for compatibility
            'accuracy': cls_metrics.accuracy,
            'precision': cls_metrics.precision,
            'recall': cls_metrics.recall,
            'f1': cls_metrics.f1,
            'auc': cls_metrics.roc_auc,
            'mcc': cls_metrics.mcc,
            'rmse': reg_metrics.rmse if reg_metrics else 0.0,
            'r2': reg_metrics.r2 if reg_metrics else 0.0,
            'pearson': reg_metrics.pearson_r if reg_metrics else 0.0,
            'spearman': reg_metrics.spearman_r if reg_metrics else 0.0
        }


def run_matrix_pipeline(
    data_path: str,
    output_dir: str = 'results/matrix_pipeline',
    protein_model: str = 'esm2_t33',
    ligand_model: str = 'smited',
    **kwargs
) -> Dict[str, Any]:
    """
    Run the complete matrix-based affinity prediction pipeline.
    
    Args:
        data_path: Path to input CSV file
        output_dir: Output directory
        protein_model: Protein embedding model
        ligand_model: Ligand embedding model
        **kwargs: Additional configuration options
    
    Returns:
        Results dictionary
    """
    # Load data
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} samples from {data_path}")
    
    # Create config
    config = MatrixPipelineConfig(
        data_path=data_path,
        output_dir=output_dir,
        protein_model=protein_model,
        ligand_model=ligand_model,
        **kwargs
    )
    
    # Initialize pipeline
    pipeline = MatrixAffinityPipeline(config)
    
    # Generate embeddings
    pipeline.generate_embeddings(df)
    
    # Prepare data
    train_df, val_df, test_df = pipeline.prepare_data(df)
    
    # Train
    history = pipeline.train(train_df, val_df, test_df)
    
    # Predict on test set
    predictions = pipeline.predict(test_df)
    pipeline.save_predictions(test_df, predictions, f"{output_dir}/test_predictions.csv")
    
    return {
        'history': history,
        'test_predictions': predictions,
        'config': config
    }


if __name__ == "__main__":
    # Example usage
    print("Matrix Affinity Pipeline")
    print("=" * 50)
    print("\nExample usage:")
    print("""
from src.classifier.utils.matrix_pipeline import MatrixAffinityPipeline, MatrixPipelineConfig

# Create configuration
config = MatrixPipelineConfig(
    data_path='data/kinase_data.csv',
    output_dir='results/matrix_pipeline',
    protein_model='esm2_t33',
    ligand_model='smited',
    hidden_dim=256,
    num_epochs=50
)

# Initialize pipeline
pipeline = MatrixAffinityPipeline(config)

# Load data
df = pd.read_csv(config.data_path)

# Generate embeddings
pipeline.generate_embeddings(df)

# Split data
train_df, val_df, test_df = pipeline.prepare_data(df)

# Train
history = pipeline.train(train_df, val_df, test_df)

# Predict
predictions = pipeline.predict(test_df)
    """)
