"""
Pipeline Integration for Matrix-Based Affinity Prediction.

This module provides high-level functions to:
1. Generate matrix embeddings from proteins and ligands
2. Train the CrossAttentionAffinityModel
3. Make predictions on new data

Author: DockTKinase Team
Date: November 2025
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
import pandas as pd
import numpy as np
import torch
from dataclasses import dataclass

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
        from sklearn.model_selection import train_test_split
        
        # Ensure required columns
        required_cols = ['seq_id', 'chembl_id', label_col]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Split data
        train_val_df, test_df = train_test_split(
            df, test_size=test_split, random_state=random_state, stratify=df[label_col]
        )
        
        val_size = val_split / (1 - test_split)
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_size, random_state=random_state, stratify=train_val_df[label_col]
        )
        
        logger.info(f"Data split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        
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
