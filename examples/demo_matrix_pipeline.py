#!/usr/bin/env python
"""
Example: Matrix-Based Affinity Prediction Pipeline

This example demonstrates how to use the CrossAttentionAffinityModel
with matrix embeddings [seq_len, dim] for protein-ligand affinity prediction.

Author: DockTKinase Team
Date: November 2025
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_data(output_dir: Path) -> pd.DataFrame:
    """Create sample data for demonstration."""
    
    # Sample protein sequences (short for demo)
    proteins = {
        'P001': 'MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNL',
        'P002': 'MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDL',
        'P003': 'MGSSHHHHHHSSGLVPRGSHMGKVKVGVNGFGRIGRLVTRAAFNSG'
    }
    
    # Sample SMILES
    ligands = {
        'C001': 'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
        'C002': 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',  # Ibuprofen
        'C003': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
    }
    
    # Create interaction data
    data = []
    np.random.seed(42)
    
    for prot_id, seq in proteins.items():
        for lig_id, smiles in ligands.items():
            # Random binding label and affinity
            label = np.random.randint(0, 2)
            pchembl = np.random.uniform(5.0, 9.0) if label == 1 else np.random.uniform(3.0, 5.0)
            
            data.append({
                'seq_id': prot_id,
                'sequence': seq,
                'chembl_id': lig_id,
                'smiles': smiles,
                'label': label,
                'pchembl_value': round(pchembl, 2)
            })
    
    df = pd.DataFrame(data)
    
    # Save to file
    data_path = output_dir / 'sample_kinase_data.csv'
    df.to_csv(data_path, index=False)
    logger.info(f"Created sample data with {len(df)} samples at {data_path}")
    
    return df


def create_dummy_embeddings(df: pd.DataFrame, output_dir: Path):
    """
    Create dummy matrix embeddings for demonstration.
    In a real scenario, use the ProteinEmbedding and LigandEmbedding classes.
    """
    
    prot_dir = output_dir / 'protein_matrix_embeddings'
    lig_dir = output_dir / 'ligand_matrix_embeddings'
    prot_dir.mkdir(parents=True, exist_ok=True)
    lig_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate protein matrix embeddings
    # Shape: [seq_len, embed_dim] - using ESM-2 small dimension for demo
    embed_dim_protein = 320  # ESM-2 t6 dimension
    
    unique_proteins = df[['seq_id', 'sequence']].drop_duplicates()
    for _, row in unique_proteins.iterrows():
        seq_len = len(row['sequence'])
        # Simulate ESM-2 output: random embeddings for each residue
        matrix = np.random.randn(seq_len, embed_dim_protein).astype(np.float32)
        np.save(prot_dir / f"{row['seq_id']}_matrix.npy", matrix)
        logger.info(f"Saved protein matrix: {row['seq_id']} shape={matrix.shape}")
    
    # Generate ligand matrix embeddings
    # Shape: [num_tokens, embed_dim] - using SMI-TED dimension
    embed_dim_ligand = 768  # SMI-TED dimension
    
    unique_ligands = df[['chembl_id', 'smiles']].drop_duplicates()
    for _, row in unique_ligands.iterrows():
        # Approximate token count (simplified)
        num_tokens = len(row['smiles']) // 2 + 1
        matrix = np.random.randn(num_tokens, embed_dim_ligand).astype(np.float32)
        np.save(lig_dir / f"{row['chembl_id']}_matrix.npy", matrix)
        logger.info(f"Saved ligand matrix: {row['chembl_id']} shape={matrix.shape}")
    
    return str(prot_dir), str(lig_dir)


def example_basic_usage():
    """Example 1: Basic usage with dummy data."""
    print("\n" + "="*60)
    print("Example 1: Basic Usage")
    print("="*60 + "\n")
    
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create sample data
        df = create_sample_data(tmpdir)
        print(f"Sample data:\n{df.head()}\n")
        
        # Create dummy embeddings
        prot_dir, lig_dir = create_dummy_embeddings(df, tmpdir)
        
        # Import pipeline components
        from src.classifier.utils.matrix_dataloader import create_matrix_dataloader
        from src.classifier.models.cross_attention_model import CrossAttentionAffinityModel
        
        # Create dataloader
        loader = create_matrix_dataloader(
            df, prot_dir, lig_dir,
            batch_size=4,
            shuffle=True
        )
        
        # Create model (small for demo)
        model = CrossAttentionAffinityModel(
            protein_dim=320,
            ligand_dim=768,
            hidden_dim=64,
            num_cnn_layers=2,
            num_cross_attn_layers=1,
            num_heads=4
        )
        
        print(f"Model parameters: {model.count_parameters():,}")
        
        # Get a batch and run forward pass
        batch = next(iter(loader))
        print(f"\nBatch shapes:")
        print(f"  Protein matrices: {batch['protein_matrix'].shape}")
        print(f"  Ligand matrices: {batch['ligand_matrix'].shape}")
        print(f"  Labels: {batch['labels'].shape}")
        
        # Forward pass
        output = model(
            batch['protein_matrix'],
            batch['ligand_matrix'],
            batch['protein_mask'],
            batch['ligand_mask']
        )
        
        print(f"\nModel output:")
        print(f"  Classification logits: {output['classification'].shape}")
        print(f"  Regression predictions: {output['regression'].shape}")
        
        # Convert to predictions
        binding_probs = torch.sigmoid(output['classification'])
        print(f"\nPredictions:")
        print(f"  Binding probabilities: {binding_probs.detach().numpy().flatten()}")
        print(f"  Predicted affinities: {output['regression'].detach().numpy().flatten()}")


def example_training_loop():
    """Example 2: Complete training loop."""
    print("\n" + "="*60)
    print("Example 2: Training Loop")
    print("="*60 + "\n")
    
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create data
        df = create_sample_data(tmpdir)
        prot_dir, lig_dir = create_dummy_embeddings(df, tmpdir)
        
        # Split data
        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(df, test_size=0.3, random_state=42)
        
        print(f"Train samples: {len(train_df)}")
        print(f"Val samples: {len(val_df)}")
        
        # Import components
        from src.classifier.utils.matrix_dataloader import create_matrix_dataloader
        from src.classifier.utils.matrix_trainer import CrossAttentionTrainer, TrainingConfig
        from src.classifier.models.cross_attention_model import CrossAttentionAffinityModel
        
        # Create dataloaders
        train_loader = create_matrix_dataloader(train_df, prot_dir, lig_dir, batch_size=3)
        val_loader = create_matrix_dataloader(val_df, prot_dir, lig_dir, batch_size=3)
        
        # Create model
        model = CrossAttentionAffinityModel(
            protein_dim=320,
            ligand_dim=768,
            hidden_dim=64,
            num_cnn_layers=2,
            num_cross_attn_layers=1,
            num_heads=4
        )
        
        # Training config
        config = TrainingConfig(
            protein_dim=320,
            ligand_dim=768,
            hidden_dim=64,
            num_cnn_layers=2,
            num_cross_attn_layers=1,
            batch_size=3,
            num_epochs=3,  # Short for demo
            learning_rate=1e-3,
            patience=5,
            save_dir=str(tmpdir / 'checkpoints')
        )
        
        # Create trainer
        trainer = CrossAttentionTrainer(
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader
        )
        
        print(f"\nStarting training for {config.num_epochs} epochs...")
        
        # Train
        history = trainer.train()
        
        print(f"\nTraining complete!")
        print(f"Best validation loss: {history['best_val_loss']:.4f}")
        print(f"Final metrics: {list(history['history'].keys())}")


def example_full_pipeline():
    """Example 3: Using MatrixAffinityPipeline for end-to-end workflow."""
    print("\n" + "="*60)
    print("Example 3: Full Pipeline (Conceptual)")
    print("="*60 + "\n")
    
    print("""
The MatrixAffinityPipeline provides an end-to-end workflow:

1. EMBEDDING GENERATION
   - Uses ProteinEmbedding with generate_embedding_matrix()
   - Uses LigandEmbedding with SMI-TED
   - Outputs: [seq_len, embed_dim] matrices per sequence

2. DATA PREPARATION
   - Splits data into train/val/test
   - Creates DataLoaders with padding

3. MODEL TRAINING
   - CNN encoders for variable-length sequences
   - Cross-attention for protein-ligand interaction
   - Multi-task loss (classification + regression)

4. PREDICTION
   - Binding probability (0-1)
   - Predicted affinity (pIC50)

Example code:

```python
from src.classifier.utils.matrix_pipeline import (
    MatrixPipelineConfig, 
    MatrixAffinityPipeline
)

# Configure pipeline
config = MatrixPipelineConfig(
    data_path='data/kinase_interactions.csv',
    output_dir='results/matrix_pipeline',
    protein_model='esm2_t33',  # or 'boltz', 'esmc'
    ligand_model='smited',
    hidden_dim=256,
    num_epochs=100,
    batch_size=32
)

# Initialize pipeline
pipeline = MatrixAffinityPipeline(config)

# Load and prepare data
df = pd.read_csv(config.data_path)

# Generate embeddings (skip if already exist)
pipeline.generate_embeddings(df)

# Split data
train_df, val_df, test_df = pipeline.prepare_data(df)

# Train model
history = pipeline.train(train_df, val_df, test_df)

# Make predictions
predictions = pipeline.predict(test_df)
print(f"Binding probabilities: {predictions['classification_prob']}")
print(f"Predicted affinities: {predictions['regression_pred']}")

# Save results
pipeline.save_predictions(test_df, predictions, 'predictions.csv')
```
""")


def main():
    """Run all examples."""
    print("\n" + "#"*60)
    print("# Matrix-Based Affinity Prediction Pipeline Examples")
    print("#"*60)
    
    print("""
This example demonstrates the CNN + Cross-Attention model
for protein-ligand affinity prediction using matrix embeddings.

Key advantages of matrix embeddings [seq_len, dim]:
- Preserves positional information
- Enables attention over specific residues/atoms  
- Better captures binding site interactions
- Supports variable-length sequences with padding

Model architecture:
- CNN Encoders: Extract local patterns from sequences
- Cross-Attention: Model protein-ligand interactions
- Multi-Task Heads: Classification + Regression
    """)
    
    # Run examples
    example_basic_usage()
    example_training_loop()
    example_full_pipeline()
    
    print("\n" + "="*60)
    print("Examples complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
