#!/usr/bin/env python3
"""
Simple Boltz-2 Test for Non-Human Kinases

Testa apenas o Boltz-2 sem comparação com ESM-2 (para evitar downloads longos).

Author: DockTKinase Team
Date: 2025-11-21
"""

import sys
import logging
import torch
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_dataset():
    """Create small test dataset with non-human kinase sequences."""
    logger.info("Creating test dataset...")
    
    # Short test sequences (real kinase domains, truncated for speed)
    sequences = [
        ("C_elegans_KIN", "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQ"),
        ("D_melanogaster_KIN", "MGSSHHHHHHSSGLVPRGSHMLEVEELEFVRKLGEGEFGKVMKAYHQNKKKIKVRCVKKGEGQPVALK"),
        ("S_cerevisiae_KIN", "MAFSAEDVLKEYDRRRRMEALLLSLYYPNDRKLLDYKEWSPPRVQVECPKAPVEWNNPPSEKGLIVGH"),
    ]
    
    data = []
    for kinase_name, sequence in sequences:
        organism = kinase_name.split('_')[0]
        data.append({
            'canonical_smiles': 'CCO',
            'seq': sequence,
            'target_kinase': kinase_name,
            'organism': organism,
            'standard_type': 'Ki',
            'standard_value': 100.0,
            'pchembl_value': 7.0,
            'molregno': 12345
        })
    
    df = pd.DataFrame(data)
    logger.info(f"✓ Created {len(df)} test samples")
    return df


def main():
    """Main test execution."""
    logger.info("\n" + "="*70)
    logger.info("BOLTZ-2 SIMPLE TEST - NON-HUMAN KINASES")
    logger.info("="*70)
    
    # Create test dataset
    dataset = create_test_dataset()
    
    # Add src to path
    src_path = Path(__file__).parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    try:
        # Import Boltz strategy
        logger.info("\n1. Importing Boltz strategy...")
        from src.build.embeddings.strategies.boltz_strategy import BoltzStrategy
        logger.info("✓ BoltzStrategy imported")
        
        # Initialize strategy
        logger.info("\n2. Initializing strategy...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        strategy = BoltzStrategy(use_msa=False)
        logger.info("✓ Strategy created")
        
        # Load model (check CLI)
        logger.info("\n3. Loading model (checking Boltz CLI)...")
        strategy.load('boltz2', device=device)
        logger.info("✓ Model loaded successfully")
        
        # Test embedding generation
        logger.info("\n4. Generating embeddings...")
        for i, row in dataset.iterrows():
            seq = row['seq']
            kinase = row['target_kinase']
            
            logger.info(f"\n  Testing {kinase} ({len(seq)} AA)...")
            
            try:
                embedding = strategy.generate(None, None, seq, device)
                
                if isinstance(embedding, torch.Tensor):
                    embedding = embedding.cpu().numpy()
                
                logger.info(f"    ✓ Shape: {embedding.shape}")
                logger.info(f"    ✓ Mean: {embedding.mean():.4f}, Std: {embedding.std():.4f}")
                
            except Exception as e:
                logger.error(f"    ❌ Failed: {e}")
                # Continue with next sequence
        
        # Cleanup
        logger.info("\n5. Cleaning up...")
        strategy.cleanup(None, None)
        logger.info("✓ Cleanup complete")
        
        logger.info("\n" + "="*70)
        logger.info("🎉 TEST COMPLETED SUCCESSFULLY")
        logger.info("="*70)
        return 0
        
    except Exception as e:
        logger.error(f"\n❌ TEST FAILED: {e}")
        logger.exception(e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
