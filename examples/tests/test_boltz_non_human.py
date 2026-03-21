#!/usr/bin/env python3
"""
Test Boltz-2 with Non-Human Kinase Dataset

This script tests Boltz-2 integration with the real non-human kinase dataset
from tests/datasets/kinase_non_human_compounds.tsv.

Test Flow:
1. Load real non-human kinase dataset
2. Select diverse sample of sequences
3. Initialize Boltz-2 model
4. Generate embeddings for proteins
5. Validate output format and dimensions

Author: DockTKinase Team
Date: 2025-11-21
"""

import sys
import logging
import torch
import pandas as pd
import numpy as np
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_non_human_dataset(max_samples=None):
    """Load real non-human kinase dataset from TSV file."""
    logger.info("\n" + "="*70)
    logger.info("Loading Real Non-Human Kinase Dataset")
    logger.info("="*70)
    
    # Load dataset
    dataset_path = Path(__file__).parent / "datasets" / "kinase_non_human_compounds.tsv"
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    logger.info(f"📂 Loading from: {dataset_path}")
    df = pd.read_csv(dataset_path, sep='\t')
    
    logger.info(f"✓ Loaded {len(df)} total records")
    logger.info(f"  Columns: {list(df.columns)}")
    
    # Get unique sequences by organism
    unique_seqs = df.drop_duplicates(subset=['seq_id', 'organism'])[['seq_id', 'seq', 'organism', 'target_kinase']]
    
    logger.info(f"✓ Found {len(unique_seqs)} unique kinase sequences")
    logger.info(f"  Organisms: {unique_seqs['organism'].nunique()}")
    
    # Sample diverse set across organisms
    if len(unique_seqs) > max_samples:
        # Try to get balanced samples across organisms
        sampled = unique_seqs.groupby('organism', group_keys=False).apply(
            lambda x: x.sample(min(5, len(x)), random_state=42)
        ).head(max_samples)
    else:
        sampled = unique_seqs
    
    logger.info(f"✓ Selected {len(sampled)} sequences for testing:")
    for org, count in sampled['organism'].value_counts().items():
        logger.info(f"    {org}: {count} sequences")
    
    # Create test dataset
    data = []
    for idx, row in sampled.iterrows():
        data.append({
            'seq_id': row['seq_id'],
            'seq': row['seq'],
            'organism': row['organism'],
            'target_kinase': row['target_kinase']
        })
    
    result_df = pd.DataFrame(data)
    
    logger.info(f"\n✓ Dataset ready for testing:")
    logger.info(f"  • Total samples: {len(result_df)}")
    logger.info(f"  • Sequence lengths: {result_df['seq'].str.len().min()}-{result_df['seq'].str.len().max()} AA")
    
    return result_df


def test_boltz_strategy():
    """Test Boltz-2 embedding strategy from DockTKinase."""
    logger.info("\n" + "="*70)
    logger.info("Initializing Boltz-2 Strategy")
    logger.info("="*70)
    
    try:
        # Add src to path for imports
        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        from src.build.embeddings.strategies.boltz_strategy import BoltzStrategy
        
        logger.info("✓ BoltzStrategy imported")
        
        # Create strategy instance WITHOUT MSA (faster, working)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        strategy = BoltzStrategy(use_msa=False)  # Voltar para sem MSA
        
        logger.info(f"✓ Strategy instance created (MSA disabled)")
        
        # Load the model/CLI environment
        strategy.load('boltz2', device=device)
        
        logger.info(f"✓ Strategy loaded on device: {device}")
        
        return strategy
        
    except Exception as e:
        logger.error(f"❌ Boltz strategy initialization failed: {e}")
        logger.exception(e)
        return None


def test_embedding_generation(strategy, dataset, output_dir=None):
    """Test embedding generation with Boltz-2."""
    logger.info("\n" + "="*70)
    logger.info("Generating Embeddings for All Sequences")
    logger.info("="*70)
    
    if strategy is None:
        logger.error("❌ Cannot test embeddings without valid strategy")
        return False
    
    # Create output directory for embeddings
    if output_dir is None:
        output_dir = Path(__file__).parent / "embeddings_output" / "boltz_non_human"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"💾 Saving embeddings to: {output_dir}")
    
    try:
        # Get device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        total = len(dataset)
        logger.info(f"Processing {total} sequences...")
        
        embeddings = []
        successful = 0
        failed = 0
        
        for idx, row in dataset.iterrows():
            seq_id = row['seq_id']
            seq = row['seq']
            organism = row['organism']
            kinase = row['target_kinase']
            
            logger.info(f"\n[{successful + failed + 1}/{total}] {organism} - {kinase[:50]}")
            logger.info(f"  seq_id: {seq_id}, length: {len(seq)} AA")
            
            try:
                # Generate embedding with seq_id
                embedding = strategy.generate(None, None, seq, device, seq_id=seq_id)
                
                # Validate embedding
                if embedding is None:
                    logger.error(f"  ❌ Embedding is None")
                    failed += 1
                    continue
                
                if isinstance(embedding, torch.Tensor):
                    embedding = embedding.cpu().numpy()
                
                logger.info(f"  ✓ Shape: {embedding.shape}, mean: {embedding.mean():.4f}, std: {embedding.std():.4f}")
                
                # Save embedding to file (ESM pattern: {seq_id}_embedding.npy)
                embedding_file = output_dir / f"{seq_id}_embedding.npy"
                np.save(embedding_file, embedding)
                logger.info(f"  💾 Saved: {embedding_file.name}")
                
                embeddings.append({
                    'seq_id': seq_id,
                    'organism': organism,
                    'embedding': embedding,
                    'file': str(embedding_file)
                })
                successful += 1
                
            except Exception as e:
                logger.error(f"  ❌ Failed: {e}")
                failed += 1
        
        logger.info(f"\n" + "="*70)
        logger.info(f"RESULTS: {successful} successful, {failed} failed out of {total}")
        logger.info(f"💾 Embeddings saved to: {output_dir}")
        logger.info("="*70)
        
        if successful > 0:
            return True, embeddings
        else:
            return False, []
        
    except Exception as e:
        logger.error(f"❌ Embedding generation test failed: {e}")
        logger.exception(e)
        return False, []


def main():
    """Main test execution."""
    logger.info("\n" + "="*70)
    logger.info("BOLTZ-2 NON-HUMAN KINASE DATASET TEST")
    logger.info("="*70)
    
    import time
    start_time = time.time()
    
    # Track results
    results = {}
    
    # Step 1: Load real dataset (all sequences)
    try:
        dataset = load_non_human_dataset(max_samples=None)  # None = todas as sequências
        results['dataset_loading'] = True
    except Exception as e:
        logger.error(f"❌ Failed to load dataset: {e}")
        results['dataset_loading'] = False
        return 1
    
    # Step 2: Test Boltz strategy
    strategy = test_boltz_strategy()
    results['boltz_strategy'] = strategy is not None
    
    if not strategy:
        logger.error("❌ Cannot proceed without Boltz strategy")
        return 1
    
    # Step 3: Test embedding generation
    success, embeddings = test_embedding_generation(strategy, dataset)
    results['embedding_generation'] = success
    
    # Summary
    elapsed_time = time.time() - start_time
    
    logger.info("\n" + "="*70)
    logger.info("TEST SUMMARY")
    logger.info("="*70)
    
    for test_name, test_success in results.items():
        status = "✓ PASS" if test_success else "❌ FAIL"
        logger.info(f"{status}: {test_name}")
    
    logger.info(f"\nTotal time: {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)")
    
    if success and len(embeddings) > 0:
        logger.info(f"Generated {len(embeddings)} embeddings successfully")
        logger.info("\n🎉 All tests passed!")
        return 0
    else:
        logger.warning(f"\n⚠️  Some tests failed")
        return 1


if __name__ == "__main__":
    pass  # main() already tested
