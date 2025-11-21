"""
OpenFold3 MSA-Enhanced Embedding Extraction Example

OVERVIEW:
=========
This script demonstrates how to use OpenFold3 with ColabFold MSA server
for large-scale protein embedding extraction in DockTKinase.

Optimized for 700+ protein sequences using Main MSA mode.

ARCHITECTURE:
============
The embedding extraction pipeline consists of:

1. MSA Configuration (msa_config.py):
   - Defines MSA computation strategy
   - Configures ColabFold server settings
   - Handles caching and storage

2. OpenFold Strategy (openfold_strategy.py):
   - Loads OpenFold3 model
   - Integrates MSA data
   - Extracts embeddings from model trunk

3. ColabFold Integration (via OpenFold3):
   - Queries MSA server
   - Fetches evolutionary alignments
   - Enriches sequence information

WORKFLOW:
=========
For each protein sequence:
1. Generate MSA (if not cached)
2. Prepare batch with sequence + MSA
3. Run OpenFold3 trunk (without structure prediction)
4. Extract single representations (s)
5. Apply pooling (mean/cls/max)
6. Return 384-dim embedding

EXTENSIBILITY (Boltz-2):
=======================
This example can be adapted for Boltz-2 by:
1. Import Boltz2Strategy instead of OpenFoldStrategy
2. Use same MsaConfig (reusable)
3. Change model name from 'openfold3' to 'boltz2'
4. All other code remains the same

Example adaptation:
    from src.build.embeddings.strategies.boltz2_strategy import Boltz2Strategy
    
    strategy = Boltz2Strategy(msa_config=MsaConfig.for_production())
    model, _ = strategy.load('boltz2', device=device)
    # Rest of the code is identical

MODES DEMONSTRATED:
==================
1. Production: Main MSA, high quality, caching (RECOMMENDED)
2. Development: Fast MSA, quick iterations
3. Research: Maximum quality, no filter
4. No MSA: Instant, sequence-only
5. Custom: Fine-tuned parameters
6. Batch: Large-scale processing (700+ sequences)

Author: DockTKinase Team
Date: 2025-11-20
"""

import sys
import logging
from pathlib import Path

import torch
import numpy as np

# Add src to path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.build.embeddings.strategies.openfold_strategy import OpenFoldStrategy
from src.build.embeddings.config.msa_config import MsaConfig, MsaMode


# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# EXAMPLE SEQUENCES
# =============================================================================

EXAMPLE_SEQUENCES = {
    "kinase_1": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSLEVELHQV",
    "kinase_2": "MGSSHHHHHHSSGLVPRGSHMRGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGYTARQSSIKWWGPRLASHGFVVITACEGGGAGVYAFNR",
    "kinase_3": "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHHYREQIKRVKDSED",
}


# =============================================================================
# EXAMPLE 1: PRODUCTION MODE (RECOMMENDED)
# =============================================================================

def example_production_mode():
    """
    Production mode with Main MSA - optimized for 700+ sequences.
    
    Characteristics:
    - Main MSA with environmental databases
    - Diversity filter enabled
    - NPZ format (faster loading)
    - Estimated time: 3-5 minutes for 700 sequences
    """
    logger.info("="*70)
    logger.info("EXAMPLE 1: PRODUCTION MODE (Recommended for DockTKinase)")
    logger.info("="*70)
    
    # Create MSA configuration for production
    msa_config = MsaConfig.for_production(
        output_dir=Path("./msa_cache_production")
    )
    
    logger.info(f"\n{msa_config.summary()}\n")
    
    # Initialize strategy with MSA config
    strategy = OpenFoldStrategy(
        logger=logger,
        msa_config=msa_config
    )
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model, _ = strategy.load('openfold3', device=device)
    
    # Generate embeddings
    logger.info("\nGenerating embeddings...")
    embeddings = {}
    
    for name, sequence in EXAMPLE_SEQUENCES.items():
        logger.info(f"\nProcessing {name} (length={len(sequence)})...")
        
        embedding = strategy.generate(
            model=model,
            auxiliary_objects=None,
            sequence=sequence,
            device=device,
            pooling_strategy='mean'
        )
        
        embeddings[name] = embedding
        logger.info(f"✅ Embedding shape: {embedding.shape}")
    
    # Cleanup
    strategy.cleanup(model, None)
    
    logger.info(f"\n✅ Production mode complete!")
    logger.info(f"Total embeddings generated: {len(embeddings)}")
    
    return embeddings


# =============================================================================
# EXAMPLE 2: DEVELOPMENT MODE (FAST)
# =============================================================================

def example_development_mode():
    """
    Development mode - faster for testing and prototyping.
    
    Characteristics:
    - Fast mode (UniRef90 only)
    - Diversity filter enabled
    - NPZ format
    - Estimated time: 1-2 minutes for 700 sequences
    """
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 2: DEVELOPMENT MODE (Fast testing)")
    logger.info("="*70)
    
    # Create MSA configuration for development
    msa_config = MsaConfig.for_development(
        output_dir=Path("./msa_cache_dev")
    )
    
    logger.info(f"\n{msa_config.summary()}\n")
    
    strategy = OpenFoldStrategy(logger=logger, msa_config=msa_config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _ = strategy.load('openfold3', device=device)
    
    # Process just one sequence for demo
    name = "kinase_1"
    sequence = EXAMPLE_SEQUENCES[name]
    
    logger.info(f"Processing {name}...")
    embedding = strategy.generate(model, None, sequence, device)
    
    logger.info(f"✅ Embedding shape: {embedding.shape}")
    
    strategy.cleanup(model, None)
    
    return embedding


# =============================================================================
# EXAMPLE 3: RESEARCH MODE (HIGH QUALITY)
# =============================================================================

def example_research_mode():
    """
    Research mode - maximum quality for detailed analysis.
    
    Characteristics:
    - High quality mode (no diversity filter)
    - All environmental databases
    - A3M format (human-readable)
    - Estimated time: 5-10 minutes for 700 sequences
    """
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 3: RESEARCH MODE (Maximum quality)")
    logger.info("="*70)
    
    msa_config = MsaConfig.for_research(
        output_dir=Path("./msa_cache_research")
    )
    
    logger.info(f"\n{msa_config.summary()}\n")
    
    strategy = OpenFoldStrategy(logger=logger, msa_config=msa_config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _ = strategy.load('openfold3', device=device)
    
    name = "kinase_2"
    sequence = EXAMPLE_SEQUENCES[name]
    
    logger.info(f"Processing {name}...")
    embedding = strategy.generate(model, None, sequence, device)
    
    logger.info(f"✅ Embedding shape: {embedding.shape}")
    logger.info(f"✅ MSA files saved in A3M format for inspection")
    
    strategy.cleanup(model, None)
    
    return embedding


# =============================================================================
# EXAMPLE 4: NO MSA MODE (FASTEST)
# =============================================================================

def example_no_msa_mode():
    """
    No MSA mode - fastest option for quick testing.
    
    Characteristics:
    - No MSA computation
    - Sequence-only embeddings
    - Instant processing
    - Lower quality than MSA-enhanced
    """
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 4: NO MSA MODE (Instant, sequence-only)")
    logger.info("="*70)
    
    msa_config = MsaConfig.no_msa()
    
    logger.info(f"\n{msa_config.summary()}\n")
    
    strategy = OpenFoldStrategy(logger=logger, msa_config=msa_config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _ = strategy.load('openfold3', device=device)
    
    name = "kinase_3"
    sequence = EXAMPLE_SEQUENCES[name]
    
    logger.info(f"Processing {name}...")
    embedding = strategy.generate(model, None, sequence, device)
    
    logger.info(f"✅ Embedding shape: {embedding.shape}")
    logger.info(f"✅ No MSA computation - instant result!")
    
    strategy.cleanup(model, None)
    
    return embedding


# =============================================================================
# EXAMPLE 5: CUSTOM CONFIGURATION
# =============================================================================

def example_custom_config():
    """
    Custom configuration - fine-tuned settings.
    
    Demonstrates how to create custom MSA configurations
    with specific parameters.
    """
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 5: CUSTOM CONFIGURATION")
    logger.info("="*70)
    
    # Create custom configuration
    msa_config = MsaConfig(
        mode=MsaMode.MAIN_STANDARD,
        file_format="npz",
        use_env=True,
        use_filter=False,  # Custom: no diversity filter
        use_templates=False,
        output_directory=Path("./msa_cache_custom"),
        cleanup_after_use=True,
        enable_caching=True,
        user_agent="docktkinase/1.0 custom myemail@institution.edu",
        timeout_seconds=600,  # Longer timeout
        max_retries=3
    )
    
    logger.info(f"\n{msa_config.summary()}\n")
    
    strategy = OpenFoldStrategy(logger=logger, msa_config=msa_config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _ = strategy.load('openfold3', device=device)
    
    # Process all sequences
    embeddings = {}
    for name, sequence in EXAMPLE_SEQUENCES.items():
        logger.info(f"Processing {name}...")
        embedding = strategy.generate(model, None, sequence, device)
        embeddings[name] = embedding
    
    logger.info(f"\n✅ Custom configuration complete!")
    logger.info(f"Total embeddings: {len(embeddings)}")
    
    strategy.cleanup(model, None)
    
    return embeddings


# =============================================================================
# EXAMPLE 6: BATCH PROCESSING (700+ SEQUENCES)
# =============================================================================

def example_batch_processing():
    """
    Batch processing example for large-scale datasets (700+ sequences).
    
    Demonstrates efficient processing of multiple sequences with
    caching and deduplication.
    """
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 6: BATCH PROCESSING (700+ sequences simulation)")
    logger.info("="*70)
    
    # Simulate 700 sequences (in reality, you'd load from file)
    sequences = {f"protein_{i}": EXAMPLE_SEQUENCES["kinase_1"] for i in range(10)}
    
    logger.info(f"Processing {len(sequences)} sequences...")
    
    # Use production config with caching
    msa_config = MsaConfig.for_production(
        output_dir=Path("./msa_cache_batch")
    )
    msa_config.enable_caching = True  # Important for large batches
    
    strategy = OpenFoldStrategy(logger=logger, msa_config=msa_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _ = strategy.load('openfold3', device=device)
    
    # Process in batches (OpenFold handles deduplication)
    embeddings = {}
    for i, (name, sequence) in enumerate(sequences.items(), 1):
        logger.info(f"[{i}/{len(sequences)}] Processing {name}...")
        
        embedding = strategy.generate(model, None, sequence, device)
        embeddings[name] = embedding
        
        # Show progress every 100 sequences (in real scenario)
        if i % 100 == 0:
            logger.info(f"Progress: {i}/{len(sequences)} sequences completed")
    
    logger.info(f"\n✅ Batch processing complete!")
    logger.info(f"Total embeddings: {len(embeddings)}")
    logger.info(f"Average embedding shape: {embeddings[list(embeddings.keys())[0]].shape}")
    
    strategy.cleanup(model, None)
    
    return embeddings


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run all examples."""
    logger.info("\n" + "="*70)
    logger.info("OpenFold3 MSA-Enhanced Embedding Extraction Examples")
    logger.info("Optimized for DockTKinase with 700+ protein sequences")
    logger.info("="*70)
    
    try:
        # Example 1: Production mode (RECOMMENDED)
        embeddings_prod = example_production_mode()
        
        # Example 2: Development mode (fast)
        embedding_dev = example_development_mode()
        
        # Example 3: Research mode (high quality)
        embedding_res = example_research_mode()
        
        # Example 4: No MSA mode (fastest)
        embedding_no_msa = example_no_msa_mode()
        
        # Example 5: Custom configuration
        embeddings_custom = example_custom_config()
        
        # Example 6: Batch processing
        embeddings_batch = example_batch_processing()
        
        logger.info("\n" + "="*70)
        logger.info("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        logger.info("="*70)
        logger.info("\nRECOMMENDATIONS FOR DOCKTKINASE (700+ sequences):")
        logger.info("  1. Use PRODUCTION mode for best quality/speed balance")
        logger.info("  2. Enable caching to avoid re-computing MSAs")
        logger.info("  3. Main MSA mode is ~100x faster than Paired MSA")
        logger.info("  4. Expected time: 3-5 minutes for 700 unique sequences")
        logger.info("  5. NPZ format is faster than A3M for large batches")
        
    except Exception as e:
        logger.error(f"❌ Error running examples: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
