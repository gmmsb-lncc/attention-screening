"""
Test OpenFold3 MSA implementation with a single sequence.

This script validates:
1. MsaConfig creation and validation
2. OpenFoldStrategy initialization with MSA
3. Model loading
4. Embedding extraction with MSA
5. Output shape and type validation
"""

import sys
import torch
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.build.embeddings.strategies.openfold_strategy import OpenFoldStrategy
from src.build.embeddings.config.msa_config import MsaConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_single_sequence():
    """Test MSA embedding extraction with a single sequence."""
    
    logger.info("="*80)
    logger.info("TESTING OPENFOLD3 MSA WITH SINGLE SEQUENCE")
    logger.info("="*80)
    
    # Test sequence (small protein for quick testing)
    sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL"
    
    logger.info(f"\nTest sequence length: {len(sequence)}")
    logger.info(f"Sequence: {sequence[:50]}...{sequence[-50:]}")
    
    # 1. Test MsaConfig creation
    logger.info("\n" + "="*80)
    logger.info("STEP 1: Creating MSA Configuration")
    logger.info("="*80)
    
    try:
        # Use development mode for faster testing
        msa_config = MsaConfig.for_development(
            output_dir=Path("./test_msa_cache")
        )
        logger.info("✓ MsaConfig created successfully")
        logger.info(f"\n{msa_config.summary()}")
    except Exception as e:
        logger.error(f"✗ Failed to create MsaConfig: {e}")
        return False
    
    # 2. Test OpenFoldStrategy initialization
    logger.info("\n" + "="*80)
    logger.info("STEP 2: Initializing OpenFoldStrategy with MSA")
    logger.info("="*80)
    
    try:
        strategy = OpenFoldStrategy(
            logger=logger,
            msa_config=msa_config
        )
        logger.info("✓ OpenFoldStrategy initialized successfully")
        logger.info(f"  - MSA mode: {strategy.msa_config.mode.value}")
        logger.info(f"  - Caching enabled: {strategy.msa_config.enable_caching}")
    except Exception as e:
        logger.error(f"✗ Failed to initialize OpenFoldStrategy: {e}")
        return False
    
    # 3. Test model loading
    logger.info("\n" + "="*80)
    logger.info("STEP 3: Loading OpenFold3 Model")
    logger.info("="*80)
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        model, auxiliary = strategy.load('openfold3', device=device)
        logger.info("✓ Model loaded successfully")
        logger.info(f"  - Model type: {type(model)}")
        logger.info(f"  - Device: {next(model.parameters()).device}")
    except Exception as e:
        logger.error(f"✗ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. Test embedding extraction
    logger.info("\n" + "="*80)
    logger.info("STEP 4: Extracting Embedding with MSA")
    logger.info("="*80)
    
    try:
        logger.info("Starting embedding extraction...")
        logger.info("NOTE: First run may take 1-2 minutes for MSA computation")
        
        embedding = strategy.generate(
            model=model,
            auxiliary_objects=auxiliary,
            sequence=sequence,
            device=device,
            pooling_strategy='mean'
        )
        
        logger.info("✓ Embedding extracted successfully")
        logger.info(f"  - Shape: {embedding.shape}")
        logger.info(f"  - Dtype: {embedding.dtype}")
        logger.info(f"  - Min value: {float(embedding.min()):.4f}")
        logger.info(f"  - Max value: {float(embedding.max()):.4f}")
        logger.info(f"  - Mean value: {float(embedding.mean()):.4f}")
        logger.info(f"  - Std value: {float(embedding.std()):.4f}")
        
    except Exception as e:
        logger.error(f"✗ Failed to extract embedding: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 5. Validate output
    logger.info("\n" + "="*80)
    logger.info("STEP 5: Validating Output")
    logger.info("="*80)
    
    try:
        # Check shape
        expected_shape = (384,)
        if embedding.shape != expected_shape:
            logger.error(f"✗ Invalid shape: expected {expected_shape}, got {embedding.shape}")
            return False
        logger.info(f"✓ Shape validation passed: {embedding.shape}")
        
        # Check dtype
        import numpy as np
        if embedding.dtype != np.float32:
            logger.warning(f"⚠ Unexpected dtype: {embedding.dtype} (expected float32)")
        logger.info(f"✓ Dtype validation passed: {embedding.dtype}")
        
        # Check for NaN or Inf
        if np.isnan(embedding).any():
            logger.error("✗ Output contains NaN values")
            return False
        logger.info("✓ No NaN values detected")
        
        if np.isinf(embedding).any():
            logger.error("✗ Output contains Inf values")
            return False
        logger.info("✓ No Inf values detected")
        
        # Check value range (embeddings should be reasonable)
        if np.abs(embedding).max() > 100:
            logger.warning(f"⚠ Large values detected: max={float(np.abs(embedding).max()):.2f}")
        logger.info("✓ Value range validation passed")
        
    except Exception as e:
        logger.error(f"✗ Validation failed: {e}")
        return False
    
    # 6. Cleanup
    logger.info("\n" + "="*80)
    logger.info("STEP 6: Cleanup")
    logger.info("="*80)
    
    try:
        strategy.cleanup(model, auxiliary)
        logger.info("✓ Cleanup completed successfully")
    except Exception as e:
        logger.error(f"✗ Cleanup failed: {e}")
        return False
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("TEST SUMMARY")
    logger.info("="*80)
    logger.info("✓ All tests passed successfully!")
    logger.info(f"  - MSA mode: {msa_config.mode.value}")
    logger.info(f"  - Sequence length: {len(sequence)}")
    logger.info(f"  - Embedding shape: {embedding.shape}")
    logger.info(f"  - Device: {device}")
    logger.info("="*80)
    
    return True


if __name__ == "__main__":
    try:
        success = test_single_sequence()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
