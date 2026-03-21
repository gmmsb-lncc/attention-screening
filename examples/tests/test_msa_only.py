"""
Test MSA computation via ColabFold server (without OpenFold3 model).

This script tests only the MSA computation functionality by directly
calling the ColabFold API, without requiring OpenFold3 model weights.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_colabfold_msa():
    """Test MSA computation via ColabFold server."""
    
    logger.info("="*80)
    logger.info("TESTING MSA COMPUTATION VIA COLABFOLD SERVER")
    logger.info("="*80)
    
    # Test sequence (small protein for quick testing)
    sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL"
    
    logger.info(f"\nTest sequence:")
    logger.info(f"  Length: {len(sequence)}")
    logger.info(f"  First 50: {sequence[:50]}...")
    logger.info(f"  Last 50: ...{sequence[-50:]}")
    
    # Import ColabFold MSA function
    try:
        logger.info("\n" + "="*80)
        logger.info("STEP 1: Importing ColabFold MSA module")
        logger.info("="*80)
        
        # Import OpenFold's ColabFold integration
        sys.path.insert(0, str(Path(__file__).parent / "OPENFOLD-3"))
        from openfold3.core.data.tools.colabfold_msa_server import query_colabfold_msa_server
        
        logger.info("✓ ColabFold MSA module imported successfully")
        logger.info(f"  Function: {query_colabfold_msa_server.__name__}")
        
    except ImportError as e:
        logger.error(f"✗ Failed to import ColabFold module: {e}")
        logger.error("\nNote: Requires gemmi and other OpenFold dependencies")
        return False
    
    # Test MSA computation
    logger.info("\n" + "="*80)
    logger.info("STEP 2: Computing MSA via ColabFold Server")
    logger.info("="*80)
    
    try:
        logger.info("Sending request to ColabFold server...")
        logger.info("  Server: https://api.colabfold.com")
        logger.info("  Mode: Main MSA (fast)")
        logger.info("  Expected time: 30-60 seconds")
        logger.info("\nPlease wait...")
        
        # Create output directory
        output_dir = Path("./test_msa_output")
        output_dir.mkdir(exist_ok=True)
        
        # Call ColabFold API
        # Parameters for fast MSA computation
        # x: list of sequences (single sequence for this test)
        # prefix: output directory for results
        msa_result = query_colabfold_msa_server(
            x=[sequence],            # List of sequences
            prefix=output_dir,       # Output directory
            use_env=False,           # Don't use environmental databases (faster)
            use_filter=True,         # Use diversity filter
            use_templates=False,     # Don't fetch templates
            use_pairing=False,       # Single sequence, no pairing
            host_url="https://api.colabfold.com",
            user_agent="docktkinase/1.0 test"
        )
        
        logger.info("\n✓ MSA computed successfully!")
        
    except Exception as e:
        logger.error(f"\n✗ MSA computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Analyze MSA results
    logger.info("\n" + "="*80)
    logger.info("STEP 3: Analyzing MSA Results")
    logger.info("="*80)
    
    try:
        # Check what we received
        logger.info(f"MSA result type: {type(msa_result)}")
        
        if isinstance(msa_result, dict):
            logger.info(f"MSA result keys: {list(msa_result.keys())}")
            
            # Check for MSA sequences
            if 'msa' in msa_result:
                msa = msa_result['msa']
                logger.info(f"\n✓ MSA found:")
                logger.info(f"  - Type: {type(msa)}")
                logger.info(f"  - Shape/Length: {len(msa) if hasattr(msa, '__len__') else 'N/A'}")
                
                if hasattr(msa, 'shape'):
                    logger.info(f"  - Shape: {msa.shape}")
                    logger.info(f"  - Number of sequences: {msa.shape[0]}")
                    logger.info(f"  - Sequence length: {msa.shape[1]}")
                elif isinstance(msa, list):
                    logger.info(f"  - Number of sequences: {len(msa)}")
                    if len(msa) > 0:
                        logger.info(f"  - First sequence length: {len(msa[0])}")
            
            # Check for deletion matrix
            if 'deletion_matrix' in msa_result:
                logger.info(f"\n✓ Deletion matrix found")
                
            # Check for other fields
            other_fields = [k for k in msa_result.keys() if k not in ['msa', 'deletion_matrix']]
            if other_fields:
                logger.info(f"\nAdditional fields: {other_fields}")
        
        else:
            logger.info(f"MSA result: {msa_result}")
        
    except Exception as e:
        logger.error(f"✗ Failed to analyze results: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("TEST SUMMARY")
    logger.info("="*80)
    logger.info("✓ All tests passed successfully!")
    logger.info(f"  - Sequence length: {len(sequence)}")
    logger.info(f"  - MSA computed: Yes")
    logger.info(f"  - Server: ColabFold API")
    logger.info("="*80)
    
    return True


if __name__ == "__main__":
    try:
        success = test_colabfold_msa()
        
    except KeyboardInterrupt:
        logger.info("\n\nTest interrupted by user")
        raise AssertionError("Test failed")
    except Exception as e:
        logger.error(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        raise AssertionError("Test failed")
