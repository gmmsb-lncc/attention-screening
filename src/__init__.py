"""
DockTKinase Source Package
==========================

This module initializes the DockTKinase package with proper ESM configuration.

IMPORTANT: ESM (Evolutionary Scale Modeling) must be loaded from the local
repository at llm/ESM/ to avoid segmentation faults that occur with pip-installed
versions on some systems (especially macOS).

This module automatically:
1. Adds llm/ESM to sys.path
2. Pre-imports ESM before any other module can import a conflicting version
3. Verifies ESM is properly loaded

Usage:
    # Simply importing src will ensure ESM is configured
    import src
    
    # Or import submodules directly
    from src.integrated_pipeline import IntegratedPipeline
"""

import sys
import warnings
from pathlib import Path

# ==============================================================================
# CRITICAL: Configure ESM before any other imports
# ==============================================================================

def _setup_esm():
    """
    Configure local ESM from llm/ESM repository.
    
    This function must be called before any other module imports ESM.
    It adds the local ESM path to sys.path and pre-imports the module
    to prevent pip-installed versions from being used.
    
    Returns:
        bool: True if ESM was successfully configured, False otherwise
    """
    # Calculate project root (parent of src/)
    project_root = Path(__file__).parent.parent
    esm_path = project_root / "llm" / "ESM"
    
    # Check if local ESM exists
    if not esm_path.exists():
        # Fallback to ESM/ in root
        esm_path = project_root / "ESM"
    
    if not esm_path.exists():
        warnings.warn(
            "Local ESM repository not found at llm/ESM or ESM/. "
            "Clone it with: git clone https://github.com/facebookresearch/esm.git llm/ESM",
            RuntimeWarning
        )
        return False
    
    # Check if ESM module structure exists
    esm_module_path = esm_path / "esm" / "__init__.py"
    if not esm_module_path.exists():
        warnings.warn(
            f"ESM module not found at {esm_module_path}. "
            "The ESM repository may be incomplete.",
            RuntimeWarning
        )
        return False
    
    # Add to sys.path if not already present
    esm_str = str(esm_path)
    if esm_str not in sys.path:
        sys.path.insert(0, esm_str)
    
    # Pre-import ESM to lock in the local version
    try:
        import esm
        
        # Verify it's the local version
        if hasattr(esm, '__file__') and esm.__file__:
            esm_file = Path(esm.__file__)
            if esm_path in esm_file.parents or esm_path == esm_file.parent.parent:
                return True
            else:
                warnings.warn(
                    f"ESM loaded from unexpected location: {esm.__file__}. "
                    f"Expected: {esm_path}",
                    RuntimeWarning
                )
        
        # Check for pretrained module (indicates valid ESM)
        if hasattr(esm, 'pretrained'):
            return True
        else:
            warnings.warn("ESM module loaded but missing 'pretrained' attribute", RuntimeWarning)
            return False
            
    except ImportError as e:
        warnings.warn(f"Failed to import ESM: {e}", RuntimeWarning)
        return False


# Run ESM setup at import time
_ESM_CONFIGURED = _setup_esm()

# Export configuration status
__all__ = ['_ESM_CONFIGURED']
