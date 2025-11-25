#!/usr/bin/env python3
"""
Download ESM-C models for DockTKinase

This script downloads ESM-C models (esmc-300m-2024-12 and esmc-600m-2024-12)
to the local model cache, resolving the namespace conflict between fair-esm and ESM-3.

Usage:
    python scripts/download_esmc_models.py [--model MODEL_NAME] [--cache-dir DIR]

Examples:
    # Download both models (default)
    python scripts/download_esmc_models.py
    
    # Download only esmc-300m
    python scripts/download_esmc_models.py --model esmc-300m-2024-12
    
    # Custom cache directory
    python scripts/download_esmc_models.py --cache-dir /path/to/cache
"""

import sys
import os
from pathlib import Path
import argparse


def setup_esm3_environment():
    """
    Setup ESM-3 environment to resolve namespace conflict with fair-esm.
    
    This prioritizes ESM-3 in sys.path and clears the module cache.
    """
    # Get workspace root
    script_dir = Path(__file__).resolve().parent
    workspace_root = script_dir.parent
    esm3_path = workspace_root / 'ESM' / 'esm-3' / 'esm-main'
    
    if not esm3_path.exists():
        print(f"❌ ESM-3 not found at: {esm3_path}")
        print(f"   Please clone ESM-3 repository first:")
        print(f"   git clone https://github.com/evolutionaryscale/esm.git {esm3_path}")
        sys.exit(1)
    
    # Clear any existing esm modules from cache
    esm_modules = [key for key in list(sys.modules.keys()) if key.startswith('esm')]
    for mod_key in esm_modules:
        del sys.modules[mod_key]
    
    # Prioritize ESM-3 in sys.path
    esm3_path_str = str(esm3_path)
    if esm3_path_str in sys.path:
        sys.path.remove(esm3_path_str)
    sys.path.insert(0, esm3_path_str)
    
    print(f"✅ ESM-3 environment configured")
    print(f"   Path: {esm3_path}")
    print(f"   Cleared {len(esm_modules)} esm modules from cache")
    print()
    
    return esm3_path, workspace_root


def download_model(model_name: str, cache_dir: Path, device: str = 'cpu'):
    """
    Download ESM-C model using ESMC.from_pretrained().
    
    Args:
        model_name: Model identifier (e.g., 'esmc-300m-2024-12')
        cache_dir: Directory to store downloaded models
        device: Device to load model on ('cpu' or 'cuda')
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Import ESMC (after environment setup)
        from esm.models.esmc import ESMC
        
        print(f"📥 Downloading {model_name}...")
        print(f"   Target directory: {cache_dir}")
        print(f"   This may take several minutes depending on your connection...")
        print()
        
        # Set cache directory
        os.environ['ESM_DATA_ROOT'] = str(cache_dir)
        
        # Download model (from_pretrained auto-downloads if not found)
        model = ESMC.from_pretrained(model_name, device=device)
        
        print(f"✅ {model_name} downloaded successfully!")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Location: {cache_dir}")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to download {model_name}")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_model(model_name: str, cache_dir: Path):
    """
    Verify that model was downloaded correctly by loading it.
    
    Args:
        model_name: Model identifier
        cache_dir: Cache directory where model should be
    
    Returns:
        True if model loads successfully, False otherwise
    """
    try:
        from esm.models.esmc import ESMC
        
        print(f"🔍 Verifying {model_name}...")
        os.environ['ESM_DATA_ROOT'] = str(cache_dir)
        
        # Try to load model
        model = ESMC.from_pretrained(model_name, device='cpu')
        
        # Test tokenizer
        tokenizer = model.tokenizer
        test_seq = "MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV"
        tokens = tokenizer.encode(test_seq)
        
        print(f"   ✅ Model loads successfully")
        print(f"   ✅ Tokenizer works (tokenized {len(tokens)} tokens)")
        print()
        
        return True
        
    except Exception as e:
        print(f"   ❌ Verification failed: {e}")
        print()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Download ESM-C models for DockTKinase',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['esmc-300m-2024-12', 'esmc-600m-2024-12', 'both'],
        default='both',
        help='Model to download (default: both)'
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        default=None,
        help='Cache directory for models (default: ./llm/models_cache/ESM3)'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda'],
        default='cpu',
        help='Device to load models on (default: cpu)'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify models after download'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("ESM-C Model Downloader for DockTKinase")
    print("=" * 70)
    print()
    
    # Setup ESM-3 environment
    esm3_path, workspace_root = setup_esm3_environment()
    
    # Determine cache directory
    if args.cache_dir:
        cache_dir = Path(args.cache_dir).resolve()
    else:
        cache_dir = workspace_root / 'llm' / 'models_cache' / 'ESM3'
    
    # Create cache directory if needed
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Cache directory: {cache_dir}")
    print()
    
    # Determine which models to download
    if args.model == 'both':
        models = ['esmc-300m-2024-12', 'esmc-600m-2024-12']
    else:
        models = [args.model]
    
    print(f"📦 Models to download: {', '.join(models)}")
    print()
    
    # Download models
    success_count = 0
    for model_name in models:
        if download_model(model_name, cache_dir, args.device):
            success_count += 1
            
            # Verify if requested
            if args.verify:
                verify_model(model_name, cache_dir)
    
    # Summary
    print("=" * 70)
    print("Download Summary")
    print("=" * 70)
    print(f"✅ Successfully downloaded: {success_count}/{len(models)} models")
    if success_count == len(models):
        print()
        print("🎉 All models downloaded successfully!")
        print()
        print("Next steps:")
        print("1. Test ESM-C with: python examples/demo_esmc_phase1.py")
        print("2. Run full test suite: pytest tests/test_esmc_strategy.py")
        print("3. Integrate with pipeline: see docs/05-development/PHASE1_ESMC_IMPLEMENTATION.md")
    else:
        print(f"⚠️ Some downloads failed. Check errors above.")
        sys.exit(1)


if __name__ == '__main__':
    main()
