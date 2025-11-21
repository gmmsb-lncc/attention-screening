"""
Test OpenFold3 basic loading without MSA.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.build.embeddings.strategies.openfold_strategy import OpenFoldStrategy

def test_basic_import():
    """Test basic OpenFold3 import."""
    print("Testing OpenFold3 basic import...")
    
    try:
        strategy = OpenFoldStrategy()
        print("✓ OpenFoldStrategy created (no MSA)")
        
        # Just test the import, don't load model yet
        print("\nChecking OpenFold-3 directory...")
        openfold_dir = Path("/Users/sulfierry/docktkinase/OPENFOLD-3")
        print(f"  - OPENFOLD-3 exists: {openfold_dir.exists()}")
        print(f"  - __init__.py exists: {(openfold_dir / 'openfold3' / '__init__.py').exists()}")
        
        print("\n✅ Basic setup is working!")
        print("\nNote: Model loading requires downloading weights first.")
        print("Use: OPENFOLD-3/openfold3/scripts/download_openfold3_params.sh")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_basic_import()
    sys.exit(0 if success else 1)
