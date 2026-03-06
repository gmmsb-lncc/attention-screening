#!/usr/bin/env python3
"""
Launcher for DockTKinase - Configures environment and starts system.
"""

import sys
import os
from pathlib import Path

def setup_environment():
    """Configure Python environment."""
    # Add src to path
    src_path = Path(__file__).parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    print("🚀 DockTKinase Launcher")
    print("=" * 30)
    print(f"📁 Project: {Path.cwd()}")
    print(f"🐍 Python: {sys.version}")
    print(f"📦 Src path: {src_path}")

def test_system():
    """Test if the system is working."""
    try:
        from classifier.modular_classifier import main as classifier_main
        from classifier.modular_pipeline import ModularMLPPipeline
        from classifier.models.mlp_classifier import MLPEmbeddingClassifier
        from classifier.utils.import_utils import safe_import_optional
        
        print("✅ Modularized system loaded successfully!")
        
        # Test pipeline
        pipeline = ModularMLPPipeline()
        print("✅ Pipeline: OK")
        
        # Test model
        model = MLPEmbeddingClassifier(input_dim=100, hidden_dims=[64, 32])
        print("✅ MLP Model: OK")
        
        # Check optional dependencies
        optuna_available = safe_import_optional("optuna", "optimization")
        pyspark_available = safe_import_optional("pyspark", "distributed processing")
        
        print(f"🔧 Optuna: {'✅ Available' if optuna_available else '⚠️  Not available'}")
        print(f"🔧 PySpark: {'✅ Available' if pyspark_available else '⚠️  Not available'}")
        
        print("")
        print("System ready for use!")
        print("To get started:")
        print("  from classifier.modular_pipeline import ModularMLPPipeline")
        print("  pipeline = ModularMLPPipeline()")
        print("  # or use CLI:")
        print("  python src/classifier/modular_classifier.py --help")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading system: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    setup_environment()
    if test_system():
        print("\n🎉 DockTKinase is ready!")
    else:
        print("\n⚠️  Check installation")
        sys.exit(1)
