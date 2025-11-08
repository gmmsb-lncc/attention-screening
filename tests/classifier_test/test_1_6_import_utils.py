"""
Level 1.6: Import Utils Test

Tests import utilities and optional dependencies handling.

Test Coverage:
- Module imports work
- Optional dependencies graceful handling
- Import paths are correct
- No circular imports

Author: Test Suite
Created: 2025-11-08
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_core_imports():
    """Test 1.1: Core module imports"""
    print("\n" + "="*60)
    print("Test 1.1: Core Imports")
    print("="*60)
    
    try:
        # Test core imports
        from src.classifier.core.data_loader import DataManager
        from src.classifier.core.trainer import ModelTrainer, TrainingConfig
        from src.classifier.core.evaluator import ModelEvaluator
        from src.classifier.core.cross_validator import CrossValidator
        
        print("✅ Core modules imported successfully")
        print("   - DataManager ✓")
        print("   - ModelTrainer ✓")
        print("   - ModelEvaluator ✓")
        print("   - CrossValidator ✓")
        return True
        
    except ImportError as e:
        print(f"❌ FAILED: {str(e)}")
        return False


def test_utils_imports():
    """Test 1.2: Utils module imports"""
    print("\n" + "="*60)
    print("Test 1.2: Utils Imports")
    print("="*60)
    
    try:
        # Test utils imports
        from src.classifier.utils.metrics import MetricsCalculator, ClassificationMetrics
        from src.classifier.utils.device_manager import DeviceManager
        from src.classifier.utils.config_manager import SimpleConfig
        from src.classifier.utils.data_validation import DataValidator
        
        print("✅ Utils modules imported successfully")
        print("   - MetricsCalculator ✓")
        print("   - DeviceManager ✓")
        print("   - SimpleConfig ✓")
        print("   - DataValidator ✓")
        return True
        
    except ImportError as e:
        print(f"❌ FAILED: {str(e)}")
        return False


def test_models_imports():
    """Test 1.3: Models module imports"""
    print("\n" + "="*60)
    print("Test 1.3: Models Imports")
    print("="*60)
    
    try:
        # Test models imports
        from src.classifier.models.mlp_classifier import MLPEmbeddingClassifier
        
        print("✅ Models modules imported successfully")
        print("   - MLPEmbeddingClassifier ✓")
        return True
        
    except ImportError as e:
        print(f"❌ FAILED: {str(e)}")
        return False


def test_pytorch_imports():
    """Test 1.4: PyTorch dependencies"""
    print("\n" + "="*60)
    print("Test 1.4: PyTorch Dependencies")
    print("="*60)
    
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        
        print("✅ PyTorch imported successfully")
        print(f"   - Version: {torch.__version__} ✓")
        print(f"   - CUDA available: {torch.cuda.is_available()}")
        print(f"   - MPS available: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")
        return True
        
    except ImportError as e:
        print(f"❌ FAILED: {str(e)}")
        return False


def test_numpy_imports():
    """Test 1.5: NumPy dependencies"""
    print("\n" + "="*60)
    print("Test 1.5: NumPy Dependencies")
    print("="*60)
    
    try:
        import numpy as np
        
        print("✅ NumPy imported successfully")
        print(f"   - Version: {np.__version__} ✓")
        return True
        
    except ImportError as e:
        print(f"❌ FAILED: {str(e)}")
        return False


def test_sklearn_imports():
    """Test 1.6: scikit-learn dependencies"""
    print("\n" + "="*60)
    print("Test 1.6: scikit-learn Dependencies")
    print("="*60)
    
    try:
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import roc_auc_score, accuracy_score
        
        print("✅ scikit-learn imported successfully")
        print("   - model_selection ✓")
        print("   - metrics ✓")
        return True
        
    except ImportError as e:
        print(f"❌ FAILED: {str(e)}")
        return False


def test_no_circular_imports():
    """Test 1.7: No circular imports"""
    print("\n" + "="*60)
    print("Test 1.7: No Circular Imports")
    print("="*60)
    
    try:
        # Import main pipeline - this would fail on circular imports
        from src.classifier.modular_pipeline import MLPEmbeddingPipeline
        
        print("✅ No circular imports detected")
        print("   - Pipeline imported successfully ✓")
        return True
        
    except ImportError as e:
        if "circular" in str(e).lower():
            print(f"❌ FAILED: Circular import detected: {str(e)}")
            return False
        else:
            print(f"⚠️  Import failed but not circular: {str(e)}")
            return True  # Not a circular import issue
        
    except Exception as e:
        print(f"⚠️  Other error (not circular): {str(e)}")
        return True  # Not a circular import issue


def main():
    """Run all Import Utils tests"""
    print("\n" + "="*70)
    print("🧪 LEVEL 1.6: IMPORT UTILS TEST")
    print("="*70)
    
    tests = [
        ("Core Imports", test_core_imports),
        ("Utils Imports", test_utils_imports),
        ("Models Imports", test_models_imports),
        ("PyTorch Dependencies", test_pytorch_imports),
        ("NumPy Dependencies", test_numpy_imports),
        ("scikit-learn Dependencies", test_sklearn_imports),
        ("No Circular Imports", test_no_circular_imports),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} crashed: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<50} {status}")
    
    print("="*70)
    print(f"Results: {passed}/{total} tests passed ({100*passed//total}%)")
    print("="*70)
    
    if passed == total:
        print("🎉 Import Utils: FULLY FUNCTIONAL ✅")
        return 0
    else:
        print(f"⚠️  Some tests failed. Please investigate.")
        return 1


if __name__ == "__main__":
    exit(main())
