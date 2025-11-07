"""
Test 3: ModelRegistry - Model catalog and information
======================================================
Tests the model registry with 7 ESM models + FM4M.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from build.embeddings.models.model_registry import ModelRegistry, ModelInfo


def test_model_registry_basic():
    """Test 3.1: Basic model registry operations"""
    print("\n" + "="*70)
    print("TEST 3.1: Model Registry - Basic Operations")
    print("="*70)
    
    # Get all ESM models
    esm_models = ModelRegistry.get_models_by_type('esm')
    print(f"\n📊 ESM Models Available: {len(esm_models)}")
    for name in esm_models.keys():
        print(f"   - {name}")
    
    assert len(esm_models) == 7, f"Expected 7 ESM models, got {len(esm_models)}"
    
    # Get all FM4M models
    fm4m_models = ModelRegistry.get_models_by_type('fm4m')
    print(f"\n📊 FM4M Models Available: {len(fm4m_models)}")
    for name in fm4m_models.keys():
        print(f"   - {name}")
    
    assert len(fm4m_models) >= 1, f"Expected at least 1 FM4M model"
    
    # Test default models
    default_esm = ModelRegistry.get_default_model('esm')
    default_fm4m = ModelRegistry.get_default_model('fm4m')
    
    print(f"\n📊 Default Models:")
    print(f"   - ESM: {default_esm}")
    print(f"   - FM4M: {default_fm4m}")
    
    assert default_esm in esm_models
    assert default_fm4m in fm4m_models
    
    print("\n✅ TEST 3.1 PASSED!")


def test_model_info():
    """Test 3.2: Model information retrieval"""
    print("\n" + "="*70)
    print("TEST 3.2: Model Information")
    print("="*70)
    
    # Test ESM model info
    esm_model_name = "esm2_t33_650M_UR50D"
    esm_info = ModelRegistry.get_model_info(esm_model_name)
    
    print(f"\n📊 ESM Model: {esm_model_name}")
    print(f"   - Type: {esm_info.type}")
    print(f"   - Embedding Dim: {esm_info.embedding_dim}")
    print(f"   - Default Layer: {esm_info.default_layer}")
    print(f"   - GPU Required: {esm_info.requires_gpu}")
    print(f"   - Description: {esm_info.description[:60]}...")
    
    assert esm_info.type == "esm"
    assert esm_info.embedding_dim == 1280
    assert esm_info.default_layer == 33
    
    # Test FM4M model info
    fm4m_model_name = "smi_ted_light"
    fm4m_info = ModelRegistry.get_model_info(fm4m_model_name)
    
    print(f"\n📊 FM4M Model: {fm4m_model_name}")
    print(f"   - Type: {fm4m_info.type}")
    print(f"   - Embedding Dim: {fm4m_info.embedding_dim}")
    print(f"   - Description: {fm4m_info.description[:60]}...")
    
    assert fm4m_info.type == "fm4m"
    assert fm4m_info.embedding_dim == 768
    
    print("\n✅ TEST 3.2 PASSED!")


def test_model_validation():
    """Test 3.3: Model name validation"""
    print("\n" + "="*70)
    print("TEST 3.3: Model Validation")
    print("="*70)
    
    # Valid models
    valid_esm = "esm2_t33_650M_UR50D"
    valid_fm4m = "smi_ted_light"
    
    print(f"\n📊 Validating ESM: {valid_esm}")
    assert ModelRegistry.is_valid_model(valid_esm, 'esm'), \
        f"{valid_esm} should be valid ESM model"
    
    print(f"   ✅ Valid ESM model")
    
    print(f"\n📊 Validating FM4M: {valid_fm4m}")
    assert ModelRegistry.is_valid_model(valid_fm4m, 'fm4m'), \
        f"{valid_fm4m} should be valid FM4M model"
    
    print(f"   ✅ Valid FM4M model")
    
    # Invalid models
    invalid_model = "invalid_model_name"
    
    print(f"\n📊 Validating invalid: {invalid_model}")
    assert not ModelRegistry.is_valid_model(invalid_model, 'esm'), \
        f"{invalid_model} should be invalid"
    assert not ModelRegistry.is_valid_model(invalid_model, 'fm4m'), \
        f"{invalid_model} should be invalid"
    
    print(f"   ✅ Correctly rejected invalid model")
    
    print("\n✅ TEST 3.3 PASSED!")


def test_all_esm_models_info():
    """Test 3.4: Info for all ESM models"""
    print("\n" + "="*70)
    print("TEST 3.4: All ESM Models Info")
    print("="*70)
    
    esm_models = ModelRegistry.get_models_by_type('esm')
    
    print(f"\n📊 Checking info for {len(esm_models)} ESM models:\n")
    
    for model_name in esm_models.keys():
        info = ModelRegistry.get_model_info(model_name)
        print(f"   {model_name}:")
        print(f"      - Dim: {info.embedding_dim}, Layer: {info.default_layer}, "
              f"GPU: {info.requires_gpu}")
    
    print("\n✅ TEST 3.4 PASSED!")


if __name__ == "__main__":
    print("\n" + "🧪 RUNNING MODEL REGISTRY TESTS ".center(70, "="))
    
    try:
        test_model_registry_basic()
        test_model_info()
        test_model_validation()
        test_all_esm_models_info()
        
        print("\n" + "="*70)
        print("✅ ALL MODEL REGISTRY TESTS PASSED!".center(70))
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise
