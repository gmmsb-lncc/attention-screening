#!/usr/bin/env python3
"""
Test script to verify that the SMI-TED model is working correctly with local files.
"""

import os
import sys

# Add FM4M directory to Python path
fm4m_path = os.path.join(os.path.dirname(__file__), 'FM4M')
sys.path.append(fm4m_path)
sys.path.append(os.path.join(fm4m_path, 'models'))

def test_smi_ted_model():
    """Test the SMI-TED model loading and inference."""
    try:
        print("Testing SMI-TED model loading...")
        
        # Import the model loading function
        from models.smi_ted.smi_ted_light.load import load_smi_ted
        
        # Load the model
        model = load_smi_ted(folder="./materials/model_files")
        print("Model loaded successfully!")
        
        # Test with a simple SMILES string
        test_smiles = ["CCO", "CCN"]  # Ethanol and ethylamine
        print(f"Testing with SMILES: {test_smiles}")
        
        # Encode the SMILES
        embeddings = model.encode(test_smiles, return_torch=False)
        print(f"Embeddings shape: {embeddings.shape}")
        print("Embeddings generated successfully!")
        
        return True
        
    except Exception as e:
        print(f"Error testing SMI-TED model: {e}")
        return False

def test_fm4m_integration():
    """Test the FM4M integration."""
    try:
        print("\nTesting FM4M integration...")
        
        import fm4m
        
        # Test getting representations
        test_smiles = ["CCO", "CCN"]
        representations, _ = fm4m.get_representation(
            train_data=test_smiles,
            test_data=test_smiles,
            model_type="SMI-TED",
            return_tensor=False
        )
        
        print(f"FM4M representations shape: {representations.shape}")
        print("FM4M integration working correctly!")
        
        return True
        
    except Exception as e:
        print(f"Error testing FM4M integration: {e}")
        return False

if __name__ == "__main__":
    print("DockTKinase Model Test Script")
    print("=" * 40)
    
    # Test SMI-TED model
    smi_ted_success = test_smi_ted_model()
    
    # Test FM4M integration
    fm4m_success = test_fm4m_integration()
    
    print("\n" + "=" * 40)
    if smi_ted_success and fm4m_success:
        print("All tests passed! Models are working correctly.")
        sys.exit(0)
    else:
        print("Some tests failed. Please check the errors above.")
        sys.exit(1)