#!/usr/bin/env python3
"""
Post-install script to download required model files for DockTKinase.
This script runs automatically after creating the conda environment.
"""

import os
import sys
import subprocess
import time
import random
from pathlib import Path

def download_model_files():
    """Download required model files with retry logic."""
    # Get the materials directory
    script_dir = Path(__file__).parent.absolute()
    materials_dir = script_dir / "materials"
    model_files_dir = materials_dir / "model_files"
    
    # Create model_files directory if it doesn't exist
    model_files_dir.mkdir(parents=True, exist_ok=True)
    
    # Add materials to Python path
    sys.path.append(str(materials_dir))
    
    try:
        # Try to import huggingface_hub
        from huggingface_hub import hf_hub_download
        print("Downloading required model files...")
        
        # Files to download
        files = [
            ("ibm/materials.smi-ted", "bert_vocab_curated.txt"),
            ("ibm/materials.smi-ted", "smi-ted-Light_40.pt")
        ]
        
        downloaded_files = []
        
        for repo_id, filename in files:
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    print(f"Downloading {filename} (attempt {attempt + 1}/{max_retries})...")
                    file_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        local_dir=str(model_files_dir),
                        resume_download=True
                    )
                    downloaded_files.append((filename, file_path))
                    print(f"✓ Successfully downloaded {filename}")
                    break
                except Exception as e:
                    if "429" in str(e) or "Too Many Requests" in str(e):
                        if attempt < max_retries - 1:
                            wait_time = (2 ** attempt) + random.uniform(0, 1)
                            print(f"Rate limited. Waiting {wait_time:.2f} seconds before retry...")
                            time.sleep(wait_time)
                        else:
                            print(f"✗ Failed to download {filename} after {max_retries} attempts")
                            raise e
                    else:
                        print(f"✗ Error downloading {filename}: {e}")
                        if attempt == max_retries - 1:
                            raise e
                        else:
                            wait_time = (2 ** attempt) + random.uniform(0, 1)
                            print(f"Waiting {wait_time:.2f} seconds before retry...")
                            time.sleep(wait_time)
        
        print("\nDownload summary:")
        for filename, path in downloaded_files:
            print(f"✓ {filename}: {path}")
        
        print(f"\nAll model files downloaded successfully to {model_files_dir}")
        return True
        
    except Exception as e:
        print(f"Error during model file download: {e}")
        return False

def verify_downloads():
    """Verify that the downloaded files work correctly."""
    try:
        print("\nVerifying downloaded model files...")
        
        # Add materials directory to Python path
        materials_path = Path(__file__).parent.absolute() / "materials"
        sys.path.append(str(materials_path))
        sys.path.append(str(materials_path / "models"))
        
        # Import the model loading function
        from models.smi_ted.smi_ted_light.load import load_smi_ted
        
        # Load the model
        model = load_smi_ted(folder=str(materials_path / "model_files"))
        print("✓ Model loaded successfully!")
        
        # Test with a simple SMILES string
        test_smiles = ["CCO"]  # Ethanol
        embeddings = model.encode(test_smiles, return_torch=False)
        print(f"✓ Embeddings generated successfully! Shape: {embeddings.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error verifying model files: {e}")
        return False

if __name__ == "__main__":
    print("DockTKinase Post-Install Setup")
    print("=" * 40)
    
    # Download model files
    download_success = download_model_files()
    
    if download_success:
        # Verify downloads
        verify_success = verify_downloads()
        
        if verify_success:
            print("\n" + "=" * 40)
            print("✅ Post-install setup completed successfully!")
            print("DockTKinase is ready to use.")
            sys.exit(0)
        else:
            print("\n" + "=" * 40)
            print("⚠️  Model files downloaded but verification failed.")
            print("You may need to check the downloaded files.")
            sys.exit(1)
    else:
        print("\n" + "=" * 40)
        print("❌ Post-install setup failed!")
        print("Please check your internet connection and try again.")
        sys.exit(1)