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

def download_fm4m_model_files():
    """Download required FM4M model files with retry logic."""
    # Get the materials directory
    script_dir = Path(__file__).parent.absolute()
    project_dir = script_dir.parent
    materials_dir = project_dir / "materials"
    model_files_dir = materials_dir / "model_files"
    
    # Create model_files directory if it doesn't exist
    model_files_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Try to import huggingface_hub
        from huggingface_hub import hf_hub_download
        print("Downloading required FM4M model files...")
        
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
        
        print("\nFM4M model files download summary:")
        for filename, path in downloaded_files:
            print(f"✓ {filename}: {path}")
        
        print(f"\nAll FM4M model files downloaded successfully to {model_files_dir}")
        return True
        
    except Exception as e:
        print(f"Error during FM4M model file download: {e}")
        return False

def download_esm_model():
    """Download ESM model files by loading the model."""
    try:
        print("\nDownloading required ESM model files...")
        
        # Import ESM
        import torch
        import esm
        
        # Load the default ESM model (this will trigger download if needed)
        model_name = "esm2_t33_650M_UR50D"
        print(f"Loading ESM model {model_name}...")
        
        # This will download the model if it's not already present
        model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        
        # Move to device and set to eval mode
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()
        
        print(f"✓ Successfully downloaded and loaded ESM model {model_name}")
        return True
        
    except Exception as e:
        print(f"✗ Error downloading ESM model: {e}")
        return False

def verify_downloads():
    """Verify that the downloaded files exist and are accessible."""
    try:
        print("\nVerifying downloaded model files...")
        
        # Get the materials directory
        script_dir = Path(__file__).parent.absolute()
        project_dir = script_dir.parent
        materials_dir = project_dir / "materials"
        model_files_dir = materials_dir / "model_files"
        
        # Check if required FM4M files exist
        required_fm4m_files = [
            "bert_vocab_curated.txt",
            "smi-ted-Light_40.pt"
        ]
        
        all_files_exist = True
        for filename in required_fm4m_files:
            file_path = model_files_dir / filename
            if file_path.exists():
                print(f"✓ FM4M {filename}: Found")
            else:
                print(f"✗ FM4M {filename}: Not found")
                all_files_exist = False
        
        # Check if ESM model is accessible
        try:
            import esm
            import torch
            model, alphabet = esm.pretrained.load_model_and_alphabet("esm2_t33_650M_UR50D")
            print("✓ ESM model: Accessible")
        except Exception as e:
            print(f"✗ ESM model: Not accessible - {e}")
            all_files_exist = False
        
        if all_files_exist:
            print("✓ All required model files are present and accessible")
            return True
        else:
            print("✗ Some required model files are missing or not accessible")
            return False
        
    except Exception as e:
        print(f"✗ Error verifying model files: {e}")
        return False

if __name__ == "__main__":
    print("DockTKinase Post-Install Setup")
    print("=" * 40)
    
    # Download FM4M model files
    fm4m_success = download_fm4m_model_files()
    
    # Download ESM model files
    esm_success = download_esm_model()
    
    if fm4m_success and esm_success:
        # Verify downloads
        verify_success = verify_downloads()
        
        if verify_success:
            print("\n" + "=" * 40)
            print("✅ Post-install setup completed successfully!")
            print("DockTKinase is ready to use.")
            sys.exit(0)
        else:
            print("\n" + "=" * 40)
            print("⚠️  Model files downloaded but verification indicates some issues.")
            print("The pipeline may still work, but you should check the downloaded files.")
            # Exit with success code since downloading was successful
            sys.exit(0)
    else:
        print("\n" + "=" * 40)
        print("❌ Post-install setup failed!")
        print("Please check your internet connection and try again.")
        sys.exit(1)