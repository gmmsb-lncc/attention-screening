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
    # Get the FM4M directory
    script_dir = Path(__file__).parent.absolute()
    project_dir = script_dir.parent
    fm4m_dir = project_dir / "FM4M"
    model_files_dir = fm4m_dir / "model_files"
    
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

def install_fm4m_dependencies():
    """Install FM4M dependencies (torch_nl, ase, and torch-scatter)."""
    try:
        print("\nInstalling FM4M dependencies...")
        
        # List of required packages for FM4M
        required_packages = [
            ("torch_nl", "0.3"),  # Neural network layers
            ("ase", None),        # Atomic Simulation Environment
        ]
        
        installed_packages = []
        failed_packages = []
        
        for package_name, version in required_packages:
            try:
                # Try to import the package first
                try:
                    __import__(package_name)
                    print(f"✓ {package_name} is already installed")
                    installed_packages.append(package_name)
                    continue
                except ImportError:
                    pass
                
                # Install the package
                package_spec = f"{package_name}=={version}" if version else package_name
                print(f"Installing {package_spec}...")
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", package_spec],
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minutes timeout per package
                )
                
                if result.returncode == 0:
                    print(f"✓ Successfully installed {package_spec}")
                    installed_packages.append(package_name)
                else:
                    print(f"✗ Error installing {package_spec}: {result.stderr}")
                    failed_packages.append(package_name)
                    
            except subprocess.TimeoutExpired:
                print(f"✗ Timeout installing {package_name}")
                failed_packages.append(package_name)
            except Exception as e:
                print(f"✗ Error installing {package_name}: {e}")
                failed_packages.append(package_name)
        
        # Install torch-scatter (PyTorch Geometric extension)
        try:
            import torch_scatter
            print("✓ torch-scatter is already installed")
            installed_packages.append("torch-scatter")
        except ImportError:
            print("Installing torch-scatter...")
            # Get PyTorch version to match torch-scatter wheel
            try:
                import torch
                torch_version = torch.__version__.split("+")[0]  # e.g., "2.5.0"
                cuda_version = torch.version.cuda.replace(".", "")  # e.g., "121" for CUDA 12.1
                
                # Install torch-scatter from PyG wheel index
                wheel_url = f"https://data.pyg.org/whl/torch-{torch_version}+cu{cuda_version}.html"
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "torch-scatter", "-f", wheel_url],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if result.returncode == 0:
                    print("✓ Successfully installed torch-scatter")
                    installed_packages.append("torch-scatter")
                else:
                    print(f"✗ Error installing torch-scatter: {result.stderr}")
                    failed_packages.append("torch-scatter")
            except Exception as e:
                print(f"✗ Error installing torch-scatter: {e}")
                failed_packages.append("torch-scatter")
        
        # Print summary
        print(f"\nFM4M dependencies installation summary:")
        print(f"  Installed: {len(installed_packages)}/{len(required_packages) + 1}")  # +1 for torch-scatter
        if failed_packages:
            print(f"  Failed: {', '.join(failed_packages)}")
        
        # Return success if all packages were installed
        return len(failed_packages) == 0
        
    except Exception as e:
        print(f"✗ Error installing FM4M dependencies: {e}")
        return False


def install_openfold_dependencies():
    """Install OpenFold3 and MSA dependencies."""
    try:
        print("\nInstalling OpenFold3 and MSA dependencies...")
        
        # List of required packages for OpenFold3 + MSA
        required_packages = [
            "gemmi",              # Crystal structure library (required by OpenFold3)
            "ml-collections",     # Configuration management (required by OpenFold3)
            "einops",            # Tensor operations (required by OpenFold3)
            "biopython",         # Biological sequence analysis (required by OpenFold3)
            "pydantic",          # Data validation (required by ColabFold API)
            "lmdb",              # Database (required by OpenFold3 data pipeline)
            "biotite",           # Bioinformatics toolkit (required by OpenFold3)
            "memory_profiler",   # Memory profiling (required by OpenFold3)
            "lightning",         # PyTorch Lightning (required by OpenFold3 training)
        ]
        
        installed_packages = []
        failed_packages = []
        
        for package in required_packages:
            try:
                # Try to import the package first
                try:
                    if package == "ml-collections":
                        import ml_collections
                    elif package == "memory_profiler":
                        import memory_profiler
                    else:
                        __import__(package)
                    print(f"✓ {package} is already installed")
                    installed_packages.append(package)
                    continue
                except ImportError:
                    pass
                
                # Install the package
                print(f"Installing {package}...")
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", package],
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minutes timeout per package
                )
                
                if result.returncode == 0:
                    print(f"✓ Successfully installed {package}")
                    installed_packages.append(package)
                else:
                    print(f"✗ Error installing {package}: {result.stderr}")
                    failed_packages.append(package)
                    
            except subprocess.TimeoutExpired:
                print(f"✗ Timeout installing {package}")
                failed_packages.append(package)
            except Exception as e:
                print(f"✗ Error installing {package}: {e}")
                failed_packages.append(package)
        
        # Print summary
        print(f"\nOpenFold3 dependencies installation summary:")
        print(f"  Installed: {len(installed_packages)}/{len(required_packages)}")
        if failed_packages:
            print(f"  Failed: {', '.join(failed_packages)}")
        
        # Return success if all packages were installed
        return len(failed_packages) == 0
        
    except Exception as e:
        print(f"✗ Error installing OpenFold3 dependencies: {e}")
        return False


def install_boltz_dependencies():
    """Install Boltz-2 specific dependencies."""
    try:
        print("\nInstalling Boltz-2 dependencies...")
        
        # List of required packages for Boltz-2
        required_packages = [
            "einx",                    # Tensor operations extension
            "fairscale",               # Distributed training
            "hydra-core",              # Configuration management
            "omegaconf",               # Config files (installed with hydra)
            "mashumaro",               # Data serialization
            "chembl-structure-pipeline",  # Chemical structure processing
            "numba",                   # JIT compilation
        ]
        
        installed_packages = []
        failed_packages = []
        
        for package in required_packages:
            try:
                # Try to import the package first
                try:
                    if package == "hydra-core":
                        import hydra
                    elif package == "chembl-structure-pipeline":
                        import chembl_structure_pipeline
                    else:
                        __import__(package.replace("-", "_"))
                    print(f"✓ {package} is already installed")
                    installed_packages.append(package)
                    continue
                except ImportError:
                    pass
                
                # Install the package
                print(f"Installing {package}...")
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", package],
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minutes timeout per package
                )
                
                if result.returncode == 0:
                    print(f"✓ Successfully installed {package}")
                    installed_packages.append(package)
                else:
                    print(f"✗ Error installing {package}: {result.stderr}")
                    failed_packages.append(package)
                    
            except subprocess.TimeoutExpired:
                print(f"✗ Timeout installing {package}")
                failed_packages.append(package)
            except Exception as e:
                print(f"✗ Error installing {package}: {e}")
                failed_packages.append(package)
        
        # Print summary
        print(f"\nBoltz-2 dependencies installation summary:")
        print(f"  Installed: {len(installed_packages)}/{len(required_packages)}")
        if failed_packages:
            print(f"  Failed: {', '.join(failed_packages)}")
        
        # Return success if all packages were installed
        return len(failed_packages) == 0
        
    except Exception as e:
        print(f"✗ Error installing Boltz-2 dependencies: {e}")
        return False


def install_esm3_dependencies():
    """Install ESM-3 (ESM-C) from local repository."""
    try:
        print("\nInstalling ESM-3 (ESM-C) dependencies...")
        
        # Get ESM-3 path
        script_dir = Path(__file__).parent.absolute()
        project_dir = script_dir.parent
        esm3_path = project_dir / "llm" / "ESM" / "esm-3" / "esm-main"
        
        if not esm3_path.exists():
            print(f"⚠️  ESM-3 not found at: {esm3_path}")
            print("   ESM-C models will not be available.")
            print("   To install ESM-3, run:")
            print(f"   mkdir -p {esm3_path.parent}")
            print(f"   git clone https://github.com/evolutionaryscale/esm.git {esm3_path}")
            print(f"   cd {esm3_path} && pip install -e .")
            return False
        
        # Check if already installed by trying to import
        try:
            # Clear any cached esm modules
            import sys as sys_module
            esm_modules = [k for k in list(sys_module.modules.keys()) if k.startswith('esm')]
            for mod in esm_modules:
                del sys_module.modules[mod]
            
            # Temporarily add ESM-3 to path
            sys_module.path.insert(0, str(esm3_path))
            from esm.models.esmc import ESMC
            print("✓ ESM-3 (ESM-C) is already installed")
            return True
        except ImportError:
            pass
        
        # Install ESM-3 in editable mode
        print(f"Installing ESM-3 from {esm3_path}...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", str(esm3_path)],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes timeout
            cwd=str(esm3_path)
        )
        
        if result.returncode == 0:
            print("✓ Successfully installed ESM-3 (ESM-C)")
            return True
        else:
            print(f"✗ Error installing ESM-3: {result.stderr}")
            # Try to provide more helpful error message
            if "einops" in result.stderr.lower():
                print("   Missing dependency: einops")
                print("   Run: pip install einops")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Timeout installing ESM-3")
        return False
    except Exception as e:
        print(f"✗ Error installing ESM-3: {e}")
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
        
        # Get the FM4M directory
        script_dir = Path(__file__).parent.absolute()
        project_dir = script_dir.parent
        fm4m_dir = project_dir / "FM4M"
        model_files_dir = fm4m_dir / "model_files"
        
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
    
    # Install FM4M dependencies (torch_nl and ase)
    fm4m_deps_success = install_fm4m_dependencies()
    
    # Install OpenFold3 and MSA dependencies
    openfold_success = install_openfold_dependencies()
    
    # Install Boltz-2 dependencies
    boltz_success = install_boltz_dependencies()
    
    # Install ESM-3 (ESM-C) dependencies
    esm3_success = install_esm3_dependencies()
    
    # Download FM4M model files
    fm4m_success = download_fm4m_model_files()
    
    # Download ESM model files
    esm_success = download_esm_model()
    
    if fm4m_deps_success and openfold_success and boltz_success and fm4m_success and esm_success:
        # Verify downloads
        verify_success = verify_downloads()
        
        if verify_success:
            print("\n" + "=" * 40)
            print("✅ Post-install setup completed successfully!")
            print("DockTKinase is ready to use.")
            print("\nInstalled components:")
            print("  ✓ FM4M dependencies (torch_nl, ase, torch-scatter)")
            print("  ✓ OpenFold3 dependencies (gemmi, ml-collections, einops, etc.)")
            print("  ✓ Boltz-2 dependencies (einx, fairscale, hydra-core, etc.)")
            if esm3_success:
                print("  ✓ ESM-3 (ESM-C) - esmc-300m, esmc-600m, esmc-6b models")
            else:
                print("  ⚠️  ESM-3 (ESM-C) - not installed (ESM-C models unavailable)")
            print("  ✓ FM4M model files")
            print("  ✓ ESM model files")
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
        print("\nFailed components:")
        if not fm4m_deps_success:
            print("  ✗ FM4M dependencies (torch_nl, ase, torch-scatter)")
        if not openfold_success:
            print("  ✗ OpenFold3 dependencies")
        if not boltz_success:
            print("  ✗ Boltz-2 dependencies")
        if not esm3_success:
            print("  ✗ ESM-3 (ESM-C) dependencies")
        if not fm4m_success:
            print("  ✗ FM4M model files")
        if not esm_success:
            print("  ✗ ESM model files")
        print("\nYou can try installing failed components manually:")
        if not fm4m_deps_success:
            print("  pip install torch_nl==0.3 ase")
            print("  pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cu121.html")
        if not openfold_success:
            print("  pip install gemmi ml-collections einops biopython pydantic lmdb biotite memory-profiler lightning")
        if not esm3_success:
            print("  cd llm/ESM/esm-3/esm-main && pip install -e .")
        if not boltz_success:
            print("  pip install einx fairscale hydra-core mashumaro chembl-structure-pipeline numba")
        sys.exit(1)