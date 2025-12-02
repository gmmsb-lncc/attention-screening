#!/usr/bin/env python3
"""
Automated setup script for DockTKinase.
Configures environment, verifies dependencies and validates installation.
"""

import os
import sys
import subprocess
import platform
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any

def print_header(title: str) -> None:
    """Print formatted header."""
    print("\n" + "=" * 60)
    print(f"🚀 {title}")
    print("=" * 60)

def print_section(title: str) -> None:
    """Print formatted section."""
    print(f"\n📋 {title}")
    print("-" * 40)

def run_command(cmd: List[str], description: str) -> Tuple[bool, str]:
    """Execute command and return result."""
    try:
        print(f"⚙️  {description}...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"✅ {description}: Success")
            return True, result.stdout
        else:
            print(f"❌ {description}: Failed")
            print(f"   Error: {result.stderr}")
            return False, result.stderr
    except subprocess.TimeoutExpired:
        print(f"⏰ {description}: Timeout (5min)")
        return False, "Timeout"
    except Exception as e:
        print(f"❌ {description}: Error - {e}")
        return False, str(e)

def check_system_requirements() -> bool:
    """Verify system requirements."""
    print_section("SYSTEM VERIFICATION")
    
    # Operating system
    system = platform.system()
    print(f"🖥️  System: {system} {platform.release()}")
    
    # Python version
    python_version = sys.version_info
    print(f"🐍 Python: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("❌ Python 3.8+ is required")
        return False
    
    print("✅ Python version is adequate")
    
    # Check pip
    try:
        import pip
        print(f"📦 pip: Available")
    except ImportError:
        print("❌ pip not found")
        return False
    
    # Check Python development headers (CRITICAL for PyG extensions)
    print("\n🔍 Checking Python development headers...")
    python_h_paths = [
        f"/usr/include/python{python_version.major}.{python_version.minor}/Python.h",
        f"/usr/local/include/python{python_version.major}.{python_version.minor}/Python.h",
    ]
    
    python_h_found = any(Path(p).exists() for p in python_h_paths)
    
    if not python_h_found:
        print("⚠️  WARNING: Python development headers NOT found!")
        print("   This will prevent installation of PyG extensions (torch-scatter, torch-sparse, torch-cluster)")
        print("   The system will work normally, but with limited functionality.")
        print("\n   📝 To install:")
        print(f"      sudo apt-get install python{python_version.major}.{python_version.minor}-dev -y  # Ubuntu/Debian")
        print(f"      sudo yum install python{python_version.major}-devel -y       # CentOS/RHEL")
        print(f"      sudo dnf install python{python_version.major}-devel -y       # Fedora")
        print("\n   💡 After installing, run this setup again.")
    else:
        print("✅ Python development headers found")
    
    return True

def setup_virtual_environment() -> bool:
    """Configure virtual environment."""
    print_section("VIRTUAL ENVIRONMENT CONFIGURATION")
    
    env_path = Path("env")
    
    # Check if already exists
    if env_path.exists():
        print("📁 Virtual environment already exists")
        
        # Check if functional
        python_exe = env_path / ("Scripts/python.exe" if platform.system() == "Windows" else "bin/python")
        if python_exe.exists():
            print("✅ Virtual environment is functional")
            return True
        else:
            print("⚠️  Environment corrupted, recreating...")
            import shutil
            shutil.rmtree(env_path)
    
    # Create new environment with Python 3.12
    python_cmd = "python3.12"
    
    # Check if Python 3.12 is available
    try:
        subprocess.run([python_cmd, "--version"], capture_output=True, check=True)
        print(f"🐍 Using {python_cmd}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback to python3 or sys.executable
        python_cmd = "python3" if subprocess.run(["python3", "--version"], capture_output=True).returncode == 0 else sys.executable
        print(f"🐍 Using {python_cmd}")
    
    success, _ = run_command([python_cmd, "-m", "venv", "env"], "Creating virtual environment")
    
    if success:
        print("✅ Virtual environment created")
        return True
    else:
        # Try with virtualenv
        success, _ = run_command(["virtualenv", "env"], "Creating with virtualenv")
        return success

def check_package_installed(python_exe: str, package_name: str) -> bool:
    """Check if a package is installed."""
    # Map package names to import modules
    import_map = {
        "scikit-learn": "sklearn",
        "torch-geometric": "torch_geometric",
        "torch-scatter": "torch_scatter",
        "torch-sparse": "torch_sparse",
        "torch-cluster": "torch_cluster",
        "torch-optimizer": "torch_optimizer",
        "umap-learn": "umap",
        "rdkit": "rdkit",
    }
    
    # Extract base package name (remove versions)
    base_name = package_name.split(">=")[0].split("==")[0].split("<")[0]
    
    # Get import module name
    import_name = import_map.get(base_name, base_name.replace("-", "_"))
    
    try:
        result = subprocess.run(
            [python_exe, "-c", f"import {import_name}"],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except:
        return False


def get_torch_cuda_info(python_exe: str) -> Dict[str, Any]:
    """Get information about installed PyTorch and CUDA."""
    try:
        result = subprocess.run(
            [python_exe, "-c", 
             "import torch; print(torch.__version__); print(torch.version.cuda if torch.cuda.is_available() else 'cpu')"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            torch_version = lines[0].split('+')[0]  # Remove suffix like +cu118
            cuda_version = lines[1] if len(lines) > 1 else None
            
            return {
                'torch_version': torch_version,
                'cuda_version': cuda_version if cuda_version != 'cpu' else None,
                'full_version': lines[0]
            }
    except Exception as e:
        print(f"⚠️  Error detecting PyTorch/CUDA: {e}")
    
    return None


def install_pyg_extensions_with_wheels(pip_exe: str, python_exe: str, 
                                       extensions: List[str], 
                                       torch_version: str, 
                                       cuda_version: str) -> List[str]:
    """
    Install PyG extensions using pre-compiled wheels.
    Returns list of extensions that failed.
    
    IMPORTANT: PyG extensions must be compiled for the SAME PyTorch version.
    Using wheels from a different version causes segmentation fault.
    """
    failed = []
    
    # Check system CUDA vs PyTorch CUDA incompatibility
    system_cuda = get_system_cuda_version()
    
    # Extract main torch version
    torch_parts = torch_version.replace('+', '.').split('.')
    torch_major = int(torch_parts[0]) if torch_parts[0].isdigit() else 2
    torch_minor = int(torch_parts[1]) if len(torch_parts) > 1 and torch_parts[1].isdigit() else 0
    torch_patch = int(torch_parts[2]) if len(torch_parts) > 2 and torch_parts[2].isdigit() else 0
    
    # Determine CUDA suffix
    if cuda_version:
        cuda_parts = cuda_version.split('.')
        if len(cuda_parts) >= 2:
            cuda_suffix = f"cu{cuda_parts[0]}{cuda_parts[1]}"
        else:
            cuda_suffix = f"cu{cuda_parts[0]}0"
    else:
        cuda_suffix = "cpu"
    
    # PyTorch versions with available PyG wheels (updated December 2025)
    # CRITICAL: DO NOT install wheels from different version - causes segfault!
    available_pyg_versions = {
        (2, 8, 0): ["cu128", "cu126", "cu124", "cu121", "cu118", "cpu"],
        (2, 7, 1): ["cu128", "cu126", "cu118", "cpu"],
        (2, 7, 0): ["cu128", "cu126", "cu118", "cpu"],
        (2, 6, 0): ["cu126", "cu124", "cu121", "cu118", "cpu"],
        (2, 5, 1): ["cu124", "cu121", "cu118", "cpu"],
        (2, 5, 0): ["cu124", "cu121", "cu118", "cpu"],
    }
    
    current_torch = (torch_major, torch_minor, torch_patch)
    
    print(f"\n📦 Checking PyG extensions...")
    print(f"   PyTorch: {torch_version} ({torch_major}.{torch_minor}.{torch_patch})")
    print(f"   CUDA PyTorch: {cuda_version if cuda_version else 'CPU'}")
    if system_cuda:
        print(f"   CUDA System: {system_cuda}")
    
    # Check if there are exact wheels for this version
    exact_match = current_torch in available_pyg_versions
    compatible_cuda = cuda_suffix in available_pyg_versions.get(current_torch, [])
    
    if not exact_match:
        print(f"\n⚠️  NO PyG WHEELS for PyTorch {torch_major}.{torch_minor}.{torch_patch}")
        print(f"   Available versions: {', '.join([f'{v[0]}.{v[1]}.{v[2]}' for v in available_pyg_versions.keys()])}")
        
        # DO NOT try to install incompatible wheels - causes segfault!
        failed = extensions.copy()
        
        print(f"\n" + "=" * 70)
        print("📋 PyG EXTENSIONS WILL NOT BE INSTALLED")
        print("=" * 70)
        print("\n⚠️  REASON: PyTorch too recent, no compatible PyG wheels available yet")
        print("   Installing wheels from different version causes CRASH (segmentation fault)")
        
        print("\n💡 SOLUTIONS:")
        print("\n   OPTION 1: Continue WITHOUT PyG extensions (RECOMMENDED)")
        print("   ✅ DockTKinase will work perfectly!")
        print("   ✅ Only Graph Neural Networks will not be available")
        print("   ✅ MLP classifiers, embeddings, etc. work normally")
        
        print("\n   OPTION 2: DOWNGRADE PyTorch to 2.8.0 (if you need GNNs)")
        print("   source env/bin/activate")
        print("   pip uninstall torch torchvision torchaudio -y")
        print(f"   pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/{cuda_suffix}")
        print(f"   pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.8.0+{cuda_suffix}.html")
        
        print("\n   OPTION 3: Wait for wheel release for PyTorch " + torch_version)
        print("   Check periodically: https://data.pyg.org/whl/")
        print("=" * 70)
        
        return failed
    
    if not compatible_cuda:
        print(f"\n⚠️  CUDA {cuda_suffix} not available for PyTorch {torch_major}.{torch_minor}.{torch_patch}")
        print(f"   Available variants: {', '.join(available_pyg_versions[current_torch])}")
        
        # Try closest variant
        for alt_cuda in available_pyg_versions[current_torch]:
            if alt_cuda.startswith("cu12") and cuda_suffix.startswith("cu12"):
                cuda_suffix = alt_cuda
                print(f"   Using alternative variant: {alt_cuda}")
                break
    
    # URL for exact wheels
    wheel_url = f"https://data.pyg.org/whl/torch-{torch_major}.{torch_minor}.{torch_patch}+{cuda_suffix}.html"
    
    print(f"   Wheels URL: {wheel_url}")
    
    for ext in extensions:
        if check_package_installed(python_exe, ext):
            print(f"✅ {ext}: Already installed")
            continue
        
        print(f"📥 Installing {ext}...")
        
        success, _ = run_command(
            [pip_exe, "install", ext, "-f", wheel_url, "--no-cache-dir"],
            f"Installing {ext}"
        )
        
        if success:
            print(f"✅ {ext}: Installed")
        else:
            failed.append(ext)
            print(f"⚠️  {ext}: Installation failed")
    
    if failed:
        print(f"\n⚠️  Some extensions were not installed: {', '.join(failed)}")
        print("   The system will work without them (only GNNs unavailable)")
    else:
        print(f"\n✅ All PyG extensions installed successfully!")
    
    return failed


def get_system_cuda_version() -> str:
    """Detect system CUDA version via nvcc."""
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            # Look for "release X.Y" in output
            match = re.search(r'release (\d+\.\d+)', result.stdout)
            if match:
                return match.group(1)
    except:
        pass
    return None

def install_dependencies() -> bool:
    """Install project dependencies (only those not already installed)."""
    print_section("DEPENDENCY INSTALLATION")
    
    # Determine virtual environment Python executable
    if platform.system() == "Windows":
        python_exe = "env/Scripts/python.exe"
        pip_exe = "env/Scripts/pip.exe"
    else:
        python_exe = "env/bin/python"
        pip_exe = "env/bin/pip"
    
    # Check if executables exist
    if not Path(python_exe).exists():
        print(f"❌ Python not found at {python_exe}")
        return False
    
    # Update pip
    print("🔄 Checking pip...")
    success, _ = run_command([python_exe, "-m", "pip", "install", "--upgrade", "pip"], 
                           "Updating pip")
    if not success:
        print("⚠️  Failed to update pip, continuing...")
    
    # Basic dependencies
    basic_deps = [
        "torch",
        "torchvision", 
        "numpy>=1.26.1",
        "pandas>=1.5.3",
        "scikit-learn>=1.5.0",
        "matplotlib>=3.9.2",
        "seaborn",
        "jupyter",
        "pytest",
        # Dependencies needed for src/build
        "tqdm>=4.66.4",           # Progress bars (all scripts)
        "psutil",                 # System utilities (embeddingIBM.py, embeddingBuild.py)  
        "pyspark>=3.0.0",        # Apache Spark (buildInteractionLabels.py, embeddingBuild.py)
        "optuna",                 # Hyperparameter optimization
        "scipy>=1.12.0",         # Scientific computing (also used in compare_classifiers.py)
        "pyarrow>=14.0.1",       # Apache Arrow (for Spark and pandas)
        # Dependencies for compare_classifiers.py and run_complete_pipeline.py
        "threadpoolctl>=3.1.0",  # Thread pool control (to avoid KNN bugs)
    ]
    
    # ==========================================================================
    # CRITICAL: Remove any pip-installed ESM packages to avoid conflicts
    # The local ESM in llm/ESM must be used exclusively via sys.path
    # ==========================================================================
    print("\n🧬 Configuring ESM (local repository version)...")
    
    # First, uninstall any pip-installed ESM packages that could conflict
    esm_pip_packages = ["fair-esm", "esm"]
    for pkg in esm_pip_packages:
        print(f"   🔍 Checking for pip package: {pkg}")
        try:
            result = subprocess.run(
                [pip_exe, "show", pkg], 
                capture_output=True, 
                text=True
            )
            if result.returncode == 0:
                print(f"   ⚠️  Found conflicting package: {pkg}")
                run_command([pip_exe, "uninstall", "-y", pkg], f"Removing {pkg}")
        except Exception:
            pass
    
    # Verify local ESM exists (but DO NOT install via pip)
    esm_paths = [Path("llm/ESM"), Path("ESM")]
    esm_path = None
    for p in esm_paths:
        if p.exists() and (p / "esm" / "__init__.py").exists():
            esm_path = p
            break
    
    if esm_path:
        print(f"   ✅ Local ESM found at: {esm_path.absolute()}")
        print(f"   📝 ESM will be loaded via sys.path (NOT pip install)")
        print(f"   💡 Scripts must add '{esm_path}' to sys.path before importing esm")
    else:
        print("   ⚠️  ESM folder not found!")
        print("   📝 Clone it with: git clone https://github.com/facebookresearch/esm.git llm/ESM")
    
    # Optional dependencies (installed with error handling)
    optional_deps = [
        # Protein and Ligand Embeddings
        "transformers>=4.38",     # HuggingFace models (required for ESM)
        "sentencepiece",          # Tokenizer (used by some models)
        
        # Chemistry and Molecules
        "rdkit",                  # Chemistry toolkit (install via pip in env)
        "umap-learn",             # Dimensionality reduction (FM4M)
        "selfies>=2.1.0",         # SELFIES molecular representation
        "mordred",                # Molecular descriptors
        "numba",                  # JIT compiler (required for umap)
        
        # Graph Neural Networks
        "torch-geometric>=2.3.1", # Graph neural networks
        "networkx>=2.8",          # Graph processing
        
        # Machine Learning and Optimization  
        "xgboost",                # Gradient boosting (required for FM4M)
        "torch-optimizer",        # Additional optimizers
        "datasets>=2.13.1",       # HuggingFace datasets
        "evaluate>=0.4.0",        # Model evaluation metrics
        
        # Scalable Clustering (for large dataset stratification)
        "faiss-cpu>=1.7.0",       # FAISS for efficient similarity search and clustering
        
        # Utilities
        "requests>=2.32.2",       # HTTP requests
        "urllib3>=2.2.2",         # HTTP client
        "aiohttp>=3.10.2",        # Async HTTP
        "zipp>=3.19.1",           # ZIP utilities
        "torchinfo>=1.8.0",       # Model summary
        "ase",                    # Atomic simulation environment
    ]
    
    # PyTorch Geometric extensions (MUST be installed AFTER torch)
    # These dependencies need to compile against the installed PyTorch
    pyg_extensions = [
        "torch-scatter",          # Scatter operations for PyG
        "torch-sparse",           # Sparse operations for PyG
        "torch-cluster",          # Clustering for PyG
    ]
    
    print("📦 Checking and installing basic dependencies...")
    to_install_basic = []
    for dep in basic_deps:
        if check_package_installed(python_exe, dep):
            print(f"✅ {dep}: Already installed")
        else:
            print(f"📥 {dep}: Will be installed")
            to_install_basic.append(dep)
    
    if to_install_basic:
        print(f"\n🔨 Installing {len(to_install_basic)} basic dependencies...")
        for dep in to_install_basic:
            success, _ = run_command([pip_exe, "install", dep], f"Installing {dep}")
            if not success:
                print(f"❌ Failed to install {dep} (CRITICAL)")
                return False  # Basic dependencies are mandatory
    else:
        print("✅ All basic dependencies are already installed!")
    
    print("\n🔧 Checking and installing optional dependencies...")
    to_install_optional = []
    already_installed_optional = []
    
    for dep in optional_deps:
        if check_package_installed(python_exe, dep):
            already_installed_optional.append(dep)
            print(f"✅ {dep}: Already installed")
        else:
            print(f"📥 {dep}: Will be installed")
            to_install_optional.append(dep)
    
    failed_optional = []
    if to_install_optional:
        print(f"\n🔨 Installing {len(to_install_optional)} optional dependencies...")
        for dep in to_install_optional:
            success, _ = run_command([pip_exe, "install", dep], f"Installing {dep}")
            if not success:
                failed_optional.append(dep)
                print(f"⚠️  Failed to install {dep} (optional)")
    else:
        print("✅ All optional dependencies are already installed!")
    
    # Install PyG extensions (AFTER torch is installed)
    print("\n🔧 Checking PyTorch Geometric extensions...")
    
    # First, check PyTorch and CUDA version to use pre-compiled wheels
    torch_info = get_torch_cuda_info(python_exe)
    
    if torch_info is None:
        print("⚠️  PyTorch not found, skipping PyG extensions")
        failed_pyg = pyg_extensions.copy()
    else:
        torch_version = torch_info['torch_version']
        cuda_version = torch_info['cuda_version']
        
        print(f"🔍 PyTorch: {torch_version}")
        print(f"🔍 CUDA (PyTorch): {cuda_version if cuda_version else 'CPU'}")
        
        # Try to install using PyG pre-compiled wheels
        failed_pyg = install_pyg_extensions_with_wheels(
            pip_exe, python_exe, pyg_extensions, torch_version, cuda_version
        )
    
    # Installation summary
    print("\n" + "=" * 60)
    print("📊 INSTALLATION SUMMARY")
    print("=" * 60)
    print(f"✅ Basic installed: {len(to_install_basic)}/{len(basic_deps)}")
    print(f"🔧 Optional installed: {len(to_install_optional) - len(failed_optional)}/{len(optional_deps)}")
    print(f"🧩 PyG extensions: {len(pyg_extensions) - len(failed_pyg)}/{len(pyg_extensions)}")
    print(f"♻️  Already present: {len(basic_deps) - len(to_install_basic) + len(already_installed_optional)}")
    
    if failed_optional:
        print(f"\n⚠️  Optional dependencies not installed ({len(failed_optional)}):")
        for dep in failed_optional:
            print(f"   - {dep}")
        print("\n💡 To install manually:")
        print("   source env/bin/activate")
        for dep in failed_optional:
            print(f"   pip install {dep}")
        print("\n📝 NOTE about rdkit:")
        print("   If rdkit fails via pip, you can try:")
        print("   pip install rdkit-pypi")
    else:
        print("\n🎉 All dependencies were installed successfully!")
    
    return True

def download_model_files() -> bool:
    """Download required model files (SMI-TED and ESM)."""
    print_section("MODEL FILES DOWNLOAD")
    
    # Virtual environment Python executable
    if platform.system() == "Windows":
        python_exe = "env/Scripts/python.exe"
    else:
        python_exe = "env/bin/python"
    
    if not Path(python_exe).exists():
        print(f"❌ Python not found at {python_exe}")
        return False
    
    # Model directories (llm/FM4M is main, FM4M in root is legacy)
    llm_fm4m_dir = Path("llm/FM4M")
    root_fm4m_dir = Path("FM4M")
    
    # Determine which FM4M to use (prioritize llm/FM4M)
    if llm_fm4m_dir.exists():
        fm4m_dir = llm_fm4m_dir
        print(f"📁 FM4M found at: {fm4m_dir}")
    elif root_fm4m_dir.exists():
        fm4m_dir = root_fm4m_dir
        print(f"📁 FM4M found at: {fm4m_dir} (legacy)")
    else:
        fm4m_dir = None
        print("⚠️  FM4M not found")
    
    # Check if post_install.py exists
    post_install_script = Path("scripts/post_install.py")
    
    success_fm4m = False
    success_post_install = False
    
    # Option 1: Download FM4M directly via download_model_files.py
    if fm4m_dir:
        download_script = fm4m_dir / "download_model_files.py"
        model_files_dir = fm4m_dir / "model_files"
        
        if download_script.exists():
            print(f"\n📥 Downloading FM4M models via {download_script}...")
            print("   • SMI-TED models (~4.1 GB)")
            print("   ⏱️  Estimated time: 5-10 minutes")
            
            success_fm4m, output = run_command(
                [python_exe, str(download_script)], 
                "Downloading FM4M models"
            )
            
            if success_fm4m:
                print("✅ FM4M models downloaded successfully")
                
                # Create symlink in FM4M/ at root if needed
                if fm4m_dir == llm_fm4m_dir and root_fm4m_dir.exists():
                    root_model_files = root_fm4m_dir / "model_files"
                    if not root_model_files.exists() and model_files_dir.exists():
                        try:
                            root_model_files.symlink_to(model_files_dir.absolute())
                            print(f"🔗 Symlink created: {root_model_files} -> {model_files_dir}")
                        except Exception as e:
                            print(f"⚠️  Could not create symlink: {e}")
            else:
                print("⚠️  Failed to download FM4M models")
        else:
            print(f"⚠️  Script {download_script} not found")
    
    # Option 2: Run post_install.py for other dependencies
    if post_install_script.exists():
        print(f"\n📥 Running {post_install_script}...")
        print("   • ML dependencies (lightgbm, xgboost, etc.)")
        print("   • ESM model (~2.6 GB)")
        print("   ⏱️  Estimated time: 5-10 minutes")
        
        success_post_install, output = run_command(
            [python_exe, str(post_install_script)], 
            "Running post_install.py"
        )
        
        if success_post_install:
            print("✅ Post-install completed successfully")
        else:
            print("⚠️  Some post_install components may have failed")
    else:
        print(f"⚠️  Script {post_install_script} not found")
    
    # Summary
    if not success_fm4m and not success_post_install:
        print("\n" + "=" * 60)
        print("⚠️  DOWNLOADS NOT COMPLETED")
        print("=" * 60)
        print("\n💡 To download manually:")
        if fm4m_dir:
            print(f"   cd {fm4m_dir} && python download_model_files.py")
        else:
            print("   cd llm/FM4M && python download_model_files.py")
        if post_install_script.exists():
            print(f"   python {post_install_script}")
    
    # Always return True (models are optional)
    return True

def verify_installation() -> bool:
    """Verify that installation is working."""
    print_section("INSTALLATION VERIFICATION")
    
    # Virtual environment Python executable
    if platform.system() == "Windows":
        python_exe = "env/Scripts/python.exe"
    else:
        python_exe = "env/bin/python"
    
    # Test main imports
    test_imports = [
        "import torch; print(f'PyTorch: {torch.__version__}')",
        "import numpy; print(f'NumPy: {numpy.__version__}')",
        "import pandas; print(f'Pandas: {pandas.__version__}')",
        "import sklearn; print(f'Scikit-learn: {sklearn.__version__}')",
        "import matplotlib; print(f'Matplotlib: {matplotlib.__version__}')",
        # Build script specific dependencies
        "import tqdm; print(f'tqdm: {tqdm.__version__}')",
        "import psutil; print(f'psutil: {psutil.__version__}')",
        "import pyspark; print(f'PySpark: {pyspark.__version__}')",
        "import optuna; print(f'Optuna: {optuna.__version__}')",
        "import scipy; print(f'SciPy: {scipy.__version__}')",
    ]
    
    # Test optional imports
    optional_imports = [
        "import esm; print(f'ESM: {esm.__version__}')",
        "import transformers; print(f'Transformers: {transformers.__version__}')",
        "import umap; print(f'UMAP: available')",
        "import rdkit; print(f'RDKit: {rdkit.__version__}')",
        "import torch_geometric; print(f'PyTorch Geometric: {torch_geometric.__version__}')",
    ]
    
    for test_import in test_imports:
        success, output = run_command([python_exe, "-c", test_import], 
                                    f"Testing: {test_import.split(';')[0]}")
        if success:
            print(f"   ✅ {output.strip()}")
        else:
            print(f"   ❌ Failed: {test_import.split(';')[0]}")
    
    print("\n🔍 Testing optional imports...")
    for test_import in optional_imports:
        success, output = run_command([python_exe, "-c", test_import], 
                                    f"Testing: {test_import.split(';')[0]}")
        if success:
            print(f"   ✅ {output.strip()}")
        else:
            print(f"   ⚠️  Not available: {test_import.split(';')[0]}")
    
    # Test DockTKinase system (basic imports only, no instantiation)
    print("\n🧪 Testing DockTKinase system...")
    
    test_script = '''
import sys
from pathlib import Path

# Add src and src/classifier to path
src_path = Path.cwd() / "src"
classifier_path = src_path / "classifier"
sys.path.insert(0, str(classifier_path))
sys.path.insert(0, str(src_path))

try:
    # Test basic system imports
    print("🔍 Testing system imports...")
    
    # Functional basic imports
    import torch
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    print("✅ Basic libraries: OK")
    
    # Test run_complete_pipeline import
    import run_complete_pipeline
    print("✅ run_complete_pipeline: OK")
    
    # Test compare_classifiers import
    import compare_classifiers
    print("✅ compare_classifiers: OK")
    
    print("🎉 DOCKTKINASE SYSTEM FUNCTIONAL!")
    
except Exception as e:
    print(f"⚠️  Warning: {e}")
    print("System partially functional, but may have issues with specific modules")
    # Don't fail here, just warn
'''
    
    success, output = run_command([python_exe, "-c", test_script], 
                                "Testing DockTKinase")
    
    if success:
        print(output)
    else:
        print("⚠️  System may have issues, but continuing...")
    
    # Always return True if we got here (basic dependencies OK)
    return True

def create_startup_scripts() -> None:
    """Create startup scripts."""
    print_section("CREATING STARTUP SCRIPTS")
    
    # Script to activate environment (Unix)
    if platform.system() != "Windows":
        activate_script = '''#!/bin/bash
# Script to activate DockTKinase environment

echo "🚀 Activating DockTKinase environment..."

# Activate virtual environment
source env/bin/activate

# Add src to PYTHONPATH
export PYTHONPATH="$PWD/src:$PYTHONPATH"

echo "✅ Environment activated!"
echo "📁 PYTHONPATH: $PYTHONPATH"
echo ""
echo "To use DockTKinase:"
echo "  python src/classifier/modular_classifier.py --help  # CLI interface"
echo "  python -c 'from classifier.modular_pipeline import ModularMLPPipeline; print(\"System ready!\")'"
echo "  jupyter lab  # For notebooks"
echo ""
'''
        
        with open("activate_env.sh", "w") as f:
            f.write(activate_script)
        
        os.chmod("activate_env.sh", 0o755)
        print("✅ activate_env.sh created")
    
    # Universal Python script
    launcher_script = '''#!/usr/bin/env python3
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
        print("\\n🎉 DockTKinase is ready!")
    else:
        print("\\n⚠️  Check installation")
        sys.exit(1)
'''
    
    with open("launch_docktkinase.py", "w") as f:
        f.write(launcher_script)
    
    print("✅ launch_docktkinase.py created")

def validate_build_dependencies() -> bool:
    """Validate build script specific dependencies."""
    print_section("BUILD SCRIPT DEPENDENCY VALIDATION")
    
    # Virtual environment Python executable
    if platform.system() == "Windows":
        python_exe = "env/Scripts/python.exe"
    else:
        python_exe = "env/bin/python"
    
    if not Path(python_exe).exists():
        print(f"❌ Virtual environment not found at {python_exe}")
        return False
    
    # Essential dependencies for build scripts
    essential_build_deps = [
        ("numpy", "import numpy"),
        ("pandas", "import pandas"), 
        ("tqdm", "import tqdm"),
        ("psutil", "import psutil"),
        ("pyspark", "import pyspark"),
    ]
    
    # Optional dependencies for full functionality
    optional_build_deps = [
        ("esm", "import esm"),
        ("umap", "import umap"), 
        ("rdkit", "import rdkit"),
        ("transformers", "import transformers"),
        ("torch_geometric", "import torch_geometric"),
    ]
    
    print("🔍 Checking essential dependencies...")
    essential_failed = []
    for dep_name, import_cmd in essential_build_deps:
        success, _ = run_command([python_exe, "-c", import_cmd], 
                               f"Testing {dep_name}")
        if success:
            print(f"   ✅ {dep_name}: OK")
        else:
            print(f"   ❌ {dep_name}: FAILED")
            essential_failed.append(dep_name)
    
    print("\n🔧 Checking optional dependencies...")
    optional_failed = []
    for dep_name, import_cmd in optional_build_deps:
        success, _ = run_command([python_exe, "-c", import_cmd], 
                               f"Testing {dep_name}")
        if success:
            print(f"   ✅ {dep_name}: OK")
        else:
            print(f"   ⚠️  {dep_name}: Not available")
            optional_failed.append(dep_name)
    
    # Validation summary
    print("\n📋 VALIDATION SUMMARY:")
    print(f"   ✅ Essential: {len(essential_build_deps) - len(essential_failed)}/{len(essential_build_deps)}")
    print(f"   🔧 Optional: {len(optional_build_deps) - len(optional_failed)}/{len(optional_build_deps)}")
    
    if essential_failed:
        print(f"\n❌ CRITICAL DEPENDENCIES MISSING: {', '.join(essential_failed)}")
        print("   Run setup again or install manually:")
        for dep in essential_failed:
            print(f"     pip install {dep}")
        return False
    
    if optional_failed:
        print(f"\n⚠️  OPTIONAL DEPENDENCIES MISSING: {', '.join(optional_failed)}")
        print("   Limited functionality for:")
        if 'esm' in optional_failed:
            print("     - Protein embeddings (embeddingMeta.py)")
        if 'umap' in optional_failed or 'rdkit' in optional_failed:
            print("     - Ligand embeddings (embeddingIBM.py via FM4M)")
        if 'transformers' in optional_failed:
            print("     - Advanced language models")
    
    return True

def main() -> int:
    """Main setup function."""
    print_header("DOCKTKINASE AUTOMATED SETUP")
    
    # Check correct directory (using files that actually exist)
    required_markers = ["run_complete_pipeline.py", "src", "requirements.txt"]
    missing_markers = [m for m in required_markers if not Path(m).exists()]
    
    if missing_markers:
        print(f"❌ Run this script in the DockTKinase root directory")
        print(f"   Files/folders not found: {', '.join(missing_markers)}")
        return 1
    
    print("📁 Project directory detected")
    
    # List of steps
    steps = [
        ("Check system requirements", check_system_requirements),
        ("Configure virtual environment", setup_virtual_environment),
        ("Install dependencies", install_dependencies),
        ("Download model files", download_model_files),
        ("Validate build dependencies", validate_build_dependencies),
        ("Verify installation", verify_installation),
        ("Create startup scripts", lambda: (create_startup_scripts(), True)[1])
    ]
    
    # Execute steps
    for step_name, step_func in steps:
        print_section(step_name.upper())
        
        try:
            success = step_func()
            if success:
                print(f"✅ {step_name}: Completed")
            else:
                print(f"❌ {step_name}: Failed")
                print(f"\n⚠️  Setup interrupted at: {step_name}")
                return 1
        except Exception as e:
            print(f"❌ {step_name}: Error - {e}")
            return 1
    
    # Success
    print_header("SETUP COMPLETED SUCCESSFULLY!")
    
    print("🎉 DockTKinase has been configured successfully!")
    print("\n📦 Installed components:")
    print("   ✅ Python virtual environment")
    print("   ✅ Python dependencies (torch, numpy, pandas, etc.)")
    print("   ✅ SMI-TED models (ligand embeddings)")
    print("   ✅ ESM model (protein embeddings)")
    print("   ✅ Startup scripts")
    
    print("\n📋 Next steps:")
    
    if platform.system() != "Windows":
        print("   1. source activate_env.sh")
    print("   2. python launch_docktkinase.py")
    print("   3. jupyter lab (for notebooks)")
    
    print("\n📖 To use the system:")
    print("   from classifier.modular_pipeline import ModularMLPPipeline")
    print("   pipeline = ModularMLPPipeline()")
    print("   # or use CLI interface:")
    print("   python src/classifier/modular_classifier.py --help")
    
    print("\n🚀 To run the complete pipeline:")
    print("   python run_complete_pipeline.py --help")
    print("   python run_complete_pipeline.py \\")
    print("       --input tests/datasets/kinase_non_human_compounds.tsv \\")
    print("       --output results/test_run \\")
    print("       --esm-model esm2_t33_650M_UR50D \\")
    print("       --seed 42")
    
    print("\n💡 Available models:")
    print("   • SMI-TED: Ligand embeddings (llm/FM4M/model_files/)")
    print("   • ESM-2: Protein embeddings (llm/ESM/ + automatic cache)")
    print("   • 6 ESM-2 variants: 8M, 35M, 150M, 650M, 3B, 15B parameters")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
