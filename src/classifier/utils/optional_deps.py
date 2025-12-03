"""
Optional dependencies manager with graceful degradation.
Allows the system to work even without all dependencies installed.
"""

import logging
from typing import Optional, Any, Dict
import warnings

logger = logging.getLogger(__name__)

# Track de dependências disponíveis
AVAILABLE_DEPS = {}


def check_dependency(name: str, package: str, fallback_msg: str = None) -> Optional[Any]:
    """
    Checks if a dependency is available and imports it.
    
    Args:
        name: Dependency name for cache
        package: Package name to import
        fallback_msg: Optional fallback message
        
    Returns:
        Imported module or None if not available
    """
    if name in AVAILABLE_DEPS:
        return AVAILABLE_DEPS[name]
    
    try:
        module = __import__(package)
        AVAILABLE_DEPS[name] = module
        logger.debug(f"✅ {name} available")
        return module
    except ImportError:
        AVAILABLE_DEPS[name] = None
        msg = fallback_msg or f"⚠️  {name} not available - limited functionality"
        logger.warning(msg)
        return None


# Check main dependencies at initialization
def check_main_dependencies() -> Dict[str, bool]:
    """Checks all main dependencies."""
    deps_status = {}
    
    # PyTorch - CRITICAL
    torch = check_dependency('torch', 'torch', 
                           "PyTorch not found - system will not work!")
    deps_status['torch'] = torch is not None
    
    # Sklearn - CRITICAL
    sklearn = check_dependency('sklearn', 'sklearn',
                             "Scikit-learn not found - limited metrics")  
    deps_status['sklearn'] = sklearn is not None
    
    # Optuna - OPTIONAL
    optuna = check_dependency('optuna', 'optuna',
                            "Optuna not available - hyperparameter optimization disabled")
    deps_status['optuna'] = optuna is not None
    
    # NumPy - CRITICAL  
    numpy = check_dependency('numpy', 'numpy',
                           "NumPy not found - system will not work!")
    deps_status['numpy'] = numpy is not None
    
    # Pandas - SEMI-OPTIONAL
    pandas = check_dependency('pandas', 'pandas',
                            "Pandas not available - limited CSV loading")
    deps_status['pandas'] = pandas is not None
    
    return deps_status


def require_dependency(name: str) -> Any:
    """
    Requires a critical dependency, fails gracefully if not available.
    """
    if name not in AVAILABLE_DEPS:
        raise RuntimeError(f"Dependency {name} must be checked first with check_dependency()")
    
    if AVAILABLE_DEPS[name] is None:
        raise RuntimeError(f"Critical dependency {name} is not available!")
    
    return AVAILABLE_DEPS[name]


def is_available(name: str) -> bool:
    """Checks if a dependency is available."""
    return AVAILABLE_DEPS.get(name, False) is not None


def get_available_features() -> Dict[str, str]:
    """Returns list of available features based on dependencies."""
    features = {}
    
    if is_available('torch'):
        features['neural_networks'] = "PyTorch neural networks"
    else:
        features['neural_networks'] = "❌ PyTorch required"
        
    if is_available('optuna'):
        features['hyperopt'] = "Hyperparameter optimization"
    else:
        features['hyperopt'] = "❌ Install Optuna for optimization"
        
    if is_available('sklearn'):
        features['metrics'] = "Complete evaluation metrics"
    else:
        features['metrics'] = "❌ Limited metrics without sklearn"
        
    if is_available('pandas'):
        features['data_loading'] = "Complete CSV/Excel loading"
    else:
        features['data_loading'] = "⚠️ NumPy arrays only"
    
    return features


# Initialize check on import
_DEPS_STATUS = check_main_dependencies()

# Warnings about missing critical dependencies
if not _DEPS_STATUS.get('torch', False):
    warnings.warn("PyTorch not found! System will not work.", 
                  ImportWarning, stacklevel=2)

if not _DEPS_STATUS.get('sklearn', False):
    warnings.warn("Scikit-learn not found! Limited functionality.", 
                  ImportWarning, stacklevel=2)
