"""
Gerenciador de dependências opcionais com graceful degradation.
Permite que o sistema funcione mesmo sem todas as dependências instaladas.
"""

import logging
from typing import Optional, Any, Dict
import warnings

logger = logging.getLogger(__name__)

# Track de dependências disponíveis
AVAILABLE_DEPS = {}


def check_dependency(name: str, package: str, fallback_msg: str = None) -> Optional[Any]:
    """
    Verifica se uma dependência está disponível e a importa.
    
    Args:
        name: Nome da dependência para cache
        package: Nome do pacote para importar
        fallback_msg: Mensagem de fallback opcional
        
    Returns:
        Módulo importado ou None se não disponível
    """
    if name in AVAILABLE_DEPS:
        return AVAILABLE_DEPS[name]
    
    try:
        module = __import__(package)
        AVAILABLE_DEPS[name] = module
        logger.debug(f"✅ {name} disponível")
        return module
    except ImportError:
        AVAILABLE_DEPS[name] = None
        msg = fallback_msg or f"⚠️  {name} não disponível - funcionalidade limitada"
        logger.warning(msg)
        return None


# Verificar dependências principais na inicialização
def check_main_dependencies() -> Dict[str, bool]:
    """Verifica todas as dependências principais."""
    deps_status = {}
    
    # PyTorch - CRÍTICO
    torch = check_dependency('torch', 'torch', 
                           "PyTorch não encontrado - sistema não funcionará!")
    deps_status['torch'] = torch is not None
    
    # Sklearn - CRÍTICO
    sklearn = check_dependency('sklearn', 'sklearn',
                             "Scikit-learn não encontrado - métricas limitadas")  
    deps_status['sklearn'] = sklearn is not None
    
    # Optuna - OPCIONAL
    optuna = check_dependency('optuna', 'optuna',
                            "Optuna não disponível - otimização de hiperparâmetros desabilitada")
    deps_status['optuna'] = optuna is not None
    
    # NumPy - CRÍTICO  
    numpy = check_dependency('numpy', 'numpy',
                           "NumPy não encontrado - sistema não funcionará!")
    deps_status['numpy'] = numpy is not None
    
    # Pandas - SEMI-OPCIONAL
    pandas = check_dependency('pandas', 'pandas',
                            "Pandas não disponível - carregamento de CSV limitado")
    deps_status['pandas'] = pandas is not None
    
    return deps_status


def require_dependency(name: str) -> Any:
    """
    Requer uma dependência crítica, falha gracefully se não disponível.
    """
    if name not in AVAILABLE_DEPS:
        raise RuntimeError(f"Dependência {name} deve ser verificada primeiro com check_dependency()")
    
    if AVAILABLE_DEPS[name] is None:
        raise RuntimeError(f"Dependência crítica {name} não está disponível!")
    
    return AVAILABLE_DEPS[name]


def is_available(name: str) -> bool:
    """Verifica se uma dependência está disponível."""
    return AVAILABLE_DEPS.get(name, False) is not None


def get_available_features() -> Dict[str, str]:
    """Retorna lista de funcionalidades disponíveis baseada nas dependências."""
    features = {}
    
    if is_available('torch'):
        features['neural_networks'] = "Redes neurais PyTorch"
    else:
        features['neural_networks'] = "❌ PyTorch requerido"
        
    if is_available('optuna'):
        features['hyperopt'] = "Otimização de hiperparâmetros"
    else:
        features['hyperopt'] = "❌ Instale Optuna para otimização"
        
    if is_available('sklearn'):
        features['metrics'] = "Métricas completas de avaliação"
    else:
        features['metrics'] = "❌ Métricas limitadas sem sklearn"
        
    if is_available('pandas'):
        features['data_loading'] = "Carregamento CSV/Excel completo"
    else:
        features['data_loading'] = "⚠️ Apenas arrays NumPy"
    
    return features


# Inicializar verificação na importação
_DEPS_STATUS = check_main_dependencies()

# Avisos sobre dependências críticas faltantes
if not _DEPS_STATUS.get('torch', False):
    warnings.warn("PyTorch não encontrado! Sistema não funcionará.", 
                  ImportWarning, stacklevel=2)

if not _DEPS_STATUS.get('sklearn', False):
    warnings.warn("Scikit-learn não encontrado! Funcionalidade limitada.", 
                  ImportWarning, stacklevel=2)
