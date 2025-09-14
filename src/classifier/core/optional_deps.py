"""
Fallbacks graciais para módulos com dependências opcionais.
Permite que o sistema funcione mesmo sem todas as dependências ML instaladas.
"""

import logging
from typing import Any, Dict, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class OptionalDependencyManager:
    """Gerencia dependências opcionais e fornece fallbacks."""
    
    def __init__(self):
        self.dependencies = {}
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Verifica disponibilidade de dependências opcionais."""
        
        # Verificar numpy
        try:
            import numpy
            self.dependencies['numpy'] = {
                'available': True,
                'version': numpy.__version__,
                'module': numpy
            }
        except ImportError:
            self.dependencies['numpy'] = {'available': False, 'module': None}
        
        # Verificar pandas
        try:
            import pandas
            self.dependencies['pandas'] = {
                'available': True,
                'version': pandas.__version__,
                'module': pandas
            }
        except ImportError:
            self.dependencies['pandas'] = {'available': False, 'module': None}
        
        # Verificar torch
        try:
            import torch
            self.dependencies['torch'] = {
                'available': True,
                'version': torch.__version__,
                'module': torch
            }
        except ImportError:
            self.dependencies['torch'] = {'available': False, 'module': None}
        
        # Verificar sklearn
        try:
            import sklearn
            self.dependencies['sklearn'] = {
                'available': True,
                'version': sklearn.__version__,
                'module': sklearn
            }
        except ImportError:
            self.dependencies['sklearn'] = {'available': False, 'module': None}
        
        # Verificar psutil
        try:
            import psutil
            self.dependencies['psutil'] = {
                'available': True,
                'version': psutil.__version__,
                'module': psutil
            }
        except ImportError:
            self.dependencies['psutil'] = {'available': False, 'module': None}
    
    def is_available(self, dep_name: str) -> bool:
        """Verifica se uma dependência está disponível."""
        return self.dependencies.get(dep_name, {}).get('available', False)
    
    def get_module(self, dep_name: str) -> Optional[Any]:
        """Retorna módulo se disponível."""
        return self.dependencies.get(dep_name, {}).get('module')
    
    def get_summary(self) -> Dict[str, Any]:
        """Retorna resumo das dependências."""
        summary = {
            'core_available': True,  # Core sempre disponível
            'ml_available': self.is_available('torch') and self.is_available('numpy'),
            'data_available': self.is_available('pandas') and self.is_available('numpy'),
            'monitoring_available': self.is_available('psutil'),
            'full_stack': all(self.is_available(dep) for dep in self.dependencies)
        }
        
        summary['details'] = {
            dep: {
                'available': info['available'],
                'version': info.get('version', 'N/A')
            }
            for dep, info in self.dependencies.items()
        }
        
        return summary
    
    def get_installation_instructions(self) -> List[str]:
        """Retorna instruções para instalar dependências faltantes."""
        missing = [dep for dep, info in self.dependencies.items() if not info['available']]
        
        if not missing:
            return ["✅ Todas as dependências estão instaladas!"]
        
        instructions = ["📦 Para instalar dependências faltantes:"]
        
        if 'numpy' in missing or 'pandas' in missing or 'sklearn' in missing:
            instructions.append("   pip install numpy pandas scikit-learn")
        
        if 'torch' in missing:
            instructions.append("   pip install torch")
        
        if 'psutil' in missing:
            instructions.append("   pip install psutil")
        
        # Comando completo
        all_missing = ' '.join(missing)
        instructions.append(f"   # Ou tudo de uma vez: pip install {all_missing}")
        
        return instructions


# Instância global do gerenciador
dependency_manager = OptionalDependencyManager()


# Funções de conveniência
def check_ml_dependencies() -> bool:
    """Verifica se dependências ML estão disponíveis."""
    return dependency_manager.is_available('torch') and dependency_manager.is_available('numpy')


def check_data_dependencies() -> bool:
    """Verifica se dependências de dados estão disponíveis."""
    return dependency_manager.is_available('pandas') and dependency_manager.is_available('numpy')


def check_monitoring_dependencies() -> bool:
    """Verifica se dependências de monitoramento estão disponíveis."""
    return dependency_manager.is_available('psutil')


def get_dependency_summary() -> Dict[str, Any]:
    """Retorna resumo das dependências."""
    return dependency_manager.get_summary()


def print_dependency_status():
    """Imprime status das dependências."""
    summary = dependency_manager.get_summary()
    
    print("📋 Status das Dependências DockTKinase:")
    print(f"   ✅ Core: Sempre disponível")
    print(f"   {'✅' if summary['ml_available'] else '❌'} ML Stack: {'Disponível' if summary['ml_available'] else 'Limitado'}")
    print(f"   {'✅' if summary['data_available'] else '❌'} Data Processing: {'Disponível' if summary['data_available'] else 'Limitado'}")
    print(f"   {'✅' if summary['monitoring_available'] else '❌'} Memory Monitoring: {'Disponível' if summary['monitoring_available'] else 'Indisponível'}")
    
    print("\n📊 Detalhes:")
    for dep, info in summary['details'].items():
        status = "✅" if info['available'] else "❌"
        version = f"v{info['version']}" if info['available'] else "Não instalado"
        print(f"   {status} {dep}: {version}")
    
    if not summary['full_stack']:
        print("\n" + "\n".join(dependency_manager.get_installation_instructions()))


# Classes de fallback para quando dependências não estão disponíveis
class MockDataFrame:
    """Mock simples para pandas.DataFrame quando pandas não disponível."""
    
    def __init__(self, data=None):
        self.data = data if data is not None else []
        self._columns = []
    
    def __len__(self):
        return len(self.data) if hasattr(self.data, '__len__') else 0
    
    def __getitem__(self, key):
        return self.data[key] if hasattr(self.data, '__getitem__') else None
    
    @property
    def shape(self):
        if hasattr(self.data, '__len__'):
            return (len(self.data), len(self._columns))
        return (0, 0)
    
    def head(self, n=5):
        return self
    
    def info(self):
        print(f"MockDataFrame with {len(self)} rows")


class MockArray:
    """Mock simples para numpy.array quando numpy não disponível."""
    
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data) if hasattr(self.data, '__len__') else 0
    
    def __getitem__(self, key):
        return self.data[key] if hasattr(self.data, '__getitem__') else None
    
    @property
    def shape(self):
        if hasattr(self.data, '__len__'):
            return (len(self.data),)
        return (0,)


# Funções utilitárias com fallbacks
def safe_import_or_mock(module_name: str, mock_class=None):
    """Importa módulo ou retorna mock se não disponível."""
    if dependency_manager.is_available(module_name):
        return dependency_manager.get_module(module_name)
    
    logger.warning(f"Módulo {module_name} não disponível - usando fallback")
    return mock_class if mock_class else object


def ensure_array(data, fallback_type=list):
    """Converte dados para array numpy ou fallback."""
    if dependency_manager.is_available('numpy'):
        np = dependency_manager.get_module('numpy')
        return np.array(data)
    
    logger.warning("NumPy não disponível - usando lista Python")
    return fallback_type(data)


def ensure_dataframe(data, fallback_type=dict):
    """Converte dados para DataFrame pandas ou fallback."""
    if dependency_manager.is_available('pandas'):
        pd = dependency_manager.get_module('pandas')
        return pd.DataFrame(data)
    
    logger.warning("Pandas não disponível - usando dicionário Python")
    return fallback_type(data) if data else fallback_type()


# Exemplo de uso
if __name__ == "__main__":
    print_dependency_status()
