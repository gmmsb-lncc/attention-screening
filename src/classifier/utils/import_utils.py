"""
Utility for managing relative and absolute imports robustly.

This module solves the common problem of relative imports that fail when
modules are executed directly vs. as part of a package.
"""

import sys
import os
from pathlib import Path
from typing import Any, Optional
import importlib.util


def get_classifier_root() -> Path:
    """Returns the classifier root directory."""
    current_file = Path(__file__).resolve()
    # Go up until finding the classifier directory
    classifier_root = current_file.parent.parent
    if classifier_root.name != 'classifier':
        raise RuntimeError(f"Could not find classifier directory. Current: {classifier_root}")
    return classifier_root


def ensure_classifier_in_path():
    """Ensures the classifier directory is in sys.path."""
    classifier_root = get_classifier_root()
    classifier_str = str(classifier_root)
    
    if classifier_str not in sys.path:
        sys.path.insert(0, classifier_str)
        return True
    return False


def safe_import(module_name: str, package: Optional[str] = None, fallback_module: Optional[str] = None):
    """
    Imports a module safely, trying relative and absolute imports.
    
    Args:
        module_name: Module name (e.g.: '.config.mlp_config' or 'config.mlp_config')
        package: Base package for relative imports
        fallback_module: Alternative module if main fails
    
    Returns:
        Imported module
    
    Raises:
        ImportError: If no import method works
    """
    errors = []
    
    # Attempt 1: Relative import
    if module_name.startswith('.') and package:
        try:
            return importlib.import_module(module_name, package)
        except ImportError as e:
            errors.append(f"Relative import failed: {e}")
    
    # Attempt 2: Absolute import (remove leading dot if present)
    abs_module_name = module_name.lstrip('.')
    try:
        ensure_classifier_in_path()
        return importlib.import_module(abs_module_name)
    except ImportError as e:
        errors.append(f"Absolute import failed: {e}")
    
    # Attempt 3: Fallback module
    if fallback_module:
        try:
            ensure_classifier_in_path()
            return importlib.import_module(fallback_module)
        except ImportError as e:
            errors.append(f"Fallback import failed: {e}")
    
    # All failed
    error_msg = f"Failed to import {module_name}:\n" + "\n".join(errors)
    raise ImportError(error_msg)


def safe_import_from(module_name: str, *items, package: Optional[str] = None, fallback_module: Optional[str] = None):
    """
    Imports specific items from a module safely.
    
    Args:
        module_name: Module name
        *items: Items to import from the module
        package: Base package for relative imports
        fallback_module: Alternative module
    
    Returns:
        Tuple with imported items in the same order
        
    Example:
        MLPConfig, create_default_config = safe_import_from(
            '.config.mlp_config', 'MLPConfig', 'create_default_config',
            package=__package__
        )
    """
    module = safe_import(module_name, package, fallback_module)
    
    results = []
    for item in items:
        if hasattr(module, item):
            results.append(getattr(module, item))
        else:
            raise ImportError(f"Item '{item}' not found in module {module_name}")
    
    return tuple(results) if len(results) > 1 else results[0]


class ImportContext:
    """Context manager para gerenciar imports temporários."""
    
    def __init__(self, add_to_path: Optional[str] = None):
        self.add_to_path = add_to_path
        self.path_added = False
        self.original_path = None
    
    def __enter__(self):
        if self.add_to_path and self.add_to_path not in sys.path:
            self.original_path = sys.path.copy()
            sys.path.insert(0, self.add_to_path)
            self.path_added = True
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.path_added and self.original_path is not None:
            sys.path = self.original_path


# Função de conveniência para uso comum no projeto
def setup_classifier_imports():
    """
    Configura o ambiente para imports do classificador.
    Deve ser chamada no início de scripts que usam o classificador.
    """
    try:
        ensure_classifier_in_path()
        return True
    except Exception as e:
        print(f"⚠️ Aviso: Não foi possível configurar imports do classificador: {e}")
        return False


if __name__ == "__main__":
    # Teste do sistema de imports
    print("🧪 Testando sistema de imports...")
    
    # Teste 1: Verificar diretório raiz
    try:
        root = get_classifier_root()
        print(f"✅ Diretório classifier encontrado: {root}")
    except Exception as e:
        print(f"❌ Erro ao encontrar diretório classifier: {e}")
    
    # Teste 2: Adicionar ao path
    try:
        added = ensure_classifier_in_path()
        print(f"✅ Path configurado (adicionado: {added})")
    except Exception as e:
        print(f"❌ Erro ao configurar path: {e}")
    
    # Teste 3: Import seguro
    try:
        from config.mlp_config import MLPConfig
        print("✅ Import de teste funcionou")
    except Exception as e:
        print(f"❌ Import de teste falhou: {e}")
    
    print("🏁 Teste de imports concluído")


def safe_import_optional(module_name: str, purpose: str = "") -> Optional[Any]:
    """
    Tenta importar um módulo opcional de forma segura.
    
    Args:
        module_name: Nome do módulo para importar
        purpose: Descrição do propósito (para logs)
        
    Returns:
        O módulo importado ou None se falhar
    """
    try:
        module = importlib.import_module(module_name)
        return module
    except ImportError:
        if purpose:
            print(f"⚠️  Módulo opcional '{module_name}' não disponível para {purpose}")
        else:
            print(f"⚠️  Módulo opcional '{module_name}' não disponível")
        return None
    except Exception as e:
        print(f"❌ Erro ao importar '{module_name}': {e}")
        return None
