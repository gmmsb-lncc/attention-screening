"""
Utilitário para gerenciar imports relativos e absolutos de forma robusta.

Este módulo resolve o problema comum de imports relativos que falham quando
módulos são executados diretamente vs. como parte de um pacote.
"""

import sys
import os
from pathlib import Path
from typing import Any, Optional
import importlib.util


def get_classifier_root() -> Path:
    """Retorna o diretório raiz do classificador."""
    current_file = Path(__file__).resolve()
    # Subir até encontrar o diretório classifier
    classifier_root = current_file.parent.parent
    if classifier_root.name != 'classifier':
        raise RuntimeError(f"Não foi possível encontrar o diretório classifier. Atual: {classifier_root}")
    return classifier_root


def ensure_classifier_in_path():
    """Garante que o diretório classifier está no sys.path."""
    classifier_root = get_classifier_root()
    classifier_str = str(classifier_root)
    
    if classifier_str not in sys.path:
        sys.path.insert(0, classifier_str)
        return True
    return False


def safe_import(module_name: str, package: Optional[str] = None, fallback_module: Optional[str] = None):
    """
    Importa um módulo de forma segura, tentando imports relativos e absolutos.
    
    Args:
        module_name: Nome do módulo (ex: '.config.mlp_config' ou 'config.mlp_config')
        package: Pacote base para imports relativos
        fallback_module: Módulo alternativo se o principal falhar
    
    Returns:
        Módulo importado
    
    Raises:
        ImportError: Se nenhuma forma de import funcionar
    """
    errors = []
    
    # Tentativa 1: Import relativo
    if module_name.startswith('.') and package:
        try:
            return importlib.import_module(module_name, package)
        except ImportError as e:
            errors.append(f"Import relativo falhou: {e}")
    
    # Tentativa 2: Import absoluto (remover ponto inicial se houver)
    abs_module_name = module_name.lstrip('.')
    try:
        ensure_classifier_in_path()
        return importlib.import_module(abs_module_name)
    except ImportError as e:
        errors.append(f"Import absoluto falhou: {e}")
    
    # Tentativa 3: Fallback module
    if fallback_module:
        try:
            ensure_classifier_in_path()
            return importlib.import_module(fallback_module)
        except ImportError as e:
            errors.append(f"Import de fallback falhou: {e}")
    
    # Falhou tudo
    error_msg = f"Falha ao importar {module_name}:\n" + "\n".join(errors)
    raise ImportError(error_msg)


def safe_import_from(module_name: str, *items, package: Optional[str] = None, fallback_module: Optional[str] = None):
    """
    Importa itens específicos de um módulo de forma segura.
    
    Args:
        module_name: Nome do módulo
        *items: Itens para importar do módulo
        package: Pacote base para imports relativos
        fallback_module: Módulo alternativo
    
    Returns:
        Tupla com os itens importados na mesma ordem
        
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
            raise ImportError(f"Item '{item}' não encontrado no módulo {module_name}")
    
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
