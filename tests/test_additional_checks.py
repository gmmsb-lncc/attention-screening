#!/usr/bin/env python3
"""
Script para verificar inconsistências adicionais no sistema DockTKinase.
"""

import sys
from pathlib import Path
import ast
import re

# Adicionar src ao path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def check_init_files():
    """Verifica se todos os __init__.py estão consistentes."""
    print("🔍 Verificando arquivos __init__.py...")
    
    init_files = list(Path("src/classifier").rglob("__init__.py"))
    
    for init_file in init_files:
        print(f"📁 {init_file}")
        try:
            with open(init_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if content.strip():
                    # Verificar se há imports problemáticos
                    if "from classifier" in content and not content.startswith("from ."):
                        print(f"  ⚠️  Import absoluto encontrado")
                    else:
                        print(f"  ✅ OK")
                else:
                    print(f"  ℹ️  Vazio")
        except Exception as e:
            print(f"  ❌ Erro ao ler: {e}")

def check_duplicate_functions():
    """Verifica se há funções duplicadas."""
    print("\n🔍 Verificando funções duplicadas...")
    
    py_files = list(Path("src/classifier").glob("**/*.py"))
    function_registry = {}
    
    for py_file in py_files:
        if "test_" in str(py_file) or "__pycache__" in str(py_file):
            continue
            
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Encontrar definições de função
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_name = node.name
                    if func_name.startswith('_'):  # Skip private functions
                        continue
                        
                    if func_name not in function_registry:
                        function_registry[func_name] = []
                    function_registry[func_name].append(str(py_file))
                        
        except Exception as e:
            print(f"❌ Erro ao analisar {py_file}: {e}")
    
    # Verificar duplicatas
    duplicates_found = False
    for func_name, files in function_registry.items():
        if len(files) > 1:
            print(f"⚠️  Função '{func_name}' encontrada em: {files}")
            duplicates_found = True
    
    if not duplicates_found:
        print("✅ Nenhuma função duplicada encontrada")

def check_config_consistency():
    """Verifica consistência nas configurações."""
    print("\n🔍 Verificando consistência de configurações...")
    
    try:
        from classifier.config.mlp_config import MLPConfig
        from classifier.utils.config_manager import ConfigManager
        
        # Verificar se campos do MLPConfig batem com os usados no ConfigManager
        config = MLPConfig()
        config_manager = ConfigManager()
        
        # Testar criação de template
        template_config = config_manager.create_config("development")
        
        # Verificar se os campos são compatíveis
        mlp_fields = set(config.__dataclass_fields__.keys())
        template_mlp_fields = set(template_config.model.__dataclass_fields__.keys())
        
        if mlp_fields == template_mlp_fields:
            print("✅ Campos MLPConfig consistentes entre sistemas")
        else:
            print(f"⚠️  Diferenças encontradas:")
            print(f"   MLPConfig: {mlp_fields - template_mlp_fields}")
            print(f"   Template:  {template_mlp_fields - mlp_fields}")
        
    except Exception as e:
        print(f"❌ Erro na verificação de configuração: {e}")

def check_imports_cycles():
    """Verifica imports circulares básicos."""
    print("\n🔍 Verificando possíveis imports circulares...")
    
    py_files = list(Path("src/classifier").glob("**/*.py"))
    import_graph = {}
    
    for py_file in py_files:
        if "__pycache__" in str(py_file):
            continue
            
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Encontrar imports relativos locais
            relative_imports = re.findall(r'from \.([\w.]+) import', content)
            
            file_key = str(py_file.relative_to(Path("src/classifier")))
            import_graph[file_key] = relative_imports
            
        except Exception as e:
            print(f"❌ Erro ao analisar imports em {py_file}: {e}")
    
    # Análise básica de ciclos
    cycles_found = []
    for file, imports in import_graph.items():
        for imp in imports:
            # Verificar se o import tenta importar de volta
            imp_file = imp.replace('.', '/') + '.py'
            if imp_file in import_graph:
                if any(back_imp for back_imp in import_graph[imp_file] 
                       if file.replace('.py', '').replace('/', '.') in back_imp):
                    cycles_found.append((file, imp_file))
    
    if cycles_found:
        print(f"⚠️  Possíveis imports circulares encontrados: {cycles_found}")
    else:
        print("✅ Nenhum import circular óbvio encontrado")

def main():
    """Executa todas as verificações."""
    print("🔧 VERIFICAÇÕES ADICIONAIS DE INTEGRIDADE")
    print("=" * 60)
    
    checks = [
        check_init_files,
        check_duplicate_functions,
        check_config_consistency,
        check_imports_cycles
    ]
    
    for check in checks:
        try:
            check()
        except Exception as e:
            print(f"❌ Erro na verificação {check.__name__}: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Verificações adicionais concluídas")

if __name__ == "__main__":
    main()
