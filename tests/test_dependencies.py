#!/usr/bin/env python3
"""
Teste simples de dependências para scripts de build
"""

import sys
import os

def test_basic_imports():
    """Testa imports básicos necessários."""
    print("🔍 Testando imports básicos...")
    
    # Imports essenciais
    basic_imports = [
        ('os', 'Sistema operacional'),
        ('sys', 'Sistema'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
    ]
    
    success_count = 0
    for module_name, description in basic_imports:
        try:
            __import__(module_name)
            print(f"✅ {description} ({module_name}) disponível")
            success_count += 1
        except ImportError:
            print(f"❌ {description} ({module_name}) não disponível")
    
    print(f"\n📊 Imports básicos: {success_count}/{len(basic_imports)} disponíveis")
    
    # Imports opcionais
    optional_imports = [
        ('torch', 'PyTorch'),
        ('esm', 'ESM (Facebook)'),
        ('pyspark', 'PySpark'),
        ('tqdm', 'Progress bars'),
        ('psutil', 'System utilities'),
    ]
    
    print("\n🔍 Testando imports opcionais...")
    optional_success = 0
    for module_name, description in optional_imports:
        try:
            __import__(module_name)
            print(f"✅ {description} ({module_name}) disponível")
            optional_success += 1
        except ImportError:
            print(f"⚠️ {description} ({module_name}) não disponível (opcional)")
    
    print(f"\n📊 Imports opcionais: {optional_success}/{len(optional_imports)} disponíveis")
    
    return success_count == len(basic_imports)

def test_script_syntax():
    """Testa se todos os scripts têm sintaxe válida."""
    print("\n🔍 Testando sintaxe dos scripts...")
    
    build_dir = 'src/build'
    scripts = [
        'buildbinaryLabels.py',
        'buildInteractionLabels.py', 
        'buildEmbeddingMatrix.py',
        'buildKinaseMatrix.py',
        'checkConcatenate.py',
        'checkEmbedding.py',
        'embeddingIBM.py',
        'embeddingMeta.py',
        'build.py'
    ]
    
    syntax_errors = []
    for script in scripts:
        script_path = os.path.join(build_dir, script)
        if os.path.exists(script_path):
            try:
                with open(script_path, 'r') as f:
                    compile(f.read(), script_path, 'exec')
                print(f"✅ {script} - sintaxe OK")
            except SyntaxError as e:
                print(f"❌ {script} - erro de sintaxe: {e}")
                syntax_errors.append((script, str(e)))
        else:
            print(f"⚠️ {script} - arquivo não encontrado")
            syntax_errors.append((script, "Arquivo não encontrado"))
    
    print(f"\n📊 Sintaxe: {len(scripts) - len(syntax_errors)}/{len(scripts)} scripts OK")
    
    if syntax_errors:
        print("\n❌ Erros encontrados:")
        for script, error in syntax_errors:
            print(f"  - {script}: {error}")
    
    return len(syntax_errors) == 0

def main():
    print("🚀 TESTE RÁPIDO DE DEPENDÊNCIAS E SINTAXE")
    print("="*50)
    
    # Teste 1: Imports básicos
    basic_ok = test_basic_imports()
    
    # Teste 2: Sintaxe
    syntax_ok = test_script_syntax()
    
    # Resultado final
    print("\n" + "="*50)
    print("📊 RESULTADO FINAL")
    print("="*50)
    
    if basic_ok and syntax_ok:
        print("🎉 TODOS OS TESTES BÁSICOS PASSARAM!")
        print("✅ Imports básicos disponíveis")
        print("✅ Sintaxe dos scripts válida")
        return True
    else:
        print("⚠️ PROBLEMAS ENCONTRADOS:")
        if not basic_ok:
            print("❌ Alguns imports básicos estão faltando")
        if not syntax_ok:
            print("❌ Alguns scripts têm problemas de sintaxe")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
