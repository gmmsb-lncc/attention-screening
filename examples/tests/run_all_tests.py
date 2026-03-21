#!/usr/bin/env python3
"""
Suite de Testes Completa - Executa todos os testes principais.
"""

import sys
import os
import subprocess
import time
from pathlib import Path

print("🧪 SUITE COMPLETA DE TESTES - DOCKTKINASE")
print("=" * 60)
print(f"📅 Data: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"📂 Executando da pasta: {Path.cwd()}")
print("=" * 60)

def run_test(test_name: str, test_file: str) -> bool:
    """Executa um teste e retorna se passou."""
    print(f"\n🔍 {test_name}")
    print("-" * 40)
    
    try:
        result = subprocess.run(
            [sys.executable, test_file], 
            capture_output=True, 
            text=True, 
            timeout=30
        )
        
        if result.returncode == 0:
            print("✅ PASSOU")
            return True
        else:
            print("❌ FALHOU")
            # Mostrar apenas última linha do erro se houver
            if result.stderr:
                error_lines = result.stderr.strip().split('\n')
                print(f"   Erro: {error_lines[-1]}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏱️ TIMEOUT (>30s)")
        return False
    except Exception as e:
        print(f"💥 ERRO: {e}")
        return False

def main():
    """Executa suite completa de testes."""
    
    # Lista de testes principais
    tests = [
        ("Funcionalidade Básica", "test_basic_functionality.py"),
        ("Device Manager", "test_device_manager.py"),
        ("Pipeline Completo", "test_complete_pipeline.py"),
        ("Divisão Robusta", "simple_test_robust_split.py"),
        ("Relatório Final", "test_final_report.py"),
    ]
    
    results = []
    start_time = time.time()
    
    for test_name, test_file in tests:
        success = run_test(test_name, test_file)
        results.append((test_name, success))
    
    elapsed = time.time() - start_time
    
    # Relatório final
    print("\n" + "=" * 60)
    print("📊 RELATÓRIO FINAL DA SUITE")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSOU" if success else "❌ FALHOU"
        print(f"{test_name:.<30} {status}")
    
    print("-" * 60)
    print(f"📈 Resultado: {passed}/{total} testes passaram ({passed/total*100:.1f}%)")
    print(f"⏱️  Tempo total: {elapsed:.1f}s")
    
    if passed == total:
        print("🎉 TODOS OS TESTES PASSARAM!")
        print("✅ Sistema DockTKinase totalmente funcional")
        print("✅ Pronto para uso em produção")
    elif passed >= total * 0.8:
        print("⚠️  MAIORIA DOS TESTES PASSOU")
        print("🔧 Alguns ajustes podem ser necessários")
    else:
        print("❌ MUITOS TESTES FALHARAM")
        print("🛠️  Correções necessárias antes do uso")
    
    print("=" * 60)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
