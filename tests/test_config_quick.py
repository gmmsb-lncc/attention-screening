#!/usr/bin/env python3
"""
Teste rápido do sistema de configuração.
"""

import sys
import os
from pathlib import Path

# Adicionar src ao path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src" / "classifier"))

print("🧪 Teste Rápido do Sistema de Configuração")
print("=" * 50)

def test_config_system():
    """Testa sistema de configuração."""
    try:
        from utils.config_manager import ConfigManager
        
        # Criar manager
        config_manager = ConfigManager()
        print("✅ ConfigManager criado")
        
        # Criar configuração
        config = config_manager.create_config(template="development")
        print("✅ Configuração development criada")
        
        # Verificar componentes
        print(f"✅ Modelo: {config.model.get_architecture_summary()}")
        print(f"✅ Device: {config.device.requirement}")
        print(f"✅ Batch size: {config.data.batch_size}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no sistema de configuração: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pipeline_integration():
    """Testa integração com pipeline."""
    try:
        from main import MLPPipeline
        
        # Criar pipeline básico
        pipeline = MLPPipeline(config_template="development")
        print("✅ Pipeline criado")
        print(f"✅ Device: {pipeline.device}")
        
        # Verificar componentes
        print(f"✅ Config manager: {type(pipeline.config_manager).__name__}")
        print(f"✅ Device manager: {type(pipeline.device_manager).__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro na integração: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Executa testes rápidos."""
    tests = [
        ("Sistema de Configuração", test_config_system),
        ("Integração Pipeline", test_pipeline_integration),
    ]
    
    results = []
    
    for name, test_func in tests:
        print(f"\n🔍 {name}...")
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"❌ {name}: ERRO - {e}")
            results.append((name, False))
    
    # Relatório
    print("\n" + "=" * 50)
    print("📋 RESULTADO DOS TESTES")
    print("=" * 50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✅ PASSOU" if success else "❌ FALHOU"
        print(f"{name}: {status}")
    
    print(f"\n📊 {passed}/{total} testes passaram")
    
    if passed == total:
        print("🎉 Sistema funcionando!")
    else:
        print("⚠️  Alguns problemas detectados")
    
    return passed == total

if __name__ == "__main__":
    main()
