#!/usr/bin/env python3
"""
Script de teste de integridade para validar todos os sistemas do DockTKinase Classifier.
Este script deve ser executado a partir do diretório raiz do projeto.
"""

import sys
from pathlib import Path

# Adicionar o diretório src ao path para imports funcionarem
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def test_basic_imports():
    """Testa imports básicos dos sistemas principais."""
    print("🔍 Testando imports básicos...")
    
    try:
        from classifier.config.mlp_config import MLPConfig, create_default_config
        print("✅ MLPConfig importado com sucesso")
        
        from classifier.utils.config_manager import ConfigManager
        print("✅ ConfigManager importado com sucesso")
        
        from classifier.utils.device_manager import SmartDeviceManager
        print("✅ SmartDeviceManager importado com sucesso")
        
        from classifier.utils.data_manager import DataManager
        print("✅ DataManager importado com sucesso")
        
        return True
    except Exception as e:
        print(f"❌ Erro no import básico: {e}")
        return False

def test_class_instantiation():
    """Testa criação de instâncias das classes principais."""
    print("\n🔍 Testando criação de instâncias...")
    
    try:
        from classifier.config.mlp_config import MLPConfig
        from classifier.utils.config_manager import ConfigManager
        from classifier.utils.device_manager import SmartDeviceManager
        
        # Teste MLPConfig
        config = MLPConfig()
        print(f"✅ MLPConfig criado: {config.hidden_layers}")
        
        # Teste ConfigManager
        config_manager = ConfigManager()
        print("✅ ConfigManager criado")
        
        # Teste SmartDeviceManager
        device_manager = SmartDeviceManager()
        print("✅ SmartDeviceManager criado")
        
        return True
    except Exception as e:
        print(f"❌ Erro na criação de instâncias: {e}")
        return False

def test_config_templates():
    """Testa os templates de configuração."""
    print("\n🔍 Testando templates de configuração...")
    
    try:
        from classifier.utils.config_manager import ConfigManager
        
        config_manager = ConfigManager()
        templates = config_manager.list_templates()
        print(f"✅ Templates encontrados: {templates}")
        
        # Teste criação de configuração development
        dev_config = config_manager.create_config("development")
        print(f"✅ Config development criado: modelo {dev_config.model.hidden_layers}")
        
        return True
    except Exception as e:
        print(f"❌ Erro nos templates: {e}")
        return False

def test_device_detection():
    """Testa detecção de dispositivos."""
    print("\n🔍 Testando detecção de dispositivos...")
    
    try:
        from classifier.utils.device_manager import SmartDeviceManager
        
        device_manager = SmartDeviceManager()
        
        # Teste de detecção de dispositivos via validator
        devices = device_manager.validator.detect_available_devices()
        print(f"✅ Dispositivos detectados: {len(devices)}")
        
        # Teste de seleção do melhor dispositivo
        best_device = device_manager.get_device()
        print(f"✅ Melhor dispositivo: {best_device}")
        
        # Teste de informações do dispositivo
        device_info = device_manager.get_device_info()
        if device_info:
            print(f"✅ Info do dispositivo: {device_info.name}")
        
        return True
    except Exception as e:
        print(f"❌ Erro na detecção de dispositivos: {e}")
        return False

def main():
    """Executa todos os testes de integridade."""
    print("🚀 TESTE DE INTEGRIDADE DO DOCKTKINASE CLASSIFIER")
    print("=" * 60)
    
    tests = [
        test_basic_imports,
        test_class_instantiation,
        test_config_templates,
        test_device_detection
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Erro inesperado no teste {test.__name__}: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 RESULTADO FINAL: {passed}/{total} testes passaram")
    
    if passed == total:
        print("🎉 TODOS OS TESTES PASSARAM! Sistema íntegro.")
        return 0
    else:
        print("⚠️  Alguns testes falharam. Verificar problemas acima.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
