#!/usr/bin/env python3
"""
Teste básico para verificar funcionamento do pipeline.
"""

import sys
import os
from pathlib import Path

# Adicionar src ao path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src" / "classifier"))

print("🚀 Teste Básico do DockTKinase Pipeline")
print("=" * 50)

def test_imports():
    """Testa imports básicos."""
    print("\n📦 Testando imports básicos...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy: {e}")
        return False
    
    try:
        from config.mlp_config import MLPConfig, create_default_config
        print("✅ MLPConfig")
    except ImportError as e:
        print(f"❌ MLPConfig: {e}")
        return False
    
    try:
        from utils.device_manager import SimpleDeviceManager
        print("✅ SimpleDeviceManager")
    except ImportError as e:
        print(f"❌ SimpleDeviceManager: {e}")
        return False
    
    return True

def test_basic_model():
    """Testa criação e funcionamento básico do modelo."""
    print("\n🧠 Testando modelo básico...")
    
    try:
        import torch
        from config.mlp_config import create_default_config
        from models.mlp import MLPEmbeddingClassifier
        
        # Configuração básica
        config = create_default_config(input_size=100)
        print(f"✅ Configuração criada: {config.get_architecture_summary()}")
        
        # Criar modelo
        model = MLPEmbeddingClassifier(config)
        print(f"✅ Modelo criado com {sum(p.numel() for p in model.parameters())} parâmetros")
        
        # Teste forward
        x = torch.randn(32, 100)
        with torch.no_grad():
            output = model(x)
        print(f"✅ Forward pass: input {x.shape} -> output {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no modelo: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_split():
    """Testa divisão de dados."""
    print("\n📊 Testando divisão de dados...")
    
    try:
        import torch
        from utils.train_test_split import robust_train_test_split
        
        # Dados sintéticos
        X = torch.randn(1000, 50)
        y = torch.randint(0, 2, (1000,))
        
        # Split
        X_train, X_test, y_train, y_test = robust_train_test_split(
            X, y, test_size=0.2, verbose=False
        )
        
        print(f"✅ Split realizado:")
        print(f"  - Train: {X_train.shape[0]} amostras")
        print(f"  - Test: {X_test.shape[0]} amostras")
        print(f"  - Proporções mantidas: {y_train.bincount()}, {y_test.bincount()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro na divisão: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_device_manager():
    """Testa gerenciamento de dispositivos."""
    print("\n🔧 Testando device manager...")
    
    try:
        from utils.device_manager import SimpleDeviceManager
        
        manager = SimpleDeviceManager()
        device = manager.get_device()
        
        print(f"✅ Device selecionado: {device}")
        
        # Teste de tensor
        import torch
        x = torch.randn(10, 10)
        x_device = manager.to_device(x)
        print(f"✅ Tensor movido para device: {x_device.device}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no device manager: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Executa todos os testes."""
    print("Iniciando bateria de testes...\n")
    
    tests = [
        ("Imports Básicos", test_imports),
        ("Modelo Básico", test_basic_model),
        ("Divisão de Dados", test_data_split),
        ("Device Manager", test_device_manager),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"❌ {name}: ERRO CRÍTICO - {e}")
            results.append((name, False))
    
    # Relatório final
    print("\n" + "=" * 50)
    print("📋 RELATÓRIO FINAL")
    print("=" * 50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✅ PASSOU" if success else "❌ FALHOU"
        print(f"{name}: {status}")
    
    print(f"\n📊 Resultado: {passed}/{total} testes passaram ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 TODOS OS TESTES PASSARAM!")
        print("✅ Pipeline funcionando corretamente!")
    elif passed > total // 2:
        print("⚠️  MAIORIA DOS TESTES PASSOU")
        print("🔧 Alguns ajustes podem ser necessários")
    else:
        print("❌ MUITOS TESTES FALHARAM")
        print("🛠️  Correções necessárias antes do uso")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
