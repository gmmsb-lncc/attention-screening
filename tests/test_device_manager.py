#!/usr/bin/env python3
"""
Teste do novo DeviceManager consolidado.
"""

import sys
import os
from pathlib import Path

# Adicionar src ao path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src" / "classifier"))

import torch

print("🔧 Teste do DeviceManager Consolidado")
print("=" * 50)

def test_simple_mode():
    """Testa modo simples."""
    print("\n1️⃣ MODO SIMPLE")
    print("-" * 30)
    
    try:
        from utils.device_manager import DeviceManager
        
        manager = DeviceManager(mode="simple")
        device = manager.get_device()
        info = manager.get_device_info()
        
        print(f"✅ Device: {device}")
        print(f"✅ Info: {info.name if info else 'N/A'}")
        print(f"✅ Validação: {manager.validate_device_status()}")
        
        return True
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False

def test_smart_mode():
    """Testa modo smart."""
    print("\n2️⃣ MODO SMART")
    print("-" * 30)
    
    try:
        from utils.device_manager import DeviceManager
        
        manager = DeviceManager(mode="smart")
        device = manager.get_device()
        info = manager.get_device_info()
        devices = manager.get_available_devices()
        
        print(f"✅ Device: {device}")
        print(f"✅ Info: {info.get_summary() if info else 'N/A'}")
        print(f"✅ Devices disponíveis: {len(devices)}")
        for d in devices:
            print(f"   - {d.get_summary()}")
        
        return True
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_complex_mode():
    """Testa modo complexo."""
    print("\n3️⃣ MODO COMPLEX")
    print("-" * 30)
    
    try:
        from utils.device_manager import DeviceManager
        
        manager = DeviceManager(mode="complex", enable_benchmarking=False)  # Sem benchmark para ser rápido
        device = manager.get_device()
        info = manager.get_device_info()
        
        print(f"✅ Device: {device}")
        print(f"✅ Info: {info.get_summary() if info else 'N/A'}")
        if info and info.warnings:
            print(f"⚠️  Warnings: {info.warnings}")
        
        return True
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False

def test_aliases():
    """Testa aliases para compatibilidade."""
    print("\n4️⃣ ALIASES (COMPATIBILIDADE)")
    print("-" * 30)
    
    try:
        from utils.device_manager import SimpleDeviceManager, SmartDeviceManager
        
        # Simple
        simple = SimpleDeviceManager()
        device1 = simple.get_device()
        print(f"✅ SimpleDeviceManager: {device1}")
        
        # Smart
        smart = SmartDeviceManager()
        device2 = smart.get_device()
        print(f"✅ SmartDeviceManager: {device2}")
        
        return True
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False

def test_convenience_function():
    """Testa função de conveniência."""
    print("\n5️⃣ FUNÇÃO DE CONVENIÊNCIA")
    print("-" * 30)
    
    try:
        from utils.device_manager import get_best_device
        
        device1 = get_best_device("auto", "simple")
        device2 = get_best_device("auto", "smart")
        
        print(f"✅ get_best_device (simple): {device1}")
        print(f"✅ get_best_device (smart): {device2}")
        
        return True
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False

def test_tensor_operations():
    """Testa operações com tensores."""
    print("\n6️⃣ OPERAÇÕES COM TENSORES")
    print("-" * 30)
    
    try:
        from utils.device_manager import DeviceManager
        
        manager = DeviceManager(mode="smart")
        device = manager.get_device()
        
        # Criar tensor e mover para device
        x = torch.randn(100, 100)
        x_device = manager.to_device(x)
        
        # Operação no device
        y = x_device @ x_device
        
        print(f"✅ Tensor original: {x.device}")
        print(f"✅ Tensor no device: {x_device.device}")
        print(f"✅ Resultado: {y.shape} no {y.device}")
        
        return True
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False

def main():
    """Executa todos os testes."""
    tests = [
        ("Modo Simple", test_simple_mode),
        ("Modo Smart", test_smart_mode),
        ("Modo Complex", test_complex_mode),
        ("Aliases", test_aliases),
        ("Função Conveniência", test_convenience_function),
        ("Operações Tensores", test_tensor_operations)
    ]
    
    results = []
    for name, test_func in tests:
        success = test_func()
        results.append((name, success))
    
    # Relatório
    print("\n" + "=" * 50)
    print("📊 RESULTADO FINAL")
    print("=" * 50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✅ PASSOU" if success else "❌ FALHOU"
        print(f"{name}: {status}")
    
    print(f"\n📈 {passed}/{total} testes passaram ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 DEVICEMANAGER CONSOLIDADO FUNCIONANDO!")
    else:
        print("⚠️  Alguns problemas detectados")

if __name__ == "__main__":
    main()
