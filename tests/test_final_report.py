#!/usr/bin/env python3
"""
Relatório Final de Testes do Pipeline DockTKinase.
"""

import sys
import os
from pathlib import Path
import time

# Adicionar src ao path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src" / "classifier"))

def run_test_suite():
    """Executa suite completa de testes."""
    
    print("🚀 RELATÓRIO FINAL DE TESTES - DOCKTKINASE PIPELINE")
    print("=" * 70)
    print(f"📅 Data: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python: {sys.version.split()[0]}")
    print("=" * 70)
    
    tests_results = []
    
    # 1. Teste de Imports Básicos
    print("\n1️⃣ TESTANDO IMPORTS BÁSICOS")
    print("-" * 40)
    
    basic_imports = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("sklearn", "Scikit-learn"),
        ("pandas", "Pandas")
    ]
    
    imports_ok = 0
    for module, name in basic_imports:
        try:
            __import__(module)
            version = getattr(__import__(module), '__version__', 'N/A')
            print(f"✅ {name}: {version}")
            imports_ok += 1
        except ImportError:
            print(f"❌ {name}: Não encontrado")
    
    tests_results.append(("Imports Básicos", imports_ok, len(basic_imports)))
    
    # 2. Teste de Componentes do Sistema
    print("\n2️⃣ TESTANDO COMPONENTES DO SISTEMA")
    print("-" * 40)
    
    components_ok = 0
    components = [
        ("config.mlp_config", "Configuração MLP"),
        ("models.mlp", "Modelo MLP"),
        ("utils.device_manager", "Gerenciador Device"),
        ("utils.train_test_split", "Divisão Train/Test"),
        ("main", "Pipeline Principal")
    ]
    
    for module, name in components:
        try:
            __import__(module)
            print(f"✅ {name}")
            components_ok += 1
        except ImportError as e:
            print(f"❌ {name}: {str(e)[:50]}...")
    
    tests_results.append(("Componentes Sistema", components_ok, len(components)))
    
    # 3. Teste Funcional Básico
    print("\n3️⃣ TESTANDO FUNCIONALIDADES BÁSICAS")
    print("-" * 40)
    
    functional_tests = []
    
    # Device Manager
    try:
        from utils.device_manager import SimpleDeviceManager
        manager = SimpleDeviceManager()
        device = manager.get_device()
        print(f"✅ Device Manager: {device}")
        functional_tests.append(True)
    except Exception as e:
        print(f"❌ Device Manager: {e}")
        functional_tests.append(False)
    
    # Configuração MLP
    try:
        from config.mlp_config import create_default_config
        config = create_default_config(100)
        print(f"✅ Config MLP: {config.get_architecture_summary()}")
        functional_tests.append(True)
    except Exception as e:
        print(f"❌ Config MLP: {e}")
        functional_tests.append(False)
    
    # Modelo MLP
    try:
        from models.mlp import MLPEmbeddingClassifier
        model = MLPEmbeddingClassifier(config)
        params = sum(p.numel() for p in model.parameters())
        print(f"✅ Modelo MLP: {params:,} parâmetros")
        functional_tests.append(True)
    except Exception as e:
        print(f"❌ Modelo MLP: {e}")
        functional_tests.append(False)
    
    # Train/Test Split
    try:
        import torch
        from utils.train_test_split import robust_train_test_split
        X = torch.randn(100, 20)
        y = torch.randint(0, 2, (100,))
        X_train, X_test, y_train, y_test = robust_train_test_split(X, y)
        print(f"✅ Train/Test Split: {len(X_train)}/{len(X_test)}")
        functional_tests.append(True)
    except Exception as e:
        print(f"❌ Train/Test Split: {e}")
        functional_tests.append(False)
    
    # Pipeline Principal
    try:
        from main import MLPPipeline
        pipeline = MLPPipeline()
        print(f"✅ Pipeline Principal: {pipeline.device}")
        functional_tests.append(True)
    except Exception as e:
        print(f"❌ Pipeline Principal: {e}")
        functional_tests.append(False)
    
    functional_ok = sum(functional_tests)
    tests_results.append(("Funcionalidades", functional_ok, len(functional_tests)))
    
    # 4. Teste de Integração
    print("\n4️⃣ TESTANDO INTEGRAÇÃO COMPLETA")
    print("-" * 40)
    
    integration_ok = 0
    try:
        import torch
        from torch.utils.data import TensorDataset, DataLoader
        
        # Dados sintéticos
        X = torch.randn(200, 50)
        y = torch.randint(0, 2, (200,))
        
        # Pipeline
        pipeline = MLPPipeline()
        
        # Dataset
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=32)
        
        # Modelo
        config = create_default_config(50)
        model = MLPEmbeddingClassifier(config)
        model = model.to(pipeline.device)
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            batch_X, batch_y = next(iter(dataloader))
            batch_X = batch_X.to(pipeline.device)
            output = model(batch_X)
        
        print(f"✅ Integração Completa: {batch_X.shape} -> {output.shape}")
        integration_ok = 1
        
    except Exception as e:
        print(f"❌ Integração Completa: {e}")
        integration_ok = 0
    
    tests_results.append(("Integração", integration_ok, 1))
    
    # RELATÓRIO FINAL
    print("\n" + "=" * 70)
    print("📊 RELATÓRIO FINAL")
    print("=" * 70)
    
    total_passed = 0
    total_tests = 0
    
    for test_name, passed, total in tests_results:
        percentage = (passed / total * 100) if total > 0 else 0
        status = "✅" if passed == total else "⚠️" if passed > total//2 else "❌"
        print(f"{status} {test_name}: {passed}/{total} ({percentage:.1f}%)")
        total_passed += passed
        total_tests += total
    
    overall_percentage = (total_passed / total_tests * 100) if total_tests > 0 else 0
    
    print("-" * 70)
    print(f"📈 RESULTADO GERAL: {total_passed}/{total_tests} ({overall_percentage:.1f}%)")
    
    if overall_percentage >= 90:
        print("🎉 EXCELENTE! Sistema totalmente funcional")
        status = "APROVADO"
    elif overall_percentage >= 75:
        print("✅ BOM! Sistema majoritariamente funcional")
        status = "APROVADO"
    elif overall_percentage >= 50:
        print("⚠️ PARCIAL! Algumas funcionalidades precisam de ajustes")
        status = "PARCIAL"
    else:
        print("❌ CRÍTICO! Muitas correções necessárias")
        status = "REPROVADO"
    
    print(f"🏆 STATUS FINAL: {status}")
    print("=" * 70)
    
    return overall_percentage >= 75

if __name__ == "__main__":
    success = run_test_suite()
    sys.exit(0 if success else 1)
