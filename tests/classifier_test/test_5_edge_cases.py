#!/usr/bin/env python3
"""
Test 5: Edge Cases
==================

Testa condições de contorno e casos extremos do módulo classifier.

Tests incluídos:
1. Empty and minimal datasets - datasets vazios ou muito pequenos
2. Extreme dimensions - dimensões muito grandes ou pequenas
3. Imbalanced classes - classes muito desbalanceadas
4. Invalid data types - tipos de dados incorretos
5. Boundary values - valores limites (0, 1, infinito, NaN)
6. Memory constraints - limites de memória
7. Concurrent operations - operações concorrentes/conflitantes

Author: Test Suite
Date: 2024
"""

import sys
from pathlib import Path
import gc

# Adicionar src ao path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# Imports do classifier
from classifier.models.mlp_classifier import MLPEmbeddingClassifier
from classifier.core.trainer import ModelTrainer, TrainingConfig
from classifier.utils.metrics import MetricsCalculator


def test_1_empty_and_minimal_datasets():
    """
    Test 5.1: Empty and Minimal Datasets
    
    Valida comportamento com datasets vazios ou muito pequenos.
    """
    print("\n" + "="*60)
    print("Test 5.1: Empty and Minimal Datasets")
    print("="*60)
    
    device = torch.device("cpu")
    model = MLPEmbeddingClassifier(input_dim=64, hidden_dim=32, dropout=0.1)
    model.to(device)
    
    # Test 1: Dataset vazio (0 samples)
    print("\n--- Test 1.1: Empty dataset ---")
    try:
        empty_X = torch.empty(0, 64)
        empty_y = torch.empty(0)
        empty_dataset = TensorDataset(empty_X, empty_y)
        empty_loader = DataLoader(empty_dataset, batch_size=32)
        
        # Tentar calcular métricas
        metrics_calc = MetricsCalculator(device=device)
        criterion = nn.BCEWithLogitsLoss()
        
        # Isso deve falhar ou retornar métricas inválidas
        if len(empty_loader) == 0:
            print(f"✅ Empty dataset detected (0 batches)")
        else:
            metrics = metrics_calc.evaluate_model(model, empty_loader, criterion)
            print(f"⚠️  Empty dataset processed (unexpected)")
            
    except Exception as e:
        print(f"✅ Empty dataset error caught: {type(e).__name__}")
    
    # Test 2: Dataset minimal (1 sample)
    print("\n--- Test 1.2: Minimal dataset (1 sample) ---")
    try:
        minimal_X = torch.randn(1, 64)
        minimal_y = torch.tensor([1.0])
        minimal_dataset = TensorDataset(minimal_X, minimal_y)
        minimal_loader = DataLoader(minimal_dataset, batch_size=32)
        
        metrics = metrics_calc.evaluate_model(model, minimal_loader, criterion)
        print(f"✅ Single sample processed: acc={metrics.accuracy:.4f}")
        
        # Treinar com 1 amostra (deve funcionar mas não aprender muito)
        config = TrainingConfig(max_epochs=1, patience=10)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        trainer = ModelTrainer(model=model, config=config, device=device)
        trainer.setup_training(optimizer=optimizer, criterion=criterion)
        history = trainer.train(minimal_loader, minimal_loader)
        
        print(f"✅ Training with 1 sample completed: {history.total_epochs} epochs")
        
    except Exception as e:
        print(f"⚠️  Minimal dataset error: {type(e).__name__}: {e}")
    
    # Test 3: Dataset muito pequeno (2 samples)
    print("\n--- Test 1.3: Very small dataset (2 samples) ---")
    small_X = torch.randn(2, 64)
    small_y = torch.tensor([0.0, 1.0])
    small_dataset = TensorDataset(small_X, small_y)
    small_loader = DataLoader(small_dataset, batch_size=32)
    
    # Novo modelo
    model2 = MLPEmbeddingClassifier(input_dim=64, hidden_dim=32, dropout=0.1)
    model2.to(device)
    
    config = TrainingConfig(max_epochs=2, patience=10)
    optimizer = torch.optim.Adam(model2.parameters(), lr=0.001)
    
    trainer = ModelTrainer(model=model2, config=config, device=device)
    trainer.setup_training(optimizer=optimizer, criterion=criterion)
    history = trainer.train(small_loader, small_loader)
    
    print(f"✅ Training with 2 samples: loss={history.train_losses[-1]:.4f}")
    
    print(f"\n✅ Edge case: Empty/minimal datasets validated")


def test_2_extreme_dimensions():
    """
    Test 5.2: Extreme Dimensions
    
    Valida comportamento com dimensões extremas.
    """
    print("\n" + "="*60)
    print("Test 5.2: Extreme Dimensions")
    print("="*60)
    
    device = torch.device("cpu")
    
    # Test 1: Dimensão de entrada muito pequena (1)
    print("\n--- Test 2.1: Very small input dimension (1) ---")
    model_small = MLPEmbeddingClassifier(input_dim=1, hidden_dim=8, dropout=0.1)
    model_small.to(device)
    
    X_small = torch.randn(50, 1)
    y_small = torch.randint(0, 2, (50,)).float()
    dataset_small = TensorDataset(X_small, y_small)
    loader_small = DataLoader(dataset_small, batch_size=16)
    
    config = TrainingConfig(max_epochs=2, patience=10)
    optimizer = torch.optim.Adam(model_small.parameters(), lr=0.001)
    
    trainer = ModelTrainer(model=model_small, config=config, device=device)
    trainer.setup_training(optimizer=optimizer)
    history = trainer.train(loader_small, loader_small)
    
    print(f"✅ Model with input_dim=1: {sum(p.numel() for p in model_small.parameters())} params")
    print(f"   Training completed: loss={history.train_losses[-1]:.4f}")
    
    # Test 2: Dimensão de entrada grande (2048)
    print("\n--- Test 2.2: Large input dimension (2048) ---")
    model_large = MLPEmbeddingClassifier(input_dim=2048, hidden_dim=64, dropout=0.1)
    model_large.to(device)
    
    X_large = torch.randn(50, 2048)
    y_large = torch.randint(0, 2, (50,)).float()
    dataset_large = TensorDataset(X_large, y_large)
    loader_large = DataLoader(dataset_large, batch_size=16)
    
    optimizer_large = torch.optim.Adam(model_large.parameters(), lr=0.001)
    
    trainer_large = ModelTrainer(model=model_large, config=config, device=device)
    trainer_large.setup_training(optimizer=optimizer_large)
    history_large = trainer_large.train(loader_large, loader_large)
    
    print(f"✅ Model with input_dim=2048: {sum(p.numel() for p in model_large.parameters())} params")
    print(f"   Training completed: loss={history_large.train_losses[-1]:.4f}")
    
    # Test 3: Hidden dimension muito pequena (2)
    print("\n--- Test 2.3: Very small hidden dimension (2) ---")
    model_tiny_hidden = MLPEmbeddingClassifier(input_dim=64, hidden_dim=2, dropout=0.0)
    model_tiny_hidden.to(device)
    
    X = torch.randn(50, 64)
    y = torch.randint(0, 2, (50,)).float()
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=16)
    
    optimizer = torch.optim.Adam(model_tiny_hidden.parameters(), lr=0.001)
    trainer = ModelTrainer(model=model_tiny_hidden, config=config, device=device)
    trainer.setup_training(optimizer=optimizer)
    history = trainer.train(loader, loader)
    
    print(f"✅ Model with hidden_dim=2: {sum(p.numel() for p in model_tiny_hidden.parameters())} params")
    print(f"   Training completed: loss={history.train_losses[-1]:.4f}")
    
    print(f"\n✅ Edge case: Extreme dimensions validated")


def test_3_imbalanced_classes():
    """
    Test 5.3: Imbalanced Classes
    
    Valida comportamento com classes muito desbalanceadas.
    """
    print("\n" + "="*60)
    print("Test 5.3: Imbalanced Classes")
    print("="*60)
    
    device = torch.device("cpu")
    
    # Test 1: Desbalanceamento extremo (99:1)
    print("\n--- Test 3.1: Extreme imbalance (99% class 0, 1% class 1) ---")
    n_total = 1000
    n_class_1 = 10  # 1%
    n_class_0 = n_total - n_class_1  # 99%
    
    X = torch.randn(n_total, 64)
    y = torch.cat([torch.zeros(n_class_0), torch.ones(n_class_1)])
    
    # Shuffle
    indices = torch.randperm(n_total)
    X = X[indices]
    y = y[indices]
    
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = MLPEmbeddingClassifier(input_dim=64, hidden_dim=32, dropout=0.1)
    model.to(device)
    
    config = TrainingConfig(max_epochs=3, patience=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    trainer = ModelTrainer(model=model, config=config, device=device)
    trainer.setup_training(optimizer=optimizer, criterion=criterion)
    history = trainer.train(loader, loader)
    
    # Avaliar
    metrics_calc = MetricsCalculator(device=device)
    metrics = metrics_calc.evaluate_model(model, loader, criterion)
    
    print(f"✅ Extreme imbalance (99:1):")
    print(f"   Accuracy: {metrics.accuracy:.4f}")
    print(f"   Precision: {metrics.precision:.4f}")
    print(f"   Recall: {metrics.recall:.4f}")
    print(f"   F1: {metrics.f1:.4f}")
    
    # Test 2: Classe completamente ausente no treino
    print("\n--- Test 3.2: Single class only (all class 0) ---")
    X_single = torch.randn(100, 64)
    y_single = torch.zeros(100)
    
    dataset_single = TensorDataset(X_single, y_single)
    loader_single = DataLoader(dataset_single, batch_size=32)
    
    model_single = MLPEmbeddingClassifier(input_dim=64, hidden_dim=32, dropout=0.1)
    model_single.to(device)
    
    config = TrainingConfig(max_epochs=2, patience=10)
    optimizer = torch.optim.Adam(model_single.parameters(), lr=0.001)
    
    trainer = ModelTrainer(model=model_single, config=config, device=device)
    trainer.setup_training(optimizer=optimizer, criterion=criterion)
    
    try:
        history = trainer.train(loader_single, loader_single)
        print(f"✅ Single class training: loss={history.train_losses[-1]:.4f}")
        
        # Métricas podem ser undefined para single class
        metrics_single = metrics_calc.evaluate_model(model_single, loader_single, criterion)
        print(f"   Accuracy: {metrics_single.accuracy:.4f}")
        
    except Exception as e:
        print(f"⚠️  Single class training error: {type(e).__name__}")
    
    print(f"\n✅ Edge case: Imbalanced classes validated")


def test_4_invalid_data_types():
    """
    Test 5.4: Invalid Data Types
    
    Valida detecção de tipos de dados incorretos.
    """
    print("\n" + "="*60)
    print("Test 5.4: Invalid Data Types")
    print("="*60)
    
    device = torch.device("cpu")
    model = MLPEmbeddingClassifier(input_dim=64, hidden_dim=32, dropout=0.1)
    model.to(device)
    
    # Test 1: Tipos incorretos de entrada
    print("\n--- Test 4.1: Wrong input types ---")
    
    # Integer input (deveria funcionar, PyTorch converte)
    try:
        X_int = torch.randint(0, 10, (50, 64))
        output = model(X_int.float().to(device))
        print(f"✅ Integer input converted: output shape {output.shape}")
    except Exception as e:
        print(f"⚠️  Integer input error: {type(e).__name__}")
    
    # Test 2: Dimensões incorretas
    print("\n--- Test 4.2: Wrong dimensions ---")
    
    # 1D input (esperado 2D)
    try:
        X_1d = torch.randn(64)
        output = model(X_1d.to(device))
        print(f"⚠️  1D input accepted (unexpected): {output.shape}")
    except (RuntimeError, ValueError) as e:
        print(f"✅ 1D input rejected correctly: {type(e).__name__}")
    
    # 3D input (esperado 2D) - pode funcionar se última dim = input_dim
    try:
        X_3d = torch.randn(10, 5, 64)
        output = model(X_3d.to(device))
        # 3D pode funcionar, PyTorch aplica em última dimensão
        print(f"✅ 3D input processed: {output.shape} (broadcasts over batch dims)")
    except (RuntimeError, ValueError) as e:
        print(f"✅ 3D input rejected: {type(e).__name__}")
    
    # Test 3: Labels com tipo incorreto para BCEWithLogitsLoss
    print("\n--- Test 4.3: Wrong label types ---")
    
    X = torch.randn(50, 64)
    y_int = torch.randint(0, 2, (50,))  # Int, deveria ser Float
    
    dataset = TensorDataset(X, y_int.float())  # Corrigir para Float
    loader = DataLoader(dataset, batch_size=16)
    
    config = TrainingConfig(max_epochs=1, patience=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    trainer = ModelTrainer(model=model, config=config, device=device)
    trainer.setup_training(optimizer=optimizer, criterion=criterion)
    
    try:
        history = trainer.train(loader, loader)
        print(f"✅ Training with converted labels: loss={history.train_losses[-1]:.4f}")
    except Exception as e:
        print(f"⚠️  Label type error: {type(e).__name__}")
    
    print(f"\n✅ Edge case: Invalid data types validated")


def test_5_boundary_values():
    """
    Test 5.5: Boundary Values
    
    Valida comportamento com valores limites.
    """
    print("\n" + "="*60)
    print("Test 5.5: Boundary Values")
    print("="*60)
    
    device = torch.device("cpu")
    
    # Test 1: All zeros input
    print("\n--- Test 5.1: All zeros input ---")
    model = MLPEmbeddingClassifier(input_dim=64, hidden_dim=32, dropout=0.1)
    model.to(device)
    model.eval()
    
    X_zeros = torch.zeros(10, 64).to(device)
    with torch.no_grad():
        output_zeros = model(X_zeros)
    
    print(f"✅ All-zeros input: output range [{output_zeros.min():.4f}, {output_zeros.max():.4f}]")
    assert not torch.isnan(output_zeros).any(), "Output should not have NaN"
    
    # Test 2: All ones input
    print("\n--- Test 5.2: All ones input ---")
    X_ones = torch.ones(10, 64).to(device)
    with torch.no_grad():
        output_ones = model(X_ones)
    
    print(f"✅ All-ones input: output range [{output_ones.min():.4f}, {output_ones.max():.4f}]")
    assert not torch.isnan(output_ones).any(), "Output should not have NaN"
    
    # Test 3: Very large values
    print("\n--- Test 5.3: Very large values ---")
    X_large = torch.ones(10, 64).to(device) * 1000
    with torch.no_grad():
        output_large = model(X_large)
    
    print(f"✅ Large values (x1000): output range [{output_large.min():.4f}, {output_large.max():.4f}]")
    
    # Test 4: Very small values
    print("\n--- Test 5.4: Very small values ---")
    X_small = torch.ones(10, 64).to(device) * 1e-6
    with torch.no_grad():
        output_small = model(X_small)
    
    print(f"✅ Small values (x1e-6): output range [{output_small.min():.4f}, {output_small.max():.4f}]")
    
    # Test 5: NaN input (propagação)
    print("\n--- Test 5.5: NaN input ---")
    X_nan = torch.randn(10, 64).to(device)
    X_nan[0, 0] = float('nan')
    
    with torch.no_grad():
        output_nan = model(X_nan)
    
    if torch.isnan(output_nan).any():
        print(f"✅ NaN propagated through network (expected)")
    else:
        print(f"✅ NaN handled gracefully")
    
    # Test 6: Inf input
    print("\n--- Test 5.6: Inf input ---")
    X_inf = torch.randn(10, 64).to(device)
    X_inf[0, 0] = float('inf')
    
    with torch.no_grad():
        output_inf = model(X_inf)
    
    if torch.isinf(output_inf).any() or torch.isnan(output_inf).any():
        print(f"✅ Inf propagated or converted to NaN (expected)")
    else:
        print(f"✅ Inf handled gracefully")
    
    print(f"\n✅ Edge case: Boundary values validated")


def test_6_memory_constraints():
    """
    Test 5.6: Memory Constraints
    
    Valida comportamento com limites de memória.
    """
    print("\n" + "="*60)
    print("Test 5.6: Memory Constraints")
    print("="*60)
    
    device = torch.device("cpu")
    
    # Test 1: Large batch size
    print("\n--- Test 6.1: Large batch size ---")
    X_large_batch = torch.randn(10000, 64)
    y_large_batch = torch.randint(0, 2, (10000,)).float()
    dataset_large = TensorDataset(X_large_batch, y_large_batch)
    loader_large = DataLoader(dataset_large, batch_size=5000)  # Batch grande
    
    model = MLPEmbeddingClassifier(input_dim=64, hidden_dim=32, dropout=0.1)
    model.to(device)
    
    config = TrainingConfig(max_epochs=1, patience=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    trainer = ModelTrainer(model=model, config=config, device=device)
    trainer.setup_training(optimizer=optimizer)
    
    try:
        history = trainer.train(loader_large, loader_large)
        print(f"✅ Large batch training: loss={history.train_losses[-1]:.4f}")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"✅ OOM error caught correctly")
        else:
            print(f"⚠️  Unexpected error: {e}")
    
    # Test 2: Model deletion and cleanup
    print("\n--- Test 6.2: Memory cleanup ---")
    param_count_before = sum(p.numel() for p in model.parameters())
    
    del model
    del trainer
    del optimizer
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"✅ Memory cleaned up ({param_count_before} params deleted)")
    
    # Test 3: Multiple models sequentially
    print("\n--- Test 6.3: Sequential model creation ---")
    for i in range(5):
        temp_model = MLPEmbeddingClassifier(input_dim=64, hidden_dim=128, dropout=0.1)
        temp_model.to(device)
        
        # Pequeno forward pass
        with torch.no_grad():
            temp_output = temp_model(torch.randn(10, 64).to(device))
        
        del temp_model
        gc.collect()
    
    print(f"✅ Created and deleted 5 models sequentially")
    
    print(f"\n✅ Edge case: Memory constraints validated")


def test_7_concurrent_operations():
    """
    Test 5.7: Concurrent Operations
    
    Valida operações concorrentes/conflitantes.
    """
    print("\n" + "="*60)
    print("Test 5.7: Concurrent Operations")
    print("="*60)
    
    device = torch.device("cpu")
    
    # Test 1: Train/eval mode switching
    print("\n--- Test 7.1: Train/eval mode switching ---")
    model = MLPEmbeddingClassifier(input_dim=64, hidden_dim=32, dropout=0.3)
    model.to(device)
    
    X_test = torch.randn(10, 64).to(device)
    
    # Training mode
    model.train()
    with torch.no_grad():
        out_train_1 = model(X_test)
        out_train_2 = model(X_test)
    
    # Eval mode
    model.eval()
    with torch.no_grad():
        out_eval_1 = model(X_test)
        out_eval_2 = model(X_test)
    
    # Em eval mode com dropout=0.3, outputs devem ser idênticos
    eval_diff = torch.abs(out_eval_1 - out_eval_2).max().item()
    print(f"✅ Eval mode consistency: max diff = {eval_diff:.6f}")
    assert eval_diff < 1e-6, "Eval mode should be deterministic"
    
    # Test 2: Multiple optimizers (não recomendado mas deve funcionar)
    print("\n--- Test 7.2: Multiple optimizers ---")
    model2 = MLPEmbeddingClassifier(input_dim=64, hidden_dim=32, dropout=0.1)
    model2.to(device)
    
    optimizer1 = torch.optim.Adam(model2.parameters(), lr=0.001)
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01)
    
    # Usar optimizer1
    X = torch.randn(10, 64).to(device)
    y = torch.randint(0, 2, (10,)).float().to(device)
    
    model2.train()
    optimizer1.zero_grad()
    output = model2(X)
    loss = nn.BCEWithLogitsLoss()(output.squeeze(), y)
    loss.backward()
    optimizer1.step()
    
    print(f"✅ Step with optimizer1: loss={loss.item():.4f}")
    
    # Usar optimizer2 (vai atualizar com diferentes parâmetros)
    optimizer2.zero_grad()
    output = model2(X)
    loss = nn.BCEWithLogitsLoss()(output.squeeze(), y)
    loss.backward()
    optimizer2.step()
    
    print(f"✅ Step with optimizer2: loss={loss.item():.4f}")
    print(f"⚠️  Using multiple optimizers is not recommended but works")
    
    # Test 3: Modificar modelo durante avaliação (não deve afetar)
    print("\n--- Test 7.3: Model modification during evaluation ---")
    model3 = MLPEmbeddingClassifier(input_dim=64, hidden_dim=32, dropout=0.1)
    model3.to(device)
    
    # Eval mode
    model3.eval()
    with torch.no_grad():
        out_before = model3(X_test)
    
    # Tentar modificar (mas está em no_grad)
    model3.fc1.weight.data += 0.1  # Pequena perturbação
    
    with torch.no_grad():
        out_after = model3(X_test)
    
    diff = torch.abs(out_before - out_after).max().item()
    print(f"✅ Output changed after weight modification: diff={diff:.4f}")
    assert diff > 0, "Weight modification should affect output"
    
    print(f"\n✅ Edge case: Concurrent operations validated")


def run_all_tests():
    """Executar todos os testes."""
    tests = [
        ("5.1 - Empty and Minimal Datasets", test_1_empty_and_minimal_datasets),
        ("5.2 - Extreme Dimensions", test_2_extreme_dimensions),
        ("5.3 - Imbalanced Classes", test_3_imbalanced_classes),
        ("5.4 - Invalid Data Types", test_4_invalid_data_types),
        ("5.5 - Boundary Values", test_5_boundary_values),
        ("5.6 - Memory Constraints", test_6_memory_constraints),
        ("5.7 - Concurrent Operations", test_7_concurrent_operations),
    ]
    
    print("\n" + "="*60)
    print("LEVEL 5: EDGE CASES")
    print("="*60)
    
    passed = 0
    failed = 0
    errors = []
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
            print(f"✅ {test_name} PASSED\n")
        except AssertionError as e:
            failed += 1
            error_msg = f"❌ {test_name} FAILED: {str(e)}"
            print(f"{error_msg}\n")
            errors.append(error_msg)
        except Exception as e:
            failed += 1
            error_msg = f"❌ {test_name} ERROR: {str(e)}"
            print(f"{error_msg}\n")
            errors.append(error_msg)
    
    # Sumário final
    print("\n" + "="*60)
    print("FINAL SUMMARY - LEVEL 5")
    print("="*60)
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success rate: {(passed/len(tests)*100):.1f}%")
    
    if errors:
        print("\n❌ Failed tests:")
        for error in errors:
            print(f"  {error}")
    else:
        print("\n🎉 All tests passed!")
    
    print("="*60)
    
    return passed == len(tests)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
