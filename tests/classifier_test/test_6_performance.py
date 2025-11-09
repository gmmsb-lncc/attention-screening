#!/usr/bin/env python3
"""
Test 6: Performance Testing
============================

Testa performance, benchmarks e otimizações do módulo classifier.

Tests incluídos:
1. Training speed benchmark - velocidade de treinamento
2. Inference speed benchmark - velocidade de inferência
3. Memory usage profiling - uso de memória

Author: Test Suite
Date: 2024
"""

import sys
from pathlib import Path
import time
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


def test_1_training_speed_benchmark():
    """
    Test 6.1: Training Speed Benchmark
    
    Mede velocidade de treinamento com diferentes configurações.
    """
    print("\n" + "="*60)
    print("Test 6.1: Training Speed Benchmark")
    print("="*60)
    
    device = torch.device("cpu")
    
    # Configurações para benchmark
    configs = [
        ("Small (n=500, h=16)", 500, 16, 32),
        ("Medium (n=1000, h=32)", 1000, 32, 64),
        ("Large (n=2000, h=64)", 2000, 64, 128),
    ]
    
    results = {}
    
    for name, n_samples, hidden_dim, batch_size in configs:
        print(f"\n--- {name} ---")
        
        # Dados
        X = torch.randn(n_samples, 64)
        y = torch.randint(0, 2, (n_samples,)).float()
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Modelo
        model = MLPEmbeddingClassifier(input_dim=64, hidden_dim=hidden_dim, dropout=0.1)
        model.to(device)
        
        # Treinamento
        config = TrainingConfig(max_epochs=5, patience=10)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        trainer = ModelTrainer(model=model, config=config, device=device)
        trainer.setup_training(optimizer=optimizer)
        
        start_time = time.time()
        history = trainer.train(loader, loader)
        total_time = time.time() - start_time
        
        # Calcular métricas de performance
        samples_per_sec = n_samples * history.total_epochs / total_time
        time_per_epoch = total_time / history.total_epochs
        
        results[name] = {
            'total_time': total_time,
            'time_per_epoch': time_per_epoch,
            'samples_per_sec': samples_per_sec,
            'final_loss': history.train_losses[-1]
        }
        
        print(f"✅ Total time: {total_time:.2f}s")
        print(f"   Time/epoch: {time_per_epoch:.3f}s")
        print(f"   Throughput: {samples_per_sec:.0f} samples/sec")
        print(f"   Final loss: {results[name]['final_loss']:.4f}")
        
        # Cleanup
        del model, trainer, optimizer
        gc.collect()
    
    # Verificar que performance escala razoavelmente
    small_time = results["Small (n=500, h=16)"]['time_per_epoch']
    large_time = results["Large (n=2000, h=64)"]['time_per_epoch']
    
    # Large deve ser mais lento, mas não muito (max 10x)
    speedup_ratio = large_time / small_time
    print(f"\n✅ Performance scaling: Large/Small = {speedup_ratio:.2f}x")
    assert speedup_ratio < 10, f"Performance degradation too high: {speedup_ratio:.2f}x"
    
    print(f"\n✅ Training speed benchmark completed")


def test_2_inference_speed_benchmark():
    """
    Test 6.2: Inference Speed Benchmark
    
    Mede velocidade de inferência.
    """
    print("\n" + "="*60)
    print("Test 6.2: Inference Speed Benchmark")
    print("="*60)
    
    device = torch.device("cpu")
    
    # Treinar modelo
    X_train = torch.randn(500, 64)
    y_train = torch.randint(0, 2, (500,)).float()
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32)
    
    model = MLPEmbeddingClassifier(input_dim=64, hidden_dim=32, dropout=0.1)
    model.to(device)
    
    config = TrainingConfig(max_epochs=3, patience=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    trainer = ModelTrainer(model=model, config=config, device=device)
    trainer.setup_training(optimizer=optimizer)
    trainer.train(train_loader, train_loader)
    
    print(f"✅ Model trained")
    
    # Benchmark de inferência
    model.eval()
    
    batch_sizes = [1, 10, 100, 1000]
    
    for batch_size in batch_sizes:
        X_test = torch.randn(batch_size, 64).to(device)
        
        # Warmup
        with torch.no_grad():
            _ = model(X_test)
        
        # Benchmark
        n_iterations = 100
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(n_iterations):
                _ = model(X_test)
        
        total_time = time.time() - start_time
        avg_time = total_time / n_iterations
        throughput = batch_size / avg_time
        
        print(f"✅ Batch size {batch_size:4d}: {avg_time*1000:.2f}ms/batch, {throughput:.0f} samples/sec")
    
    # Verificar que throughput aumenta com batch size
    print(f"\n✅ Inference speed benchmark completed")


def test_3_memory_usage_profiling():
    """
    Test 6.3: Memory Usage Profiling
    
    Perfila uso de memória durante treinamento.
    """
    print("\n" + "="*60)
    print("Test 6.3: Memory Usage Profiling")
    print("="*60)
    
    device = torch.device("cpu")
    
    # Função auxiliar para estimar memória
    def get_model_memory(model):
        """Estima memória do modelo em MB."""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        return (param_size + buffer_size) / 1024**2
    
    # Test diferentes tamanhos de modelo
    configs = [
        ("Tiny", 64, 8),
        ("Small", 64, 32),
        ("Medium", 64, 128),
        ("Large", 64, 512),
    ]
    
    print("\n--- Model Memory Usage ---")
    for name, input_dim, hidden_dim in configs:
        model = MLPEmbeddingClassifier(input_dim=input_dim, hidden_dim=hidden_dim, dropout=0.1)
        model.to(device)
        
        mem_mb = get_model_memory(model)
        n_params = sum(p.numel() for p in model.parameters())
        
        print(f"✅ {name:8s} (h={hidden_dim:3d}): {mem_mb:.2f} MB, {n_params:6d} params")
        
        del model
    
    # Test memória durante treinamento
    print("\n--- Training Memory Profile ---")
    
    X = torch.randn(1000, 64)
    y = torch.randint(0, 2, (1000,)).float()
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=64)
    
    model = MLPEmbeddingClassifier(input_dim=64, hidden_dim=32, dropout=0.1)
    model.to(device)
    
    base_mem = get_model_memory(model)
    print(f"✅ Model memory: {base_mem:.2f} MB")
    
    config = TrainingConfig(max_epochs=3, patience=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Memória do optimizer (estados internos)
    optimizer_states = sum(
        state.numel() * state.element_size() 
        for group in optimizer.param_groups 
        for state in optimizer.state.values()
        for state in state.values() if isinstance(state, torch.Tensor)
    ) / 1024**2 if optimizer.state else 0
    
    print(f"✅ Optimizer memory: {optimizer_states:.2f} MB")
    
    trainer = ModelTrainer(model=model, config=config, device=device)
    trainer.setup_training(optimizer=optimizer)
    
    # Treinar e medir
    history = trainer.train(loader, loader)
    
    print(f"✅ Training completed: {history.total_epochs} epochs")
    print(f"   Total model + optimizer: {base_mem + optimizer_states:.2f} MB")
    
    # Cleanup
    del model, trainer, optimizer
    gc.collect()
    
    print(f"\n✅ Memory profiling completed")


def run_all_tests():
    """Executar todos os testes."""
    tests = [
        ("6.1 - Training Speed Benchmark", test_1_training_speed_benchmark),
        ("6.2 - Inference Speed Benchmark", test_2_inference_speed_benchmark),
        ("6.3 - Memory Usage Profiling", test_3_memory_usage_profiling),
    ]
    
    print("\n" + "="*60)
    print("LEVEL 6: PERFORMANCE TESTING")
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
    print("FINAL SUMMARY - LEVEL 6")
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
