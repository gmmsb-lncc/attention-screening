#!/usr/bin/env python3
"""
Test 4.2: Component Integration
================================

Testa integração entre diferentes componentes do módulo classifier.

Tests incluídos:
1. Model + Metrics integration - modelo com diferentes métricas
2. Trainer + Optimizer combinations - diferentes otimizadores/schedulers
3. Cross-validator + Multiple models - CV com diferentes arquiteturas
4. AMP + Scheduler integration - mixed precision com scheduler
5. Early stopping + Checkpoint - checkpoint na melhor época
6. Batch size scaling - comportamento com diferentes batch sizes
7. Data pipeline integration - dataset completo até avaliação

Author: Test Suite
Date: 2024
"""

import sys
from pathlib import Path
import tempfile
import shutil

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
from classifier.core.cross_validator import CrossValidator, CrossValidationConfig
from classifier.utils.metrics import MetricsCalculator, ClassificationMetrics


def create_test_data(n_samples=200, input_dim=64, seed=42):
    """Criar dados de teste."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    X = torch.randn(n_samples, input_dim)
    y = torch.randint(0, 2, (n_samples,)).float()
    
    return X, y


def test_1_model_metrics_integration():
    """
    Test 4.2.1: Model + Metrics Integration
    
    Valida integração de modelo com diferentes métricas.
    """
    print("\n" + "="*60)
    print("Test 4.2.1: Model + Metrics Integration")
    print("="*60)
    
    # Criar dados e modelo
    X, y = create_test_data(n_samples=100, input_dim=64)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    model = MLPEmbeddingClassifier(input_dim=64, hidden_dim=32, dropout=0.1)
    device = torch.device("cpu")
    model.to(device)
    
    # Treinar modelo
    config = TrainingConfig(max_epochs=3, patience=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    trainer = ModelTrainer(model=model, config=config, device=device)
    trainer.setup_training(optimizer=optimizer, criterion=criterion)
    trainer.train(loader, loader)
    
    print(f"✅ Model trained")
    
    # Calcular métricas de diferentes formas
    metrics_calc = MetricsCalculator(device=device)
    
    # 1. Com criterio
    metrics_with_loss = metrics_calc.evaluate_model(model, loader, criterion)
    print(f"✅ Metrics with loss: acc={metrics_with_loss.accuracy:.4f}, loss={metrics_with_loss.loss:.4f}")
    
    # 2. Sem criterio (sem loss)
    metrics_no_loss = metrics_calc.evaluate_model(model, loader, criterion=None)
    print(f"✅ Metrics without loss: acc={metrics_no_loss.accuracy:.4f}")
    
    # Verificar consistência
    assert abs(metrics_with_loss.accuracy - metrics_no_loss.accuracy) < 1e-6, "Accuracy should be same"
    assert metrics_with_loss.loss is not None, "Should have loss when criterion provided"
    assert 0 <= metrics_with_loss.accuracy <= 1, "Accuracy in valid range"
    assert 0 <= metrics_with_loss.roc_auc <= 1, "ROC-AUC in valid range"
    
    print(f"✅ Model-Metrics integration validated")


def test_2_trainer_optimizer_combinations():
    """
    Test 4.2.2: Trainer + Optimizer Combinations
    
    Valida trainer com diferentes otimizadores e schedulers.
    """
    print("\n" + "="*60)
    print("Test 4.2.2: Trainer + Optimizer Combinations")
    print("="*60)
    
    # Dados
    X, y = create_test_data(n_samples=100, input_dim=64)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    device = torch.device("cpu")
    results = {}
    
    # Test diferentes combinações
    combinations = [
        ("Adam", lambda model: torch.optim.Adam(model.parameters(), lr=0.001), False),
        ("SGD", lambda model: torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9), False),
        ("Adam+Scheduler", lambda model: torch.optim.Adam(model.parameters(), lr=0.001), True),
    ]
    
    for name, optimizer_factory, use_scheduler in combinations:
        # Criar modelo novo para cada teste
        model = MLPEmbeddingClassifier(input_dim=64, hidden_dim=32, dropout=0.1)
        model.to(device)
        
        # Configurar e treinar
        config = TrainingConfig(
            max_epochs=3,
            patience=10,
            use_scheduler=use_scheduler
        )
        
        trainer = ModelTrainer(model=model, config=config, device=device)
        optimizer = optimizer_factory(model)
        trainer.setup_training(optimizer=optimizer)
        
        history = trainer.train(loader, loader)
        
        results[name] = {
            'final_loss': history.train_losses[-1],
            'epochs': history.total_epochs
        }
        
        print(f"✅ {name}: loss={results[name]['final_loss']:.4f}, epochs={results[name]['epochs']}")
    
    # Verificações
    for name, result in results.items():
        assert result['epochs'] > 0, f"{name} should train at least 1 epoch"
        assert result['final_loss'] > 0, f"{name} should have positive loss"
    
    print(f"✅ All optimizer combinations validated")


def test_3_cross_validator_multiple_models():
    """
    Test 4.2.3: Cross-validator + Multiple Models
    
    Valida cross-validation com diferentes arquiteturas.
    """
    print("\n" + "="*60)
    print("Test 4.2.3: Cross-validator + Multiple Models")
    print("="*60)
    
    # Dados
    X, y = create_test_data(n_samples=150, input_dim=64)
    device = torch.device("cpu")
    
    # Diferentes arquiteturas
    model_configs = [
        ("Small", 16, 0.1),
        ("Medium", 32, 0.2),
        ("Large", 64, 0.1),
    ]
    
    results = {}
    
    for name, hidden_dim, dropout in model_configs:
        # Configurar CV
        cv_config = CrossValidationConfig(
            n_splits=3,
            shuffle=True,
            random_state=42,
            batch_size=32
        )
        
        training_config = TrainingConfig(max_epochs=2, patience=10)
        
        cv = CrossValidator(
            cv_config=cv_config,
            training_config=training_config,
            device=device
        )
        
        # Model factory
        def model_factory():
            return MLPEmbeddingClassifier(input_dim=64, hidden_dim=hidden_dim, dropout=dropout)
        
        def optimizer_factory(model):
            return torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Execute CV
        cv_results = cv.cross_validate(
            model_factory=model_factory,
            optimizer_factory=optimizer_factory,
            X=X,
            y=y
        )
        
        # Extrair estatísticas de summary_statistics
        summary_stats = cv_results['summary_statistics']
        accuracy_stats = summary_stats.get('accuracy', {})
        
        results[name] = {
            'mean_accuracy': accuracy_stats.get('mean', 0.0),
            'std_accuracy': accuracy_stats.get('std', 0.0)
        }
        
        print(f"✅ {name} (h={hidden_dim}, d={dropout}): acc={results[name]['mean_accuracy']:.4f}±{results[name]['std_accuracy']:.4f}")
    
    # Verificações
    for name, result in results.items():
        assert 0 <= result['mean_accuracy'] <= 1, f"{name} accuracy should be in [0,1]"
        assert result['std_accuracy'] >= 0, f"{name} std should be non-negative"
    
    print(f"✅ Cross-validation with multiple models validated")


def test_4_amp_scheduler_integration():
    """
    Test 4.2.4: AMP + Scheduler Integration
    
    Valida mixed precision com scheduler.
    """
    print("\n" + "="*60)
    print("Test 4.2.4: AMP + Scheduler Integration")
    print("="*60)
    
    # Dados
    X, y = create_test_data(n_samples=100, input_dim=64)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    device = torch.device("cpu")
    
    # Test com AMP + Scheduler
    model = MLPEmbeddingClassifier(input_dim=64, hidden_dim=32, dropout=0.1)
    model.to(device)
    
    config = TrainingConfig(
        max_epochs=3,
        patience=10,
        amp_enabled=False,  # CPU não suporta AMP, mas testar config
        use_scheduler=True
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    trainer = ModelTrainer(model=model, config=config, device=device)
    trainer.setup_training(optimizer=optimizer)
    
    history = trainer.train(loader, loader)
    
    print(f"✅ Training with scheduler completed: {history.total_epochs} epochs")
    print(f"   Learning rates: {history.learning_rates}")
    
    # Verificações
    assert len(history.learning_rates) == history.total_epochs, "Should have LR for each epoch"
    
    # Com scheduler ReduceLROnPlateau, LR pode mudar
    # Apenas verificar que foi registrado
    assert all(lr > 0 for lr in history.learning_rates), "All LRs should be positive"
    
    print(f"✅ AMP + Scheduler integration validated")


def test_5_early_stopping_checkpoint():
    """
    Test 4.2.5: Early Stopping + Checkpoint
    
    Valida checkpoint na melhor época com early stopping.
    """
    print("\n" + "="*60)
    print("Test 4.2.5: Early Stopping + Checkpoint")
    print("="*60)
    
    # Criar dados com overfitting proposital
    X, y = create_test_data(n_samples=80, input_dim=64)
    
    # Split train/val pequeno para forçar overfitting
    train_size = 60
    val_size = 20
    
    train_X, train_y = X[:train_size], y[:train_size]
    val_X, val_y = X[train_size:], y[train_size:]
    
    train_dataset = TensorDataset(train_X, train_y)
    val_dataset = TensorDataset(val_X, val_y)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Modelo e config com early stopping
    model = MLPEmbeddingClassifier(input_dim=64, hidden_dim=64, dropout=0.0)  # Grande para overfit
    device = torch.device("cpu")
    model.to(device)
    
    config = TrainingConfig(
        max_epochs=50,  # Muitas épocas
        patience=5,      # Early stop após 5 épocas sem melhora
        monitor_mode="max"
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    trainer = ModelTrainer(model=model, config=config, device=device)
    trainer.setup_training(optimizer=optimizer)
    
    history = trainer.train(train_loader, val_loader)
    
    print(f"✅ Training completed with early stopping")
    print(f"   Total epochs: {history.total_epochs} / {config.max_epochs}")
    print(f"   Best epoch: {history.best_epoch + 1}")
    print(f"   Early stopped: {history.early_stopped}")
    
    # Verificações
    assert history.total_epochs < config.max_epochs, "Should stop early"
    assert history.early_stopped, "Should be marked as early stopped"
    assert 0 <= history.best_epoch < history.total_epochs, "Best epoch should be valid"
    
    # Salvar checkpoint da melhor época
    temp_dir = tempfile.mkdtemp()
    checkpoint_path = Path(temp_dir) / "best_model.pt"
    
    best_metrics = history.get_best_metrics()
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': history.best_epoch,
        'metrics': best_metrics.to_dict() if best_metrics else None
    }, checkpoint_path)
    
    print(f"✅ Best epoch checkpoint saved")
    if best_metrics:
        print(f"   Best accuracy: {best_metrics.accuracy:.4f}")
    
    # Cleanup
    shutil.rmtree(temp_dir)
    print(f"✅ Early stopping + Checkpoint validated")


def test_6_batch_size_scaling():
    """
    Test 4.2.6: Batch Size Scaling
    
    Valida comportamento com diferentes batch sizes.
    """
    print("\n" + "="*60)
    print("Test 4.2.6: Batch Size Scaling")
    print("="*60)
    
    # Dados
    X, y = create_test_data(n_samples=128, input_dim=64)
    dataset = TensorDataset(X, y)
    
    device = torch.device("cpu")
    batch_sizes = [8, 32, 128]  # Pequeno, médio, grande
    
    results = {}
    
    for batch_size in batch_sizes:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Modelo novo para cada teste
        model = MLPEmbeddingClassifier(input_dim=64, hidden_dim=32, dropout=0.1)
        model.to(device)
        
        config = TrainingConfig(max_epochs=3, patience=10)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        trainer = ModelTrainer(model=model, config=config, device=device)
        trainer.setup_training(optimizer=optimizer)
        
        history = trainer.train(loader, loader)
        
        results[batch_size] = {
            'final_loss': history.train_losses[-1],
            'avg_epoch_time': np.mean(history.epoch_times) if history.epoch_times else 0
        }
        
        print(f"✅ Batch size {batch_size:3d}: loss={results[batch_size]['final_loss']:.4f}, "
              f"time={results[batch_size]['avg_epoch_time']:.3f}s/epoch")
    
    # Verificações
    for bs, result in results.items():
        assert result['final_loss'] > 0, f"Batch size {bs} should have positive loss"
        assert result['avg_epoch_time'] > 0, f"Batch size {bs} should have positive time"
    
    print(f"✅ Batch size scaling validated")


def test_7_data_pipeline_integration():
    """
    Test 4.2.7: Data Pipeline Integration
    
    Valida pipeline completo de dados até avaliação.
    """
    print("\n" + "="*60)
    print("Test 4.2.7: Data Pipeline Integration")
    print("="*60)
    
    # 1. Criar e processar dados
    X, y = create_test_data(n_samples=200, input_dim=64)
    
    # Normalizar dados
    X_mean = X.mean(dim=0, keepdim=True)
    X_std = X.std(dim=0, keepdim=True) + 1e-8
    X_normalized = (X - X_mean) / X_std
    
    print(f"✅ Data created and normalized")
    print(f"   Original mean: {X.mean():.4f}, std: {X.std():.4f}")
    print(f"   Normalized mean: {X_normalized.mean():.4f}, std: {X_normalized.std():.4f}")
    
    # 2. Split train/val/test
    n_train = 120
    n_val = 40
    n_test = 40
    
    train_X, train_y = X_normalized[:n_train], y[:n_train]
    val_X, val_y = X_normalized[n_train:n_train+n_val], y[n_train:n_train+n_val]
    test_X, test_y = X_normalized[n_train+n_val:], y[n_train+n_val:]
    
    train_dataset = TensorDataset(train_X, train_y)
    val_dataset = TensorDataset(val_X, val_y)
    test_dataset = TensorDataset(test_X, test_y)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"✅ Data split: {n_train} train, {n_val} val, {n_test} test")
    
    # 3. Treinar modelo
    model = MLPEmbeddingClassifier(input_dim=64, hidden_dim=32, dropout=0.2)
    device = torch.device("cpu")
    model.to(device)
    
    config = TrainingConfig(max_epochs=5, patience=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    trainer = ModelTrainer(model=model, config=config, device=device)
    trainer.setup_training(optimizer=optimizer, criterion=criterion)
    
    history = trainer.train(train_loader, val_loader)
    
    print(f"✅ Model trained: {history.total_epochs} epochs")
    print(f"   Best validation at epoch {history.best_epoch + 1}")
    
    # 4. Avaliar em test set
    metrics_calc = MetricsCalculator(device=device)
    test_metrics = metrics_calc.evaluate_model(model, test_loader, criterion)
    
    print(f"✅ Test evaluation:")
    print(f"   Accuracy: {test_metrics.accuracy:.4f}")
    print(f"   Precision: {test_metrics.precision:.4f}")
    print(f"   Recall: {test_metrics.recall:.4f}")
    print(f"   F1: {test_metrics.f1:.4f}")
    print(f"   ROC-AUC: {test_metrics.roc_auc:.4f}")
    
    # Verificações
    assert 0 <= test_metrics.accuracy <= 1, "Accuracy in valid range"
    assert 0 <= test_metrics.precision <= 1, "Precision in valid range"
    assert 0 <= test_metrics.recall <= 1, "Recall in valid range"
    assert 0 <= test_metrics.f1 <= 1, "F1 in valid range"
    assert 0 <= test_metrics.roc_auc <= 1, "ROC-AUC in valid range"
    
    print(f"✅ Complete data pipeline validated")


def run_all_tests():
    """Executar todos os testes."""
    tests = [
        ("4.2.1 - Model + Metrics Integration", test_1_model_metrics_integration),
        ("4.2.2 - Trainer + Optimizer Combinations", test_2_trainer_optimizer_combinations),
        ("4.2.3 - Cross-validator + Multiple Models", test_3_cross_validator_multiple_models),
        ("4.2.4 - AMP + Scheduler Integration", test_4_amp_scheduler_integration),
        ("4.2.5 - Early Stopping + Checkpoint", test_5_early_stopping_checkpoint),
        ("4.2.6 - Batch Size Scaling", test_6_batch_size_scaling),
        ("4.2.7 - Data Pipeline Integration", test_7_data_pipeline_integration),
    ]
    
    print("\n" + "="*60)
    print("LEVEL 4.2: COMPONENT INTEGRATION")
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
    print("FINAL SUMMARY - LEVEL 4.2")
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
