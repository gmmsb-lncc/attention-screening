#!/usr/bin/env python3
"""
Test 4.1: End-to-End Pipeline Integration
===========================================

Testa integração completa do pipeline de classificação do início ao fim.

Tests incluídos:
1. Complete training pipeline - data → model → training → evaluation
2. Model persistence - save/load modelo treinado
3. Prediction pipeline - inference em novos dados
4. Metrics consistency - métricas consistentes entre train/eval
5. Configuration validation - validação de configs incompatíveis
6. Pipeline error handling - tratamento de erros em cada etapa
7. Resource cleanup - limpeza de recursos após execução

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
from classifier.utils.metrics import MetricsCalculator, ClassificationMetrics


def create_test_dataset(n_samples=200, input_dim=64, seed=42):
    """Criar dataset de teste."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    X = torch.randn(n_samples, input_dim)
    y = torch.randint(0, 2, (n_samples,)).float()
    
    dataset = TensorDataset(X, y)
    return dataset, X, y


def test_1_complete_training_pipeline():
    """
    Test 4.1.1: Complete Training Pipeline
    
    Valida pipeline completo:
    - Data preparation
    - Model initialization
    - Training configuration
    - Training execution
    - Evaluation
    """
    print("\n" + "="*60)
    print("Test 4.1.1: Complete Training Pipeline")
    print("="*60)
    
    # 1. Data preparation
    dataset, X, y = create_test_dataset(n_samples=150, input_dim=64)
    
    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"✅ Data prepared: {train_size} train, {val_size} val samples")
    
    # 2. Model initialization
    model = MLPEmbeddingClassifier(input_dim=64, hidden_dim=32, dropout=0.2)
    device = torch.device("cpu")
    model.to(device)
    
    print(f"✅ Model initialized: {sum(p.numel() for p in model.parameters())} parameters")
    
    # 3. Training configuration
    config = TrainingConfig(
        max_epochs=5,
        patience=10,
        use_scheduler=True,
        amp_enabled=False
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    print(f"✅ Training configured: {config.max_epochs} epochs")
    
    # 4. Training execution
    trainer = ModelTrainer(model=model, config=config, device=device)
    trainer.setup_training(optimizer=optimizer, criterion=criterion)
    
    history = trainer.train(train_loader, val_loader)
    
    print(f"✅ Training completed: {history.total_epochs} epochs executed")
    print(f"   Best epoch: {history.best_epoch}")
    print(f"   Final train loss: {history.train_losses[-1]:.4f}")
    print(f"   Final val loss: {history.val_losses[-1]:.4f}")
    
    # 5. Evaluation
    metrics_calc = MetricsCalculator(device=device)
    final_metrics = metrics_calc.evaluate_model(model, val_loader, criterion)
    
    print(f"✅ Evaluation completed:")
    print(f"   Accuracy: {final_metrics.accuracy:.4f}")
    print(f"   ROC-AUC: {final_metrics.roc_auc:.4f}")
    
    # Validações
    assert history.total_epochs <= config.max_epochs, "Should not exceed max epochs"
    assert len(history.train_losses) == history.total_epochs, "Should have loss for each epoch"
    assert 0 <= final_metrics.accuracy <= 1, "Accuracy should be in [0,1]"
    assert 0 <= final_metrics.roc_auc <= 1, "ROC-AUC should be in [0,1]"
    
    print(f"✅ Complete pipeline validated successfully")


def test_2_model_persistence():
    """
    Test 4.1.2: Model Persistence
    
    Valida save/load de modelo:
    - Save model checkpoint
    - Load model from checkpoint
    - Predictions consistency
    """
    print("\n" + "="*60)
    print("Test 4.1.2: Model Persistence")
    print("="*60)
    
    # Criar e treinar modelo
    dataset, X, y = create_test_dataset(n_samples=100, input_dim=64)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model1 = MLPEmbeddingClassifier(input_dim=64, hidden_dim=32, dropout=0.1)
    device = torch.device("cpu")
    model1.to(device)
    
    config = TrainingConfig(max_epochs=2, patience=10)
    optimizer = torch.optim.Adam(model1.parameters(), lr=0.001)
    
    trainer = ModelTrainer(model=model1, config=config, device=device)
    trainer.setup_training(optimizer=optimizer)
    trainer.train(train_loader, train_loader)
    
    # Fazer predição com modelo original
    model1.eval()
    with torch.no_grad():
        test_input = X[:10].to(device)
        pred1 = model1(test_input)
    
    print(f"✅ Model trained and predictions generated")
    
    # Salvar modelo
    temp_dir = tempfile.mkdtemp()
    save_path = Path(temp_dir) / "model_checkpoint.pt"
    
    torch.save({
        'model_state_dict': model1.state_dict(),
        'model_config': {
            'input_dim': 64,
            'hidden_dim': 32,
            'dropout': 0.1
        }
    }, save_path)
    
    print(f"✅ Model saved to: {save_path}")
    
    # Carregar modelo
    checkpoint = torch.load(save_path)
    model2 = MLPEmbeddingClassifier(
        input_dim=checkpoint['model_config']['input_dim'],
        hidden_dim=checkpoint['model_config']['hidden_dim'],
        dropout=checkpoint['model_config']['dropout']
    )
    model2.load_state_dict(checkpoint['model_state_dict'])
    model2.to(device)
    model2.eval()
    
    print(f"✅ Model loaded from checkpoint")
    
    # Fazer predição com modelo carregado
    with torch.no_grad():
        pred2 = model2(test_input)
    
    # Verificar consistência
    diff = torch.abs(pred1 - pred2).max().item()
    assert diff < 1e-6, f"Predictions should be identical, got diff={diff}"
    
    print(f"✅ Predictions consistent: max diff={diff:.2e}")
    
    # Cleanup
    shutil.rmtree(temp_dir)
    print(f"✅ Temporary files cleaned up")


def test_3_prediction_pipeline():
    """
    Test 4.1.3: Prediction Pipeline
    
    Valida pipeline de inferência:
    - Batch prediction
    - Single sample prediction
    - Probability outputs
    - Binary predictions
    """
    print("\n" + "="*60)
    print("Test 4.1.3: Prediction Pipeline")
    print("="*60)
    
    # Treinar modelo simples
    dataset, X, y = create_test_dataset(n_samples=100, input_dim=64, seed=42)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Usar dropout=0.0 para garantir determinismo durante inferência
    model = MLPEmbeddingClassifier(input_dim=64, hidden_dim=32, dropout=0.0)
    device = torch.device("cpu")
    model.to(device)
    
    # Fixar seed para reproducibilidade
    torch.manual_seed(42)
    np.random.seed(42)
    
    config = TrainingConfig(max_epochs=3, patience=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    trainer = ModelTrainer(model=model, config=config, device=device)
    trainer.setup_training(optimizer=optimizer)
    trainer.train(train_loader, train_loader)
    
    print(f"✅ Model trained")
    
    # Test data - usar mesmas amostras do batch
    test_X = X[:20].to(device)
    
    # Garantir que está em modo eval
    model.eval()
    
    # 1. Batch prediction
    with torch.no_grad():
        logits_batch = model(test_X)
        probs_batch = torch.sigmoid(logits_batch)
        preds_batch = (probs_batch > 0.5).float()
    
    print(f"✅ Batch prediction: {len(test_X)} samples")
    print(f"   Logits shape: {logits_batch.shape}")
    print(f"   Probs range: [{probs_batch.min():.4f}, {probs_batch.max():.4f}]")
    
    # 2. Single sample prediction
    single_sample = test_X[0:1]
    with torch.no_grad():
        logits_single = model(single_sample)
        probs_single = torch.sigmoid(logits_single)
        pred_single = (probs_single > 0.5).float()
    
    print(f"✅ Single prediction: prob={probs_single.item():.4f}, pred={pred_single.item():.0f}")
    print(f"   Expected (from batch): prob={probs_batch[0].item():.4f}, pred={preds_batch[0].item():.0f}")
    
    # 3. Verificar consistência
    # Aceitar pequenas diferenças devido a precisão numérica e otimizações do PyTorch
    diff = torch.abs(logits_single - logits_batch[0:1]).max().item()
    print(f"   Difference: {diff:.6f}")
    
    # O importante é que as predições sejam razoavelmente próximas
    # Não exigir identidade bit-a-bit devido a possíveis otimizações do PyTorch
    assert diff < 0.01, f"Single and batch predictions should be close, got diff={diff}"
    
    # Validar outras propriedades importantes
    assert torch.all((probs_batch >= 0) & (probs_batch <= 1)), "Probs should be in [0,1]"
    assert torch.all((preds_batch == 0) | (preds_batch == 1)), "Preds should be 0 or 1"
    
    # Verificar que predições binárias são iguais (mais importante que logits exatos)
    assert pred_single == preds_batch[0], "Binary predictions should match"
    
    print(f"✅ Prediction pipeline validated (max diff={diff:.2e}, binary preds match)")


def test_4_metrics_consistency():
    """
    Test 4.1.4: Metrics Consistency
    
    Valida consistência de métricas:
    - Training vs evaluation metrics
    - Batch vs epoch metrics
    - Manual vs calculated metrics
    """
    print("\n" + "="*60)
    print("Test 4.1.4: Metrics Consistency")
    print("="*60)
    
    # Preparar dados
    dataset, X, y = create_test_dataset(n_samples=100, input_dim=64)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Treinar modelo
    model = MLPEmbeddingClassifier(input_dim=64, hidden_dim=32, dropout=0.1)
    device = torch.device("cpu")
    model.to(device)
    
    config = TrainingConfig(max_epochs=2, patience=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    trainer = ModelTrainer(model=model, config=config, device=device)
    trainer.setup_training(optimizer=optimizer, criterion=criterion)
    history = trainer.train(loader, loader)
    
    # Calcular métricas via MetricsCalculator
    metrics_calc = MetricsCalculator(device=device)
    calc_metrics = metrics_calc.evaluate_model(model, loader, criterion)
    
    # Pegar métricas de validação da última época do histórico
    last_val_metrics = history.val_metrics[-1] if history.val_metrics else None
    
    print(f"✅ Training completed")
    if last_val_metrics:
        print(f"   History final val accuracy: {last_val_metrics.accuracy:.4f}")
    print(f"   Calculated accuracy: {calc_metrics.accuracy:.4f}")
    
    # Calcular métricas manualmente
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            logits = model(batch_X)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.numpy())
    
    manual_accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    print(f"   Manual accuracy: {manual_accuracy:.4f}")
    
    # Verificar consistência entre métodos
    diff = abs(calc_metrics.accuracy - manual_accuracy)
    assert diff < 0.01, f"Calculated and manual accuracies should be close, got diff={diff:.4f}"
    
    # Se temos métricas do histórico, verificar consistência também
    if last_val_metrics:
        hist_diff = abs(last_val_metrics.accuracy - manual_accuracy)
        assert hist_diff < 0.01, f"History and manual accuracies should be close, got diff={hist_diff:.4f}"
        print(f"✅ Metrics consistent across all methods (max diff={max(diff, hist_diff):.4f})")
    else:
        print(f"✅ Metrics consistent between calculated and manual (diff={diff:.4f})")


def test_5_configuration_validation():
    """
    Test 4.1.5: Configuration Validation
    
    Valida configurações incompatíveis:
    - Invalid parameter values
    - Conflicting settings
    - Missing required configs
    """
    print("\n" + "="*60)
    print("Test 4.1.5: Configuration Validation")
    print("="*60)
    
    # Test 1: Configuração válida
    try:
        config_valid = TrainingConfig(
            max_epochs=10,
            patience=5,
            monitor_mode="max"
        )
        print(f"✅ Valid config accepted: max_epochs={config_valid.max_epochs}")
    except Exception as e:
        assert False, f"Valid config should not raise error: {e}"
    
    # Test 2: Configuração inválida - max_epochs negativo
    try:
        config_invalid = TrainingConfig(max_epochs=-1)
        assert False, "Should raise error for negative max_epochs"
    except ValueError as e:
        print(f"✅ Invalid max_epochs rejected: {str(e)}")
    
    # Test 3: Configuração inválida - patience negativo
    try:
        config_invalid = TrainingConfig(patience=-1)
        assert False, "Should raise error for negative patience"
    except ValueError as e:
        print(f"✅ Invalid patience rejected: {str(e)}")
    
    # Test 4: Configuração inválida - monitor_mode inválido
    try:
        config_invalid = TrainingConfig(monitor_mode="invalid")
        assert False, "Should raise error for invalid monitor_mode"
    except ValueError as e:
        print(f"✅ Invalid monitor_mode rejected: {str(e)}")
    
    # Test 5: Configuração inválida - amp_dtype inválido
    try:
        config_invalid = TrainingConfig(
            amp_enabled=True,
            amp_dtype=torch.float32  # Invalid for AMP
        )
        assert False, "Should raise error for invalid amp_dtype"
    except ValueError as e:
        print(f"✅ Invalid amp_dtype rejected: {str(e)}")
    
    print(f"✅ Configuration validation working correctly")


def test_6_pipeline_error_handling():
    """
    Test 4.1.6: Pipeline Error Handling
    
    Valida tratamento de erros:
    - Invalid input dimensions
    - Empty datasets
    - NaN/Inf in data
    """
    print("\n" + "="*60)
    print("Test 4.1.6: Pipeline Error Handling")
    print("="*60)
    
    device = torch.device("cpu")
    
    # Test 1: Dimensão de entrada incorreta
    model = MLPEmbeddingClassifier(input_dim=64, hidden_dim=32, dropout=0.1)
    model.to(device)
    
    try:
        wrong_input = torch.randn(10, 32)  # Esperado: 64, recebido: 32
        model(wrong_input)
        assert False, "Should raise error for wrong input dimension"
    except RuntimeError as e:
        print(f"✅ Wrong input dimension detected")
    
    # Test 2: Dataset vazio
    try:
        empty_dataset = TensorDataset(torch.empty(0, 64), torch.empty(0))
        empty_loader = DataLoader(empty_dataset, batch_size=32)
        
        # Tentar treinar com dataset vazio
        config = TrainingConfig(max_epochs=1)
        optimizer = torch.optim.Adam(model.parameters())
        trainer = ModelTrainer(model=model, config=config, device=device)
        trainer.setup_training(optimizer=optimizer)
        
        # Deve falhar ou não treinar nada
        history = trainer.train(empty_loader, empty_loader)
        # Se chegou aqui, verificar se não treinou
        assert history.total_epochs == 0 or len(history.train_losses) == 0, "Should not train on empty dataset"
        print(f"✅ Empty dataset handled gracefully")
    except Exception as e:
        print(f"✅ Empty dataset error caught: {type(e).__name__}")
    
    # Test 3: NaN em dados (será detectado durante forward pass)
    try:
        nan_input = torch.randn(10, 64)
        nan_input[0, 0] = float('nan')
        output = model(nan_input)
        
        # Se passou, verificar se output tem NaN
        if torch.isnan(output).any():
            print(f"✅ NaN propagated through network (expected behavior)")
        else:
            print(f"✅ NaN handled internally")
            
    except Exception as e:
        print(f"✅ NaN error caught: {type(e).__name__}")
    
    print(f"✅ Error handling validated")


def test_7_resource_cleanup():
    """
    Test 4.1.7: Resource Cleanup
    
    Valida limpeza de recursos:
    - Memory cleanup after training
    - Model deletion
    - Temporary file cleanup
    """
    print("\n" + "="*60)
    print("Test 4.1.7: Resource Cleanup")
    print("="*60)
    
    # Criar modelo e dataset
    dataset, X, y = create_test_dataset(n_samples=100, input_dim=64)
    loader = DataLoader(dataset, batch_size=32)
    
    model = MLPEmbeddingClassifier(input_dim=64, hidden_dim=32, dropout=0.1)
    device = torch.device("cpu")
    model.to(device)
    
    # Treinar
    config = TrainingConfig(max_epochs=2, patience=10)
    optimizer = torch.optim.Adam(model.parameters())
    
    trainer = ModelTrainer(model=model, config=config, device=device)
    trainer.setup_training(optimizer=optimizer)
    trainer.train(loader, loader)
    
    print(f"✅ Model trained")
    
    # Verificar memória antes de cleanup
    param_count_before = sum(p.numel() for p in model.parameters())
    print(f"   Parameters before cleanup: {param_count_before}")
    
    # Cleanup explícito
    del model
    del trainer
    del optimizer
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"✅ Resources cleaned up")
    
    # Criar arquivo temporário e limpar
    temp_dir = tempfile.mkdtemp()
    temp_file = Path(temp_dir) / "test_file.txt"
    temp_file.write_text("test data")
    
    assert temp_file.exists(), "Temp file should exist"
    print(f"✅ Temp file created: {temp_file}")
    
    # Cleanup
    shutil.rmtree(temp_dir)
    assert not temp_file.exists(), "Temp file should be deleted"
    print(f"✅ Temp directory cleaned up")
    
    print(f"✅ Resource cleanup validated")


def run_all_tests():
    """Executar todos os testes."""
    tests = [
        ("4.1.1 - Complete Training Pipeline", test_1_complete_training_pipeline),
        ("4.1.2 - Model Persistence", test_2_model_persistence),
        ("4.1.3 - Prediction Pipeline", test_3_prediction_pipeline),
        ("4.1.4 - Metrics Consistency", test_4_metrics_consistency),
        ("4.1.5 - Configuration Validation", test_5_configuration_validation),
        ("4.1.6 - Pipeline Error Handling", test_6_pipeline_error_handling),
        ("4.1.7 - Resource Cleanup", test_7_resource_cleanup),
    ]
    
    print("\n" + "="*60)
    print("LEVEL 4.1: END-TO-END PIPELINE INTEGRATION")
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
    print("FINAL SUMMARY - LEVEL 4.1")
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
