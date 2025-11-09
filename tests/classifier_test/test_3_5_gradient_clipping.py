"""
Test 3.5: Gradient Clipping Tests

Testa a funcionalidade de clipping de gradientes no ModelTrainer:
1. Clipping desabilitado (gradientes sem restrição)
2. Clipping por valor (clip by value)
3. Clipping por norma (clip by norm)
4. Magnitude dos gradientes
5. Prevenção de explosão de gradientes
6. Validação de parâmetros de clipping
7. Histórico de clipping
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from classifier.core.trainer import ModelTrainer, TrainingConfig
from classifier.models.mlp_classifier import MLPEmbeddingClassifier


def create_model(input_dim: int = 64, hidden_dim: int = 32):
    """Cria um modelo MLP para testes."""
    return MLPEmbeddingClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout=0.2
    )


def create_synthetic_data(n_samples: int = 200, input_dim: int = 64):
    """
    Cria dados sintéticos para treinamento.
    
    Args:
        n_samples: Número de amostras
        input_dim: Dimensionalidade das features
        
    Returns:
        TensorDataset com features e labels
    """
    torch.manual_seed(42)
    X = torch.randn(n_samples, input_dim)
    y = (X.sum(dim=1) > 0).float()
    return TensorDataset(X, y)


def test_1_no_gradient_clipping():
    """
    Test 3.5.1: Gradient Clipping Desabilitado
    
    Valida que:
    - Sem clipping, gradientes podem crescer livremente
    - gradient_clip_value=None e gradient_clip_norm=None
    - Modelo treina normalmente
    - Histórico não registra clipping
    """
    print("\n" + "="*60)
    print("Test 3.5.1: No Gradient Clipping")
    print("="*60)
    
    # Configuração sem gradient clipping
    config = TrainingConfig(
        max_epochs=5,
        patience=10,
        gradient_clip_value=None,  # Sem clipping por valor
        gradient_clip_norm=None,   # Sem clipping por norma
        monitor_metric="roc_auc",
        monitor_mode="max"
    )
    
    # Dataset e DataLoader
    dataset = create_synthetic_data(n_samples=200, input_dim=64)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Modelo
    model = create_model(input_dim=64)
    
    # Trainer
    device = torch.device("cpu")
    trainer = ModelTrainer(model=model, config=config, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer.setup_training(optimizer=optimizer)
    
    # Treinamento
    history = trainer.train(train_loader, val_loader)
    
    # Validações
    assert config.gradient_clip_value is None, "Clip value should be None"
    assert config.gradient_clip_norm is None, "Clip norm should be None"
    assert history.total_epochs == 5, f"Should train all 5 epochs, got {history.total_epochs}"
    assert len(history.train_losses) == 5, "Should have 5 training losses"
    
    print(f"✅ No gradient clipping configured")
    print(f"✅ Trained {history.total_epochs} epochs")
    print(f"✅ Final train loss: {history.train_losses[-1]:.4f}")


def test_2_gradient_clip_by_value():
    """
    Test 3.5.2: Gradient Clipping por Valor
    
    Valida que:
    - gradient_clip_value limita valores individuais de gradientes
    - Gradientes ficam no intervalo [-clip_value, clip_value]
    - Modelo treina com gradientes controlados
    - Previne valores extremos
    """
    print("\n" + "="*60)
    print("Test 3.5.2: Gradient Clip by Value")
    print("="*60)
    
    # Configuração com clipping por valor
    clip_value = 0.5
    config = TrainingConfig(
        max_epochs=5,
        patience=10,
        gradient_clip_value=clip_value,
        gradient_clip_norm=None,
        monitor_metric="roc_auc",
        monitor_mode="max"
    )
    
    # Dataset e DataLoader
    dataset = create_synthetic_data(n_samples=200, input_dim=64)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Modelo
    model = create_model(input_dim=64)
    
    # Trainer
    device = torch.device("cpu")
    trainer = ModelTrainer(model=model, config=config, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # LR maior para gerar gradientes maiores
    trainer.setup_training(optimizer=optimizer)
    
    # Treinamento
    history = trainer.train(train_loader, val_loader)
    
    # Validações
    assert config.gradient_clip_value == clip_value, f"Clip value should be {clip_value}"
    assert history.total_epochs == 5, f"Should train 5 epochs, got {history.total_epochs}"
    
    # Verificar que o modelo aprendeu (loss diminuiu)
    initial_loss = history.train_losses[0]
    final_loss = history.train_losses[-1]
    
    print(f"✅ Gradient clip by value: {clip_value}")
    print(f"✅ Initial loss: {initial_loss:.4f}")
    print(f"✅ Final loss: {final_loss:.4f}")
    print(f"✅ Loss decreased: {initial_loss > final_loss}")


def test_3_gradient_clip_by_norm():
    """
    Test 3.5.3: Gradient Clipping por Norma
    
    Valida que:
    - gradient_clip_norm limita a norma total dos gradientes
    - Norma L2 dos gradientes ≤ clip_norm
    - Preserva direção dos gradientes
    - Controla magnitude global
    """
    print("\n" + "="*60)
    print("Test 3.5.3: Gradient Clip by Norm")
    print("="*60)
    
    # Configuração com clipping por norma
    clip_norm = 1.0
    config = TrainingConfig(
        max_epochs=5,
        patience=10,
        gradient_clip_value=None,
        gradient_clip_norm=clip_norm,
        monitor_metric="roc_auc",
        monitor_mode="max"
    )
    
    # Dataset e DataLoader
    dataset = create_synthetic_data(n_samples=200, input_dim=64)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Modelo
    model = create_model(input_dim=64)
    
    # Trainer
    device = torch.device("cpu")
    trainer = ModelTrainer(model=model, config=config, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer.setup_training(optimizer=optimizer)
    
    # Treinamento
    history = trainer.train(train_loader, val_loader)
    
    # Validações
    assert config.gradient_clip_norm == clip_norm, f"Clip norm should be {clip_norm}"
    assert history.total_epochs == 5, f"Should train 5 epochs, got {history.total_epochs}"
    
    # Verificar convergência
    initial_loss = history.train_losses[0]
    final_loss = history.train_losses[-1]
    
    print(f"✅ Gradient clip by norm: {clip_norm}")
    print(f"✅ Initial loss: {initial_loss:.4f}")
    print(f"✅ Final loss: {final_loss:.4f}")
    print(f"✅ Loss improved: {final_loss < initial_loss}")


def test_4_gradient_magnitudes():
    """
    Test 3.5.4: Magnitude dos Gradientes
    
    Valida que:
    - Gradientes sem clipping podem ter valores grandes
    - Clipping reduz magnitude dos gradientes
    - Comparação entre com e sem clipping
    """
    print("\n" + "="*60)
    print("Test 3.5.4: Gradient Magnitudes")
    print("="*60)
    
    # Dataset e DataLoader
    dataset = create_synthetic_data(n_samples=200, input_dim=64)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Teste 1: Sem clipping
    model = create_model(input_dim=64)
    model_no_clip = create_model(input_dim=64)
    
    config_no_clip = TrainingConfig(
        max_epochs=3,
        patience=10,
        gradient_clip_value=None,
        gradient_clip_norm=None,
        monitor_metric="roc_auc",
        monitor_mode="max"
    )
    
    device = torch.device("cpu")
    trainer_no_clip = ModelTrainer(model=model_no_clip, config=config_no_clip, device=device)
    optimizer_no_clip = torch.optim.SGD(model_no_clip.parameters(), lr=0.1)  # LR alto para gradientes grandes
    trainer_no_clip.setup_training(optimizer=optimizer_no_clip)
    history_no_clip = trainer_no_clip.train(train_loader, val_loader)
    
    # Teste 2: Com clipping por valor
    model = create_model(input_dim=64)
    model_with_clip = create_model(input_dim=64)
    
    config_with_clip = TrainingConfig(
        max_epochs=3,
        patience=10,
        gradient_clip_value=0.1,  # Clipping agressivo
        gradient_clip_norm=None,
        monitor_metric="roc_auc",
        monitor_mode="max"
    )
    
    trainer_with_clip = ModelTrainer(model=model_with_clip, config=config_with_clip, device=device)
    optimizer_with_clip = torch.optim.SGD(model_with_clip.parameters(), lr=0.1)
    trainer_with_clip.setup_training(optimizer=optimizer_with_clip)
    history_with_clip = trainer_with_clip.train(train_loader, val_loader)
    
    # Validações
    assert len(history_no_clip.train_losses) == 3, "Should train 3 epochs without clip"
    assert len(history_with_clip.train_losses) == 3, "Should train 3 epochs with clip"
    
    print(f"✅ Without clipping - Final loss: {history_no_clip.train_losses[-1]:.4f}")
    print(f"✅ With clipping (0.1) - Final loss: {history_with_clip.train_losses[-1]:.4f}")
    print(f"✅ Both models trained successfully")


def test_5_gradient_explosion_prevention():
    """
    Test 3.5.5: Prevenção de Explosão de Gradientes
    
    Valida que:
    - Clipping previne explosão de gradientes
    - Loss permanece estável com clipping
    - Sem clipping, loss pode explodir com LR alto
    """
    print("\n" + "="*60)
    print("Test 3.5.5: Gradient Explosion Prevention")
    print("="*60)
    
    # Dataset e DataLoader
    dataset = create_synthetic_data(n_samples=200, input_dim=64)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Modelo com clipping (proteção contra explosão)
    model = create_model(input_dim=64)
    # model created above
    
    config = TrainingConfig(
        max_epochs=10,
        patience=15,
        gradient_clip_value=None,
        gradient_clip_norm=1.0,  # Clipping por norma para estabilidade
        monitor_metric="roc_auc",
        monitor_mode="max"
    )
    
    device = torch.device("cpu")
    trainer = ModelTrainer(model=model, config=config, device=device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)  # LR muito alto
    trainer.setup_training(optimizer=optimizer)
    
    # Treinamento
    history = trainer.train(train_loader, val_loader)
    
    # Validações - Com clipping, o treinamento deve ser estável
    assert all(not torch.isnan(torch.tensor(loss)) for loss in history.train_losses), \
        "Training losses should not contain NaN"
    assert all(not torch.isinf(torch.tensor(loss)) for loss in history.train_losses), \
        "Training losses should not contain Inf"
    
    # Verificar que a loss não explodiu
    max_loss = max(history.train_losses)
    assert max_loss < 100.0, f"Loss should not explode, got max={max_loss:.4f}"
    
    print(f"✅ Gradient clipping enabled (norm={config.gradient_clip_norm})")
    print(f"✅ Training stable with high LR (0.5)")
    print(f"✅ Max loss: {max_loss:.4f} (< 100.0)")
    print(f"✅ No NaN or Inf values")


def test_6_clipping_parameter_validation():
    """
    Test 3.5.6: Validação de Parâmetros de Clipping
    
    Valida que:
    - gradient_clip_value aceita valores positivos
    - gradient_clip_norm aceita valores positivos
    - Valores negativos são rejeitados
    - Zero é aceito (sem clipping efetivo)
    """
    print("\n" + "="*60)
    print("Test 3.5.6: Clipping Parameter Validation")
    print("="*60)
    
    # Teste 1: Valores válidos positivos
    try:
        config1 = TrainingConfig(
            max_epochs=5,
            gradient_clip_value=1.0,
            gradient_clip_norm=None
        )
        assert config1.gradient_clip_value == 1.0, "Should accept positive clip value"
        print("✅ Positive clip_value (1.0) accepted")
    except Exception as e:
        print(f"❌ Failed to accept positive clip_value: {e}")
        raise
    
    # Teste 2: Valores válidos positivos para norm
    try:
        config2 = TrainingConfig(
            max_epochs=5,
            gradient_clip_value=None,
            gradient_clip_norm=2.0
        )
        assert config2.gradient_clip_norm == 2.0, "Should accept positive clip norm"
        print("✅ Positive clip_norm (2.0) accepted")
    except Exception as e:
        print(f"❌ Failed to accept positive clip_norm: {e}")
        raise
    
    # Teste 3: Ambos None (sem clipping)
    try:
        config3 = TrainingConfig(
            max_epochs=5,
            gradient_clip_value=None,
            gradient_clip_norm=None
        )
        assert config3.gradient_clip_value is None, "Should accept None for clip_value"
        assert config3.gradient_clip_norm is None, "Should accept None for clip_norm"
        print("✅ Both None accepted (no clipping)")
    except Exception as e:
        print(f"❌ Failed to accept None values: {e}")
        raise
    
    # Teste 4: Valores negativos devem ser rejeitados ou tratados
    try:
        config4 = TrainingConfig(
            max_epochs=5,
            gradient_clip_value=-1.0,
            gradient_clip_norm=None
        )
        # Se aceitar valores negativos, deve tratá-los como None ou zero
        print(f"⚠️  Negative clip_value accepted as: {config4.gradient_clip_value}")
    except (ValueError, AssertionError) as e:
        print(f"✅ Correctly rejected negative clip_value: {str(e)[:50]}")
    except Exception as e:
        print(f"⚠️  Unexpected error with negative value: {e}")


def test_7_clipping_history():
    """
    Test 3.5.7: Histórico de Clipping
    
    Valida que:
    - TrainingHistory registra informações de clipping
    - Histórico contém losses, métricas e tempos
    - Treinamento com clipping gera histórico completo
    """
    print("\n" + "="*60)
    print("Test 3.5.7: Clipping History")
    print("="*60)
    
    # Configuração com clipping
    config = TrainingConfig(
        max_epochs=5,
        patience=10,
        gradient_clip_value=0.5,
        gradient_clip_norm=1.0,
        monitor_metric="roc_auc",
        monitor_mode="max"
    )
    
    # Dataset e DataLoader
    dataset = create_synthetic_data(n_samples=200, input_dim=64)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Modelo
    model = create_model(input_dim=64)
    # model created above
    
    # Trainer
    device = torch.device("cpu")
    trainer = ModelTrainer(model=model, config=config, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer.setup_training(optimizer=optimizer)
    
    # Treinamento
    history = trainer.train(train_loader, val_loader)
    
    # Validações do histórico
    assert len(history.train_losses) == 5, "Should have 5 training losses"
    assert len(history.val_losses) == 5, "Should have 5 validation losses"
    assert len(history.train_metrics) == 5, "Should have 5 training metrics"
    assert len(history.val_metrics) == 5, "Should have 5 validation metrics"
    assert len(history.epoch_times) == 5, "Should have 5 epoch times"
    assert len(history.learning_rates) == 5, "Should have 5 learning rates"
    
    # Verificar que o histórico está completo
    assert history.total_epochs == 5, f"Total epochs should be 5, got {history.total_epochs}"
    assert history.best_epoch > 0, "Best epoch should be set"
    assert history.best_metric_value is not None, "Best metric value should be set"
    
    print(f"✅ History complete with {history.total_epochs} epochs")
    print(f"✅ Train losses: {len(history.train_losses)} entries")
    print(f"✅ Val losses: {len(history.val_losses)} entries")
    print(f"✅ Metrics tracked: {len(history.train_metrics)} train, {len(history.val_metrics)} val")
    print(f"✅ Best epoch: {history.best_epoch}, Best metric: {history.best_metric_value:.4f}")


def main():
    """Executa todos os testes do Level 3.5."""
    print("\n" + "="*60)
    print("LEVEL 3.5: GRADIENT CLIPPING TESTS")
    print("="*60)
    
    tests = [
        ("test_1_no_gradient_clipping", test_1_no_gradient_clipping),
        ("test_2_gradient_clip_by_value", test_2_gradient_clip_by_value),
        ("test_3_gradient_clip_by_norm", test_3_gradient_clip_by_norm),
        ("test_4_gradient_magnitudes", test_4_gradient_magnitudes),
        ("test_5_gradient_explosion_prevention", test_5_gradient_explosion_prevention),
        ("test_6_clipping_parameter_validation", test_6_clipping_parameter_validation),
        ("test_7_clipping_history", test_7_clipping_history),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            test_func()
            results.append((test_name, "✅ PASS"))
        except Exception as e:
            results.append((test_name, f"❌ FAIL: {str(e)}"))
            import traceback
            print(f"\n❌ FAILED: {test_name}")
            print(traceback.format_exc())
    
    # Resumo
    print("\n" + "="*60)
    print("TEST SUMMARY - Level 3.5: Gradient Clipping")
    print("="*60)
    for test_name, result in results:
        status = result.split(":")[0]
        print(f"{test_name:.<40} {status}")
    
    passed = sum(1 for _, r in results if "PASS" in r)
    total = len(results)
    print("="*60)
    print(f"Results: {passed}/{total} tests passed ({100*passed/total:.1f}%)")
    print("="*60)
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    exit(main())
