#!/usr/bin/env python3
"""
Test 7: Serialization
======================

Testa save/load de modelos e configurações.

Tests incluídos:
1. Model checkpoint save/load - checkpoints completos
2. Configuration persistence - salvamento de configs

Author: Test Suite
Date: 2024
"""

import sys
from pathlib import Path
import tempfile
import shutil
import json

# Adicionar src ao path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Imports do classifier
from classifier.models.mlp_classifier import MLPEmbeddingClassifier
from classifier.core.trainer import ModelTrainer, TrainingConfig


def test_1_model_checkpoint_save_load():
    """
    Test 7.1: Model Checkpoint Save/Load
    
    Valida salvamento e carregamento completo de checkpoints.
    """
    print("\n" + "="*60)
    print("Test 7.1: Model Checkpoint Save/Load")
    print("="*60)
    
    device = torch.device("cpu")
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Criar e treinar modelo
        print("\n--- Training original model ---")
        X = torch.randn(200, 64)
        y = torch.randint(0, 2, (200,)).float()
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=32)
        
        model1 = MLPEmbeddingClassifier(input_dim=64, hidden_dim=32, dropout=0.2)
        model1.to(device)
        
        config = TrainingConfig(max_epochs=5, patience=10)
        optimizer = torch.optim.Adam(model1.parameters(), lr=0.001)
        
        trainer1 = ModelTrainer(model=model1, config=config, device=device)
        trainer1.setup_training(optimizer=optimizer)
        history1 = trainer1.train(loader, loader)
        
        print(f"✅ Original model trained: {history1.total_epochs} epochs")
        print(f"   Final loss: {history1.train_losses[-1]:.4f}")
        
        # Fazer predição
        model1.eval()
        test_X = X[:10].to(device)
        with torch.no_grad():
            pred1 = model1(test_X)
        
        print(f"✅ Original predictions generated")
        
        # Salvar checkpoint completo
        checkpoint_path = Path(temp_dir) / "full_checkpoint.pt"
        
        checkpoint = {
            'epoch': history1.total_epochs,
            'model_state_dict': model1.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': history1.train_losses[-1],
            'history': {
                'train_losses': history1.train_losses,
                'val_losses': history1.val_losses,
                'total_epochs': history1.total_epochs,
                'best_epoch': history1.best_epoch
            },
            'model_config': {
                'input_dim': 64,
                'hidden_dim': 32,
                'dropout': 0.2
            },
            'training_config': {
                'max_epochs': config.max_epochs,
                'patience': config.patience
            }
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"✅ Checkpoint saved: {checkpoint_path}")
        
        # Carregar checkpoint
        print("\n--- Loading checkpoint ---")
        loaded_checkpoint = torch.load(checkpoint_path, weights_only=False)  # Necessário para dicts complexos
        
        # Recriar modelo
        model2 = MLPEmbeddingClassifier(
            input_dim=loaded_checkpoint['model_config']['input_dim'],
            hidden_dim=loaded_checkpoint['model_config']['hidden_dim'],
            dropout=loaded_checkpoint['model_config']['dropout']
        )
        model2.load_state_dict(loaded_checkpoint['model_state_dict'])
        model2.to(device)
        model2.eval()
        
        print(f"✅ Model loaded from checkpoint")
        print(f"   Saved epoch: {loaded_checkpoint['epoch']}")
        print(f"   Saved loss: {loaded_checkpoint['train_loss']:.4f}")
        
        # Verificar predições idênticas
        with torch.no_grad():
            pred2 = model2(test_X)
        
        diff = torch.abs(pred1 - pred2).max().item()
        print(f"✅ Predictions match: max diff = {diff:.2e}")
        assert diff < 1e-6, f"Predictions should be identical, got diff={diff}"
        
        # Recarregar optimizer
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
        optimizer2.load_state_dict(loaded_checkpoint['optimizer_state_dict'])
        print(f"✅ Optimizer state restored")
        
        # Continuar treinamento
        print("\n--- Continuing training from checkpoint ---")
        config2 = TrainingConfig(max_epochs=2, patience=10)
        trainer2 = ModelTrainer(model=model2, config=config2, device=device)
        trainer2.setup_training(optimizer=optimizer2)
        history2 = trainer2.train(loader, loader)
        
        print(f"✅ Training continued: {history2.total_epochs} additional epochs")
        print(f"   Final loss: {history2.train_losses[-1]:.4f}")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
        print(f"\n✅ Temporary directory cleaned")
    
    print(f"\n✅ Model checkpoint save/load validated")


def test_2_configuration_persistence():
    """
    Test 7.2: Configuration Persistence
    
    Valida salvamento e carregamento de configurações.
    """
    print("\n" + "="*60)
    print("Test 7.2: Configuration Persistence")
    print("="*60)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Criar configuração complexa
        print("\n--- Creating configuration ---")
        
        config = TrainingConfig(
            max_epochs=100,
            patience=20,
            monitor_metric="accuracy",
            monitor_mode="max",
            use_scheduler=True,
            amp_enabled=False,
            gradient_clip_value=1.0
        )
        
        print(f"✅ Configuration created:")
        print(f"   max_epochs: {config.max_epochs}")
        print(f"   patience: {config.patience}")
        print(f"   monitor_metric: {config.monitor_metric}")
        print(f"   gradient_clip_value: {config.gradient_clip_value}")
        
        # Salvar como JSON
        config_path = Path(temp_dir) / "training_config.json"
        
        config_dict = {
            'max_epochs': config.max_epochs,
            'patience': config.patience,
            'monitor_metric': config.monitor_metric,
            'monitor_mode': config.monitor_mode,
            'use_scheduler': config.use_scheduler,
            'amp_enabled': config.amp_enabled,
            'gradient_clip_value': config.gradient_clip_value
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"✅ Configuration saved: {config_path}")
        
        # Carregar configuração
        print("\n--- Loading configuration ---")
        
        with open(config_path, 'r') as f:
            loaded_dict = json.load(f)
        
        # Recriar configuração
        config2 = TrainingConfig(**loaded_dict)
        
        print(f"✅ Configuration loaded:")
        print(f"   max_epochs: {config2.max_epochs}")
        print(f"   patience: {config2.patience}")
        print(f"   monitor_metric: {config2.monitor_metric}")
        
        # Verificar igualdade
        assert config.max_epochs == config2.max_epochs, "max_epochs should match"
        assert config.patience == config2.patience, "patience should match"
        assert config.monitor_metric == config2.monitor_metric, "monitor_metric should match"
        assert config.gradient_clip_value == config2.gradient_clip_value, "gradient_clip_value should match"
        
        print(f"✅ All configuration parameters match")
        
        # Salvar config completo (incluindo model)
        print("\n--- Saving complete experiment config ---")
        
        experiment_config = {
            'model': {
                'type': 'MLPEmbeddingClassifier',
                'input_dim': 64,
                'hidden_dim': 128,
                'dropout': 0.3
            },
            'training': config_dict,
            'optimizer': {
                'type': 'Adam',
                'lr': 0.001,
                'betas': [0.9, 0.999]
            },
            'data': {
                'batch_size': 32,
                'shuffle': True
            }
        }
        
        exp_path = Path(temp_dir) / "experiment_config.json"
        with open(exp_path, 'w') as f:
            json.dump(experiment_config, f, indent=2)
        
        print(f"✅ Complete experiment config saved")
        
        # Carregar e validar
        with open(exp_path, 'r') as f:
            loaded_exp = json.load(f)
        
        assert loaded_exp['model']['input_dim'] == 64, "Model config should match"
        assert loaded_exp['training']['max_epochs'] == 100, "Training config should match"
        assert loaded_exp['optimizer']['lr'] == 0.001, "Optimizer config should match"
        
        print(f"✅ Complete experiment config validated")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
        print(f"\n✅ Temporary directory cleaned")
    
    print(f"\n✅ Configuration persistence validated")


def run_all_tests():
    """Executar todos os testes."""
    tests = [
        ("7.1 - Model Checkpoint Save/Load", test_1_model_checkpoint_save_load),
        ("7.2 - Configuration Persistence", test_2_configuration_persistence),
    ]
    
    print("\n" + "="*60)
    print("LEVEL 7: SERIALIZATION")
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
    print("FINAL SUMMARY - LEVEL 7")
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
