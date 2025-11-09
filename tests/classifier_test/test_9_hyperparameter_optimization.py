#!/usr/bin/env python3
"""
Test 9: Hyperparameter Optimization Components
===============================================

Testa componentes de otimização de hiperparâmetros (Optuna integration).

NOTE: O módulo hyperopt.py tem incompatibilidade com MLPConfig atual.
Testamos apenas os componentes que funcionam independentemente.

Tests incluídos:
1. Optuna basic integration - integração básica com Optuna
2. Search space definition - definição de espaço de busca
3. Config classes - classes de configuração

Author: Test Suite
Date: 2024
"""

import sys
from pathlib import Path

# Adicionar src ao path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

import optuna

# Imports do classifier
from classifier.core.hyperopt import (
    OptimizationConfig,
    HyperparameterSpace
)


def test_1_optimization_config():
    """
    Test 9.1: OptimizationConfig Class
    
    Testa configuração de otimização.
    """
    print("\n" + "="*60)
    print("Test 9.1: OptimizationConfig Class")
    print("="*60)
    
    print("\n--- Step 1: Creating Valid Config ---")
    
    config = OptimizationConfig(
        study_name="test_study",
        direction="maximize",
        n_trials=100,
        optimization_metric="roc_auc",
        sampler_type="TPE",
        pruner_type="Median",
        enable_pruning=True,
        pruning_warmup_steps=5
    )
    
    print(f"✅ Config created successfully")
    print(f"   Study name: {config.study_name}")
    print(f"   Direction: {config.direction}")
    print(f"   N trials: {config.n_trials}")
    print(f"   Metric: {config.optimization_metric}")
    print(f"   Sampler: {config.sampler_type}")
    print(f"   Pruner: {config.pruner_type}")
    print(f"   Pruning enabled: {config.enable_pruning}")
    
    # Validar atributos
    assert config.study_name == "test_study"
    assert config.direction == "maximize"
    assert config.n_trials == 100
    assert config.optimization_metric == "roc_auc"
    assert config.sampler_type == "TPE"
    assert config.pruner_type == "Median"
    assert config.enable_pruning == True
    
    print(f"\n--- Step 2: Testing Invalid Direction ---")
    
    try:
        invalid_config = OptimizationConfig(direction="invalid")
        assert False, "Should raise ValueError for invalid direction"
    except ValueError as e:
        print(f"✅ Correctly rejected invalid direction: {e}")
    
    print(f"\n--- Step 3: Testing Invalid Sampler ---")
    
    try:
        invalid_config = OptimizationConfig(sampler_type="InvalidSampler")
        assert False, "Should raise ValueError for invalid sampler"
    except ValueError as e:
        print(f"✅ Correctly rejected invalid sampler: {e}")
    
    print(f"\n--- Step 4: Testing Invalid Pruner ---")
    
    try:
        invalid_config = OptimizationConfig(pruner_type="InvalidPruner")
        assert False, "Should raise ValueError for invalid pruner"
    except ValueError as e:
        print(f"✅ Correctly rejected invalid pruner: {e}")
    
    print(f"\n--- Step 5: Testing Invalid N_Trials ---")
    
    try:
        invalid_config = OptimizationConfig(n_trials=-10)
        assert False, "Should raise ValueError for negative n_trials"
    except ValueError as e:
        print(f"✅ Correctly rejected negative n_trials: {e}")
    
    print(f"\n✅ OptimizationConfig validated successfully!")


def test_2_hyperparameter_space():
    """
    Test 9.2: HyperparameterSpace Class
    
    Testa definição de espaço de busca.
    """
    print("\n" + "="*60)
    print("Test 9.2: HyperparameterSpace Class")
    print("="*60)
    
    print("\n--- Step 1: Creating Default Space ---")
    
    hp_space = HyperparameterSpace()
    
    print(f"✅ Default space created")
    
    # Validar estrutura
    assert hasattr(hp_space, 'hidden_layers')
    assert hasattr(hp_space, 'dropout_rate')
    assert hasattr(hp_space, 'activation')
    assert hasattr(hp_space, 'use_batch_norm')
    assert hasattr(hp_space, 'learning_rate')
    assert hasattr(hp_space, 'weight_decay')
    assert hasattr(hp_space, 'batch_size')
    assert hasattr(hp_space, 'max_epochs')
    assert hasattr(hp_space, 'patience')
    
    print(f"✅ All required attributes present")
    
    print(f"\n--- Step 2: Validating Parameter Types ---")
    
    # Categorical
    assert hp_space.hidden_layers["type"] == "categorical"
    assert "choices" in hp_space.hidden_layers
    assert len(hp_space.hidden_layers["choices"]) > 0
    print(f"✅ hidden_layers: categorical with {len(hp_space.hidden_layers['choices'])} choices")
    
    # Float
    assert hp_space.dropout_rate["type"] == "float"
    assert "low" in hp_space.dropout_rate
    assert "high" in hp_space.dropout_rate
    assert hp_space.dropout_rate["low"] < hp_space.dropout_rate["high"]
    print(f"✅ dropout_rate: float [{hp_space.dropout_rate['low']}, {hp_space.dropout_rate['high']}]")
    
    # Float with log
    assert hp_space.learning_rate["type"] == "float"
    assert hp_space.learning_rate.get("log", False) == True
    print(f"✅ learning_rate: float (log) [{hp_space.learning_rate['low']}, {hp_space.learning_rate['high']}]")
    
    # Int
    assert hp_space.max_epochs["type"] == "int"
    assert "low" in hp_space.max_epochs
    assert "high" in hp_space.max_epochs
    print(f"✅ max_epochs: int [{hp_space.max_epochs['low']}, {hp_space.max_epochs['high']}]")
    
    print(f"\n--- Step 3: Custom Space Definition ---")
    
    custom_space = HyperparameterSpace()
    custom_space.learning_rate = {
        "type": "float",
        "low": 1e-5,
        "high": 1e-2,
        "log": True
    }
    custom_space.batch_size = {
        "type": "categorical",
        "choices": [32, 64, 128]
    }
    
    print(f"✅ Custom space created")
    print(f"   LR range: [{custom_space.learning_rate['low']}, {custom_space.learning_rate['high']}]")
    print(f"   Batch sizes: {custom_space.batch_size['choices']}")
    
    assert custom_space.learning_rate["low"] == 1e-5
    assert len(custom_space.batch_size["choices"]) == 3
    
    print(f"\n✅ HyperparameterSpace validated successfully!")


def test_3_optuna_integration():
    """
    Test 9.3: Optuna Integration
    
    Testa integração básica com Optuna (sem hyperopt.py).
    """
    print("\n" + "="*60)
    print("Test 9.3: Optuna Integration")
    print("="*60)
    
    print("\n--- Step 1: Creating Optuna Study ---")
    
    study = optuna.create_study(
        study_name="test_optuna",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    print(f"✅ Study created: {study.study_name}")
    print(f"   Direction: {study.direction}")
    
    # Função objetivo simples
    print("\n--- Step 2: Defining Objective Function ---")
    
    def simple_objective(trial):
        # Sugerir hiperparâmetros
        x = trial.suggest_float("x", -10, 10)
        y = trial.suggest_int("y", 0, 10)
        z = trial.suggest_categorical("z", ["a", "b", "c"])
        
        # Função objetivo simples (parábola)
        score = -(x - 2)**2 - (y - 5)**2
        
        return score
    
    print(f"✅ Objective function defined")
    
    # Executar otimização
    print("\n--- Step 3: Running Optimization ---")
    
    study.optimize(simple_objective, n_trials=10)
    
    print(f"✅ Optimization completed: {len(study.trials)} trials")
    
    # Validar resultados
    print("\n--- Step 4: Validating Results ---")
    
    assert len(study.trials) == 10
    assert study.best_params is not None
    assert "x" in study.best_params
    assert "y" in study.best_params
    assert "z" in study.best_params
    
    best_x = study.best_params["x"]
    best_y = study.best_params["y"]
    
    print(f"✅ Best parameters:")
    print(f"   x: {best_x:.4f} (optimal: 2.0)")
    print(f"   y: {best_y} (optimal: 5)")
    print(f"   z: {study.best_params['z']}")
    print(f"   Best value: {study.best_value:.4f}")
    
    # Validar que encontrou valores próximos do ótimo
    assert abs(best_x - 2.0) < 3.0, "Should find x near 2.0"
    assert abs(best_y - 5) <= 2, "Should find y near 5"
    
    # Verificar que trials melhoraram
    values = [trial.value for trial in study.trials if trial.value is not None]
    assert len(values) > 0
    assert max(values) == study.best_value
    
    print(f"\n--- Step 5: Testing Pruning ---")
    
    study_pruning = optuna.create_study(
        study_name="test_pruning",
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2)
    )
    
    def prunable_objective(trial):
        # Simular treinamento progressivo
        for step in range(10):
            # Score que piora com o tempo (para testar pruning)
            score = 10 - step + trial.suggest_float("noise", -1, 1)
            
            # Reportar valor intermediário
            trial.report(score, step)
            
            # Verificar se deve parar
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return score
    
    study_pruning.optimize(prunable_objective, n_trials=5, timeout=5)
    
    n_pruned = sum(1 for trial in study_pruning.trials if trial.state == optuna.trial.TrialState.PRUNED)
    n_complete = sum(1 for trial in study_pruning.trials if trial.state == optuna.trial.TrialState.COMPLETE)
    
    print(f"✅ Pruning test completed:")
    print(f"   Total trials: {len(study_pruning.trials)}")
    print(f"   Complete: {n_complete}")
    print(f"   Pruned: {n_pruned}")
    
    assert len(study_pruning.trials) > 0
    
    print(f"\n✅ Optuna integration validated successfully!")


def run_all_tests():
    """Executar todos os testes."""
    tests = [
        ("9.1 - OptimizationConfig", test_1_optimization_config),
        ("9.2 - HyperparameterSpace", test_2_hyperparameter_space),
        ("9.3 - Optuna Integration", test_3_optuna_integration),
    ]
    
    print("\n" + "="*60)
    print("LEVEL 9: HYPERPARAMETER OPTIMIZATION COMPONENTS")
    print("="*60)
    print("\nNOTE: Testing components only - hyperopt.py has MLPConfig")
    print("      incompatibility that requires refactoring.")
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
    print("FINAL SUMMARY - LEVEL 9")
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
        print("\nNOTE: Full hyperopt.py integration requires refactoring")
        print("      to match current MLPConfig structure.")
    
    print("="*60)
    
    return passed == len(tests)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
