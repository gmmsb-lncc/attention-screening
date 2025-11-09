#!/usr/bin/env python3
"""
Test 8: End-to-End Workflows
==============================

Testa workflows completos e realistas.

Tests incluídos:
1. Complete ML workflow - workflow completo de ML
2. Production-like pipeline - pipeline estilo produção

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
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# Imports do classifier
from classifier.models.mlp_classifier import MLPEmbeddingClassifier
from classifier.core.trainer import ModelTrainer, TrainingConfig
from classifier.core.cross_validator import CrossValidator, CrossValidationConfig
from classifier.utils.metrics import MetricsCalculator


def test_1_complete_ml_workflow():
    """
    Test 8.1: Complete ML Workflow
    
    Simula workflow completo de ML do início ao fim.
    """
    print("\n" + "="*60)
    print("Test 8.1: Complete ML Workflow")
    print("="*60)
    
    device = torch.device("cpu")
    temp_dir = tempfile.mkdtemp()
    
    try:
        # STEP 1: Data Collection & Preparation
        print("\n--- Step 1: Data Collection & Preparation ---")
        
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Simular dados "reais" com padrão
        n_samples = 500
        X_raw = torch.randn(n_samples, 64)
        # Adicionar padrão: classe 1 tem valores maiores em primeiras features
        y = torch.zeros(n_samples)
        y[X_raw[:, 0] + X_raw[:, 1] > 0] = 1
        
        print(f"✅ Data collected: {n_samples} samples, {X_raw.shape[1]} features")
        print(f"   Class distribution: {(y==0).sum():.0f} / {(y==1).sum():.0f}")
        
        # STEP 2: Data Preprocessing
        print("\n--- Step 2: Data Preprocessing ---")
        
        # Normalização
        X_mean = X_raw.mean(dim=0, keepdim=True)
        X_std = X_raw.std(dim=0, keepdim=True) + 1e-8
        X_normalized = (X_raw - X_mean) / X_std
        
        print(f"✅ Data normalized")
        print(f"   Mean: {X_normalized.mean():.4f}, Std: {X_normalized.std():.4f}")
        
        # STEP 3: Train/Val/Test Split
        print("\n--- Step 3: Train/Val/Test Split ---")
        
        indices = torch.randperm(n_samples)
        n_train = int(0.6 * n_samples)
        n_val = int(0.2 * n_samples)
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train+n_val]
        test_idx = indices[n_train+n_val:]
        
        X_train, y_train = X_normalized[train_idx], y[train_idx]
        X_val, y_val = X_normalized[val_idx], y[val_idx]
        X_test, y_test = X_normalized[test_idx], y[test_idx]
        
        print(f"✅ Data split: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")
        
        # STEP 4: Model Selection via Cross-Validation
        print("\n--- Step 4: Model Selection (Cross-Validation) ---")
        
        cv_config = CrossValidationConfig(n_splits=3, shuffle=True, random_state=42, batch_size=32)
        training_config = TrainingConfig(max_epochs=5, patience=10)
        
        cv = CrossValidator(cv_config=cv_config, training_config=training_config, device=device)
        
        def model_factory():
            return MLPEmbeddingClassifier(input_dim=64, hidden_dim=32, dropout=0.2)
        
        def optimizer_factory(model):
            return torch.optim.Adam(model.parameters(), lr=0.001)
        
        cv_results = cv.cross_validate(
            model_factory=model_factory,
            optimizer_factory=optimizer_factory,
            X=X_train,
            y=y_train
        )
        
        cv_acc = cv_results['summary_statistics']['accuracy']['mean']
        cv_std = cv_results['summary_statistics']['accuracy']['std']
        
        print(f"✅ Cross-validation completed: {cv_acc:.4f} ± {cv_std:.4f}")
        
        # STEP 5: Train Final Model
        print("\n--- Step 5: Train Final Model ---")
        
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        final_model = MLPEmbeddingClassifier(input_dim=64, hidden_dim=32, dropout=0.2)
        final_model.to(device)
        
        config = TrainingConfig(max_epochs=20, patience=5, monitor_mode="max")
        optimizer = torch.optim.Adam(final_model.parameters(), lr=0.001)
        criterion = torch.nn.BCEWithLogitsLoss()
        
        trainer = ModelTrainer(model=final_model, config=config, device=device)
        trainer.setup_training(optimizer=optimizer, criterion=criterion)
        history = trainer.train(train_loader, val_loader)
        
        print(f"✅ Final model trained: {history.total_epochs} epochs")
        print(f"   Best epoch: {history.best_epoch + 1}")
        print(f"   Early stopped: {history.early_stopped}")
        
        # STEP 6: Evaluation on Test Set
        print("\n--- Step 6: Evaluation on Test Set ---")
        
        test_dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        metrics_calc = MetricsCalculator(device=device)
        test_metrics = metrics_calc.evaluate_model(final_model, test_loader, criterion)
        
        print(f"✅ Test metrics:")
        print(f"   Accuracy: {test_metrics.accuracy:.4f}")
        print(f"   Precision: {test_metrics.precision:.4f}")
        print(f"   Recall: {test_metrics.recall:.4f}")
        print(f"   F1: {test_metrics.f1:.4f}")
        print(f"   ROC-AUC: {test_metrics.roc_auc:.4f}")
        
        # STEP 7: Save Model for Production
        print("\n--- Step 7: Save Model for Production ---")
        
        model_path = Path(temp_dir) / "production_model.pt"
        
        production_checkpoint = {
            'model_state_dict': final_model.state_dict(),
            'model_config': {'input_dim': 64, 'hidden_dim': 32, 'dropout': 0.2},
            'preprocessing': {'mean': X_mean.tolist(), 'std': X_std.tolist()},
            'test_metrics': test_metrics.to_dict(),
            'training_info': {
                'total_epochs': history.total_epochs,
                'best_epoch': history.best_epoch,
                'cv_score': cv_acc
            }
        }
        
        torch.save(production_checkpoint, model_path)
        print(f"✅ Model saved for production: {model_path}")
        
        # STEP 8: Simulate Production Inference
        print("\n--- Step 8: Production Inference Simulation ---")
        
        # Carregar modelo
        loaded = torch.load(model_path, weights_only=False)
        
        prod_model = MLPEmbeddingClassifier(**loaded['model_config'])
        prod_model.load_state_dict(loaded['model_state_dict'])
        prod_model.to(device)
        prod_model.eval()
        
        # Simular novos dados de produção
        new_data = torch.randn(5, 64)
        
        # Aplicar mesmo preprocessing
        X_mean_loaded = torch.tensor(loaded['preprocessing']['mean'])
        X_std_loaded = torch.tensor(loaded['preprocessing']['std'])
        new_data_normalized = (new_data - X_mean_loaded) / X_std_loaded
        
        # Inferência
        with torch.no_grad():
            logits = prod_model(new_data_normalized.to(device))
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
        
        print(f"✅ Production inference on {len(new_data)} samples:")
        for i in range(len(new_data)):
            print(f"   Sample {i+1}: prob={probs[i].item():.4f}, pred={int(preds[i].item())}")
        
        # Validações finais
        assert test_metrics.roc_auc >= 0.5, "ROC-AUC should be better than random"
        assert 0 <= test_metrics.roc_auc <= 1, "ROC-AUC in valid range"
        
        print(f"\n✅ Complete ML workflow validated successfully!")
        
    finally:
        shutil.rmtree(temp_dir)
        print(f"✅ Cleanup completed")


def test_2_production_like_pipeline():
    """
    Test 8.2: Production-Like Pipeline
    
    Simula pipeline de produção com error handling robusto.
    """
    print("\n" + "="*60)
    print("Test 8.2: Production-Like Pipeline")
    print("="*60)
    
    device = torch.device("cpu")
    
    class ProductionPipeline:
        """Pipeline de produção simplificado."""
        
        def __init__(self, model_path=None):
            self.model = None
            self.preprocessing_params = None
            self.device = device
            
            if model_path:
                self.load_model(model_path)
        
        def load_model(self, model_path):
            """Carrega modelo salvo."""
            try:
                checkpoint = torch.load(model_path, weights_only=False)
                
                self.model = MLPEmbeddingClassifier(**checkpoint['model_config'])
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.device)
                self.model.eval()
                
                self.preprocessing_params = checkpoint.get('preprocessing', {})
                
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
                return False
        
        def preprocess(self, X):
            """Aplica preprocessing."""
            if self.preprocessing_params:
                mean = torch.tensor(self.preprocessing_params['mean'])
                std = torch.tensor(self.preprocessing_params['std'])
                return (X - mean) / std
            return X
        
        def predict(self, X, return_probs=False):
            """Faz predições."""
            if self.model is None:
                raise RuntimeError("Model not loaded")
            
            # Validação de entrada
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X, dtype=torch.float32)
            
            if X.dim() == 1:
                X = X.unsqueeze(0)
            
            # Preprocessing
            X_processed = self.preprocess(X)
            
            # Inferência
            with torch.no_grad():
                logits = self.model(X_processed.to(self.device))
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
            
            if return_probs:
                return preds.cpu(), probs.cpu()
            return preds.cpu()
    
    # Test do pipeline
    print("\n--- Testing Production Pipeline ---")
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Treinar e salvar modelo
        X = torch.randn(200, 64)
        y = torch.randint(0, 2, (200,)).float()
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=32)
        
        model = MLPEmbeddingClassifier(input_dim=64, hidden_dim=32, dropout=0.1)
        model.to(device)
        
        config = TrainingConfig(max_epochs=3, patience=10)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        trainer = ModelTrainer(model=model, config=config, device=device)
        trainer.setup_training(optimizer=optimizer)
        trainer.train(loader, loader)
        
        # Salvar
        model_path = Path(temp_dir) / "pipeline_model.pt"
        X_mean = X.mean(dim=0, keepdim=True)
        X_std = X.std(dim=0, keepdim=True) + 1e-8
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_config': {'input_dim': 64, 'hidden_dim': 32, 'dropout': 0.1},
            'preprocessing': {'mean': X_mean.tolist(), 'std': X_std.tolist()}
        }
        torch.save(checkpoint, model_path)
        
        print(f"✅ Model trained and saved")
        
        # Testar pipeline
        pipeline = ProductionPipeline(model_path)
        
        # Test 1: Single sample
        single_sample = torch.randn(64)
        pred = pipeline.predict(single_sample)
        print(f"✅ Single sample prediction: {int(pred.item())}")
        
        # Test 2: Batch
        batch_samples = torch.randn(10, 64)
        preds, probs = pipeline.predict(batch_samples, return_probs=True)
        print(f"✅ Batch prediction: {preds.shape}, probs range [{probs.min():.4f}, {probs.max():.4f}]")
        
        # Test 3: Error handling - modelo não carregado
        empty_pipeline = ProductionPipeline()
        try:
            empty_pipeline.predict(single_sample)
            assert False, "Should raise error"
        except RuntimeError as e:
            print(f"✅ Error handling: {e}")
        
        # Test 4: Numpy input
        numpy_input = np.random.randn(5, 64)
        preds_numpy = pipeline.predict(numpy_input)
        print(f"✅ Numpy input handled: {preds_numpy.shape}")
        
        print(f"\n✅ Production pipeline validated successfully!")
        
    finally:
        shutil.rmtree(temp_dir)
        print(f"✅ Cleanup completed")


def run_all_tests():
    """Executar todos os testes."""
    tests = [
        ("8.1 - Complete ML Workflow", test_1_complete_ml_workflow),
        ("8.2 - Production-Like Pipeline", test_2_production_like_pipeline),
    ]
    
    print("\n" + "="*60)
    print("LEVEL 8: END-TO-END WORKFLOWS")
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
    print("FINAL SUMMARY - LEVEL 8")
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
        print("\n" + "="*60)
        print("🎊 COMPLETE TEST SUITE FINISHED! 🎊")
        print("="*60)
    
    print("="*60)
    
    return passed == len(tests)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
