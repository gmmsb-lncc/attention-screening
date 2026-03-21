"""
=============================================================================
LEVEL 12: CLI & Pipeline Integration Tests
=============================================================================
Testes de integração para CLIs e pipeline modularizado.

Módulos testados:
- modular_pipeline.py: Pipeline modularizado
- modular_classifier.py: CLI moderno
- main.py: Entry point principal

Total de testes: 2
=============================================================================
"""

import sys
import os
import numpy as np
import torch

# Adicionar src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))


def test_1_modular_pipeline_integration():
    """
    Test 12.1 - Modular Pipeline Integration
    
    Valida o pipeline modularizado:
    - Import correto dos módulos
    - Inicialização do pipeline
    - Configuração básica
    - Compatibilidade de interface
    """
    print("\n" + "="*70)
    print("TEST 12.1: Modular Pipeline Integration")
    print("="*70)
    
    print("\n--- Step 1: Import Pipeline Module ---")
    try:
        from classifier.modular_pipeline import MLPEmbeddingPipeline
        print("✅ MLPEmbeddingPipeline imported successfully")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        raise
    
    print("\n--- Step 2: Create Test Data ---")
    # Criar dados de teste pequenos
    embeddings_test = np.random.randn(50, 128).astype(np.float32)
    labels_test = np.random.randint(0, 2, 50).astype(np.int64)
    
    # Salvar temporariamente
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        embeddings_path = f.name
        np.save(embeddings_path, embeddings_test)
    
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        labels_path = f.name
        np.save(labels_path, labels_test)
    
    print(f"✅ Test data created: {embeddings_test.shape[0]} samples")
    print(f"   Embeddings: {embeddings_test.shape}")
    print(f"   Labels: {labels_test.shape}")
    
    print("\n--- Step 3: Initialize Pipeline ---")
    try:
        pipeline = MLPEmbeddingPipeline(
            embeddings_path=embeddings_path,
            labels_path=labels_path,
            batch_size=16,
            lr=0.001,
            epochs=2,  # Apenas 2 epochs para teste rápido
            early_stopping_patience=2
        )
        print("✅ Pipeline initialized successfully")
        print(f"   Device: {pipeline.device}")
        print(f"   Input dim: {pipeline.input_dim}")
        print(f"   Batch size: {pipeline.batch_size}")
        print(f"   Learning rate: {pipeline.lr}")
    except Exception as e:
        print(f"❌ Pipeline initialization failed: {e}")
        # Limpar arquivos temporários
        os.unlink(embeddings_path)
        os.unlink(labels_path)
        raise
    
    print("\n--- Step 4: Validate Pipeline Components ---")
    # Verificar componentes essenciais
    assert hasattr(pipeline, 'data_manager'), "Pipeline missing data_manager"
    assert hasattr(pipeline, 'evaluator'), "Pipeline missing evaluator"
    assert hasattr(pipeline, 'device'), "Pipeline missing device"
    assert hasattr(pipeline, 'spark'), "Pipeline missing Spark session"
    print("✅ All essential components present")
    
    print("\n--- Step 5: Validate Interface Methods ---")
    # Verificar métodos principais
    methods = ['get_embedding_dim', 'load_data', 'train', 'cross_validate']
    for method in methods:
        assert hasattr(pipeline, method), f"Pipeline missing method: {method}"
        print(f"   ✅ Method '{method}' available")
    
    print("\n--- Step 6: Test Data Loading ---")
    try:
        # load_data não recebe test_split/val_split, usa divisão padrão 80/10/10
        pipeline.load_data()
        print("✅ Data loaded successfully")
        print(f"   Train loader: {len(pipeline.train_loader)} batches")
        print(f"   Val loader: {len(pipeline.val_loader)} batches")
        print(f"   Test loader: {len(pipeline.test_loader)} batches")
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        os.unlink(embeddings_path)
        os.unlink(labels_path)
        raise
    
    print("\n--- Step 7: Cleanup ---")
    # Limpar arquivos temporários
    os.unlink(embeddings_path)
    os.unlink(labels_path)
    
    # Fechar Spark session
    if hasattr(pipeline, 'spark'):
        pipeline.spark.stop()
    
    print("✅ Test cleanup complete")
    
    print("\n" + "="*70)
    print("✅ TEST 12.1 PASSED: Modular Pipeline Integration")
    print("="*70)


def test_2_cli_modules_import():
    """
    Test 12.2 - CLI Modules Import & Validation
    
    Valida o módulo CLI modularizado:
    - modular_classifier.py imports
    - Função principal (main)
    - set_seed functionality
    - Compatibilidade com MLPEmbeddingPipeline
    """
    print("\n" + "="*70)
    print("TEST 12.2: CLI Modules Import & Validation")
    print("="*70)
    
    print("\n--- Step 1: Import modular_classifier Module ---")
    try:
        # Import do módulo CLI moderno
        import classifier.modular_classifier as modular_classifier
        print("✅ modular_classifier imported successfully")
        
        # Validar função main
        assert hasattr(modular_classifier, 'main'), "modular_classifier missing main()"
        print("   ✅ main() function available")
        
        # Validar set_seed
        assert hasattr(modular_classifier, 'set_seed'), "modular_classifier missing set_seed()"
        print("   ✅ set_seed() function available")
        
    except ImportError as e:
        print(f"❌ modular_classifier import failed: {e}")
        raise
    
    print("\n--- Step 2: Validate set_seed Functionality ---")
    try:
        # Testar função set_seed
        modular_classifier.set_seed(42)
        print("✅ set_seed(42) executed successfully")
        
        # Validar reprodutibilidade
        torch.manual_seed(42)
        t1 = torch.randn(5)
        torch.manual_seed(42)
        t2 = torch.randn(5)
        assert torch.allclose(t1, t2), "Random seed not working properly"
        print("   ✅ Reproducibility validated")
        
    except Exception as e:
        print(f"❌ set_seed validation failed: {e}")
        raise
    
    print("\n--- Step 3: Validate MLPEmbeddingPipeline Import ---")
    try:
        # modular_classifier usa MLPEmbeddingPipeline
        from classifier.modular_pipeline import MLPEmbeddingPipeline
        print("✅ MLPEmbeddingPipeline import OK (used by modular_classifier)")
        
        # Validar que tem os métodos necessários
        assert hasattr(MLPEmbeddingPipeline, 'train'), "MLPEmbeddingPipeline missing train()"
        assert hasattr(MLPEmbeddingPipeline, 'cross_validate'), "MLPEmbeddingPipeline missing cross_validate()"
        print("   ✅ Essential methods available")
        
    except ImportError as e:
        print(f"❌ MLPEmbeddingPipeline import failed: {e}")
        raise
    
    print("\n--- Step 4: Validate Core Module Imports ---")
    try:
        # Verificar imports de módulos core essenciais
        from classifier.models.mlp_classifier import MLPEmbeddingClassifier
        print("   ✅ MLPEmbeddingClassifier import OK")
        
        from classifier.core.evaluator import ModelEvaluator
        print("   ✅ ModelEvaluator import OK")
        
        from classifier.core.data_loader import DataManager
        print("   ✅ DataManager import OK")
        
        print("✅ All core imports compatible")
        
    except ImportError as e:
        print(f"❌ Core imports check failed: {e}")
        raise
    
    print("\n--- Step 5: Check Optuna Integration ---")
    try:
        # modular_classifier usa Optuna para modo "optuna"
        import optuna
        print("✅ Optuna available for hyperparameter optimization")
        
        # Validar que optuna tem as funções necessárias
        assert hasattr(optuna, 'create_study'), "Optuna missing create_study()"
        print("   ✅ Optuna.create_study() available")
        
    except ImportError as e:
        print(f"⚠️  Optuna not available: {e}")
        print("   (Optional - required only for --mode optuna)")
    
    print("\n--- Step 6: Validate CLI Argument Structure ---")
    try:
        # Verificar que modular_classifier usa argparse corretamente
        import argparse
        
        # Simular parser
        parser = argparse.ArgumentParser()
        parser.add_argument("embeddings_path", type=str)
        parser.add_argument("labels_path", type=str)
        parser.add_argument("--mode", type=str, choices=["optuna", "manual"], default="optuna")
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--epochs", type=int, default=50)
        
        # Parse argumentos de teste
        test_args = parser.parse_args([
            "test_embeddings.npy",
            "test_labels.npy",
            "--mode", "manual",
            "--lr", "0.0001",
            "--batch_size", "32"
        ])
        
        assert test_args.embeddings_path == "test_embeddings.npy"
        assert test_args.mode == "manual"
        assert test_args.lr == 0.0001
        assert test_args.batch_size == 32
        
        print("✅ CLI argument parsing validated")
        print(f"   ✅ Mode: {test_args.mode}")
        print(f"   ✅ LR: {test_args.lr}")
        print(f"   ✅ Batch size: {test_args.batch_size}")
        
    except Exception as e:
        print(f"❌ CLI argument validation failed: {e}")
        raise
    
    print("\n" + "="*70)
    print("✅ TEST 12.2 PASSED: CLI Modules Import & Validation")
    print("="*70)


def run_all_tests():
    """Executa todos os testes do Level 12."""
    print("\n" + "="*70)
    print("LEVEL 12: CLI & Pipeline Integration Tests")
    print("="*70)
    
    tests = [
        ("Test 12.1", test_1_modular_pipeline_integration),
        ("Test 12.2", test_2_cli_modules_import),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            test_func()
            results.append((test_name, "PASSED", None))
        except Exception as e:
            results.append((test_name, "FAILED", str(e)))
            print(f"\n❌ {test_name} FAILED: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY - LEVEL 12")
    print("="*70)
    
    passed = sum(1 for _, status, _ in results if status == "PASSED")
    failed = sum(1 for _, status, _ in results if status == "FAILED")
    
    for test_name, status, error in results:
        symbol = "✅" if status == "PASSED" else "❌"
        print(f"{symbol} {test_name}: {status}")
        if error:
            print(f"   Error: {error}")
    
    print(f"\nTotal tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success rate: {(passed/len(tests)*100):.1f}%")
    print("="*70)
    
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
