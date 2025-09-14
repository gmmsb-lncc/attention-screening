#!/usr/bin/env python3
"""
Testes básicos para validar as correções no classifier.py
"""
import numpy as np
import tempfile
import os
import sys

def create_dummy_data(n_samples=1000, embedding_dim=512):
    """Cria dados dummy para teste."""
    np.random.seed(42)
    
    # Embeddings aleatórios
    embeddings = np.random.randn(n_samples, embedding_dim).astype(np.float32)
    
    # Labels balanceados
    labels = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
    
    return embeddings, labels

def test_data_validation():
    """Testa a validação de dados."""
    print("🧪 Testando validação de dados...")
    
    # Cria dados de teste
    embeddings, labels = create_dummy_data(n_samples=500, embedding_dim=256)
    
    # Salva em arquivos temporários
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f_emb:
        np.save(f_emb.name, embeddings)
        emb_path = f_emb.name
    
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f_lab:
        np.save(f_lab.name, labels)
        lab_path = f_lab.name
    
    try:
        # Tenta importar e testar
        from classifier import MLPEmbeddingPipeline, MLPConfig
        
        config = MLPConfig(
            hidden_dim=128,
            epochs=5,
            batch_size=32,
            cv_folds=3,
            num_workers=0  # Para evitar problemas em testes
        )
        
        pipeline = MLPEmbeddingPipeline(
            embeddings_path=emb_path,
            labels_path=lab_path,
            **config.__dict__
        )
        
        print("✅ Pipeline criado com sucesso")
        print(f"📊 Validação de dados: {len(pipeline.data_validation['issues'])} problemas encontrados")
        
        if pipeline.data_validation['stats']:
            stats = pipeline.data_validation['stats']
            print(f"📊 Total de amostras: {stats['total_samples']}")
            print(f"📊 Distribuição: {stats['class_distribution']}")
            print(f"📊 Taxa de duplicatas: {stats['duplicate_rate']*100:.1f}%")
        
        return True
        
    except ImportError as e:
        print(f"❌ Erro de importação: {e}")
        return False
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        return False
    finally:
        # Limpa arquivos temporários
        if os.path.exists(emb_path):
            os.unlink(emb_path)
        if os.path.exists(lab_path):
            os.unlink(lab_path)

def test_cross_validation():
    """Testa se o cross-validation não tem data leakage."""
    print("\n🧪 Testando cross-validation...")
    
    # Cria dados pequenos para teste rápido
    embeddings, labels = create_dummy_data(n_samples=200, embedding_dim=64)
    
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f_emb:
        np.save(f_emb.name, embeddings)
        emb_path = f_emb.name
    
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f_lab:
        np.save(f_lab.name, labels)
        lab_path = f_lab.name
    
    try:
        from classifier import MLPEmbeddingPipeline, MLPConfig
        
        config = MLPConfig(
            hidden_dim=32,
            epochs=2,  # Muito rápido para teste
            batch_size=16,
            cv_folds=3,
            early_stopping_patience=1,
            num_workers=0
        )
        
        pipeline = MLPEmbeddingPipeline(
            embeddings_path=emb_path,
            labels_path=lab_path,
            **config.__dict__
        )
        
        print("🔄 Executando cross-validation...")
        cv_score = pipeline.cross_validate(k=3)
        print(f"✅ CV Score: {cv_score:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no CV: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if os.path.exists(emb_path):
            os.unlink(emb_path)
        if os.path.exists(lab_path):
            os.unlink(lab_path)

def main():
    """Executa todos os testes."""
    print("🚀 Iniciando testes do classifier...")
    
    tests_passed = 0
    total_tests = 2
    
    # Teste 1: Validação de dados
    if test_data_validation():
        tests_passed += 1
    
    # Teste 2: Cross-validation
    if test_cross_validation():
        tests_passed += 1
    
    print(f"\n📊 Resultado: {tests_passed}/{total_tests} testes passaram")
    
    if tests_passed == total_tests:
        print("✅ Todos os testes passaram!")
        return 0
    else:
        print("❌ Alguns testes falharam")
        return 1

if __name__ == "__main__":
    exit(main())
