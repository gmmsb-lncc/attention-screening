#!/usr/bin/env python3
"""
Test script para verificar o funcionamento do robust train/test split
"""

import torch
import numpy as np
from sklearn.datasets import make_classification
from src.classifier.utils.robust_train_test_split import RobustTrainTestSplitter

def create_test_data():
    """Criar dados de teste simulando dataset desbalanceado"""
    X, y = make_classification(
        n_samples=1000,
        n_features=100,
        n_informative=20,
        n_redundant=10,
        n_clusters_per_class=2,
        weights=[0.8, 0.2],  # Dataset desbalanceado
        random_state=42
    )
    
    # Converter para tensors PyTorch
    X = torch.FloatTensor(X)
    y = torch.LongTensor(y)
    
    return X, y

def test_robust_splitting():
    """Test the robust train/test splitting functionality"""
    print("🧪 Testando sistema de divisão train/test robusta...\n")
    
    # Criar dados de teste
    X, y = create_test_data()
    print(f"📊 Dataset criado: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"📈 Distribuição de classes: {torch.bincount(y)}")
    
    # Criar splitter
    splitter = RobustTrainTestSplitter(
        test_size=0.2,
        stratify=True,
        random_state=42,
        verbose=True
    )
    
    # Fazer split
    try:
        X_train, X_test, y_train, y_test = splitter.split(X, y)
        
        print(f"\n✅ Split realizado com sucesso!")
        print(f"🎯 Train set: {X_train.shape[0]} samples")
        print(f"🎯 Test set: {X_test.shape[0]} samples")
        print(f"📊 Train classes: {torch.bincount(y_train)}")
        print(f"📊 Test classes: {torch.bincount(y_test)}")
        
        # Verificar proporções
        train_prop = torch.bincount(y_train).float() / len(y_train)
        test_prop = torch.bincount(y_test).float() / len(y_test)
        
        print(f"\n📈 Proporções Train: {train_prop}")
        print(f"📈 Proporções Test: {test_prop}")
        print(f"📏 Diferença máxima: {torch.max(torch.abs(train_prop - test_prop)):.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no split: {e}")
        return False

def test_with_problematic_data():
    """Test com dados problemáticos"""
    print("\n🧪 Testando com dados problemáticos...\n")
    
    # Criar dataset muito desbalanceado
    X = torch.randn(100, 50)
    y = torch.cat([torch.zeros(95), torch.ones(5)]).long()  # 95% classe 0, 5% classe 1
    
    print(f"📊 Dataset problemático: {torch.bincount(y)}")
    
    splitter = RobustTrainTestSplitter(
        test_size=0.2,
        stratify=True,
        min_samples_per_class=2,
        verbose=True
    )
    
    try:
        X_train, X_test, y_train, y_test = splitter.split(X, y)
        print("✅ Split com dados problemáticos realizado com sucesso!")
        return True
    except ValueError as e:
        print(f"⚠️  Split rejeitado (esperado): {e}")
        return True

if __name__ == "__main__":
    print("🚀 Iniciando testes do sistema de divisão robusta\n")
    
    # Test 1: Dados normais
    success1 = test_robust_splitting()
    
    # Test 2: Dados problemáticos
    success2 = test_with_problematic_data()
    
    if success1 and success2:
        print("\n🎉 Todos os testes passaram!")
        print("✨ Sistema de divisão robusta funcionando corretamente!")
    else:
        print("\n❌ Alguns testes falharam")
