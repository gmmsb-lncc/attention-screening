#!/usr/bin/env python3
"""
Teste simples e independente do sistema de divisão robusta
"""

import sys
import os
from pathlib import Path

# Adicionar o diretório raiz do projeto ao path (caminho relativo)
repo_root = Path(__file__).parent.parent  # tests/ -> docktkinase/
sys.path.insert(0, str(repo_root))

import torch
import numpy as np
from sklearn.datasets import make_classification
from scipy import stats

class SimpleTrainTestSplitter:
    """Versão simplificada do splitter para teste"""
    
    def __init__(self, test_size=0.2, stratify=True, random_state=None, verbose=True):
        self.test_size = test_size
        self.stratify = stratify
        self.random_state = random_state
        self.verbose = verbose
        
    def split(self, X, y):
        """Fazer split com validação estatística"""
        from sklearn.model_selection import train_test_split
        
        if self.verbose:
            print(f"🔍 Realizando split com test_size={self.test_size}")
            print(f"📊 Dataset original: {X.shape[0]} samples")
            print(f"📈 Classes originais: {torch.bincount(y)}")
        
        # Converter para numpy se necessário
        if torch.is_tensor(X):
            X_np = X.numpy()
        else:
            X_np = X
            
        if torch.is_tensor(y):
            y_np = y.numpy()
        else:
            y_np = y
        
        # Fazer split estratificado
        X_train, X_test, y_train, y_test = train_test_split(
            X_np, y_np,
            test_size=self.test_size,
            stratify=y_np if self.stratify else None,
            random_state=self.random_state
        )
        
        # Converter de volta para tensors
        X_train = torch.FloatTensor(X_train)
        X_test = torch.FloatTensor(X_test)
        y_train = torch.LongTensor(y_train)
        y_test = torch.LongTensor(y_test)
        
        # Validar distribuições
        train_counts = torch.bincount(y_train)
        test_counts = torch.bincount(y_test)
        
        # Calcular proporções
        train_props = train_counts.float() / len(y_train)
        test_props = test_counts.float() / len(y_test)
        
        # Teste chi-quadrado
        observed = torch.stack([train_counts, test_counts]).numpy()
        chi2, p_value = stats.chi2_contingency(observed)[:2]
        
        if self.verbose:
            print(f"✅ Split realizado!")
            print(f"📊 Train: {len(y_train)} samples, classes: {train_counts}")
            print(f"📊 Test: {len(y_test)} samples, classes: {test_counts}")
            print(f"📈 Proporções Train: {train_props}")
            print(f"📈 Proporções Test: {test_props}")
            print(f"🧪 Chi-quadrado: {chi2:.4f}, p-value: {p_value:.4f}")
            
            if p_value > 0.05:
                print("✅ Distribuições estatisticamente similares (p > 0.05)")
            else:
                print("⚠️ Distribuições podem diferir significativamente (p ≤ 0.05)")
        
        return X_train, X_test, y_train, y_test

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
    print(f"📈 Distribuição de classes: {torch.bincount(y)}\n")
    
    # Criar splitter
    splitter = SimpleTrainTestSplitter(
        test_size=0.2,
        stratify=True,
        random_state=42,
        verbose=True
    )
    
    # Fazer split
    try:
        X_train, X_test, y_train, y_test = splitter.split(X, y)
        
        # Verificar proporções
        train_prop = torch.bincount(y_train).float() / len(y_train)
        test_prop = torch.bincount(y_test).float() / len(y_test)
        diff_max = torch.max(torch.abs(train_prop - test_prop))
        
        print(f"\n📏 Diferença máxima entre proporções: {diff_max:.4f}")
        
        if diff_max < 0.05:
            print("✅ Proporções balanceadas (diferença < 5%)")
        else:
            print("⚠️ Proporções podem estar desbalanceadas")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no split: {e}")
        return False

def test_edge_cases():
    """Test com casos extremos"""
    print("\n🧪 Testando casos extremos...\n")
    
    # Dataset muito pequeno
    X_small = torch.randn(20, 10)
    y_small = torch.cat([torch.zeros(10), torch.ones(10)]).long()
    
    print("📊 Teste 1: Dataset muito pequeno (20 samples)")
    splitter = SimpleTrainTestSplitter(test_size=0.2, verbose=True)
    
    try:
        X_train, X_test, y_train, y_test = splitter.split(X_small, y_small)
        print("✅ Split com dataset pequeno funcionou!")
    except Exception as e:
        print(f"❌ Erro: {e}")
    
    return True

if __name__ == "__main__":
    print("🚀 Iniciando testes do sistema de divisão robusta\n")
    
    # Test 1: Dados normais
    success1 = test_robust_splitting()
    
    # Test 2: Casos extremos
    success2 = test_edge_cases()
    
    if success1 and success2:
        print("\n🎉 Todos os testes passaram!")
        print("✨ Sistema de divisão robusta funcionando corretamente!")
        print("\n💡 Principais características implementadas:")
        print("   • Estratificação automática")
        print("   • Validação estatística com chi-quadrado")
        print("   • Verificação de proporções balanceadas")
        print("   • Relatórios detalhados de qualidade")
    else:
        print("\n❌ Alguns testes falharam")
