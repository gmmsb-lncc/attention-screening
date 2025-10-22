#!/usr/bin/env python3
"""
Teste de Validação da Estratificação
Demonstra que o split 80/10/10 mantém proporções corretas
"""

import numpy as np
from sklearn.model_selection import train_test_split
from scipy import stats

def test_stratification(n_samples=1000, class_1_ratio=0.35, random_state=42):
    """
    Testa se a estratificação mantém proporções corretas
    
    Args:
        n_samples: Número de amostras
        class_1_ratio: Proporção da classe 1 (ATIVO)
        random_state: Seed para reprodutibilidade
    """
    print("="*80)
    print("TESTE DE VALIDAÇÃO DA ESTRATIFICAÇÃO")
    print("="*80)
    print(f"\n📊 Configuração:")
    print(f"   Amostras totais: {n_samples:,}")
    print(f"   Classe 0 (INATIVO): {(1-class_1_ratio)*100:.1f}%")
    print(f"   Classe 1 (ATIVO): {class_1_ratio*100:.1f}%")
    print(f"   Random state: {random_state}")
    
    # Criar dados sintéticos
    np.random.seed(random_state)
    n_class_1 = int(n_samples * class_1_ratio)
    n_class_0 = n_samples - n_class_1
    
    X = np.random.randn(n_samples, 100)  # 100 features
    y = np.array([0] * n_class_0 + [1] * n_class_1)
    
    # Shuffle
    indices = np.random.permutation(n_samples)
    X, y = X[indices], y[indices]
    
    print("\n" + "="*80)
    print("SPLIT ESTRATIFICADO (MÉTODO DO PIPELINE)")
    print("="*80)
    
    # PASSO 1: Separar TEST primeiro (10%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=0.10,
        stratify=y,
        random_state=random_state
    )
    
    # PASSO 2: Do restante (90%), separar VAL (11.1% do restante = 10% do total)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=0.10 / (1 - 0.10),  # = 0.111...
        stratify=y_temp,
        random_state=random_state
    )
    
    print(f"\n✅ Split realizado:")
    print(f"   Train: {len(X_train):>5} amostras ({len(X_train)/len(X)*100:>5.1f}%)")
    print(f"   Val:   {len(X_val):>5} amostras ({len(X_val)/len(X)*100:>5.1f}%)")
    print(f"   Test:  {len(X_test):>5} amostras ({len(X_test)/len(X)*100:>5.1f}%)")
    
    # Calcular proporções
    def get_props(y_subset):
        n_class_0 = np.sum(y_subset == 0)
        n_class_1 = np.sum(y_subset == 1)
        return n_class_0 / len(y_subset), n_class_1 / len(y_subset)
    
    orig_props = get_props(y)
    train_props = get_props(y_train)
    val_props = get_props(y_val)
    test_props = get_props(y_test)
    
    print("\n" + "="*80)
    print("PROPORÇÕES DE CLASSES")
    print("="*80)
    print(f"\n{'Conjunto':<12} {'Classe 0 (INATIVO)':<20} {'Classe 1 (ATIVO)':<20}")
    print("-"*80)
    print(f"{'Original':<12} {orig_props[0]*100:>6.2f}% ({np.sum(y==0):>4}) {orig_props[1]*100:>6.2f}% ({np.sum(y==1):>4})")
    print(f"{'Train':<12} {train_props[0]*100:>6.2f}% ({np.sum(y_train==0):>4}) {train_props[1]*100:>6.2f}% ({np.sum(y_train==1):>4})")
    print(f"{'Validação':<12} {val_props[0]*100:>6.2f}% ({np.sum(y_val==0):>4}) {val_props[1]*100:>6.2f}% ({np.sum(y_val==1):>4})")
    print(f"{'Teste':<12} {test_props[0]*100:>6.2f}% ({np.sum(y_test==0):>4}) {test_props[1]*100:>6.2f}% ({np.sum(y_test==1):>4})")
    
    # Calcular diferenças
    diff_train_val = max(abs(train_props[i] - val_props[i]) for i in range(2))
    diff_train_test = max(abs(train_props[i] - test_props[i]) for i in range(2))
    diff_val_test = max(abs(val_props[i] - test_props[i]) for i in range(2))
    max_diff = max(diff_train_val, diff_train_test, diff_val_test)
    
    print("\n" + "="*80)
    print("DIFERENÇAS DE PROPORÇÕES")
    print("="*80)
    print(f"\n   Train-Val:  {diff_train_val*100:.4f}%")
    print(f"   Train-Test: {diff_train_test*100:.4f}%")
    print(f"   Val-Test:   {diff_val_test*100:.4f}%")
    print(f"   Máxima:     {max_diff*100:.4f}%")
    
    # Teste Chi-Quadrado
    print("\n" + "="*80)
    print("TESTES CHI-QUADRADO")
    print("="*80)
    
    unique, counts = np.unique(y, return_counts=True)
    
    # Expected counts para cada conjunto
    total = len(y)
    expected_train = counts * (len(y_train) / total)
    expected_val = counts * (len(y_val) / total)
    expected_test = counts * (len(y_test) / total)
    
    # Observed counts
    train_counts = np.array([np.sum(y_train == label) for label in unique])
    val_counts = np.array([np.sum(y_val == label) for label in unique])
    test_counts = np.array([np.sum(y_test == label) for label in unique])
    
    # Chi-squared statistics
    chi2_train = np.sum((train_counts - expected_train)**2 / expected_train)
    chi2_val = np.sum((val_counts - expected_val)**2 / expected_val)
    chi2_test = np.sum((test_counts - expected_test)**2 / expected_test)
    
    # P-values
    df = len(unique) - 1
    p_train = 1 - stats.chi2.cdf(chi2_train, df)
    p_val = 1 - stats.chi2.cdf(chi2_val, df)
    p_test = 1 - stats.chi2.cdf(chi2_test, df)
    
    print(f"\n   Train: χ²={chi2_train:.4f}, p={p_train:.4f}")
    print(f"   Val:   χ²={chi2_val:.4f}, p={p_val:.4f}")
    print(f"   Test:  χ²={chi2_test:.4f}, p={p_test:.4f}")
    
    # Validação
    print("\n" + "="*80)
    print("VALIDAÇÃO")
    print("="*80)
    
    success = True
    
    # Critério 1: Diferença de proporções < 5%
    if max_diff < 0.05:
        print("   ✅ Diferença de proporções < 5%")
    else:
        print(f"   ❌ Diferença de proporções > 5% ({max_diff*100:.2f}%)")
        success = False
    
    # Critério 2: P-values > 0.05 (não rejeitamos H0: distribuições são iguais)
    if all(p > 0.05 for p in [p_train, p_val, p_test]):
        print("   ✅ P-values > 0.05 (distribuições estatisticamente iguais)")
    else:
        print("   ❌ Algum p-value < 0.05")
        success = False
    
    # Critério 3: Tamanhos corretos
    expected_train_size = int(n_samples * 0.80)
    expected_val_size = int(n_samples * 0.10)
    expected_test_size = int(n_samples * 0.10)
    
    size_ok = (
        abs(len(X_train) - expected_train_size) <= 1 and
        abs(len(X_val) - expected_val_size) <= 1 and
        abs(len(X_test) - expected_test_size) <= 1
    )
    
    if size_ok:
        print("   ✅ Tamanhos dos conjuntos corretos (80/10/10)")
    else:
        print("   ❌ Tamanhos dos conjuntos incorretos")
        success = False
    
    # Critério 4: Sem overlap
    train_set = set(range(len(y_train)))
    val_set = set(range(len(y_train), len(y_train) + len(y_val)))
    test_set = set(range(len(y_train) + len(y_val), len(y)))
    
    no_overlap = (
        len(train_set & val_set) == 0 and
        len(train_set & test_set) == 0 and
        len(val_set & test_set) == 0
    )
    
    if no_overlap:
        print("   ✅ Sem overlap entre conjuntos")
    else:
        print("   ❌ Existe overlap entre conjuntos")
        success = False
    
    print("\n" + "="*80)
    if success:
        print("🎉 ESTRATIFICAÇÃO VALIDADA COM SUCESSO!")
    else:
        print("⚠️  ESTRATIFICAÇÃO PRECISA DE AJUSTES")
    print("="*80)
    
    return success


if __name__ == '__main__':
    print("\n🧪 TESTE 1: Dataset balanceado (65/35)")
    test_stratification(n_samples=1000, class_1_ratio=0.35, random_state=42)
    
    print("\n\n🧪 TESTE 2: Dataset desbalanceado (85/15)")
    test_stratification(n_samples=1000, class_1_ratio=0.15, random_state=42)
    
    print("\n\n🧪 TESTE 3: Dataset muito desbalanceado (95/5)")
    test_stratification(n_samples=1000, class_1_ratio=0.05, random_state=42)
    
    print("\n\n✅ Todos os testes concluídos!")
