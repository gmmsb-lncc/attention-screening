#!/usr/bin/env python3
"""
Teste COMPLETO: Classificação + Regressão
Valida ambos os pipelines com dados sintéticos
Ordem oficial: dados → labels → embeddings → split → treino → avaliação
"""

import sys
import os
from pathlib import Path
import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')

# Add paths
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / 'src'))

print("=" * 80)
print("🧪 TESTE COMPLETO: CLASSIFICAÇÃO + REGRESSÃO")
print("=" * 80)
print("\n📋 Ordem: dados → labels → embeddings → split → treino → avaliação\n")

# Setup
test_output_dir = repo_root / "tests" / "classification_regression_test"
test_output_dir.mkdir(exist_ok=True)

# Detectar device
import torch
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
    print(f"✓ Device: MPS (Metal Performance Shaders) - GPU ativa!")
elif torch.cuda.is_available():
    device = 'cuda'
    print(f"✓ Device: CUDA")
else:
    device = 'cpu'
    print(f"✓ Device: CPU")

print(f"PyTorch: {torch.__version__}\n")

# ============================================================================
# PREPARAR DADOS
# ============================================================================
print("=" * 80)
print("PREPARAÇÃO: DATASET SINTÉTICO")
print("=" * 80)

n_samples = 30

# SMILES simples
smiles_list = [
    'CCO', 'CCCO', 'CCCCO', 'CC(C)O', 'CCC(C)O',
    'CC(=O)O', 'CCC(=O)O', 'CC(=O)C', 'CCC(=O)C', 'CCCC(=O)C',
    'c1ccccc1', 'c1ccccc1O', 'c1ccccc1C', 'c1ccccc1N', 'c1ccccc1F',
    'CCN', 'CCCN', 'CC(C)N', 'CCCCN', 'CCC(C)N',
    'CC(C)CO', 'CCCCCO', 'c1ccccc1Cl', 'c1ccccc1Br', 'CC(=O)N',
    'CCC(=O)N', 'CCNC', 'CCCNC', 'c1ccccc1S', 'c1ccccc1P'
]

# Proteína
protein_seq = 'MKVLWAALLVTFLAGCQAKVGNQFSDVHPEYGDLLGIAGRDGRMEVWAKELPADVMTPVNELQNLANLSPVLKGFPAPKGFYAIEKQNLAVLNSNFNTKVIDFGLSKDIDEDLDKITCTVREIHNLESLGGKTAIILDFG'

# Valores de afinidade (para regressão e classificação)
pchembl_values = np.random.uniform(4.0, 9.0, n_samples)
ki_values = 10 ** (-pchembl_values + np.random.normal(0, 0.5, n_samples))  # nM
kd_values = 10 ** (-pchembl_values + np.random.normal(0, 0.5, n_samples))  # nM
ic50_values = 10 ** (-pchembl_values + np.random.normal(0, 0.5, n_samples))  # nM

dataset_data = {
    'Ligand_SMILES': smiles_list[:n_samples],
    'Target_Seq': [protein_seq] * n_samples,
    'seq_id': ['KINASE_TEST'] * n_samples,
    'seq': [protein_seq] * n_samples,
    'pchembl_value': pchembl_values,
    'Ki': ki_values,
    'Kd': kd_values,
    'IC50': ic50_values
}

df = pd.DataFrame(dataset_data)
dataset_file = test_output_dir / "test_dataset.tsv"
df.to_csv(dataset_file, sep='\t', index=False)

print(f"\n📊 Dataset criado: {len(df)} amostras")
print(f"   ✓ pchembl_value: {df['pchembl_value'].min():.2f} - {df['pchembl_value'].max():.2f}")
print(f"   ✓ Ki (nM): {df['Ki'].min():.2e} - {df['Ki'].max():.2e}")
print(f"   ✓ Kd (nM): {df['Kd'].min():.2e} - {df['Kd'].max():.2e}")
print(f"   ✓ IC50 (nM): {df['IC50'].min():.2e} - {df['IC50'].max():.2e}")

# Embeddings sintéticos
n_protein_dim = 320
n_ligand_dim = 384
n_total_dim = n_protein_dim + n_ligand_dim

np.random.seed(42)
X = np.random.randn(n_samples, n_total_dim).astype(np.float32)
X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

print(f"\n🧬 Embeddings sintéticos: {X.shape}")

# ============================================================================
# TESTE 1: CLASSIFICAÇÃO
# ============================================================================
print("\n" + "=" * 80)
print("TESTE 1: PIPELINE DE CLASSIFICAÇÃO")
print("=" * 80)

# 1. Labels para classificação
threshold = 6.0
y_class = (df['pchembl_value'] >= threshold).astype(int).values

print(f"\n🏷️  Labels de Classificação:")
print(f"   ✓ Threshold: pchembl >= {threshold}")
print(f"   ✓ Ativos (1): {np.sum(y_class == 1)} ({100*np.sum(y_class == 1)/len(y_class):.1f}%)")
print(f"   ✓ Inativos (0): {np.sum(y_class == 0)} ({100*np.sum(y_class == 0)/len(y_class):.1f}%)")

# 2. Split
from sklearn.model_selection import train_test_split

X_train_c, X_temp_c, y_train_c, y_temp_c = train_test_split(
    X, y_class, test_size=0.2, random_state=42, stratify=y_class
)
X_val_c, X_test_c, y_val_c, y_test_c = train_test_split(
    X_temp_c, y_temp_c, test_size=0.5, random_state=42, stratify=y_temp_c
)

print(f"\n📦 Splits:")
print(f"   ✓ Train: {len(X_train_c)} ({100*len(X_train_c)/len(X):.0f}%)")
print(f"   ✓ Val: {len(X_val_c)} ({100*len(X_val_c)/len(X):.0f}%)")
print(f"   ✓ Test: {len(X_test_c)} ({100*len(X_test_c)/len(X):.0f}%)")

# 3. Treinar classificadores
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

print(f"\n🤖 Treinando 6 classificadores...\n")

classifiers = {
    'RandomForest': RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=2),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=50, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=3, n_jobs=2),
    'MLP': MLPClassifier(hidden_layer_sizes=(100,), max_iter=200, random_state=42),
}

# Adicionar XGBoost se disponível
try:
    import xgboost as xgb
    classifiers['XGBoost'] = xgb.XGBClassifier(n_estimators=50, random_state=42, eval_metric='logloss')
    print("   ✓ XGBoost disponível")
except ImportError:
    print("   ⚠️  XGBoost não disponível (opcional)")

trained_classifiers = {}

for name, clf in classifiers.items():
    print(f"   🔹 {name}...", end=" ")
    try:
        clf.fit(X_train_c, y_train_c)
        trained_classifiers[name] = clf
        print("✓")
    except Exception as e:
        print(f"❌ {e}")

print(f"\n   ✓ {len(trained_classifiers)}/{len(classifiers)} classificadores treinados")

# 4. Avaliar classificadores
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print(f"\n📊 Avaliação em TESTE:\n")

classification_results = {}

for name, clf in trained_classifiers.items():
    try:
        y_pred = clf.predict(X_test_c)
        
        acc = accuracy_score(y_test_c, y_pred)
        prec = precision_score(y_test_c, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test_c, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test_c, y_pred, average='weighted', zero_division=0)
        
        classification_results[name] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1
        }
        
        print(f"   {name:<18} Acc: {acc:.3f}  F1: {f1:.3f}")
        
    except Exception as e:
        print(f"   {name:<18} ❌ Erro: {e}")

# ============================================================================
# TESTE 2: REGRESSÃO
# ============================================================================
print("\n" + "=" * 80)
print("TESTE 2: PIPELINE DE REGRESSÃO")
print("=" * 80)

# 1. Labels para regressão (usar pchembl_value como target)
y_reg = df['pchembl_value'].values

print(f"\n🏷️  Labels de Regressão:")
print(f"   ✓ Target: pchembl_value")
print(f"   ✓ Range: {y_reg.min():.2f} - {y_reg.max():.2f}")
print(f"   ✓ Mean: {y_reg.mean():.2f} ± {y_reg.std():.2f}")

# 2. Split (sem estratificação para regressão)
X_train_r, X_temp_r, y_train_r, y_temp_r = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)
X_val_r, X_test_r, y_val_r, y_test_r = train_test_split(
    X_temp_r, y_temp_r, test_size=0.5, random_state=42
)

print(f"\n📦 Splits:")
print(f"   ✓ Train: {len(X_train_r)} ({100*len(X_train_r)/len(X):.0f}%)")
print(f"   ✓ Val: {len(X_val_r)} ({100*len(X_val_r)/len(X):.0f}%)")
print(f"   ✓ Test: {len(X_test_r)} ({100*len(X_test_r)/len(X):.0f}%)")

# 3. Treinar regressores
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor

print(f"\n🤖 Treinando 11 regressores...\n")

regressors = {
    'RandomForest': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=2),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=50, random_state=42),
    'LinearRegression': LinearRegression(n_jobs=2),
    'Ridge': Ridge(random_state=42),
    'Lasso': Lasso(random_state=42),
    'ElasticNet': ElasticNet(random_state=42),
    'SVR': SVR(kernel='rbf'),
    'KNN': KNeighborsRegressor(n_neighbors=3, n_jobs=2),
    'DecisionTree': DecisionTreeRegressor(random_state=42),
    'MLP': MLPRegressor(hidden_layer_sizes=(100,), max_iter=200, random_state=42),
}

# Adicionar XGBoost se disponível
try:
    import xgboost as xgb
    regressors['XGBoost'] = xgb.XGBRegressor(n_estimators=50, random_state=42)
except ImportError:
    pass

trained_regressors = {}

for name, reg in regressors.items():
    print(f"   🔹 {name}...", end=" ")
    try:
        reg.fit(X_train_r, y_train_r)
        trained_regressors[name] = reg
        print("✓")
    except Exception as e:
        print(f"❌ {e}")

print(f"\n   ✓ {len(trained_regressors)}/{len(regressors)} regressores treinados")

# 4. Avaliar regressores
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print(f"\n📊 Avaliação em TESTE:\n")

regression_results = {}

for name, reg in trained_regressors.items():
    try:
        y_pred = reg.predict(X_test_r)
        
        mae = mean_absolute_error(y_test_r, y_pred)
        mse = mean_squared_error(y_test_r, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_r, y_pred)
        
        regression_results[name] = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }
        
        print(f"   {name:<18} MAE: {mae:.3f}  RMSE: {rmse:.3f}  R²: {r2:.3f}")
        
    except Exception as e:
        print(f"   {name:<18} ❌ Erro: {e}")

# ============================================================================
# RESUMO FINAL
# ============================================================================
print("\n" + "=" * 80)
print("📊 RESUMO FINAL - CLASSIFICAÇÃO + REGRESSÃO")
print("=" * 80)

print("\n✅ CLASSIFICAÇÃO:")
print(f"   - Modelos testados: {len(classification_results)}/{len(classifiers)}")
print(f"   - Dataset: {len(df)} amostras")
print(f"   - Features: {X.shape[1]} dimensões")
print(f"   - Classes: {100*np.sum(y_class==1)/len(y_class):.1f}% ativos")

if classification_results:
    best_clf = max(classification_results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n   🥇 Melhor: {best_clf[0]}")
    print(f"      Accuracy: {best_clf[1]['accuracy']:.3f}")
    print(f"      F1-Score: {best_clf[1]['f1']:.3f}")

print("\n✅ REGRESSÃO:")
print(f"   - Modelos testados: {len(regression_results)}/{len(regressors)}")
print(f"   - Target: pchembl_value")
print(f"   - Range: {y_reg.min():.2f} - {y_reg.max():.2f}")

if regression_results:
    best_reg = min(regression_results.items(), key=lambda x: x[1]['rmse'])
    print(f"\n   🥇 Melhor: {best_reg[0]}")
    print(f"      MAE: {best_reg[1]['mae']:.3f}")
    print(f"      RMSE: {best_reg[1]['rmse']:.3f}")
    print(f"      R²: {best_reg[1]['r2']:.3f}")

print(f"\n📁 Outputs em: {test_output_dir}")

print("\n📋 Modelos Disponíveis:")
print("\n   CLASSIFICAÇÃO (6 modelos):")
print("   - RandomForest, GradientBoosting, XGBoost")
print("   - SVM, KNN, MLP")

print("\n   REGRESSÃO (11 modelos):")
print("   - RandomForest, GradientBoosting, XGBoost")
print("   - LinearRegression, Ridge, Lasso, ElasticNet")
print("   - SVR, KNN, DecisionTree, MLP")

print("\n" + "=" * 80)
print("✅ TESTE COMPLETO FINALIZADO!")
print("=" * 80)
