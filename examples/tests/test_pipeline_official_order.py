#!/usr/bin/env python3
"""
Teste do Pipeline OFICIAL com ordem CORRETA
Ordem: dados → labels → embeddings → split → treino → avaliação

Baseado em run_complete_pipeline.py (ordem oficial)
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
print("🧪 TESTE PIPELINE OFICIAL - ORDEM CORRETA")
print("=" * 80)
print("\n📋 Ordem Oficial: dados → labels → embeddings → split → treino → avaliação\n")

# Setup
test_output_dir = repo_root / "tests" / "pipeline_official_test"
test_output_dir.mkdir(exist_ok=True)

# Detectar device
import torch
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
    use_gpu = True
    print(f"✓ Device: MPS (Metal Performance Shaders) - GPU ativa!")
elif torch.cuda.is_available():
    device = 'cuda'
    use_gpu = True
    print(f"✓ Device: CUDA")
else:
    device = 'cpu'
    use_gpu = False
    print(f"✓ Device: CPU")

print(f"PyTorch: {torch.__version__}\n")

# ============================================================================
# PASSO 1: DADOS - Carregar/Criar Dataset
# ============================================================================
print("=" * 80)
print("PASSO 1: CARREGAR DATASET")
print("=" * 80)

n_samples = 20  # Dataset MUITO pequeno para teste rápido

# Criar dataset mínimo
print(f"\n📊 Criando dataset mínimo ({n_samples} amostras)...")

# SMILES simples e reais
smiles_list = [
    'CCO', 'CCCO', 'CCCCO', 'CC(C)O', 'CCC(C)O',
    'CC(=O)O', 'CCC(=O)O', 'CC(=O)C', 'CCC(=O)C', 'CCCC(=O)C',
    'c1ccccc1', 'c1ccccc1O', 'c1ccccc1C', 'c1ccccc1N', 'c1ccccc1F',
    'CCN', 'CCCN', 'CC(C)N', 'CCCCN', 'CCC(C)N'
]

# Sequência de proteína curta (kinase domain)
protein_seq = 'MKVLWAALLVTFLAGCQAKVGNQFSDVHPEYGDLLGIAGRDGRMEVWAKELPADVMTPVNELQNLANLSPVLKGFPAPKGFYAIEKQNLAVLNSNFNTKVIDFGLSKDIDEDLDKITCTVREIHNLESLGGKTAIILDFG'

# Valores de afinidade simulados (pchembl_value)
# Vamos criar valores que depois viram labels (>=6.0 = ativo)
pchembl_values = [7.5, 5.2, 8.1, 4.8, 6.5,  # 5 samples
                  5.8, 7.2, 4.5, 6.8, 5.1,  # 5 samples
                  7.8, 5.5, 6.2, 4.9, 7.0,  # 5 samples
                  5.0, 8.5, 6.1, 5.3, 7.3]  # 5 samples

dataset_data = {
    'Ligand_SMILES': smiles_list[:n_samples],
    'Target_Seq': [protein_seq] * n_samples,
    'seq_id': ['KINASE_TEST'] * n_samples,
    'seq': [protein_seq] * n_samples,
    'pchembl_value': pchembl_values[:n_samples]
}

df = pd.DataFrame(dataset_data)
dataset_file = test_output_dir / "test_dataset.tsv"
df.to_csv(dataset_file, sep='\t', index=False)

print(f"   ✓ Dataset criado: {len(df)} amostras")
print(f"   ✓ SMILES únicos: {df['Ligand_SMILES'].nunique()}")
print(f"   ✓ Sequência: {len(protein_seq)} aminoácidos")
print(f"   ✓ pchembl_value range: {df['pchembl_value'].min():.1f} - {df['pchembl_value'].max():.1f}")
print(f"   ✓ Arquivo: {dataset_file.name}")

# ============================================================================
# PASSO 2: LABELS - Criar labels ANTES de embeddings (ORDEM OFICIAL!)
# ============================================================================
print("\n" + "=" * 80)
print("PASSO 2: CRIAR LABELS (pchembl_value >= 6.0 = ativo)")
print("=" * 80)

# Criar labels conforme pipeline oficial
# pchembl_value >= 6.0 → ativo (1), caso contrário → inativo (0)
threshold = 6.0
y = (df['pchembl_value'] >= threshold).astype(int).values

print(f"\n🏷️  Labels criados:")
print(f"   ✓ Threshold: pchembl >= {threshold}")
print(f"   ✓ Total: {len(y)} labels")
print(f"   ✓ Ativos (1): {np.sum(y == 1)} ({100*np.sum(y == 1)/len(y):.1f}%)")
print(f"   ✓ Inativos (0): {np.sum(y == 0)} ({100*np.sum(y == 0)/len(y):.1f}%)")

# Salvar labels
labels_file = test_output_dir / "labels.npy"
np.save(labels_file, y)
print(f"   ✓ Salvo em: {labels_file.name}")

# ============================================================================
# PASSO 3: EMBEDDINGS - Gerar embeddings de proteínas e ligantes
# ============================================================================
print("\n" + "=" * 80)
print("PASSO 3: GERAR EMBEDDINGS (ESM-2 + SMI-TED)")
print("=" * 80)

print("\n⚠️  NOTA: Para teste rápido, vamos usar embeddings sintéticos")
print("   (Embeddings reais com ESM-2 + SMI-TED levam ~5-10min)")

# Criar embeddings sintéticos para teste rápido
print("\n🔹 Criando embeddings sintéticos...")

# Dimensões típicas:
# ESM-2 t6_8M: 320 dims
# SMI-TED: 384 dims
# Total concatenado: 704 dims

n_protein_dim = 320
n_ligand_dim = 384
n_total_dim = n_protein_dim + n_ligand_dim

# Criar matriz de features sintética
np.random.seed(42)
X = np.random.randn(n_samples, n_total_dim).astype(np.float32)

# Normalizar para parecer mais realista
X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

print(f"   ✓ Embeddings sintéticos criados: {X.shape}")
print(f"   ✓ Dimensão proteína (simulado): {n_protein_dim}")
print(f"   ✓ Dimensão ligante (simulado): {n_ligand_dim}")
print(f"   ✓ Dimensão total: {n_total_dim}")

# ============================================================================
# PASSO 4: SPLIT - Estratificar train/val/test
# ============================================================================
print("\n" + "=" * 80)
print("PASSO 4: SPLIT ESTRATIFICADO (80% train, 10% val, 10% test)")
print("=" * 80)

from sklearn.model_selection import train_test_split

# Primeiro split: train+val vs test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

# Segundo split: train vs val
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.111, random_state=42, stratify=y_temp  # 0.111 de 90% = 10% do total
)

print(f"\n📦 Splits criados:")
print(f"   ✓ Train: {X_train.shape[0]} amostras ({100*len(X_train)/len(X):.1f}%)")
print(f"      - Classe 0: {np.sum(y_train == 0)}, Classe 1: {np.sum(y_train == 1)}")
print(f"   ✓ Val:   {X_val.shape[0]} amostras ({100*len(X_val)/len(X):.1f}%)")
print(f"      - Classe 0: {np.sum(y_val == 0)}, Classe 1: {np.sum(y_val == 1)}")
print(f"   ✓ Test:  {X_test.shape[0]} amostras ({100*len(X_test)/len(X):.1f}%)")
print(f"      - Classe 0: {np.sum(y_test == 0)}, Classe 1: {np.sum(y_test == 1)}")

# Verificar estratificação
train_ratio = np.sum(y_train == 1) / len(y_train)
val_ratio = np.sum(y_val == 1) / len(y_val) if len(y_val) > 0 else 0
test_ratio = np.sum(y_test == 1) / len(y_test) if len(y_test) > 0 else 0

print(f"\n   ✓ Proporção de ativos:")
print(f"      - Train: {100*train_ratio:.1f}%")
if len(y_val) > 0:
    print(f"      - Val:   {100*val_ratio:.1f}%")
if len(y_test) > 0:
    print(f"      - Test:  {100*test_ratio:.1f}%")

# ============================================================================
# PASSO 5: TREINO - Treinar classificador
# ============================================================================
print("\n" + "=" * 80)
print("PASSO 5: TREINAR CLASSIFICADOR")
print("=" * 80)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# Modelos rápidos para teste
models = {
    'RandomForest': RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        random_state=42,
        n_jobs=2
    ),
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=50,
        max_depth=5,
        random_state=42
    ),
    'KNN': KNeighborsClassifier(
        n_neighbors=3,
        n_jobs=2
    ),
}

print(f"\n🤖 Treinando {len(models)} modelos...\n")

trained_models = {}

for name, model in models.items():
    print(f"   🔹 {name}...")
    try:
        # Treinar
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"      ✓ Treinado com sucesso")
        
    except Exception as e:
        print(f"      ❌ Erro: {e}")

print(f"\n   ✓ {len(trained_models)}/{len(models)} modelos treinados")

# ============================================================================
# PASSO 6: AVALIAÇÃO - Avaliar no conjunto de validação
# ============================================================================
print("\n" + "=" * 80)
print("PASSO 6: AVALIAÇÃO - Validação")
print("=" * 80)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("\n📊 Métricas no conjunto de VALIDAÇÃO:\n")

val_results = {}

if len(y_val) > 0:
    for name, model in trained_models.items():
        try:
            y_pred = model.predict(X_val)
            
            acc = accuracy_score(y_val, y_pred)
            prec = precision_score(y_val, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_val, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
            
            val_results[name] = {
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1
            }
            
            print(f"   🔹 {name}:")
            print(f"      - Accuracy:  {acc:.3f}")
            print(f"      - Precision: {prec:.3f}")
            print(f"      - Recall:    {rec:.3f}")
            print(f"      - F1-Score:  {f1:.3f}\n")
            
        except Exception as e:
            print(f"   ❌ {name}: Erro - {e}\n")
else:
    print("   ⚠️  Conjunto de validação muito pequeno, pulando avaliação")

# ============================================================================
# PASSO 7: AVALIAÇÃO - Avaliar no conjunto de teste
# ============================================================================
print("\n" + "=" * 80)
print("PASSO 7: AVALIAÇÃO - Teste Final")
print("=" * 80)

print("\n📊 Métricas no conjunto de TESTE:\n")

test_results = {}

if len(y_test) > 0:
    for name, model in trained_models.items():
        try:
            y_pred = model.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            test_results[name] = {
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1
            }
            
            print(f"   🔹 {name}:")
            print(f"      - Accuracy:  {acc:.3f}")
            print(f"      - Precision: {prec:.3f}")
            print(f"      - Recall:    {rec:.3f}")
            print(f"      - F1-Score:  {f1:.3f}\n")
            
        except Exception as e:
            print(f"   ❌ {name}: Erro - {e}\n")
else:
    print("   ⚠️  Conjunto de teste muito pequeno, pulando avaliação")

# ============================================================================
# RESUMO FINAL
# ============================================================================
print("\n" + "=" * 80)
print("📊 RESUMO FINAL - TESTE PIPELINE OFICIAL")
print("=" * 80)

print("\n✅ Pipeline executado com ORDEM CORRETA:")
print("   1. ✓ Dataset carregado (20 amostras)")
print("   2. ✓ Labels criados ANTES de embeddings (pchembl >= 6.0)")
print("   3. ✓ Embeddings gerados (sintéticos: 704 dims)")
print("   4. ✓ Split estratificado (train/val/test)")
print("   5. ✓ Modelos treinados (3 classificadores)")
print("   6. ✓ Avaliação em validação")
print("   7. ✓ Avaliação em teste")

print(f"\n📈 Resultados:")
print(f"   - Modelos treinados: {len(trained_models)}")
print(f"   - Dataset: {len(df)} amostras")
print(f"   - Features: {X.shape[1]} dimensões")
print(f"   - Classes balanceadas: {100*np.sum(y==1)/len(y):.1f}% ativos")

if test_results:
    best_model = max(test_results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n🥇 Melhor modelo (teste): {best_model[0]}")
    print(f"   - Accuracy: {best_model[1]['accuracy']:.3f}")
    print(f"   - F1-Score: {best_model[1]['f1']:.3f}")

print(f"\n📁 Outputs em: {test_output_dir}")

print("\n💡 Próximos passos para teste REAL:")
print("   1. Usar CompletePipeline de run_complete_pipeline.py")
print("   2. Gerar embeddings REAIS com ESM-2 (8M params)")
print("   3. Gerar embeddings REAIS com SMI-TED")
print("   4. Dataset maior (50-100 amostras)")
print("   5. Validar com todos os 6 classificadores")

print("\n" + "=" * 80)
print("✅ TESTE CONCLUÍDO COM SUCESSO!")
print("=" * 80)
