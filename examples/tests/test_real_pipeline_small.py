#!/usr/bin/env python3
"""
Teste REAL do pipeline completo com dataset reduzido
Pipeline: dados → embeddings → estratificação → matriz → classificação
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
sys.path.insert(0, str(repo_root / 'FM4M'))

print("=" * 80)
print("🧪 TESTE REAL DO PIPELINE COMPLETO - DATASET REDUZIDO")
print("=" * 80)
print("\n📋 Pipeline: dados → embeddings → estratificação → matriz → classificação\n")

# Setup
test_output_dir = repo_root / "tests" / "real_pipeline_test"
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

# FASE 1: Criar dataset pequeno mas REAL
print("=" * 80)
print("FASE 1: Preparação de Dados")
print("=" * 80)

n_samples = 30  # Pequeno para teste rápido

# Criar dataset com SMILES e sequências REAIS
print(f"\n📊 Criando dataset com {n_samples} amostras...")

# SMILES reais simples
smiles_samples = [
    'CCO', 'CCCO', 'CCCCO', 'CC(C)O', 'CCC(C)O',  # Alcoois
    'CC(=O)O', 'CCC(=O)O', 'CC(=O)C', 'CCC(=O)C',  # Cetonas/Acidos
    'c1ccccc1', 'c1ccccc1O', 'c1ccccc1C', 'c1ccccc1N',  # Aromáticos
    'CCN', 'CCCN', 'CC(C)N', 'CCCCN', 'CCC(C)N'  # Aminas
] * 2  # Duplicar para ter 30+

# Sequência de proteína REAL (kinase domain curto)
protein_seq = 'MKVLWAALLVTFLAGCQAKVGNQFSDVHPEYGDLLGIAGRDGRMEVWAKELPADVMTPVNELQNLANLSPVLKGFPAPKGFYAIEKQNLAVLNSNFNTKVIDFGLSKDIDEDLDKITCTVREIHNLESLGGKTAIILDFG'

dataset_data = {
    'Ligand_SMILES': smiles_samples[:n_samples],
    'Target_Seq': [protein_seq] * n_samples,
    'seq_id': [f'KINASE_TEST'] * n_samples,
    'seq': [protein_seq] * n_samples,
    'Y': [0] * (n_samples // 2) + [1] * (n_samples - n_samples // 2)
}

df = pd.DataFrame(dataset_data)
dataset_file = test_output_dir / "test_dataset.tsv"
df.to_csv(dataset_file, sep='\t', index=False)

print(f"   ✓ Dataset criado: {len(df)} amostras")
print(f"   ✓ SMILES únicos: {df['Ligand_SMILES'].nunique()}")
print(f"   ✓ Sequência proteína: {len(protein_seq)} aa")
print(f"   ✓ Labels: Classe 0: {sum(df['Y']==0)}, Classe 1: {sum(df['Y']==1)}")
print(f"   ✓ Arquivo: {dataset_file.name}\n")

# FASE 2: Gerar Embeddings REAIS
print("=" * 80)
print("FASE 2: Geração de Embeddings (ESM + SMI-TED)")
print("=" * 80)

from build.core import BuildConfig
from build.embeddings import ProteinEmbedding, LigandEmbedding

# Configuração
config = BuildConfig(
    batch_size=4,
    use_cache=True,
    n_jobs=1,
    output_dir=str(test_output_dir),
    protein_model='esm2_t6_8M_UR50D',  # Modelo MENOR (8M params)
    ligand_model='SMI-TED',  # Modelo FM4M
    use_gpu=use_gpu,
    device=device
)

print(f"\n⚙️  Configuração:")
print(f"   - Modelo proteína: esm2_t6_8M_UR50D (8M parâmetros)")
print(f"   - Modelo ligante: SMI-TED (FM4M)")
print(f"   - Device: {device}")
print(f"   - Batch size: {config.batch_size}\n")

# 2.1: Embeddings de Proteínas
print("🔹 Gerando embeddings de proteínas...")
protein_emb = ProteinEmbedding(
    config=config,
    model_name='esm2_t6_8M_UR50D',  # Modelo PEQUENO explicitamente
    use_gpu=use_gpu
)

try:
    protein_success = protein_emb.generate_embeddings(
        tsv_path=dataset_file,
        output_dir=test_output_dir / "protein_embeddings"
    )
    
    if protein_success:
        print("   ✓ Embeddings de proteínas gerados!")
        protein_emb_dir = test_output_dir / "protein_embeddings"
        n_protein_emb = len(list(protein_emb_dir.glob("*.pt")))
        print(f"   ✓ Arquivos criados: {n_protein_emb}")
    else:
        raise RuntimeError("Falha ao gerar embeddings de proteínas")
        
except Exception as e:
    print(f"   ❌ Erro: {e}")
    raise AssertionError("Test failed")

# 2.2: Embeddings de Ligantes
print("\n🔹 Gerando embeddings de ligantes (SMI-TED)...")
ligand_emb = LigandEmbedding(
    config=config,
    model_name='SMI-TED'  # Modelo FM4M explicitamente
)

try:
    ligand_success = ligand_emb.generate_embeddings(
        tsv_path=dataset_file,
        output_dir=test_output_dir / "ligand_embeddings"
    )
    
    if ligand_success:
        print("   ✓ Embeddings de ligantes gerados!")
        ligand_emb_dir = test_output_dir / "ligand_embeddings"
        n_ligand_emb = len(list(ligand_emb_dir.glob("*.pt")))
        print(f"   ✓ Arquivos criados: {n_ligand_emb}")
    else:
        raise RuntimeError("Falha ao gerar embeddings de ligantes")
        
except Exception as e:
    print(f"   ❌ Erro: {e}")
    raise AssertionError("Test failed")

# FASE 3: Construir Matriz de Features
print("\n" + "=" * 80)
print("FASE 3: Construção da Matriz de Features")
print("=" * 80)

from build.matrix import EmbeddingMatrix

print("\n🔹 Concatenando embeddings e construindo matriz...")

matrix_builder = EmbeddingMatrix(config)

try:
    matrix_success = matrix_builder.build_matrix(
        protein_emb_dir=test_output_dir / "protein_embeddings",
        ligand_emb_dir=test_output_dir / "ligand_embeddings",
        output_dir=test_output_dir / "matrix",
        labels=df['Y'].values
    )
    
    if matrix_success:
        print("   ✓ Matriz de features construída!")
        
        # Verificar matriz
        matrix_dir = test_output_dir / "matrix"
        matrix_files = list(matrix_dir.glob("*.npz"))
        
        if matrix_files:
            import scipy.sparse as sp
            X = sp.load_npz(matrix_files[0])
            print(f"   ✓ Shape da matriz: {X.shape}")
            print(f"   ✓ Tipo: sparse matrix")
            print(f"   ✓ Densidade: {X.nnz / (X.shape[0] * X.shape[1]):.4f}")
        
except Exception as e:
    print(f"   ❌ Erro: {e}")
    import traceback
    traceback.print_exc()
    raise AssertionError("Test failed")

# FASE 4: Classificação
print("\n" + "=" * 80)
print("FASE 4: Treinamento de Classificadores")
print("=" * 80)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import scipy.sparse as sp

# Carregar matriz e labels
print("\n📦 Carregando dados...")
matrix_files = list((test_output_dir / "matrix").glob("*.npz"))
X = sp.load_npz(matrix_files[0])
y = df['Y'].values

print(f"   ✓ Features: {X.shape}")
print(f"   ✓ Labels: {len(y)}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"   ✓ Train: {X_train.shape[0]} amostras")
print(f"   ✓ Test: {X_test.shape[0]} amostras")

# Converter para dense
X_train_dense = X_train.toarray()
X_test_dense = X_test.toarray()

# Modelos
models = {
    'RandomForest': RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=2),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=50, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=3, n_jobs=2),
    'MLP': MLPClassifier(hidden_layer_sizes=(100,), max_iter=200, random_state=42)
}

print("\n🤖 Treinando classificadores...\n")

results = {}
for name, model in models.items():
    print(f"   🔹 {name}...")
    try:
        model.fit(X_train_dense, y_train)
        y_pred = model.predict(X_test_dense)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        results[name] = {'accuracy': acc, 'f1': f1, 'predictions': y_pred}
        
        print(f"      ✓ Accuracy: {acc:.3f}")
        print(f"      ✓ F1-Score: {f1:.3f}\n")
        
    except Exception as e:
        print(f"      ❌ Erro: {e}\n")
        results[name] = {'error': str(e)}

# RESULTADOS FINAIS
print("=" * 80)
print("📊 RESULTADOS FINAIS")
print("=" * 80)

print(f"\n{'Modelo':<20} {'Accuracy':<12} {'F1-Score':<12}")
print("-" * 44)

for name, metrics in results.items():
    if 'error' not in metrics:
        print(f"{name:<20} {metrics['accuracy']:<12.3f} {metrics['f1']:<12.3f}")

if results:
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    if valid_results:
        best = max(valid_results.items(), key=lambda x: x[1]['accuracy'])
        print(f"\n🥇 Melhor: {best[0]} (Acc: {best[1]['accuracy']:.3f}, F1: {best[1]['f1']:.3f})")

print("\n" + "=" * 80)
print("✅ PIPELINE COMPLETO EXECUTADO COM SUCESSO!")
print("=" * 80)
print(f"\n📁 Outputs em: {test_output_dir}")
print(f"\n✓ Embeddings de proteínas: {n_protein_emb} arquivos")
print(f"✓ Embeddings de ligantes: {n_ligand_emb} arquivos")
print(f"✓ Matriz: {X.shape[0]} × {X.shape[1]}")
print(f"✓ Classificadores: {len(valid_results)}/{len(models)} treinados")
print("\n" + "=" * 80)
