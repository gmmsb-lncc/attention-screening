#!/usr/bin/env python3
"""
Teste completo do pipeline de classificação com conjunto reduzido
Testa o fluxo end-to-end: dados → embeddings → build → classificação
"""

import sys
import os
from pathlib import Path
import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / 'src'))

print("=" * 80)
print("🧪 TESTE COMPLETO DO PIPELINE DE CLASSIFICAÇÃO - CONJUNTO REDUZIDO")
print("=" * 80)

# Setup
test_output_dir = repo_root / "tests" / "classifier_test"
test_output_dir.mkdir(exist_ok=True)

print(f"\n📁 Diretório de saída: {test_output_dir}")

# Test 1: Verificar dataset disponível
print("\n" + "=" * 80)
print("✅ FASE 1: Preparação de Dados")
print("=" * 80)

# Procurar por datasets disponíveis
dataset_paths = [
    repo_root / "tests/datasets/kinase_test_small.tsv",
    repo_root / "src/database/kinase_data.tsv",
    repo_root / "data/test_dataset_1000.tsv",
]

dataset_file = None
for path in dataset_paths:
    if path.exists():
        dataset_file = path
        print(f"✓ Dataset encontrado: {path.name}")
        break

if dataset_file is None:
    print("❌ Nenhum dataset encontrado!")
    print("Datasets procurados:")
    for path in dataset_paths:
        print(f"  - {path}")
    sys.exit(1)

# Verificar estrutura do dataset
print(f"\n📊 Analisando dataset...")
df = pd.read_csv(dataset_file, sep='\t', nrows=10)
print(f"   Colunas disponíveis: {list(df.columns)}")
print(f"   Primeiras colunas: {list(df.columns[:5])}")

# Verificar se tem as colunas necessárias
required_columns = ['Ligand_SMILES', 'Target_Seq', 'Y']
has_required = all(col in df.columns for col in required_columns)

if not has_required:
    print(f"\n⚠️  Dataset não tem colunas padrão do pipeline")
    print(f"   Necessário: {required_columns}")
    print(f"   Disponível: {list(df.columns)}")
    
    # Tentar criar dataset sintético pequeno
    print(f"\n🔧 Criando dataset sintético para teste...")
    
    # Criar dados sintéticos mínimos para teste rápido
    n_samples = 20  # Muito pequeno para validação rápida
    
    # Usar sequências e SMILES muito curtos para processamento rápido
    synthetic_data = {
        'Ligand_SMILES': ['CCO', 'CCCO', 'CC', 'CCC', 'CCCC', 
                         'C', 'CO', 'CCN', 'CCS', 'CCF'] * 2,  # 10 diferentes x 2 = 20
        'Target_Seq': ['MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV'] * n_samples,  # Sequência curta
        'Y': [0] * (n_samples // 2) + [1] * (n_samples // 2),  # Balanceado
        'seq_id': [f'SEQ_{i:03d}' for i in range(n_samples)],  # IDs necessários
        'seq': ['MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV'] * n_samples  # Coluna 'seq' necessária
    }
    
    df_synthetic = pd.DataFrame(synthetic_data)
    synthetic_file = test_output_dir / "synthetic_test_data.tsv"
    df_synthetic.to_csv(synthetic_file, sep='\t', index=False)
    
    dataset_file = synthetic_file
    print(f"   ✓ Dataset sintético criado: {n_samples} amostras")
    print(f"   ✓ Salvo em: {synthetic_file.name}")
else:
    # Criar subset pequeno do dataset real
    print(f"\n📦 Criando subset reduzido do dataset...")
    df_full = pd.read_csv(dataset_file, sep='\t')
    
    # Pegar apenas 20 amostras balanceadas para teste rápido
    n_samples = min(20, len(df_full))
    
    if 'Y' in df_full.columns:
        # Balancear classes
        df_class_0 = df_full[df_full['Y'] == 0].head(n_samples // 2)
        df_class_1 = df_full[df_full['Y'] == 1].head(n_samples // 2)
        df_subset = pd.concat([df_class_0, df_class_1]).sample(frac=1, random_state=42)
    else:
        df_subset = df_full.head(n_samples)
    
    subset_file = test_output_dir / "test_subset.tsv"
    df_subset.to_csv(subset_file, sep='\t', index=False)
    dataset_file = subset_file
    
    print(f"   ✓ Subset criado: {len(df_subset)} amostras")
    print(f"   ✓ Salvo em: {subset_file.name}")
    
    if 'Y' in df_subset.columns:
        print(f"   ✓ Distribuição: {df_subset['Y'].value_counts().to_dict()}")

# Test 2: Import e configuração
print("\n" + "=" * 80)
print("✅ FASE 2: Importação de Módulos")
print("=" * 80)

try:
    from build.core import BuildConfig
    from build.pipeline import BuildPipeline
    print("✓ Módulos build importados")
except Exception as e:
    print(f"❌ Erro ao importar módulos build: {e}")
    sys.exit(1)

try:
    from classifier.core import DataManager, ModelEvaluator
    print("✓ Módulos classifier importados")
except Exception as e:
    print(f"❌ Erro ao importar módulos classifier: {e}")
    sys.exit(1)

try:
    import torch
    # Detectar dispositivo disponível: MPS (Mac) > CUDA (Linux/Windows) > CPU
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        use_gpu = True
        print(f"✓ PyTorch disponível - Device: MPS (Metal Performance Shaders)")
        print(f"   ✨ Mac Apple Silicon detectado - Usando aceleração GPU!")
    elif torch.cuda.is_available():
        device = 'cuda'
        use_gpu = True
        print(f"✓ PyTorch disponível - Device: CUDA")
    else:
        device = 'cpu'
        use_gpu = False
        print(f"✓ PyTorch disponível - Device: CPU")
except Exception as e:
    print(f"⚠️  PyTorch não disponível: {e}")
    device = 'cpu'
    use_gpu = False

# Test 3: Executar Build Pipeline
print("\n" + "=" * 80)
print("✅ FASE 3: Build Pipeline (Embeddings + Matrix)")
print("=" * 80)

print("\n⚙️  Configurando build pipeline...")
print(f"   - Input: {dataset_file.name}")
print(f"   - Output: {test_output_dir}")
print(f"   - Device: {device}")
print(f"   - Batch size: 4 (reduzido para teste rápido)")

try:
    # Configurar pipeline com device correto e MENOR MODELO possível
    build_config = BuildConfig(
        batch_size=2,  # Batch muito pequeno
        use_cache=True,
        n_jobs=1,  # Apenas 1 job para simplicidade
        output_dir=str(test_output_dir),
        ligand_model='SMI-TED',  # Modelo FM4M - usar o mais leve
        protein_model='esm2_t6_8M_UR50D',  # ⭐ MENOR modelo ESM (8M parâmetros)
        use_gpu=use_gpu,  # Usar MPS/CUDA se disponível
        device=device  # Configurar device específico (mps, cuda ou cpu)
    )
    
    print(f"   ⭐ Usando modelo ESM PEQUENO: esm2_t6_8M_UR50D (8M parâmetros)")
    print(f"   ⭐ Usando modelo FM4M: SMI-TED")
    
    pipeline = BuildPipeline(config=build_config)
    
    print("\n🚀 Executando build pipeline...")
    print("   Fase 1: Gerando embeddings (pode levar alguns minutos)...")
    
    # Executar geração de embeddings
    embeddings_success = pipeline.run_embedding_generation(
        input_tsv_path=str(dataset_file),
        output_dir=test_output_dir
    )
    
    if not embeddings_success:
        raise RuntimeError("Falha na geração de embeddings")
    
    print("   ✓ Embeddings gerados com sucesso")
    
    # Executar construção da matriz
    print("   Fase 2: Construindo matriz de features...")
    result = pipeline.build()
    
    print("\n✅ Build pipeline concluído!")
    
    # Verificar outputs
    embeddings_dir = test_output_dir / "embeddings"
    matrix_dir = test_output_dir / "matrix"
    
    if embeddings_dir.exists():
        n_embeddings = len(list(embeddings_dir.glob("*.pt")))
        print(f"   ✓ Embeddings gerados: {n_embeddings} arquivos")
    
    if matrix_dir.exists():
        matrix_files = list(matrix_dir.glob("*.npz"))
        if matrix_files:
            print(f"   ✓ Matriz de features criada: {matrix_files[0].name}")
            
            # Verificar dimensões da matriz
            import scipy.sparse as sp
            matrix = sp.load_npz(matrix_files[0])
            print(f"   ✓ Dimensões da matriz: {matrix.shape}")
        else:
            print("   ⚠️  Matriz de features não encontrada")
    
except KeyboardInterrupt:
    print("\n\n⚠️  Processo interrompido pelo usuário")
    sys.exit(0)
except Exception as e:
    print(f"\n❌ Erro no build pipeline: {e}")
    import traceback
    traceback.print_exc()
    print("\n⚠️  Continuando com classificação se matriz existir...")

# Test 4: Classificação
print("\n" + "=" * 80)
print("✅ FASE 4: Treinamento de Classificadores")
print("=" * 80)

# Verificar se matriz existe
matrix_dir = test_output_dir / "matrix"
matrix_files = list(matrix_dir.glob("*.npz")) if matrix_dir.exists() else []

if not matrix_files:
    print("❌ Matriz de features não encontrada - não é possível treinar")
    print("   O build pipeline precisa ter sucesso primeiro")
    sys.exit(1)

matrix_file = matrix_files[0]
print(f"✓ Matriz encontrada: {matrix_file.name}")

try:
    # Carregar dados
    print("\n📊 Carregando dados para classificação...")
    import scipy.sparse as sp
    
    X = sp.load_npz(matrix_file)
    print(f"   ✓ Features carregadas: {X.shape}")
    
    # Carregar labels
    labels_dir = test_output_dir / "labels"
    label_files = list(labels_dir.glob("*.npy")) if labels_dir.exists() else []
    
    if label_files:
        y = np.load(label_files[0])
        print(f"   ✓ Labels carregados: {len(y)} amostras")
        print(f"   ✓ Distribuição: {np.bincount(y)}")
    else:
        print("   ⚠️  Labels não encontrados - usando labels sintéticos")
        y = np.array([0] * (X.shape[0] // 2) + [1] * (X.shape[0] - X.shape[0] // 2))
    
    # Treinar modelos
    print("\n🤖 Treinando classificadores...")
    print("   Testando 3 modelos rápidos: RF, KNN, MLP")
    
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   ✓ Train: {X_train.shape[0]} amostras")
    print(f"   ✓ Test: {X_test.shape[0]} amostras")
    
    # Converter para dense se necessário (para modelos que precisam)
    if sp.issparse(X_train):
        X_train_dense = X_train.toarray()
        X_test_dense = X_test.toarray()
    else:
        X_train_dense = X_train
        X_test_dense = X_test
    
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=2),
        'KNN': KNeighborsClassifier(n_neighbors=3, n_jobs=2),
        'MLP': MLPClassifier(hidden_layer_sizes=(50,), max_iter=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n   🔹 {name}...")
        try:
            # Treinar
            model.fit(X_train_dense, y_train)
            print(f"      ✓ Treinado")
            
            # Predizer
            y_pred = model.predict(X_test_dense)
            
            # Métricas
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            results[name] = {
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1
            }
            
            print(f"      ✓ Accuracy: {acc:.3f}")
            print(f"      ✓ Precision: {prec:.3f}")
            print(f"      ✓ Recall: {rec:.3f}")
            print(f"      ✓ F1-Score: {f1:.3f}")
            
        except Exception as e:
            print(f"      ❌ Erro: {e}")
            results[name] = {'error': str(e)}
    
    # Resumo
    print("\n" + "=" * 80)
    print("📊 RESUMO DOS RESULTADOS")
    print("=" * 80)
    
    print("\n🏆 Comparação de Modelos:")
    print(f"{'Modelo':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 80)
    
    for name, metrics in results.items():
        if 'error' not in metrics:
            print(f"{name:<20} "
                  f"{metrics['accuracy']:<12.3f} "
                  f"{metrics['precision']:<12.3f} "
                  f"{metrics['recall']:<12.3f} "
                  f"{metrics['f1']:<12.3f}")
        else:
            print(f"{name:<20} ERRO: {metrics['error']}")
    
    # Melhor modelo
    if results:
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        if valid_results:
            best_model = max(valid_results.items(), key=lambda x: x[1]['accuracy'])
            print(f"\n🥇 Melhor modelo: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.3f})")
    
except Exception as e:
    print(f"\n❌ Erro durante classificação: {e}")
    import traceback
    traceback.print_exc()

# Final
print("\n" + "=" * 80)
print("✅ TESTE COMPLETO FINALIZADO")
print("=" * 80)
print(f"\n📁 Outputs salvos em: {test_output_dir}")
print("\n💡 Próximos passos:")
print("   - Testar com dataset maior")
print("   - Testar todos os 6 modelos")
print("   - Executar cross-validation")
print("   - Otimizar hiperparâmetros")
print("=" * 80)
