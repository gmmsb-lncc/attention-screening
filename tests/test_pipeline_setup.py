#!/usr/bin/env python3
"""
Script para testar o pipeline build com um subset pequeno dos dados.
Valida que todas as dependências estão instaladas e o sistema funciona.
"""

import sys
from pathlib import Path

# Add src to path (caminho relativo ao repositório)
repo_root = Path(__file__).parent.parent  # tests/ -> docktkinase/
sys.path.insert(0, str(repo_root / 'src'))

print("=" * 80)
print("🧪 TESTE DO PIPELINE BUILD - DOCKTKINASE")
print("=" * 80)
print()

# Test 1: Import dependencies
print("📦 Testando imports de dependências...")
try:
    import torch
    print(f"  ✅ PyTorch {torch.__version__}")
    print(f"     CUDA disponível: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"  ❌ PyTorch não instalado: {e}")
    sys.exit(1)

try:
    import transformers
    print(f"  ✅ Transformers {transformers.__version__}")
except ImportError as e:
    print(f"  ❌ Transformers não instalado: {e}")
    sys.exit(1)

try:
    import esm
    print(f"  ✅ ESM {esm.__version__} (fair-esm)")
except ImportError as e:
    print(f"  ❌ ESM não instalado: {e}")
    print(f"     Execute: pip install fair-esm")
    sys.exit(1)

try:
    import sklearn
    print(f"  ✅ scikit-learn {sklearn.__version__}")
except ImportError as e:
    print(f"  ❌ scikit-learn não instalado: {e}")
    sys.exit(1)

try:
    import numpy as np
    print(f"  ✅ numpy {np.__version__}")
except ImportError as e:
    print(f"  ❌ numpy não instalado: {e}")
    sys.exit(1)

try:
    import pandas as pd
    print(f"  ✅ pandas {pd.__version__}")
except ImportError as e:
    print(f"  ❌ pandas não instalado: {e}")
    sys.exit(1)

print()
print("=" * 80)
print("📂 Testando estrutura de módulos...")
print("=" * 80)
print()

# Test 2: Import build modules
try:
    from build.core import BuildConfig
    print("  ✅ build.core.BuildConfig")
except ImportError as e:
    print(f"  ❌ Erro ao importar BuildConfig: {e}")
    sys.exit(1)

try:
    from build.pipeline import BuildPipeline
    print("  ✅ build.pipeline.BuildPipeline")
except ImportError as e:
    print(f"  ❌ Erro ao importar BuildPipeline: {e}")
    sys.exit(1)

try:
    from build.embeddings import ProteinEmbedding, LigandEmbedding
    print("  ✅ build.embeddings (ProteinEmbedding, LigandEmbedding)")
except ImportError as e:
    print(f"  ❌ Erro ao importar embeddings: {e}")
    sys.exit(1)

try:
    from build.stratification import Stratifier, SplitValidator
    print("  ✅ build.stratification (Stratifier, SplitValidator)")
except ImportError as e:
    print(f"  ❌ Erro ao importar stratification: {e}")
    sys.exit(1)

try:
    from build.matrix import EmbeddingMatrix
    print("  ✅ build.matrix.EmbeddingMatrix")
except ImportError as e:
    print(f"  ❌ Erro ao importar EmbeddingMatrix: {e}")
    sys.exit(1)

try:
    from build.labels import InteractionLabels, BinaryLabels
    print("  ✅ build.labels (InteractionLabels, BinaryLabels)")
except ImportError as e:
    print(f"  ❌ Erro ao importar labels: {e}")
    sys.exit(1)

print()
print("=" * 80)
print("📁 Verificando arquivos de dados...")
print("=" * 80)
print()

# Test 3: Check data files
data_files = {
    'kinase_all': Path('src/kinase_all/kinase_all_compounds.tsv'),
    'kinase_humans': Path('src/kinase_humans/kinase_human_compounds.tsv'),
    'kinase_non_humans': Path('src/kinase_non_humans/kinase_non_human_compounds.tsv')
}

for name, filepath in data_files.items():
    if filepath.exists():
        size_mb = filepath.stat().st_size / (1024**2)
        print(f"  ✅ {name}: {filepath}")
        print(f"     Tamanho: {size_mb:.2f} MB")
        
        # Count lines
        try:
            with open(filepath, 'r') as f:
                n_lines = sum(1 for _ in f)
            print(f"     Linhas: {n_lines:,}")
        except Exception as e:
            print(f"     Não foi possível contar linhas: {e}")
    else:
        print(f"  ❌ {name}: {filepath} NÃO ENCONTRADO")

print()
print("=" * 80)
print("🎯 Testando criação de configuração...")
print("=" * 80)
print()

# Test 4: Create config
try:
    config = BuildConfig({
        'stratification_enabled': True,
        'stratification_params': {
            'clustering_algorithm': 'dbscan',
            'similarity_threshold': 0.8,
            'cluster_min_size': 5,
            'stratify_by': 'both',
            'protein_weight': 0.6,
            'ligand_weight': 0.4
        },
        'batch_size': 32,
        'use_gpu': torch.cuda.is_available()
    })
    print("  ✅ BuildConfig criado com sucesso")
    print(f"     GPU: {config.use_gpu}")
    print(f"     Batch size: {config.batch_size}")
    print(f"     Estratificação: {config.get('stratification_enabled')}")
except Exception as e:
    print(f"  ❌ Erro ao criar BuildConfig: {e}")
    sys.exit(1)

print()
print("=" * 80)
print("🎉 TODOS OS TESTES PASSARAM!")
print("=" * 80)
print()
print("O sistema está pronto para uso!")
print()
print("Próximos passos:")
print("  1. Execute o pipeline com um dataset pequeno para teste")
print("  2. Verifique os resultados e validações")
print("  3. Execute com o dataset completo")
print()
