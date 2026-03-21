#!/usr/bin/env python3
"""
Script para testar o pipeline build com um SUBSET PEQUENO dos dados.
Usa apenas 1000 amostras para validar rapidamente que tudo funciona.
"""

import sys
import pandas as pd
from pathlib import Path

# Add src to path (caminho relativo ao repositório)
repo_root = Path(__file__).parent.parent  # tests/ -> docktkinase/
sys.path.insert(0, str(repo_root / 'src'))

from build.core import BuildConfig
from build.pipeline import BuildPipeline

def create_small_dataset(input_file, output_file, n_samples=1000):
    """Cria um dataset pequeno para testes."""
    print(f"📊 Criando dataset de teste com {n_samples} amostras...")
    print(f"   Input: {input_file}")
    print(f"   Output: {output_file}")
    
    # Ler arquivo completo
    df = pd.read_csv(input_file, sep='\t')
    print(f"   Total de amostras no arquivo original: {len(df):,}")
    
    # Pegar subset balanceado
    # Calcular threshold binário (1000 nM)
    df['is_active'] = df['standard_value'] < 1000
    
    # Balancear classes
    n_active = min(n_samples // 2, (df['is_active'] == True).sum())
    n_inactive = min(n_samples // 2, (df['is_active'] == False).sum())
    
    df_active = df[df['is_active'] == True].sample(n=n_active, random_state=42)
    df_inactive = df[df['is_active'] == False].sample(n=n_inactive, random_state=42)
    
    df_subset = pd.concat([df_active, df_inactive]).sample(frac=1, random_state=42)
    
    print(f"   ✅ Subset criado:")
    print(f"      - Ativas: {n_active} ({n_active/len(df_subset)*100:.1f}%)")
    print(f"      - Inativas: {n_inactive} ({n_inactive/len(df_subset)*100:.1f}%)")
    print(f"      - Total: {len(df_subset)}")
    
    # Salvar
    df_subset.to_csv(output_file, sep='\t', index=False)
    print(f"   ✅ Salvo em: {output_file}")
    
    return output_file

def test_pipeline_small():
    """Testa o pipeline com dataset pequeno."""
    print("=" * 80)
    print("🧪 TESTE DO PIPELINE BUILD - DATASET PEQUENO")
    print("=" * 80)
    print()
    
    # Configurar paths
    input_file = Path("src/kinase_humans/kinase_human_compounds.tsv")
    output_dir = Path("test_output_small")
    test_file = output_dir / "test_dataset_1000.tsv"
    
    if not input_file.exists():
        print(f"❌ Arquivo não encontrado: {input_file}")
        return False
    
    # Criar diretório de output
    output_dir.mkdir(exist_ok=True)
    
    # Criar dataset pequeno
    print("\n" + "=" * 80)
    print("📊 ETAPA 1: CRIAR DATASET DE TESTE")
    print("=" * 80)
    print()
    
    test_file = create_small_dataset(input_file, test_file, n_samples=1000)
    
    # Configurar pipeline
    print("\n" + "=" * 80)
    print("⚙️  ETAPA 2: CONFIGURAR PIPELINE")
    print("=" * 80)
    print()
    
    config = BuildConfig({
        'stratification_enabled': True,
        'stratification_params': {
            'clustering_algorithm': 'kmeans',  # KMeans é mais rápido para teste
            'similarity_threshold': 0.8,
            'cluster_min_size': 3,  # Menor para dataset pequeno
            'stratify_by': 'both',
            'protein_weight': 0.6,
            'ligand_weight': 0.4
        },
        'batch_size': 16,  # Menor batch para Mac M1
        'use_gpu': False,  # CPU no Mac
        'use_parallel': False,  # Desabilitar paralelização para debug
        
        # Modelos
        'protein_model': 'esm2_t6_8M_UR50D',  # Modelo MENOR (8M params, ~30MB)
        'ligand_model': 'SMI-TED',  # Modelo FM4M para SMILES
        
        # Paths
        'base_dir': str(output_dir),
        'ligand_output_dir': 'ligand_embeddings',
        'protein_output_dir': 'protein_embeddings',
        'matrix_output_dir': 'matrices',
        'concatenated_output_dir': 'concatenated',
    })
    
    print("✅ Configuração:")
    print(f"   - Modelo ESM: {config.get('protein_model')} (PEQUENO para teste rápido!)")
    print(f"   - Algoritmo de clustering: {config.get('stratification_params')['clustering_algorithm']}")
    print(f"   - Batch size: {config.batch_size}")
    print(f"   - GPU: {config.use_gpu}")
    print(f"   - Estratificação: {config.get('stratification_enabled')}")
    print()
    print("💡 NOTA: Usando modelo ESM menor (8M parâmetros, ~30MB)")
    print("         Para produção, usar modelo maior: esm2_t36_3B_UR50D")
    
    # Criar pipeline
    print("\n" + "=" * 80)
    print("🚀 ETAPA 3: EXECUTAR PIPELINE")
    print("=" * 80)
    print()
    
    print("⚠️  NOTA: Geração de embeddings pode demorar ~5-10 minutos no Mac M1")
    print("         Para dataset de 1000 amostras com ESM e FM4M")
    print()
    
    pipeline = BuildPipeline(config)
    
    try:
        # Executar pipeline completo
        success = pipeline.run_complete_pipeline(
            input_tsv_path=test_file,
            output_dir=output_dir,
            matrix_type='embedding',
            binary_threshold=1000.0,  # 1000 nM
            run_validation=True,
            stratify_splits=True,
            test_size=0.10,   # 10% test
            val_size=0.10     # 10% validation (80/10/10)
        )
        
        if success:
            print("\n" + "=" * 80)
            print("🎉 PIPELINE EXECUTADO COM SUCESSO!")
            print("=" * 80)
            print()
            
            # Mostrar resumo
            summary = pipeline.get_pipeline_summary()
            print("📊 RESUMO:")
            print(f"   - Etapas executadas: {summary['total_steps']}")
            print(f"   - Sucesso: {summary['success']}")
            print()
            
            # Verificar arquivos gerados
            print("📁 ARQUIVOS GERADOS:")
            
            files_to_check = [
                output_dir / "protein_embeddings",
                output_dir / "ligand_embeddings",
                output_dir / "concatenated_embeddings.npy",
                output_dir / "interaction_labels.npy",
                output_dir / "binary_labels.npy",
                output_dir / "splits" / "train_indices.npy",
                output_dir / "splits" / "val_indices.npy",
                output_dir / "splits" / "test_indices.npy",
                output_dir / "pipeline_results.json",
            ]
            
            for f in files_to_check:
                if f.exists():
                    if f.is_dir():
                        print(f"   ✅ {f.name}/ (diretório)")
                    else:
                        size = f.stat().st_size / (1024**2)
                        print(f"   ✅ {f.name} ({size:.2f} MB)")
                else:
                    print(f"   ❌ {f.name} (não encontrado)")
            
            # Ler e mostrar splits
            if (output_dir / "splits").exists():
                import numpy as np
                train_idx = np.load(output_dir / "splits" / "train_indices.npy")
                val_idx = np.load(output_dir / "splits" / "val_indices.npy")
                test_idx = np.load(output_dir / "splits" / "test_indices.npy")
                
                print()
                print("📊 SPLITS GERADOS:")
                print(f"   - Train: {len(train_idx)} amostras ({len(train_idx)/1000*100:.1f}%)")
                print(f"   - Val:   {len(val_idx)} amostras ({len(val_idx)/1000*100:.1f}%)")
                print(f"   - Test:  {len(test_idx)} amostras ({len(test_idx)/1000*100:.1f}%)")
                print(f"   - Total: {len(train_idx) + len(val_idx) + len(test_idx)}")
            
            print()
            print("✅ TESTE COMPLETO!")
            print()
            print("📋 Próximos passos:")
            print("   1. Revisar relatório em: test_output_small/pipeline_results.json")
            print("   2. Verificar qualidade dos splits")
            print("   3. Se tudo OK, executar com dataset completo")
            
            return True
        else:
            print("\n❌ Pipeline falhou!")
            return False
            
    except Exception as e:
        print(f"\n❌ ERRO durante execução: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_pipeline_small()
    
