#!/usr/bin/env python3
# ESM-2: python run_complete_pipeline.py --input tests/datasets/kinase_non_human_compounds.tsv --output results/esm2_15B_test --esm-model esm2_t48_15B_UR50D --seed 42
# ESM-C: python run_complete_pipeline.py --input tests/datasets/kinase_non_human_compounds.tsv --output results/esmc_600m_test --esm-model esmc-600m-2024-12 --seed 42
# OpenFold3: python run_complete_pipeline.py --input tests/datasets/kinase_non_human_compounds.tsv --output results/openfold3_test --esm-model openfold3 --seed 42
# OpenFold3: python run_complete_pipeline.py --input tests/datasets/kinase_non_human_compounds.tsv --output results/openfold3_test --esm-model openfold3 --seed 42

"""
DockTKinase - Pipeline Completo Integrado
=========================================

Pipeline end-to-end com checkpoints automáticos:
1. Build: Embeddings (ESM + FM4M) + Matrix
2. Classification: Multi-Model (10 modelos sklearn)
3. Regression: Multi-Model (10 modelos)

Uso:
    # Básico
    python run_complete_pipeline.py --input tests/datasets/kinase_non_human_compounds.tsv
    
    # Com opções
    python run_complete_pipeline.py \\
        --input tests/datasets/kinase_human_compounds.tsv \\
        --output results/my_experiment \\
        --device cuda \\
        --no-checkpoints
    
    # Apenas regressão (pular classificação)
    python run_complete_pipeline.py \\
        --input data.tsv \\
        --no-classification

Features:
    • Sistema de checkpoints automático (evita recálculo)
    • Suporta CPU e GPU (CUDA/MPS)
    • Multi-modelo para classificação E regressão
    • Validação robusta com 3 splits (train/val/test)
    • Métricas completas e visualizações
"""

import sys
import time
import argparse
from pathlib import Path

# Adicionar path do projeto
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

from integrated_pipeline import IntegratedPipeline, IntegratedConfig


def parse_args():
    """Parse argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description='DockTKinase - Pipeline Completo Integrado',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Teste rápido com dataset pequeno
  python run_complete_pipeline.py --input tests/datasets/kinase_non_human_compounds.tsv
  
  # Produção com dataset completo
  python run_complete_pipeline.py \\
      --input tests/datasets/kinase_all_compounds.tsv \\
      --output results/production_run \\
      --device cuda
  
  # Apenas regressão (pular classificação)
  python run_complete_pipeline.py \\
      --input data.tsv \\
      --no-classification
  
  # Modelos específicos
  python run_complete_pipeline.py \\
      --input data.tsv \\
      --classification-models RandomForest GradientBoosting \\
      --regression-models RandomForest XGBoost

Checkpoints:
  Por padrão, o pipeline salva checkpoints após cada fase.
  Em execuções subsequentes, fases completas são carregadas do cache.
  Use --no-checkpoints para forçar recálculo completo.

Dispositivos:
  - auto: Detecta automaticamente (GPU se disponível)
  - cpu: Força CPU
  - cuda: NVIDIA GPU (requer CUDA)
  - mps: Apple Silicon GPU (Mac M1/M2/M3)
        """
    )
    
    # Input/Output
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path para arquivo TSV de entrada'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='results/pipeline_output',
        help='Diretório para salvar resultados (default: results/pipeline_output)'
    )
    
    # Build (Fase 1)
    parser.add_argument(
        '--ligand-model',
        type=str,
        default='SMI-TED',
        choices=['SMI-TED'],
        help='Modelo FM4M para embeddings de ligantes (default: SMI-TED, 768-dim)'
    )
    
    parser.add_argument(
        '--esm-model',
        type=str,
        default='esm2_t6_8M_UR50D',
        choices=['esm2_t6_8M_UR50D', 'esm2_t12_35M_UR50D', 'esm2_t30_150M_UR50D',
                 'esm2_t33_650M_UR50D', 'esm2_t36_3B_UR50D', 'esm2_t48_15B_UR50D'],
        help='Modelo ESM para embeddings de proteínas (default: esm2_t6_8M_UR50D, 320-dim)'
    )
    
    parser.add_argument(
        '--esm-dim',
        type=int,
        default=None,
        help='Dimensão customizada para embeddings ESM (default: automática baseada no modelo)'
    )
    
    # Classification (Fase 2)
    parser.add_argument(
        '--no-classification',
        action='store_true',
        help='Pular fase de classificação'
    )
    
    parser.add_argument(
        '--classification-models',
        type=str,
        nargs='+',
        default=None,
        help='Modelos de classificação específicos (default: todos os 10)'
    )
    
    # Regression (Fase 3)
    parser.add_argument(
        '--no-regression',
        action='store_true',
        help='Pular fase de regressão'
    )
    
    parser.add_argument(
        '--regression-models',
        type=str,
        nargs='+',
        default=None,
        help='Modelos de regressão específicos (default: todos os 10)'
    )
    
    # Configurações gerais
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda', 'mps'],
        help='Dispositivo de computação (default: auto)'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proporção do conjunto de teste (default: 0.2 = 20%%)'
    )
    
    parser.add_argument(
        '--val-size',
        type=float,
        default=0.1,
        help='Proporção do conjunto de validação (default: 0.1 = 10%%)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed para reprodutibilidade (default: 42)'
    )
    
    parser.add_argument(
        '--no-checkpoints',
        action='store_true',
        help='Desabilitar sistema de checkpoints (forçar recálculo)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Modo silencioso (menos output)'
    )
    
    return parser.parse_args()


def main():
    """Executar pipeline completo."""
    args = parse_args()
    
    print('=' * 80)
    print('🧬 DOCKTKINASE - PIPELINE COMPLETO INTEGRADO')
    print('=' * 80)
    print()
    
    # Validar input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f'❌ Erro: Arquivo não encontrado: {args.input}')
        return 1
    
    print('📋 Configuração:')
    print(f'   Input:  {args.input}')
    print(f'   Output: {args.output}')
    print(f'   Ligand Model: {args.ligand_model} (768-dim)')
    
    # Determinar dimensão ESM
    esm_dims = {
        # ESM-2 models
        'esm2_t6_8M_UR50D': 320,
        'esm2_t12_35M_UR50D': 480,
        'esm2_t30_150M_UR50D': 640,
        'esm2_t33_650M_UR50D': 1280,
        'esm2_t36_3B_UR50D': 2560,
        'esm2_t48_15B_UR50D': 5120,
        # ESM-C models (ESM-3)
        'esmc-300m-2024-12': 960,
        'esmc-600m-2024-12': 1152,
        'esmc-6b-2024-12': 3072,
        # OpenFold models
        'openfold3': 384
    }
    
    # Usar dimensão customizada se fornecida, senão usar padrão do modelo
    if args.esm_dim is not None:
        esm_dim = args.esm_dim
        dim_source = 'customizada'
    else:
        esm_dim = esm_dims.get(args.esm_model, 320)
        dim_source = 'padrão'
    
    total_dim = 768 + esm_dim
    
    print(f'   ESM Model: {args.esm_model} ({esm_dim}-dim {dim_source})')
    print(f'   Total Embedding: {total_dim}-dim)')
    print(f'   Device: {args.device}')
    print(f'   Checkpoints: {"❌ Desabilitado" if args.no_checkpoints else "✅ Habilitado"}')
    print()
    
    # Determinar quais fases executar
    run_classification = not args.no_classification
    run_regression = not args.no_regression
    
    if not run_classification and not run_regression:
        print('❌ Erro: Pelo menos uma fase deve ser habilitada (classificação ou regressão)')
        return 1
    
    phases = []
    if run_classification:
        n_clf_models = len(args.classification_models) if args.classification_models else 10
        phases.append(f'Classification ({n_clf_models} modelos)')
    if run_regression:
        n_reg_models = len(args.regression_models) if args.regression_models else 10
        phases.append(f'Regression ({n_reg_models} modelos)')
    
    print(f'📊 Fases a executar:')
    print(f'   • Build (sempre necessário)')
    for phase in phases:
        print(f'   • {phase}')
    print()
    
    # Configuração do pipeline
    config = IntegratedConfig(
        # Input/Output
        input_tsv=str(input_path),
        output_dir=args.output,
        
        # Build (Fase 1)
        ligand_model=args.ligand_model,
        esm_model=args.esm_model,
        esm_dim=args.esm_dim,  # Dimensão customizada (None = usar padrão)
        
        # Classification (Fase 2)
        run_classification=run_classification,
        use_multi_model_classification=True,  # Sempre usar multi-modelo
        classification_models=args.classification_models,
        
        # Regression (Fase 3)
        run_regression=run_regression,
        regression_models=args.regression_models,
        
        # Geral
        device=args.device,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.seed,
        use_checkpoints=not args.no_checkpoints,
        verbose=not args.quiet
    )
    
    # Executar pipeline
    start_time = time.time()
    
    try:
        pipeline = IntegratedPipeline(config)
        results = pipeline.run()
        
        total_time = time.time() - start_time
        
        print()
        print('=' * 80)
        print('✅ PIPELINE COMPLETO - SUCESSO!')
        print('=' * 80)
        print(f'⏱️  Tempo Total: {total_time:.2f}s ({total_time/60:.2f} min)')
        print()
        
        # Resumo dos resultados
        if results.get('status') == 'completed':
            # Build
            if 'build' in results and results['build'].get('success'):
                build = results['build']
                print('📦 BUILD:')
                print(f'   ✅ Amostras processadas: {build.get("n_samples", 0):,}')
                print(f'   ✅ Dimensão final: {build.get("embedding_dim", 0)}')
                print()
            
            # Classification
            if run_classification and 'classifier' in results:
                clf = results['classifier']
                if clf.get('success'):
                    print('📊 CLASSIFICATION:')
                    print(f'   ✅ Modelos treinados: {clf.get("n_models_trained", 0)}')
                    if 'best_model' in clf:
                        print(f'   🏆 Melhor modelo: {clf["best_model"]}')
                        if 'best_metrics' in clf:
                            metrics = clf['best_metrics']
                            print(f'      ROC-AUC: {metrics.get("ROC_AUC", 0):.4f}')
                            print(f'      F1: {metrics.get("F1", 0):.4f}')
                            print(f'      Accuracy: {metrics.get("Accuracy", 0):.4f}')
                    print()
            
            # Regression
            if run_regression and 'regression' in results:
                reg = results['regression']
                if reg.get('success'):
                    print('📈 REGRESSION:')
                    print(f'   ✅ Modelos treinados: {reg.get("models_trained", 0)}')
                    if 'best_model' in reg:
                        print(f'   🏆 Melhor modelo: {reg["best_model"]}')
                        print(f'      MAE: {reg.get("best_mae", 0):.2f} nM')
                        print(f'      R²: {reg.get("best_r2", 0):.4f}')
                    print()
        
        print(f'📁 Resultados salvos em: {args.output}')
        print('=' * 80)
        
        return 0
        
    except Exception as e:
        print()
        print('=' * 80)
        print('❌ ERRO NO PIPELINE')
        print('=' * 80)
        print(f'Erro: {str(e)}')
        
        import traceback
        if not args.quiet:
            print()
            print('Traceback completo:')
            traceback.print_exc()
        
        return 1


if __name__ == '__main__':
    sys.exit(main())
