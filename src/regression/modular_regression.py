#!/usr/bin/env python3
"""
Interface CLI Modular de Regressão - DockTKinase
=================================================

Interface de linha de comando 100% compatível com o pipeline
original, mas usando a implementação modular.

Uso:
    python modular_regression.py embeddings.npy targets.npy
    python modular_regression.py embeddings.npy targets.npy --models RandomForest XGBoost
    python modular_regression.py embeddings.npy targets.npy --output results/my_test
"""

import argparse
import sys
from pathlib import Path

# Adicionar src ao path se necessário
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from regression.modular_pipeline import RegressionPipeline, run_regression_pipeline


def parse_args():
    """Parse argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description='Pipeline Modular de Regressão - DockTKinase',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:

  # Treinar todos os modelos:
  python modular_regression.py embeddings.npy targets.npy
  
  # Treinar modelos específicos:
  python modular_regression.py embeddings.npy targets.npy \\
      --models RandomForest GradientBoosting XGBoost
  
  # Especificar diretório de saída:
  python modular_regression.py embeddings.npy targets.npy \\
      --output results/my_experiment
  
  # Configurar splits:
  python modular_regression.py embeddings.npy targets.npy \\
      --test-size 0.15 --val-size 0.15
  
  # Seed customizada:
  python modular_regression.py embeddings.npy targets.npy \\
      --random-state 123
  
Modelos disponíveis:
  - RandomForest
  - GradientBoosting
  - XGBoost (se instalado)
  - Ridge
  - Lasso
  - ElasticNet
  - SVR
  - KNN
  - DecisionTree
  - MLP
  - LightGBM (se instalado)
  - CatBoost (se instalado)
        """
    )
    
    # Argumentos posicionais
    parser.add_argument(
        'embeddings',
        type=str,
        help='Caminho para arquivo de embeddings (.npy ou .npz)'
    )
    
    parser.add_argument(
        'targets',
        type=str,
        help='Caminho para arquivo de targets (.npy)'
    )
    
    # Argumentos opcionais
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=None,
        help='Lista de modelos a treinar (padrão: todos)'
    )
    
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        default='results/regression',
        help='Diretório de saída para resultados (padrão: results/regression)'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proporção do conjunto de teste (padrão: 0.2)'
    )
    
    parser.add_argument(
        '--val-size',
        type=float,
        default=0.1,
        help='Proporção do conjunto de validação (padrão: 0.1)'
    )
    
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Seed para reprodutibilidade (padrão: 42)'
    )
    
    parser.add_argument(
        '--quiet',
        '-q',
        action='store_true',
        help='Modo silencioso (sem output verboso)'
    )
    
    return parser.parse_args()


def validate_args(args):
    """Validar argumentos."""
    # Validar arquivos de entrada
    embeddings_path = Path(args.embeddings)
    if not embeddings_path.exists():
        print(f"❌ Erro: Arquivo de embeddings não encontrado: {embeddings_path}")
        sys.exit(1)
    
    targets_path = Path(args.targets)
    if not targets_path.exists():
        print(f"❌ Erro: Arquivo de targets não encontrado: {targets_path}")
        sys.exit(1)
    
    # Validar splits
    if not 0 < args.test_size < 1:
        print(f"❌ Erro: test-size deve estar entre 0 e 1")
        sys.exit(1)
    
    if not 0 < args.val_size < 1:
        print(f"❌ Erro: val-size deve estar entre 0 e 1")
        sys.exit(1)
    
    if args.test_size + args.val_size >= 1:
        print(f"❌ Erro: test-size + val-size deve ser menor que 1")
        sys.exit(1)


def main():
    """Função principal."""
    # Parse argumentos
    args = parse_args()
    
    # Validar
    validate_args(args)
    
    # Configurar verbosidade
    verbose = not args.quiet
    
    if verbose:
        print("🚀 Pipeline Modular de Regressão - DockTKinase")
        print("=" * 70)
        print(f"   Embeddings: {args.embeddings}")
        print(f"   Targets: {args.targets}")
        print(f"   Output: {args.output}")
        if args.models:
            print(f"   Modelos: {', '.join(args.models)}")
        else:
            print(f"   Modelos: Todos disponíveis")
        print(f"   Test size: {args.test_size}")
        print(f"   Val size: {args.val_size}")
        print(f"   Random state: {args.random_state}")
        print("=" * 70)
        print()
    
    # Criar e executar pipeline
    try:
        pipeline = RegressionPipeline(
            embeddings_path=args.embeddings,
            targets_path=args.targets,
            output_dir=args.output,
            models_to_train=args.models,
            test_size=args.test_size,
            val_size=args.val_size,
            random_state=args.random_state,
            verbose=verbose
        )
        
        results = pipeline.run()
        
        if verbose:
            print(f"\n✅ Pipeline executado com sucesso!")
            print(f"   Resultados salvos em: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Erro ao executar pipeline: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
