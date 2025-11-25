#!/usr/bin/env python3
"""
Teste dos 13 Modelos de Classificação - DockTKinase
====================================================

Valida a implementação dos 10 modelos base + XGBoost obrigatório + 2 opcionais.

Uso:
    # Teste rápido com dataset pequeno (100 amostras)
    python tests/test_13_models_classification.py --dataset small
    
    # Teste completo com dataset não-humano (15k amostras)
    python tests/test_13_models_classification.py --dataset full
    
    # Testar modelos específicos
    python tests/test_13_models_classification.py --models DecisionTree AdaBoost XGBoost

Features:
    • Valida todos os 13 modelos de classificação
    • Verifica XGBoost obrigatório
    • Compara performance entre modelos
    • Gera relatório completo com métricas
    • Salva resultados em tests/results/

Modelos testados:
    Base (10):
        1. RandomForest
        2. GradientBoosting
        3. LogisticRegression
        4. LinearSVC
        5. ExtraTrees
        6. KNN
        7. MLP
        8. NaiveBayes
        9. DecisionTree (NEW)
        10. AdaBoost (NEW)
    
    Gradient Boosting (3):
        11. XGBoost (OBRIGATÓRIO)
        12. LightGBM (opcional)
        13. CatBoost (opcional)
"""

import sys
import time
import argparse
import warnings
from pathlib import Path

# Adicionar path do projeto
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import numpy as np
import pandas as pd
from datetime import datetime


def parse_args():
    """Parse argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description='Teste dos 13 Modelos de Classificação',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='small',
        choices=['small', 'full'],
        help='Dataset para teste: small (100 amostras) ou full (15k amostras)'
    )
    
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=None,
        help='Modelos específicos para testar (default: todos os 13)'
    )
    
    parser.add_argument(
        '--esm-model',
        type=str,
        default='esm2_t6_8M_UR50D',
        help='Modelo ESM para embeddings (default: esm2_t6_8M_UR50D - mais rápido)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='tests/results/test_13_models',
        help='Diretório para salvar resultados'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Modo verbose (mais output)'
    )
    
    return parser.parse_args()


def validate_xgboost():
    """Valida se XGBoost está instalado."""
    try:
        import xgboost
        print(f'   ✅ XGBoost instalado: v{xgboost.__version__}')
        return True
    except ImportError:
        print('   ⚠️  XGBoost NÃO instalado!')
        print('   ⚠️  Instale com: pip install xgboost')
        return False


def check_optional_models():
    """Verifica modelos opcionais."""
    optional = {}
    
    try:
        import lightgbm
        optional['LightGBM'] = True
        print(f'   ✅ LightGBM instalado: v{lightgbm.__version__}')
    except ImportError:
        optional['LightGBM'] = False
        print('   ℹ️  LightGBM não instalado (opcional)')
    
    try:
        import catboost
        optional['CatBoost'] = True
        print(f'   ✅ CatBoost instalado: v{catboost.__version__}')
    except ImportError:
        optional['CatBoost'] = False
        print('   ℹ️  CatBoost não instalado (opcional)')
    
    return optional


def validate_models():
    """Valida disponibilidade dos modelos."""
    print('🔍 Validando disponibilidade dos modelos...')
    print()
    
    from classifier.models.classifiers import ClassificationModels
    
    # Obter todos os modelos
    all_models = ClassificationModels.get_all_models()
    all_model_names = list(all_models.keys())
    
    print(f'📊 Modelos disponíveis:')
    print(f'   Total: {len(all_model_names)}')
    print()
    
    # Validar 10 modelos base
    expected_base = [
        'RandomForest', 'GradientBoosting', 'LogisticRegression',
        'LinearSVC', 'ExtraTrees', 'KNN', 'MLP', 'NaiveBayes',
        'DecisionTree', 'AdaBoost'
    ]
    
    print('✅ Modelos Base (10 esperados):')
    for i, model in enumerate(expected_base, 1):
        status = '✅' if model in all_model_names else '❌'
        print(f'   {i:2d}. {status} {model}')
    
    missing_base = set(expected_base) - set(all_model_names)
    if missing_base:
        print(f'\n❌ ERRO: Faltam modelos base: {missing_base}')
        return False
    
    print()
    
    # Validar XGBoost obrigatório
    print('⚠️  XGBoost (OBRIGATÓRIO):')
    xgboost_ok = validate_xgboost()
    if not xgboost_ok:
        print('\n❌ ERRO: XGBoost é obrigatório e não está instalado!')
        return False
    
    print()
    
    # Validar modelos opcionais
    print('ℹ️  Modelos Opcionais:')
    optional = check_optional_models()
    
    print()
    
    # Resumo
    n_base = len(expected_base)
    total_available = n_base + (1 if xgboost_ok else 0) + sum(optional.values())
    print(f'📈 Resumo:')
    print(f'   ✅ 10 modelos base: OK')
    print(f'   ✅ XGBoost obrigatório: OK')
    print(f'   ℹ️  LightGBM: {"OK" if optional["LightGBM"] else "não instalado"}')
    print(f'   ℹ️  CatBoost: {"OK" if optional["CatBoost"] else "não instalado"}')
    print(f'   📊 Total disponível: {total_available}/13')
    
    return True


def run_pipeline(args):
    """Executa pipeline completo via subprocess (como nos testes de Boltz)."""
    import subprocess
    
    print()
    print('🚀 Executando pipeline completo...')
    print()
    
    # Determinar dataset
    if args.dataset == 'small':
        input_file = 'tests/datasets/kinase_test_small.tsv'
        expected_samples = 100
    else:
        input_file = 'tests/datasets/kinase_non_human_compounds.tsv'
        expected_samples = 15616
    
    # Verificar arquivo
    input_path = project_root / input_file
    if not input_path.exists():
        print(f'❌ Erro: Dataset não encontrado: {input_file}')
        return None
    
    print(f'📁 Dataset: {input_file}')
    print(f'   Amostras esperadas: ~{expected_samples:,}')
    print()
    
    # Configurar output
    output_dir = args.output
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = f'{output_dir}_{args.dataset}_{timestamp}'
    
    print(f'📂 Output: {output_path}')
    print()
    
    # Construir comando
    cmd = [
        'python',
        'run_complete_pipeline.py',
        '--input', str(input_path),
        '--output', output_path,
        '--esm-model', args.esm_model,
        '--ligand-model', 'SMI-TED',
        '--device', 'auto',
        '--no-regression',  # Pular regressão para teste rápido
        '--seed', str(args.seed)
    ]
    
    # Adicionar modelos específicos se fornecidos
    if args.models:
        cmd.extend(['--classification-models'] + args.models)
    
    # Modo silencioso (inverso de verbose)
    if not args.verbose:
        cmd.append('--quiet')
    
    print('📝 Comando:')
    print('  ' + ' '.join(cmd))
    print()
    
    start_time = time.time()
    
    try:
        # Executar comando
        result = subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=not args.verbose,
            text=True,
            check=False
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print()
            print(f'✅ Pipeline executado com sucesso!')
            print(f'⏱️  Tempo total: {elapsed:.2f}s ({elapsed/60:.2f} min)')
            return output_path
        else:
            print()
            print(f'❌ Pipeline falhou com código: {result.returncode}')
            if not args.verbose and result.stderr:
                print(f'Erro: {result.stderr[-500:]}')  # Últimos 500 chars
            return None
            
    except Exception as e:
        elapsed = time.time() - start_time
        print()
        print(f'❌ Erro ao executar pipeline: {e}')
        print(f'⏱️  Tempo decorrido: {elapsed:.2f}s')
        if args.verbose:
            import traceback
            traceback.print_exc()
        return None


def analyze_results(output_dir):
    """Analisa resultados do teste."""
    import json
    
    print()
    print('=' * 80)
    print('📊 ANÁLISE DOS RESULTADOS')
    print('=' * 80)
    print()
    
    results_path = Path(output_dir)
    if not results_path.exists():
        print(f'❌ Diretório de resultados não encontrado: {output_dir}')
        return
    
    # Procurar arquivo de métricas (novo formato JSON)
    metrics_file = results_path / 'classifier' / 'metrics' / 'test_metrics.json'
    
    if not metrics_file.exists():
        print(f'❌ Arquivo de métricas não encontrado: {metrics_file}')
        return
    
    print(f'📁 Arquivo de métricas: {metrics_file.relative_to(results_path.parent)}')
    print()
    
    # Carregar métricas JSON
    with open(metrics_file) as f:
        metrics_data = json.load(f)
    
    # Converter para DataFrame para análise
    models_data = []
    for model_name, model_metrics in metrics_data.items():
        if isinstance(model_metrics, dict):
            models_data.append({
                'Model': model_name,
                'ROC_AUC': model_metrics.get('ROC_AUC', 0),  # Maiúscula
                'F1': model_metrics.get('F1', 0),            # Maiúscula
                'Accuracy': model_metrics.get('Accuracy', 0),
                'Precision': model_metrics.get('Precision', 0),
                'Recall': model_metrics.get('Recall', 0)
            })
    
    df = pd.DataFrame(models_data)
    
    # Verificar modelos treinados
    models_trained = df['Model'].tolist()
    n_models = len(models_trained)
    
    print(f'📊 Modelos Treinados: {n_models}')
    print()
    
    # Tabela de performance
    print('🏆 Performance (ordenado por ROC-AUC):')
    print()
    
    # Ordenar por ROC-AUC
    df_sorted = df.sort_values('ROC_AUC', ascending=False)
    
    # Cabeçalho
    print(f'{"Rank":<6} {"Modelo":<20} {"ROC-AUC":<10} {"F1":<10} {"Accuracy":<10} {"Precision":<10} {"Recall":<10}')
    print('-' * 80)
    
    # Linhas
    for i, row in enumerate(df_sorted.itertuples(), 1):
        medal = '🥇' if i == 1 else '🥈' if i == 2 else '🥉' if i == 3 else f'{i}.'
        print(f'{medal:<6} {row.Model:<20} {row.ROC_AUC:<10.4f} {row.F1:<10.4f} {row.Accuracy:<10.4f} {row.Precision:<10.4f} {row.Recall:<10.4f}')
    
    print()
    
    # Estatísticas
    print('📈 Estatísticas:')
    print(f'   Melhor modelo: {df_sorted.iloc[0]["Model"]} (ROC-AUC: {df_sorted.iloc[0]["ROC_AUC"]:.4f})')
    print(f'   ROC-AUC médio: {df["ROC_AUC"].mean():.4f} ± {df["ROC_AUC"].std():.4f}')
    print(f'   F1 médio: {df["F1"].mean():.4f} ± {df["F1"].std():.4f}')
    print(f'   Accuracy média: {df["Accuracy"].mean():.4f} ± {df["Accuracy"].std():.4f}')
    print()
    
    # Seção de novos modelos e XGBoost removida - informação já disponível na tabela de ranking
    
    # Comparação de tempos (se disponível no pipeline_stats.json)
    stats_file = results_path / 'classifier' / 'pipeline_stats.json'
    if stats_file.exists():
        with open(stats_file) as f:
            stats = json.load(f)
        
        if 'training_times' in stats:
            times = stats['training_times']
            print('⏱️  Tempos de Treinamento:')
            sorted_times = sorted(times.items(), key=lambda x: x[1])
            for model, time_sec in sorted_times:
                print(f'   {model:<20} {time_sec:>8.2f}s')
            
            total_time = sum(times.values())
            print()
            print(f'   Total: {total_time:.2f}s ({total_time/60:.2f} min)')
    
    print()
    print('=' * 80)


def main():
    """Main."""
    args = parse_args()
    
    print('=' * 80)
    print('🧪 TESTE DOS 13 MODELOS DE CLASSIFICAÇÃO - DOCKTKINASE')
    print('=' * 80)
    print()
    
    print(f'⚙️  Configuração:')
    print(f'   Dataset: {args.dataset}')
    print(f'   ESM Model: {args.esm_model}')
    print(f'   Output: {args.output}')
    print(f'   Seed: {args.seed}')
    if args.models:
        print(f'   Modelos específicos: {", ".join(args.models)}')
    else:
        print(f'   Modelos: TODOS (13)')
    print()
    
    # Fase 1: Validar modelos
    print('=' * 80)
    print('FASE 1: VALIDAÇÃO DOS MODELOS')
    print('=' * 80)
    print()
    
    if not validate_models():
        print()
        print('❌ Validação falhou! Instale os pacotes necessários.')
        return 1
    
    print()
    print('✅ Validação completa!')
    print()
    
    # Fase 2: Executar pipeline
    print('=' * 80)
    print('FASE 2: EXECUTAR PIPELINE COMPLETO')
    print('=' * 80)
    
    output_dir = run_pipeline(args)
    if not output_dir:
        print()
        print('❌ Pipeline falhou!')
        return 1
    
    # Fase 3: Analisar resultados
    analyze_results(output_dir)
    
    print()
    print('=' * 80)
    print('✅ TESTE COMPLETO!')
    print('=' * 80)
    print(f'📁 Resultados salvos em: {output_dir}')
    print('=' * 80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
