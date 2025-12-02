#!/usr/bin/env python3
"""
Demonstração completa: Processamento do dataset kinase_non_human_compounds.tsv
================================================================================

Este script demonstra como o IntegratedPipeline processa o dataset de teste
através das três fases principais:
1. Build (Embeddings + Matrix)
2. Classification (Ativo/Inativo)
3. Regression (Predição de valores pKi/pKd/pIC50)

Dataset: tests/datasets/kinase_non_human_compounds.tsv
- 15,617 compostos ligante-proteína
- Múltiplas kinases de organismos não-humanos
- Valores de atividade: Ki, Kd, IC50

Autor: DockTKinase Team
Data: Novembro 2025
"""

import os
import sys
from pathlib import Path

# =============================================================================
# CRITICAL: Pre-import ESM before any other imports to avoid segfault
# The local ESM must be loaded before any other module imports it differently
# =============================================================================
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root / "llm" / "ESM"))
try:
    import esm as _esm_preload
    print(f"[ESM PRE-LOAD] ✅ ESM loaded from: {_esm_preload.__file__}")
except ImportError:
    print("[ESM PRE-LOAD] ⚠️ Failed to pre-load ESM, will try later")

# Now add src to path
sys.path.insert(0, str(_project_root))

from src.integrated_pipeline import IntegratedPipeline, IntegratedConfig


def analyze_dataset():
    """Análise preliminar do dataset."""
    print("=" * 80)
    print("ANÁLISE DO DATASET: kinase_non_human_compounds.tsv")
    print("=" * 80)
    
    dataset_path = "tests/datasets/kinase_non_human_compounds.tsv"
    
    import pandas as pd
    df = pd.read_csv(dataset_path, sep='\t')
    
    print(f"\n📊 Estatísticas Gerais:")
    print(f"  • Total de registros: {len(df):,}")
    print(f"  • Compostos únicos: {df['canonical_smiles'].nunique():,}")
    print(f"  • Proteínas únicas: {df['seq'].nunique():,}")
    print(f"  • Kinases únicas: {df['target_kinase'].nunique():,}")
    print(f"  • Organismos: {df['organism'].nunique():,}")
    
    print(f"\n🧬 Top 5 Organismos:")
    for org, count in df['organism'].value_counts().head(5).items():
        print(f"  • {org}: {count:,} compostos")
    
    print(f"\n🎯 Top 5 Kinases:")
    for kinase, count in df['target_kinase'].value_counts().head(5).items():
        print(f"  • {kinase}: {count:,} compostos")
    
    print(f"\n📈 Distribuição de Tipos de Medida:")
    for mtype, count in df['standard_type'].value_counts().items():
        print(f"  • {mtype}: {count:,} ({count/len(df)*100:.1f}%)")
    
    # Análise de atividade
    print(f"\n⚡ Análise de Atividade (para classificação):")
    active_threshold = 1000  # nM
    df['activity_nM'] = df['standard_value']
    active_count = (df['activity_nM'] <= active_threshold).sum()
    inactive_count = (df['activity_nM'] > active_threshold).sum()
    
    print(f"  • Limiar: {active_threshold} nM")
    print(f"  • Ativos (≤ {active_threshold} nM): {active_count:,} ({active_count/len(df)*100:.1f}%)")
    print(f"  • Inativos (> {active_threshold} nM): {inactive_count:,} ({inactive_count/len(df)*100:.1f}%)")
    
    # Análise de valores contínuos
    print(f"\n📊 Valores Contínuos (para regressão):")
    print(f"  • Média: {df['standard_value'].mean():.2f} nM")
    print(f"  • Mediana: {df['standard_value'].median():.2f} nM")
    print(f"  • Min: {df['standard_value'].min():.2f} nM")
    print(f"  • Max: {df['standard_value'].max():.2f} nM")
    
    # pChEMBL values
    if 'pchembl_value' in df.columns:
        pchembl_available = df['pchembl_value'].notna().sum()
        print(f"\n🔬 Valores pChEMBL:")
        print(f"  • Disponíveis: {pchembl_available:,} ({pchembl_available/len(df)*100:.1f}%)")
        if pchembl_available > 0:
            print(f"  • Média: {df['pchembl_value'].mean():.2f}")
            print(f"  • Mediana: {df['pchembl_value'].median():.2f}")


def run_integrated_pipeline():
    """Executa o pipeline integrado completo."""
    print("\n" + "=" * 80)
    print("EXECUÇÃO DO INTEGRATED PIPELINE")
    print("=" * 80)
    
    # Configuration
    config = IntegratedConfig(
        input_tsv="tests/datasets/kinase_non_human_compounds.tsv",
        output_dir="results/demo_kinase_non_human",
        
        # Build settings
        esm_model="esm2_t6_8M_UR50D",  # Small model for demonstration
        device="cpu",  # Use CPU (MPS causes segfault with fair-esm)
        
        # Classification settings
        run_classification=True,
        use_multi_model_classification=True,  # Use sklearn models instead of MLP (avoids PySpark)
        classification_models=['RandomForest', 'GradientBoosting', 'LogisticRegression', 'SVM'],  # Top 4 sklearn models
        binary_threshold=1000.0,  # 1000 nM
        
        # Regression settings
        run_regression=True,
        regression_models=['Ridge', 'Lasso', 'RandomForest', 'XGBoost'],  # Top 4 models
        
        # General settings
        random_state=42
    )
    
    print(f"\n⚙️  Configuration:")
    print(f"  • Input: {config.input_tsv}")
    print(f"  • Output: {config.output_dir}")
    print(f"  • ESM Model: {config.esm_model}")
    print(f"  • Device: {config.device}")
    print(f"  • Classification: {'✅ Enabled (Multi-Model)' if config.run_classification else '❌ Disabled'}")
    print(f"  • Classification Models: {', '.join(config.classification_models)}")
    print(f"  • Regression: {'✅ Enabled' if config.run_regression else '❌ Disabled'}")
    print(f"  • Binary Threshold: {config.binary_threshold} nM")
    print(f"  • Regression Models: {', '.join(config.regression_models)}")
    
    # Inicializar pipeline
    print(f"\n🚀 Inicializando IntegratedPipeline...")
    pipeline = IntegratedPipeline(config)
    
    # Executar pipeline completo
    print(f"\n▶️  Executando pipeline completo...")
    print(f"    Isso pode levar alguns minutos dependendo do tamanho do dataset...")
    
    results = pipeline.run()
    
    return results


def display_results(results):
    """Exibe os resultados de forma estruturada."""
    print("\n" + "=" * 80)
    print("RESULTADOS DO PIPELINE")
    print("=" * 80)
    
    # 1. Build Results
    print("\n" + "─" * 80)
    print("1️⃣  FASE BUILD: Geração de Embeddings e Matrizes")
    print("─" * 80)
    
    build_results = results.get('build', {})
    if build_results.get('status') == 'success':
        print("✅ Status: Sucesso")
        
        paths = build_results.get('paths', {})
        print(f"\n📁 Arquivos Gerados:")
        print(f"  • Embedding Matrix: {paths.get('embedding_matrix', 'N/A')}")
        print(f"  • Binary Labels: {paths.get('binary_labels', 'N/A')}")
        print(f"  • Continuous Labels: {paths.get('continuous_labels', 'N/A')}")
        
        stats = build_results.get('statistics', {})
        if stats:
            print(f"\n📊 Estatísticas:")
            print(f"  • Total de amostras: {stats.get('total_samples', 'N/A')}")
            print(f"  • Dimensão dos embeddings: {stats.get('embedding_dim', 'N/A')}")
            print(f"  • Ligantes únicos: {stats.get('unique_ligands', 'N/A')}")
            print(f"  • Proteínas únicas: {stats.get('unique_proteins', 'N/A')}")
    else:
        print(f"❌ Status: Falha - {build_results.get('error', 'Unknown error')}")
    
    # 2. Classification Results
    print("\n" + "─" * 80)
    print("2️⃣  FASE CLASSIFICATION: Predição Ativo/Inativo")
    print("─" * 80)
    
    clf_results = results.get('classifier', {})
    if clf_results.get('status') == 'success':
        print("✅ Status: Sucesso")
        
        test_metrics = clf_results.get('test_metrics', {})
        if test_metrics:
            print(f"\n📈 Métricas de Teste:")
            print(f"  • ROC-AUC: {test_metrics.get('roc_auc', 0):.4f}")
            print(f"  • Accuracy: {test_metrics.get('accuracy', 0):.4f}")
            print(f"  • Precision: {test_metrics.get('precision', 0):.4f}")
            print(f"  • Recall: {test_metrics.get('recall', 0):.4f}")
            print(f"  • F1-Score: {test_metrics.get('f1', 0):.4f}")
        
        val_metrics = clf_results.get('validation_metrics', {})
        if val_metrics:
            print(f"\n🔍 Métricas de Validação:")
            print(f"  • ROC-AUC: {val_metrics.get('roc_auc', 0):.4f}")
            print(f"  • Accuracy: {val_metrics.get('accuracy', 0):.4f}")
        
        paths = clf_results.get('paths', {})
        if paths:
            print(f"\n📁 Arquivos Gerados:")
            print(f"  • Modelo: {paths.get('model', 'N/A')}")
            print(f"  • Métricas: {paths.get('metrics', 'N/A')}")
            print(f"  • Plots: {paths.get('plots_dir', 'N/A')}")
    else:
        print(f"❌ Status: Falha - {clf_results.get('error', 'Unknown error')}")
    
    # 3. Regression Results
    print("\n" + "─" * 80)
    print("3️⃣  FASE REGRESSION: Predição Quantitativa")
    print("─" * 80)
    
    reg_results = results.get('regression', {})
    if reg_results.get('status') == 'success':
        print("✅ Status: Sucesso")
        
        best_model = reg_results.get('best_model', 'N/A')
        best_mae = reg_results.get('best_mae', float('inf'))
        best_rmse = reg_results.get('best_rmse', float('inf'))
        best_r2 = reg_results.get('best_r2', 0)
        
        print(f"\n🏆 Melhor Modelo: {best_model}")
        print(f"  • MAE: {best_mae:.4f}")
        print(f"  • RMSE: {best_rmse:.4f}")
        print(f"  • R²: {best_r2:.4f}")
        
        model_results = reg_results.get('model_results', {})
        if model_results:
            print(f"\n📊 Comparação de Modelos:")
            print(f"{'Modelo':<20} {'MAE':>10} {'RMSE':>10} {'R²':>10}")
            print("─" * 54)
            
            for model_name, metrics in sorted(model_results.items(), 
                                             key=lambda x: x[1].get('test_mae', float('inf'))):
                mae = metrics.get('test_mae', 0)
                rmse = metrics.get('test_rmse', 0)
                r2 = metrics.get('test_r2', 0)
                marker = "🏆" if model_name == best_model else "  "
                print(f"{marker} {model_name:<18} {mae:>10.4f} {rmse:>10.4f} {r2:>10.4f}")
        
        paths = reg_results.get('paths', {})
        if paths:
            print(f"\n📁 Arquivos Gerados:")
            print(f"  • Modelos: {paths.get('models_dir', 'N/A')}")
            print(f"  • Predições: {paths.get('predictions_dir', 'N/A')}")
            print(f"  • Métricas: {paths.get('metrics_dir', 'N/A')}")
            print(f"  • Visualizações: {paths.get('visualizations_dir', 'N/A')}")
    else:
        print(f"❌ Status: Falha - {reg_results.get('error', 'Unknown error')}")
    
    # 4. Summary
    print("\n" + "=" * 80)
    print("RESUMO FINAL")
    print("=" * 80)
    
    success_count = sum([
        results.get('build', {}).get('status') == 'success',
        results.get('classifier', {}).get('status') == 'success',
        results.get('regression', {}).get('status') == 'success'
    ])
    
    print(f"\n✅ Fases completadas com sucesso: {success_count}/3")
    
    if success_count == 3:
        print(f"\n🎉 Pipeline executado com sucesso!")
        print(f"\n📊 Principais Resultados:")
        print(f"  • Classification ROC-AUC: {clf_results.get('test_metrics', {}).get('roc_auc', 0):.4f}")
        print(f"  • Best Regression Model: {reg_results.get('best_model', 'N/A')}")
        print(f"  • Best Regression MAE: {reg_results.get('best_mae', 0):.4f}")
        
        output_dir = results.get('output_dir', 'N/A')
        print(f"\n📁 Todos os resultados salvos em: {output_dir}")
    else:
        print(f"\n⚠️  Algumas fases falharam. Verifique os logs acima.")


def main():
    """Função principal."""
    print("\n" + "🔬" * 40)
    print("DEMO: Pipeline Integrado - Kinase Non-Human Compounds")
    print("🔬" * 40)
    
    try:
        # Passo 1: Analisar dataset
        analyze_dataset()
        
        # Passo 2: Executar pipeline
        print("\n" + "⏱️ " * 40)
        print("INICIANDO PROCESSAMENTO...")
        print("⏱️ " * 40)
        
        results = run_integrated_pipeline()
        
        # Passo 3: Exibir resultados
        display_results(results)
        
        print("\n" + "✅" * 40)
        print("DEMONSTRAÇÃO CONCLUÍDA!")
        print("✅" * 40)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Execução interrompida pelo usuário.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Erro durante execução: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
