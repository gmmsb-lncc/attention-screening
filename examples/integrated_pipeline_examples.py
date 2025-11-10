#!/usr/bin/env python3
"""
Exemplo de uso do pipeline integrado.

Este script demonstra como usar o IntegratedPipeline para
executar o workflow completo end-to-end.
"""

import sys
from pathlib import Path

# Adicionar path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / 'src'))

from integrated_pipeline import IntegratedPipeline, IntegratedConfig


def example_complete_workflow():
    """
    Exemplo 1: Workflow completo (build + classification + regression).
    """
    print("\n" + "="*80)
    print("EXEMPLO 1: Workflow Completo")
    print("="*80)
    
    config = IntegratedConfig(
        input_tsv="data/kinase_data.tsv",
        output_dir="results/example_complete",
        esm_model="esm2_t6_8M_UR50D",
        device="cpu",
        run_classification=True,
        run_regression=True,
        classifier_epochs=50,
        classifier_cv_folds=5,
        regression_models=['Ridge', 'Lasso', 'ElasticNet', 'RandomForest', 'XGBoost'],
        regression_cv_folds=5,
        random_state=42,
        verbose=True
    )
    
    pipeline = IntegratedPipeline(config)
    results = pipeline.run()
    
    # Acessar resultados
    print("\n📊 RESULTADOS:")
    print(f"Status: {results['status']}")
    print(f"Tempo total: {results['total_time_seconds']:.2f}s")
    
    if results['status'] == 'completed':
        print("\n🧠 Classification:")
        print(f"  ROC-AUC: {results['classifier']['test_metrics']['roc_auc']:.4f}")
        print(f"  Accuracy: {results['classifier']['test_metrics']['accuracy']:.4f}")
        
        print("\n📈 Regression:")
        print(f"  Best Model: {results['regression']['best_model']}")
        print(f"  Best MAE: {results['regression']['best_mae']:.3f}")
        print(f"  Best R²: {results['regression']['best_r2']:.4f}")


def example_build_only():
    """
    Exemplo 2: Apenas build (gerar embeddings).
    """
    print("\n" + "="*80)
    print("EXEMPLO 2: Build Only (Embeddings)")
    print("="*80)
    
    config = IntegratedConfig(
        input_tsv="data/kinase_data.tsv",
        output_dir="results/example_build_only",
        esm_model="esm2_t6_8M_UR50D",
        device="cpu",
        run_classification=False,
        run_regression=False,
        verbose=True
    )
    
    pipeline = IntegratedPipeline(config)
    results = pipeline.run()
    
    print("\n📊 RESULTADOS:")
    print(f"Status: {results['status']}")
    print(f"Embeddings salvos em: {results['build']['embeddings']['concatenated']}")


def example_classification_only():
    """
    Exemplo 3: Build + Classification (sem regression).
    """
    print("\n" + "="*80)
    print("EXEMPLO 3: Build + Classification")
    print("="*80)
    
    config = IntegratedConfig(
        input_tsv="data/kinase_data.tsv",
        output_dir="results/example_classification",
        esm_model="esm2_t6_8M_UR50D",
        device="cpu",
        run_classification=True,
        run_regression=False,
        classifier_epochs=30,
        classifier_cv_folds=3,
        verbose=True
    )
    
    pipeline = IntegratedPipeline(config)
    results = pipeline.run()
    
    print("\n📊 RESULTADOS:")
    print(f"ROC-AUC: {results['classifier']['test_metrics']['roc_auc']:.4f}")
    print(f"CV ROC-AUC: {results['classifier']['cv_results']['mean_roc_auc']:.4f} ± {results['classifier']['cv_results']['std_roc_auc']:.4f}")


def example_regression_only():
    """
    Exemplo 4: Build + Regression (sem classification).
    """
    print("\n" + "="*80)
    print("EXEMPLO 4: Build + Regression")
    print("="*80)
    
    config = IntegratedConfig(
        input_tsv="data/kinase_data.tsv",
        output_dir="results/example_regression",
        esm_model="esm2_t6_8M_UR50D",
        device="cpu",
        run_classification=False,
        run_regression=True,
        regression_models=['Ridge', 'Lasso', 'RandomForest'],
        regression_cv_folds=3,
        verbose=True
    )
    
    pipeline = IntegratedPipeline(config)
    results = pipeline.run()
    
    print("\n📊 RESULTADOS:")
    print(f"Best Model: {results['regression']['best_model']}")
    print(f"MAE: {results['regression']['best_mae']:.3f}")
    print(f"R²: {results['regression']['best_r2']:.4f}")
    
    # Métricas individuais
    print("\n📊 Modelos individuais:")
    for model_name, metrics in results['regression']['individual_results'].items():
        print(f"  {model_name}:")
        print(f"    MAE: {metrics['mae']:.3f}")
        print(f"    R²: {metrics['r2']:.4f}")


def example_custom_config():
    """
    Exemplo 5: Configuração customizada avançada.
    """
    print("\n" + "="*80)
    print("EXEMPLO 5: Configuração Customizada")
    print("="*80)
    
    config = IntegratedConfig(
        input_tsv="data/kinase_data.tsv",
        output_dir="results/example_custom",
        
        # Build customizado
        esm_model="esm2_t12_35M_UR50D",  # Modelo maior
        ligand_model="smi-ted-large",
        batch_size=16,
        device="cpu",
        
        # Data split customizado
        test_size=0.25,
        val_size=0.15,
        random_state=123,
        
        # Classification customizado
        run_classification=True,
        classifier_epochs=100,
        classifier_cv_folds=10,
        
        # Regression customizado
        run_regression=True,
        regression_models=['Ridge', 'Lasso', 'ElasticNet', 'SVR', 'XGBoost', 'LightGBM'],
        regression_cv_folds=10,
        
        # Threshold customizado
        binary_threshold=500.0,  # 500 nM
        
        verbose=True
    )
    
    pipeline = IntegratedPipeline(config)
    results = pipeline.run()
    
    print("\n📊 RESULTADOS CUSTOMIZADOS:")
    print(f"Status: {results['status']}")
    print(f"Tempo total: {results['total_time_seconds']:.2f}s")


def example_from_dict():
    """
    Exemplo 6: Criar pipeline a partir de dict.
    """
    print("\n" + "="*80)
    print("EXEMPLO 6: Pipeline a partir de Dict")
    print("="*80)
    
    # Configuração como dict (útil para carregar de JSON/YAML)
    config_dict = {
        'input_tsv': 'data/kinase_data.tsv',
        'output_dir': 'results/example_from_dict',
        'esm_model': 'esm2_t6_8M_UR50D',
        'device': 'cpu',
        'run_classification': True,
        'run_regression': True,
        'classifier_epochs': 50,
        'regression_models': ['Ridge', 'XGBoost'],
        'verbose': True
    }
    
    pipeline = IntegratedPipeline(config_dict)
    results = pipeline.run()
    
    print("\n📊 RESULTADOS:")
    print(f"Status: {results['status']}")


def example_gpu_accelerated():
    """
    Exemplo 7: Pipeline acelerado por GPU (se disponível).
    """
    print("\n" + "="*80)
    print("EXEMPLO 7: GPU Accelerated Pipeline")
    print("="*80)
    
    import torch
    
    # Detectar device disponível
    if torch.cuda.is_available():
        device = "cuda"
        print("✅ CUDA GPU detectada")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("✅ Apple MPS (Metal) detectada")
    else:
        device = "cpu"
        print("⚠️ GPU não disponível, usando CPU")
    
    config = IntegratedConfig(
        input_tsv="data/kinase_data.tsv",
        output_dir=f"results/example_gpu_{device}",
        esm_model="esm2_t33_650M_UR50D",  # Modelo maior (aproveita GPU)
        device=device,
        batch_size=32 if device != "cpu" else 8,  # Batch maior para GPU
        run_classification=True,
        run_regression=True,
        verbose=True
    )
    
    pipeline = IntegratedPipeline(config)
    results = pipeline.run()
    
    print(f"\n📊 RESULTADOS ({device.upper()}):")
    print(f"Tempo total: {results['total_time_seconds']:.2f}s")


def main():
    """Menu interativo de exemplos."""
    examples = {
        '1': ('Workflow Completo', example_complete_workflow),
        '2': ('Build Only', example_build_only),
        '3': ('Build + Classification', example_classification_only),
        '4': ('Build + Regression', example_regression_only),
        '5': ('Configuração Customizada', example_custom_config),
        '6': ('Pipeline from Dict', example_from_dict),
        '7': ('GPU Accelerated', example_gpu_accelerated)
    }
    
    print("\n" + "="*80)
    print(" " * 20 + "🧬 INTEGRATED PIPELINE EXAMPLES 🧬")
    print("="*80)
    print("\nEscolha um exemplo:")
    
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")
    
    print("  0. Executar todos os exemplos")
    print("  q. Sair")
    
    choice = input("\nOpção: ").strip()
    
    if choice == '0':
        for key in sorted(examples.keys()):
            _, func = examples[key]
            func()
    elif choice in examples:
        _, func = examples[choice]
        func()
    elif choice.lower() == 'q':
        print("Saindo...")
    else:
        print("Opção inválida!")


if __name__ == '__main__':
    # Verificar se argumento foi passado
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        examples = {
            '1': example_complete_workflow,
            '2': example_build_only,
            '3': example_classification_only,
            '4': example_regression_only,
            '5': example_custom_config,
            '6': example_from_dict,
            '7': example_gpu_accelerated
        }
        
        if example_num in examples:
            examples[example_num]()
        else:
            print(f"Exemplo {example_num} não encontrado!")
    else:
        main()
