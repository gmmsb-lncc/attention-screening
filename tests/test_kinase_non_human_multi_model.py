#!/usr/bin/env python3
"""
Teste do Pipeline Multi-Modelo com Dataset Kinase Non-Human
===========================================================

Executa o pipeline completo (Build + Classification Multi-Model + Regression)
com o dataset kinase_non_human_compounds.tsv
"""

import sys
import time
from pathlib import Path

# Adicionar path do projeto
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from integrated_pipeline import IntegratedPipeline, IntegratedConfig


def main():
    """Executar pipeline completo com multi-modelo de classificação."""
    
    print('=' * 80)
    print('🧪 TESTE: Pipeline Multi-Modelo com Dataset Kinase Non-Human')
    print('=' * 80)
    print()
    
    # Configuração
    config = IntegratedConfig(
        # Input
        input_tsv='tests/datasets/kinase_non_human_compounds.tsv',
        output_dir='results/kinase_non_human_multi_model',
        
        # Build (Fase 1)
        ligand_model='SMI-TED',           # FM4M: 768-dim
        esm_model='esm2_t6_8M_UR50D',     # ESM: 320-dim (rápido para teste)
        
        # Classification (Fase 2) - MULTI-MODELO
        use_multi_model_classification=True,  # NOVO: usar multi-modelo
        classification_models=None,            # None = todos os 10 modelos
        
        # Regression (Fase 3) - Multi-modelo (já existente)
        regression_models=None,  # None = todos os 11 modelos
        
        # Geral
        device='cpu',
        random_state=42,
        verbose=True
    )
    
    print('📋 Configuração:')
    print(f'   Dataset: kinase_non_human_compounds.tsv')
    print(f'   Embeddings: FM4M (768) + ESM (320) = 1088 dim')
    print(f'   Classificação: 10 modelos')
    print(f'   Regressão: 11 modelos')
    print(f'   Device: CPU')
    print()
    
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
        
        # Resultados de Classificação
        if 'classification' in results:
            class_results = results['classification']
            print('📊 CLASSIFICAÇÃO (Multi-Modelo):')
            
            if 'test_metrics' in class_results:
                # Ordenar por ROC-AUC
                sorted_models = sorted(
                    class_results['test_metrics'].items(),
                    key=lambda x: x[1].get('ROC_AUC', 0),
                    reverse=True
                )
                
                print(f'   Modelos treinados: {len(sorted_models)}')
                print(f'\n   Top 5:')
                for i, (model_name, metrics) in enumerate(sorted_models[:5], 1):
                    print(f'   {i}. {model_name:<20} ROC-AUC: {metrics.get("ROC_AUC", 0):.4f}  '
                          f'F1: {metrics.get("F1", 0):.4f}  '
                          f'Acc: {metrics.get("Accuracy", 0):.4f}')
                
                # Melhor modelo
                best_model = sorted_models[0]
                print(f'\n   🏆 Melhor: {best_model[0]}')
                print(f'      ROC-AUC: {best_model[1].get("ROC_AUC", 0):.4f}')
        
        print()
        
        # Resultados de Regressão
        if 'regression' in results:
            reg_results = results['regression']
            print('📈 REGRESSÃO (Multi-Modelo):')
            
            if 'test_metrics' in reg_results:
                # Ordenar por MAE
                sorted_models = sorted(
                    reg_results['test_metrics'].items(),
                    key=lambda x: x[1].get('MAE', float('inf'))
                )
                
                print(f'   Modelos treinados: {len(sorted_models)}')
                print(f'\n   Top 5:')
                for i, (model_name, metrics) in enumerate(sorted_models[:5], 1):
                    print(f'   {i}. {model_name:<20} MAE: {metrics.get("MAE", 0):.4f}  '
                          f'RMSE: {metrics.get("RMSE", 0):.4f}  '
                          f'R²: {metrics.get("R2", 0):.4f}')
                
                # Melhor modelo
                best_model = sorted_models[0]
                print(f'\n   🏆 Melhor: {best_model[0]}')
                print(f'      MAE: {best_model[1].get("MAE", 0):.4f} (erro ~{10**best_model[1].get("MAE", 0):.1f}x em IC50)')
        
        print()
        print('📁 Resultados salvos em: results/kinase_non_human_multi_model/')
        print('=' * 80)
        
        return 0
        
    except Exception as e:
        print()
        print('=' * 80)
        print('❌ ERRO NO PIPELINE')
        print('=' * 80)
        print(f'Erro: {str(e)}')
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    pass  # main() already tested
