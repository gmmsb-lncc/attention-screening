#!/usr/bin/env python3
"""
Script para visualização de resultados da CNN+Atenção Cruzada.

Visualiza métricas de classificação e regressão dos diferentes modelos de proteína.
"""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Adicionar path para imports
sys.path.insert(0, str(Path(__file__).parent))

from visualization.basic_plots import BasicPlotter
from visualization.advanced_plots import AdvancedPlotter


def load_cnn_attention_results(files):
    """Carrega resultados da CNN+Atenção Cruzada."""
    results = {}
    
    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extrair nome do modelo baseado no arquivo ou config
            model_name = Path(file_path).stem.replace('results_', '')
            
            # Estruturar dados no formato esperado
            results[model_name] = {
                'config': data['config'],
                'classification': {
                    'best_metrics': {
                        'Accuracy': data['metrics']['classification']['accuracy'],
                        'ROC_AUC': data['metrics']['classification']['roc_auc'],
                        'F1': data['metrics']['classification']['f1'],
                        'MCC': data['metrics']['classification']['mcc']
                    }
                },
                'regression': {
                    'best_metrics': {
                        'Pearson_R': data['metrics']['regression']['pearson_r'],
                        'Pearson_P': data['metrics']['regression']['pearson_p'],
                        'Spearman_R': data['metrics']['regression']['spearman_r'],
                        'R2': data['metrics']['regression']['r2'],
                        'RMSE': data['metrics']['regression']['rmse'],
                        'MAE': data['metrics']['regression']['mae']
                    }
                },
                'training': data.get('training', {})
            }
            
            print(f"✓ {model_name}")
            
        except Exception as e:
            print(f"⚠️  Erro ao carregar {file_path}: {e}")
    
    return results


def extract_metrics(results):
    """Extrai métricas em formato tabular."""
    classification_data = []
    regression_data = []
    embedding_data = []
    
    for model_name, data in results.items():
        # Classificação
        class_metrics = data['classification']['best_metrics']
        classification_data.append({
            'Model': model_name,
            'Accuracy': class_metrics.get('Accuracy', 0),
            'F1': class_metrics.get('F1', 0),
            'ROC_AUC': class_metrics.get('ROC_AUC', 0),
            'MCC': class_metrics.get('MCC', 0)
        })
        
        # Regressão
        reg_metrics = data['regression']['best_metrics']
        regression_data.append({
            'Model': model_name,
            'Pearson_R': reg_metrics.get('Pearson_R', 0),
            'Pearson_P': reg_metrics.get('Pearson_P', 0),
            'Spearman_R': reg_metrics.get('Spearman_R', 0),
            'R2': reg_metrics.get('R2', 0),
            'RMSE': reg_metrics.get('RMSE', 0),
            'MAE': reg_metrics.get('MAE', 0)
        })
        
        # Dimensões
        config = data['config']
        embedding_data.append({
            'Model': model_name,
            'Protein_Dim': config.get('protein_dim', 0),
            'Ligand_Dim': config.get('ligand_dim', 0),
            'Total_Dim': config.get('protein_dim', 0) + config.get('ligand_dim', 0)
        })
    
    return classification_data, regression_data, embedding_data


def calculate_overall_score(class_data, reg_data):
    """Calcula score geral combinando classificação e regressão.
    
    IMPORTANTE: Valores negativos em Pearson_R e R2 são VÁLIDOS e indicam
    correlação negativa ou modelo pior que baseline. NÃO forçar para 0!
    """
    # Métricas de classificação (0-1, quanto maior melhor)
    f1 = class_data['F1']
    roc_auc = class_data['ROC_AUC']
    accuracy = class_data['Accuracy']
    mcc = class_data['MCC']
    
    # Métricas de regressão (podem ser negativas!)
    pearson_r = reg_data['Pearson_R']
    r2 = reg_data['R2']
    
    # Score de classificação (média de 4 métricas, escala 0-100)
    class_score = (f1 + roc_auc + accuracy + mcc) / 4 * 100
    
    # Score de regressão: transformar para 0-100 considerando negativos
    # Pearson: -1 a 1 -> 0 a 100
    pearson_score = (pearson_r + 1) / 2 * 100
    # R2: pode ser muito negativo, limitar a -1 para visualização
    r2_score = (max(-1, r2) + 1) / 2 * 100
    reg_score = (pearson_score + r2_score) / 2
    
    # Score geral (média ponderada 50/50)
    overall = (class_score * 0.5 + reg_score * 0.5)
    
    return overall


def create_training_curves(results, output_dir):
    """Cria gráficos de curvas de treinamento."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_models = len(results)
    fig, axes = plt.subplots(2, n_models, figsize=(5*n_models, 10))
    
    if n_models == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, (model_name, data) in enumerate(results.items()):
        training = data.get('training', {})
        train_loss = training.get('train_loss', [])
        val_loss = training.get('val_loss', [])
        
        if not train_loss:
            continue
        
        epochs = range(1, len(train_loss) + 1)
        
        # Loss curves
        axes[0, idx].plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
        axes[0, idx].plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2)
        axes[0, idx].set_title(f'{model_name}\nLoss Curves', fontweight='bold')
        axes[0, idx].set_xlabel('Epoch')
        axes[0, idx].set_ylabel('Loss')
        axes[0, idx].legend()
        axes[0, idx].grid(alpha=0.3)
        
        # Loss difference
        if len(val_loss) == len(train_loss):
            diff = np.array(val_loss) - np.array(train_loss)
            axes[1, idx].plot(epochs, diff, 'g-', linewidth=2)
            axes[1, idx].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[1, idx].set_title('Overfitting (Val - Train)', fontweight='bold')
            axes[1, idx].set_xlabel('Epoch')
            axes[1, idx].set_ylabel('Loss Difference')
            axes[1, idx].grid(alpha=0.3)
            axes[1, idx].fill_between(epochs, 0, diff, where=(diff > 0), 
                                     color='red', alpha=0.3, label='Overfitting')
            axes[1, idx].fill_between(epochs, 0, diff, where=(diff <= 0), 
                                     color='green', alpha=0.3, label='Good fit')
            axes[1, idx].legend()
    
    plt.tight_layout()
    output_path = output_dir / 'training_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(
        description='Visualização de resultados CNN+Atenção Cruzada'
    )
    parser.add_argument(
        '--files',
        nargs='+',
        required=True,
        help='Arquivos JSON com resultados'
    )
    parser.add_argument(
        '--output',
        default='results/cnn_attention_analysis',
        help='Diretório de saída'
    )
    
    args = parser.parse_args()
    
    print("🔬 Análise CNN+Atenção Cruzada")
    print("=" * 50)
    
    # 1. Carregar dados
    print("\n📂 Carregando resultados...")
    results = load_cnn_attention_results(args.files)
    
    if not results:
        print("❌ Nenhum resultado válido carregado!")
        return 1
    
    print(f"✅ {len(results)} modelos carregados")
    
    # 2. Extrair métricas
    print("\n📊 Extraindo métricas...")
    classification_data, regression_data, embedding_data = extract_metrics(results)
    
    # 3. Calcular scores gerais
    overall_data = []
    for i in range(len(classification_data)):
        model_name = classification_data[i]['Model']
        overall_score = calculate_overall_score(
            classification_data[i],
            regression_data[i]
        )
        overall_data.append({
            'Model': model_name,
            'Overall_Score': overall_score,
            'Total_Dim': embedding_data[i]['Total_Dim']
        })
    
    print(f"✅ Métricas extraídas")
    
    # 4. Criar visualizações básicas
    print("\n📈 Gerando visualizações básicas...")
    output_dir = Path(args.output)
    basic_plotter = BasicPlotter(output_dir)
    
    plot_files = []
    
    # Plot 1: Classificação
    print("  → Criando comparação de classificação...")
    path = basic_plotter.plot_classification_comparison(classification_data)
    plot_files.append(path)
    print(f"    ✓ {path.name}")
    
    # Plot 2: Regressão
    print("  → Criando comparação de regressão...")
    path = basic_plotter.plot_regression_comparison(regression_data)
    plot_files.append(path)
    print(f"    ✓ {path.name}")
    
    # Plot 3: Dimensões
    print("  → Criando visualização de dimensões...")
    path = basic_plotter.plot_embedding_dimensions(embedding_data)
    plot_files.append(path)
    print(f"    ✓ {path.name}")
    
    # Plot 4: Ranking
    print("  → Criando ranking geral...")
    path = basic_plotter.plot_overall_ranking(overall_data)
    plot_files.append(path)
    print(f"    ✓ {path.name}")
    
    # 5. Visualizações avançadas
    print("\n📈 Gerando visualizações avançadas...")
    advanced_plotter = AdvancedPlotter(output_dir)
    
    # Plot 5: Radar
    print("  → Criando radar chart...")
    path = advanced_plotter.create_radar_chart(classification_data)
    plot_files.append(path)
    print(f"    ✓ {path.name}")
    
    # Plot 6: Heatmap
    print("  → Criando heatmap...")
    path = advanced_plotter.create_heatmap(classification_data, regression_data)
    plot_files.append(path)
    print(f"    ✓ {path.name}")
    
    # Plot 7: Pareto
    print("  → Criando gráfico de Pareto...")
    path = advanced_plotter.create_pareto_chart(overall_data)
    plot_files.append(path)
    print(f"    ✓ {path.name}")
    
    # Plot 8: Heatmap de correlação de regressão
    print("  → Criando heatmap de correlação (regressão)...")
    path = advanced_plotter.create_regression_correlation_heatmap(regression_data)
    plot_files.append(path)
    print(f"    ✓ {path.name}")
    
    # Plot 9: Training curves
    print("  → Criando curvas de treinamento...")
    path = create_training_curves(results, output_dir)
    plot_files.append(path)
    print(f"    ✓ {path.name}")
    
    # 6. Salvar CSVs
    print("\n💾 Salvando dados em CSV...")
    df_class = pd.DataFrame(classification_data)
    df_reg = pd.DataFrame(regression_data)
    df_overall = pd.DataFrame(overall_data).sort_values('Overall_Score', ascending=False)
    
    df_class.to_csv(output_dir / 'classification_metrics.csv', index=False)
    df_reg.to_csv(output_dir / 'regression_metrics.csv', index=False)
    df_overall.to_csv(output_dir / 'overall_scores.csv', index=False)
    
    # Resumo final
    print("\n" + "=" * 50)
    print("✅ Análise CNN+Atenção Cruzada concluída!")
    print(f"\n📁 Resultados salvos em: {output_dir}")
    print(f"📊 {len(plot_files)} visualizações geradas")
    print(f"📄 3 arquivos CSV gerados")
    
    # Insights com análise de qualidade
    print("\n💡 Análise de Performance:")
    
    if len(overall_data) > 1:
        best = max(overall_data, key=lambda x: x['Overall_Score'])
        print(f"  🏆 Maior score geral: {best['Model']} (Score: {best['Overall_Score']:.2f})")
    else:
        best = overall_data[0]
        print(f"  📊 Modelo analisado: {best['Model']} (Score: {best['Overall_Score']:.2f})")
    
    # Classificação com avaliação de qualidade
    best_class = max(classification_data, key=lambda x: x['F1'])
    f1_val = best_class['F1']
    f1_quality = "Excelente" if f1_val > 0.8 else "Bom" if f1_val > 0.6 else "Regular" if f1_val > 0.4 else "Fraco" if f1_val > 0.2 else "Muito Fraco"
    print(f"  🎯 Classificação: {best_class['Model']} | F1={f1_val:.3f} ({f1_quality}) | Acc={best_class['Accuracy']:.3f} | ROC-AUC={best_class['ROC_AUC']:.3f}")
    
    # Regressão com interpretação correta de valores negativos
    best_reg = max(regression_data, key=lambda x: x['Pearson_R'])
    pearson_val = best_reg['Pearson_R']
    r2_val = best_reg['R2']
    
    # Interpretação de Pearson
    if pearson_val > 0.7:
        pearson_quality = "Forte positiva"
    elif pearson_val > 0.3:
        pearson_quality = "Moderada positiva"
    elif pearson_val > -0.3:
        pearson_quality = "Fraca/Ausente"
    elif pearson_val > -0.7:
        pearson_quality = "Moderada negativa"
    else:
        pearson_quality = "Forte negativa"
    
    # Interpretação de R2
    if r2_val > 0.7:
        r2_quality = "Excelente"
    elif r2_val > 0.4:
        r2_quality = "Bom"
    elif r2_val > 0:
        r2_quality = "Fraco"
    else:
        r2_quality = "Pior que baseline"
    
    print(f"  📈 Regressão: {best_reg['Model']} | Pearson={pearson_val:.3f} ({pearson_quality}) | R²={r2_val:.3f} ({r2_quality}) | RMSE={best_reg['RMSE']:.3f}")
    
    # Aviso se performance geral é ruim
    if best['Overall_Score'] < 40:
        print("\n  ⚠️  ATENÇÃO: Performance geral está ABAIXO do esperado!")
        print("     Considere: revisar arquitetura, aumentar epochs, ajustar hiperparâmetros")
    elif best['Overall_Score'] < 60:
        print("\n  ⚡ Performance moderada. Há espaço para melhorias.")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
