"""
Módulo para geração de plotagens básicas de comparação.

Princípios aplicados:
- Single Responsibility: cada função cria um tipo específico de gráfico
- DRY: código reutilizável para formatação
- KISS: lógica de plotagem simples e direta
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict


class BasicPlotter:
    """Gera visualizações básicas de comparação de modelos."""
    
    def __init__(self, output_dir: Path):
        """
        Inicializa o plotador básico.
        
        Args:
            output_dir: Diretório para salvar os gráficos
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Resetar matplotlib units registry (bug com categorical axes)
        from matplotlib import units
        units.registry.clear()
        
        self._configure_style()
    
    @staticmethod
    def _configure_style():
        """Configura o estilo dos gráficos."""
        import matplotlib
        matplotlib.use('Agg')  # Backend sem GUI
        plt.close('all')  # Limpar todas as figuras
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
    
    def plot_classification_comparison(self, data: List[Dict]) -> Path:
        """
        Cria gráfico de comparação de métricas de classificação.
        
        Args:
            data: Lista de métricas por modelo
            
        Returns:
            Caminho do arquivo salvo
        """
        df = pd.DataFrame(data)
        
        # Métricas para plotar
        # Selecionar métricas de classificação
        metrics = ['Accuracy', 'F1', 'ROC_AUC', 'MCC']
        df_plot = df[['Model'] + [m for m in metrics if m in df.columns]]
        
        # Criar subplots 2x2 - uma métrica por subplot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.subplots_adjust(hspace=0.35, wspace=0.3)
        fig.suptitle('Comparação de Classificadores', fontsize=16, fontweight='bold', y=0.995)
        
        # Informações das métricas
        metric_info = {
            'Accuracy': {'title': 'Acurácia Geral', 'color': 'Blues', 'desc': '% de predições corretas'},
            'F1': {'title': 'F1-Score', 'color': 'Greens', 'desc': 'Média harmônica Precisão/Recall'},
            'ROC_AUC': {'title': 'ROC-AUC', 'color': 'Purples', 'desc': 'Área sob curva ROC'},
            'MCC': {'title': 'Matthews Correlation Coefficient', 'color': 'Oranges', 'desc': 'Correlação entre predito e real'}
        }
        
        for idx, metric in enumerate(metrics):
            if metric not in df_plot.columns:
                continue
            
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]
            
            # Plotar barras usando índices numéricos
            x_pos = np.arange(len(df_plot))
            values = df_plot[metric].values
            bars = ax.bar(x_pos, values,
                          color=sns.color_palette(metric_info[metric]['color'], len(df_plot)),
                          alpha=0.8, edgecolor='black')
            
            # Destacar melhor modelo
            best_idx = values.argmax()
            bars[best_idx].set_color('gold')
            bars[best_idx].set_edgecolor('darkorange')
            bars[best_idx].set_linewidth(2)
            
            # Adicionar valores nas barras
            for i, (bar, val) in enumerate(zip(bars, values)):
                height = bar.get_height()
                fontweight = 'bold' if i == best_idx else 'normal'
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{val:.3f}',
                       ha='center', va='bottom', fontsize=10, fontweight=fontweight)
            
            # Configurar eixos
            ax.set_title(f"{metric_info[metric]['title']}\n({metric_info[metric]['desc']})", 
                        fontweight='bold', fontsize=12, pad=15)
            ax.set_ylabel('Score', fontweight='bold', fontsize=11)
            
            # Ajustar ylim baseado na métrica (MCC pode ser negativo)
            if metric == 'MCC':
                y_min = min(values) - 0.1 if min(values) < 0 else -0.05
                y_max = max(values) + 0.15
                ax.axhline(y=0, color='red', linestyle='--', alpha=0.3, linewidth=1)
                ax.set_ylim(y_min, y_max)
            else:
                ax.set_ylim(0, min(1.05, max(values) + 0.15))
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels(df_plot['Model'].values, rotation=45, ha='right', fontsize=10)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            # Adicionar linhas de referência
            if metric in ['Accuracy', 'F1', 'ROC_AUC']:
                ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.3, linewidth=1.5, label='Excelente (0.9)')
                ax.axhline(y=0.7, color='orange', linestyle='--', alpha=0.3, linewidth=1.5, label='Bom (0.7)')
                ax.legend(loc='lower right', fontsize=8)
        
        plt.tight_layout()
        
        # Salvar
        output_path = self.output_dir / 'classification_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close('all')  # Limpar todas as figuras
        
        # CRÍTICO: Limpar registry de units para evitar conflitos com próximo plot
        # O plot de classificação registra StrCategoryConverter que interfere
        from matplotlib import units
        units.registry.clear()
        
        return output_path
    
    def plot_regression_comparison(self, data: List[Dict]) -> Path:
        """
        Cria gráfico de comparação de métricas de regressão.
        Foco em Pearson R, P-value e RMSE.
        
        Args:
            data: Lista de métricas por modelo
            
        Returns:
            Caminho do arquivo salvo
        """
        # CRÍTICO: Limpar registry ANTES de criar figura
        # O plot anterior pode ter registrado conversores de string
        from matplotlib import units
        units.registry.clear()
        plt.close('all')
        
        df = pd.DataFrame(data)
        
        # Validar dados
        if df.empty or 'Model' not in df.columns:
            raise ValueError("Dados de regressão vazios ou inválidos")
        
        # Criar figura com tamanho maior e espaçamento adequado
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.subplots_adjust(hspace=0.35, wspace=0.3)
        fig.suptitle('Comparação de Regressores', fontsize=16, fontweight='bold', y=0.995)
        
        # Preparar posições
        n_models = len(df)
        x_pos = np.arange(n_models)
        model_labels = df['Model'].tolist()
        
        # 1. Pearson R (correlation strength)
        ax1 = axes[0, 0]
        pearson_r_values = df['Pearson_R'].tolist()
        bars1 = ax1.bar(x_pos, pearson_r_values,
                        color=sns.color_palette('RdYlGn', n_models), alpha=0.8, edgecolor='black')
        
        for i, (bar, val) in enumerate(zip(bars1, pearson_r_values)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax1.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, linewidth=2, label='Forte (0.7)')
        ax1.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, linewidth=2, label='Moderada (0.5)')
        ax1.set_title('Pearson R (Força da Correlação)', fontweight='bold', fontsize=13, pad=15)
        ax1.set_ylabel('Coeficiente', fontweight='bold', fontsize=11)
        ax1.legend(loc='upper right', fontsize=9)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(model_labels, rotation=45, ha='right', fontsize=10)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        ax1.set_ylim(bottom=min(pearson_r_values) - 0.1, top=max(pearson_r_values) + 0.15)
        ax1.set_title('Pearson R (Força da Correlação)', fontweight='bold', fontsize=13, pad=15)
        ax1.set_ylabel('Coeficiente', fontweight='bold', fontsize=11)
        ax1.legend(loc='upper right', fontsize=9)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(model_labels, rotation=45, ha='right', fontsize=10)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        ax1.set_ylim(bottom=min(pearson_r_values) - 0.1, top=max(pearson_r_values) + 0.15)
        
        # 2. Pearson P-value (significance)
        ax2 = axes[0, 1]
        pearson_p_values = df['Pearson_P'].tolist()
        # Substituir zeros muito pequenos por valor mínimo visível
        pearson_p_display = [max(p, 1e-320) for p in pearson_p_values]
        bars2 = ax2.bar(x_pos, pearson_p_display,
                        color=sns.color_palette('viridis', n_models), alpha=0.8, edgecolor='black')
        
        for i, (bar, val) in enumerate(zip(bars2, pearson_p_values)):
            height = bar.get_height()
            if val == 0:
                label = '< 1e-300'
            else:
                label = f'{val:.2e}'
            ax2.text(bar.get_x() + bar.get_width()/2., height * 2,
                    label, ha='center', va='bottom', fontsize=8, rotation=45)
        
        ax2.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, linewidth=2, label='α=0.05')
        ax2.axhline(y=0.01, color='darkred', linestyle='--', alpha=0.5, linewidth=2, label='α=0.01')
        ax2.set_title('Pearson P-value (Significância)', fontweight='bold', fontsize=13, pad=15)
        ax2.set_ylabel('P-value (escala log)', fontweight='bold', fontsize=11)
        ax2.set_yscale('log')
        ax2.legend(loc='upper right', fontsize=9)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(model_labels, rotation=45, ha='right', fontsize=10)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        # 3. RMSE (prediction error)
        ax3 = axes[1, 0]
        rmse_values = df['RMSE'].tolist()
        bars3 = ax3.bar(x_pos, rmse_values,
                        color=sns.color_palette('Reds_r', n_models), alpha=0.8, edgecolor='black')
        
        for i, (bar, val) in enumerate(zip(bars3, rmse_values)):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax3.set_title('RMSE (Erro de Predição - menor é melhor)', fontweight='bold', fontsize=13, pad=15)
        ax3.set_ylabel('RMSE', fontweight='bold', fontsize=11)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(model_labels, rotation=45, ha='right', fontsize=10)
        ax3.grid(axis='y', alpha=0.3, linestyle='--')
        ax3.set_ylim(bottom=0, top=max(rmse_values) + 0.2)
        
        # 4. R² (reference)
        ax4 = axes[1, 1]
        r2_values = df['R2'].tolist()
        bars4 = ax4.bar(x_pos, r2_values,
                        color=sns.color_palette('Blues', n_models), alpha=0.8, edgecolor='black')
        
        for i, (bar, val) in enumerate(zip(bars4, r2_values)):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax4.axhline(y=0, color='red', linestyle='-', alpha=0.3, linewidth=1)
        ax4.set_title('R² (Variância Explicada)', fontweight='bold', fontsize=13, pad=15)
        ax4.set_ylabel('R²', fontweight='bold', fontsize=11)
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(model_labels, rotation=45, ha='right', fontsize=10)
        ax4.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Ajustar limites para R2 (pode ser negativo)
        r2_min = min(r2_values)
        r2_max = max(r2_values)
        ax4.set_ylim(bottom=r2_min - 0.1 if r2_min < 0 else -0.05, top=r2_max + 0.15)
        
        plt.tight_layout()
        
        # Salvar
        output_path = self.output_dir / 'regression_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close('all')
        
        # Limpar registry após usar labels de string
        from matplotlib import units
        units.registry.clear()
        
        return output_path
    
    def plot_embedding_dimensions(self, data: List[Dict]) -> Path:
        """
        Cria gráfico de dimensões de embeddings.
        
        Args:
            data: Lista de informações de embedding por modelo
            
        Returns:
            Caminho do arquivo salvo
        """
        # Limpar registry antes
        from matplotlib import units
        units.registry.clear()
        plt.close('all')
        
        df = pd.DataFrame(data)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = range(len(df))
        width = 0.35
        
        # Barras empilhadas
        bars1 = ax.bar(x, df['Protein_Dim'], width, label='Proteína', 
                       color='steelblue')
        bars2 = ax.bar(x, df['Ligand_Dim'], width, bottom=df['Protein_Dim'],
                       label='Ligante', color='coral')
        
        # Adicionar valores totais
        for i, (protein, ligand) in enumerate(zip(df['Protein_Dim'], df['Ligand_Dim'])):
            total = protein + ligand
            ax.text(i, total, f'{total}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Modelo')
        ax.set_ylabel('Dimensão do Embedding')
        ax.set_title('Dimensões de Embeddings por Modelo', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df['Model'], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Salvar
        output_path = self.output_dir / 'embedding_dimensions.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close('all')
        
        # Limpar registry
        from matplotlib import units
        units.registry.clear()
        
        return output_path
    
    def plot_overall_ranking(self, data: List[Dict]) -> Path:
        """
        Cria gráfico de ranking geral dos modelos.
        
        Args:
            data: Lista com scores gerais por modelo
            
        Returns:
            Caminho do arquivo salvo
        """
        # Limpar registry antes
        from matplotlib import units
        units.registry.clear()
        plt.close('all')
        
        df = pd.DataFrame(data)
        df = df.sort_values('Overall_Score', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = sns.color_palette('RdYlGn', len(df))
        
        # Usar índices numéricos para evitar problemas com strings
        y_pos = np.arange(len(df))
        bars = ax.barh(y_pos, df['Overall_Score'].values, color=colors)
        
        # Adicionar valores
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f'{width:.1f}', ha='left', va='center', fontsize=10,
                   fontweight='bold')
        
        ax.set_xlabel('Score Geral', fontweight='bold')
        ax.set_title('Ranking Geral de Modelos', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 100)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df['Model'].values)
        
        plt.tight_layout()
        
        # Salvar
        output_path = self.output_dir / 'overall_ranking.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close('all')
        
        # Limpar registry
        from matplotlib import units
        units.registry.clear()
        
        return output_path
