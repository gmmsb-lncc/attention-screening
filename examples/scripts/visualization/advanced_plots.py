"""
Módulo para geração de visualizações avançadas.

Princípios aplicados:
- Single Responsibility: cada classe/método tem uma responsabilidade clara
- Open/Closed: fácil adicionar novos tipos de visualização
- KISS: lógica clara e objetiva
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple


class AdvancedPlotter:
    """Gera visualizações avançadas para análise científica."""
    
    def __init__(self, output_dir: Path):
        """
        Inicializa o plotador avançado.
        
        Args:
            output_dir: Diretório para salvar os gráficos
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Resetar matplotlib units registry
        from matplotlib import units
        units.registry.clear()
        
        self._configure_style()
    
    @staticmethod
    def _configure_style():
        """Configura o estilo dos gráficos."""
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
    
    def create_radar_chart(self, classification_data: List[Dict]) -> Path:
        """
        Cria gráfico radar para métricas de classificação.
        
        Args:
            classification_data: Lista de métricas de classificação
            
        Returns:
            Caminho do arquivo salvo
        """
        df = pd.DataFrame(classification_data)
        metrics = ['Accuracy', 'F1', 'ROC_AUC', 'Precision', 'Recall', 'MCC']
        
        # Preparar dados
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Fechar o círculo
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Plotar cada modelo
        colors = sns.color_palette('husl', len(df))
        for idx, row in df.iterrows():
            values = [row.get(m, 0) for m in metrics]
            # Normalizar MCC de [-1,1] para [0,1]
            values[5] = (values[5] + 1) / 2
            values += values[:1]  # Fechar o círculo
            
            ax.plot(angles, values, 'o-', linewidth=2, 
                   label=row['Model'], color=colors[idx])
            ax.fill(angles, values, alpha=0.15, color=colors[idx])
        
        # Configurar eixos
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, size=11)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_title('Radar de Métricas de Classificação', 
                    size=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'radar_classification.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Limpar registry
        from matplotlib import units
        units.registry.clear()
        
        return output_path
    
    def create_heatmap(self, classification_data: List[Dict], 
                      regression_data: List[Dict]) -> Path:
        """
        Cria heatmap de métricas normalizadas.
        
        Args:
            classification_data: Métricas de classificação
            regression_data: Métricas de regressão
            
        Returns:
            Caminho do arquivo salvo
        """
        # Combinar dados
        df_class = pd.DataFrame(classification_data).set_index('Model')
        df_reg = pd.DataFrame(regression_data).set_index('Model')
        
        # Selecionar métricas principais
        class_metrics = ['Accuracy', 'F1', 'ROC_AUC', 'MCC']
        reg_metrics = ['Pearson_R', 'R2']
        
        df_combined = pd.concat([
            df_class[class_metrics],
            df_reg[reg_metrics]
        ], axis=1)
        
        # Normalizar para [0, 1]
        df_normalized = df_combined.copy()
        # MCC: [-1, 1] -> [0, 1]
        if 'MCC' in df_normalized.columns:
            df_normalized['MCC'] = (df_normalized['MCC'] + 1) / 2
        
        # Criar heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(df_normalized.T, annot=True, fmt='.3f', cmap='RdYlGn',
                   center=0.5, vmin=0, vmax=1, cbar_kws={'label': 'Score Normalizado'},
                   linewidths=0.5, ax=ax)
        
        ax.set_title('Heatmap de Performance dos Modelos', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Modelo', fontweight='bold')
        ax.set_ylabel('Métrica', fontweight='bold')
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'heatmap_performance.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Limpar registry
        from matplotlib import units
        units.registry.clear()
        
        return output_path
    
    def create_tradeoff_scatter(self, classification_data: List[Dict],
                               regression_data: List[Dict]) -> Path:
        """
        Cria scatter plots de trade-offs entre métricas.
        
        Args:
            classification_data: Métricas de classificação
            regression_data: Métricas de regressão
            
        Returns:
            Caminho do arquivo salvo
        """
        df_class = pd.DataFrame(classification_data)
        df_reg = pd.DataFrame(regression_data)
        df = pd.merge(df_class, df_reg, on='Model')
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('Análise de Trade-offs', fontsize=16, fontweight='bold')
        
        # 1. Precision vs Recall
        self._create_single_scatter(axes[0, 0], df, 'Recall', 'Precision',
                                   'Precision vs Recall (Classificação)')
        
        # 2. F1 vs ROC-AUC
        self._create_single_scatter(axes[0, 1], df, 'ROC_AUC', 'F1',
                                   'F1 vs ROC-AUC (Classificação)')
        
        # 3. Pearson R vs RMSE
        self._create_single_scatter(axes[1, 0], df, 'Pearson_R', 'RMSE',
                                   'Correlação vs Erro (Regressão)',
                                   invert_y=True)
        
        # 4. Pearson R vs R²
        self._create_single_scatter(axes[1, 1], df, 'Pearson_R', 'R2',
                                   'Pearson R vs R² (Regressão)')
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'tradeoff_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Limpar registry
        from matplotlib import units
        units.registry.clear()
        
        return output_path
    
    def _create_single_scatter(self, ax, df: pd.DataFrame, 
                              x_col: str, y_col: str, 
                              title: str, invert_y: bool = False):
        """
        Cria um único scatter plot.
        
        Args:
            ax: Eixo matplotlib
            df: DataFrame com dados
            x_col: Nome da coluna X
            y_col: Nome da coluna Y
            title: Título do gráfico
            invert_y: Se deve inverter o eixo Y
        """
        colors = sns.color_palette('husl', len(df))
        
        for idx, row in df.iterrows():
            ax.scatter(row[x_col], row[y_col], s=200, 
                      color=colors[idx], alpha=0.7, edgecolors='black')
            ax.annotate(row['Model'], (row[x_col], row[y_col]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.8)
        
        ax.set_xlabel(x_col, fontweight='bold')
        ax.set_ylabel(y_col, fontweight='bold')
        ax.set_title(title, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        if invert_y:
            ax.invert_yaxis()
    
    def create_pareto_chart(self, data: List[Dict]) -> Path:
        """
        Cria gráfico de Pareto melhorado para eficiência vs performance.
        
        Args:
            data: Dados com dimensões e scores
            
        Returns:
            Caminho do arquivo salvo
        """
        df = pd.DataFrame(data)
        df = df.sort_values('Overall_Score', ascending=False)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Subplot 1: Barras com score geral
        x = np.arange(len(df))
        bars = ax1.barh(x, df['Overall_Score'], color='steelblue', alpha=0.8, edgecolor='navy')
        
        # Colorir barra do melhor modelo
        bars[0].set_color('gold')
        bars[0].set_edgecolor('darkorange')
        bars[0].set_linewidth(2)
        
        ax1.set_yticks(x)
        ax1.set_yticklabels(df['Model'], fontsize=10)
        ax1.set_xlabel('Score Geral', fontweight='bold', fontsize=12)
        ax1.set_title('Ranking de Performance', fontsize=13, fontweight='bold', pad=15)
        ax1.grid(axis='x', alpha=0.3, linestyle='--')
        ax1.invert_yaxis()
        
        # Adicionar valores nas barras
        for i, (idx, row) in enumerate(df.iterrows()):
            ax1.text(row['Overall_Score'] + 1, i, f"{row['Overall_Score']:.2f}",
                    va='center', fontsize=9, fontweight='bold')
        
        # Subplot 2: Scatter plot - Pareto frontier
        # Normalizar dimensões para melhor visualização
        max_dim = df['Total_Dim'].max() if df['Total_Dim'].max() > 0 else 1
        sizes = 300 - (df['Total_Dim'] / max_dim * 200) if max_dim > 0 else [300] * len(df)
        
        scatter = ax2.scatter(df['Total_Dim'], df['Overall_Score'], 
                            s=sizes, c=df['Overall_Score'], 
                            cmap='RdYlGn', alpha=0.7, edgecolors='black', linewidth=2)
        
        # Destacar melhor modelo
        best_idx = df['Overall_Score'].idxmax()
        ax2.scatter(df.loc[best_idx, 'Total_Dim'], df.loc[best_idx, 'Overall_Score'],
                   s=400, c='gold', marker='*', edgecolors='darkorange', linewidth=3,
                   zorder=5, label='Melhor Modelo')
        
        # Adicionar labels aos pontos
        for idx, row in df.iterrows():
            ax2.annotate(row['Model'], 
                        (row['Total_Dim'], row['Overall_Score']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8)
        
        ax2.set_xlabel('Dimensão do Embedding', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Score Geral', fontweight='bold', fontsize=12)
        ax2.set_title('Fronteira de Pareto: Performance vs Complexidade', 
                     fontsize=13, fontweight='bold', pad=15)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.legend(loc='lower right', fontsize=10)
        
        # Adicionar colorbar
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Score Geral', fontweight='bold')
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'pareto_efficiency.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Limpar registry
        from matplotlib import units
        units.registry.clear()
        
        return output_path
    
    def create_regression_correlation_heatmap(self, regression_data: List[Dict]) -> Path:
        """
        Cria heatmap de correlação entre métricas de regressão.
        
        Args:
            regression_data: Métricas de regressão
            
        Returns:
            Caminho do arquivo salvo
        """
        df = pd.DataFrame(regression_data).set_index('Model')
        
        # Selecionar todas as métricas de regressão disponíveis
        # Pearson_R, Pearson_P (p-value), R2, RMSE, MAE
        reg_metrics = ['Pearson_R', 'Pearson_P', 'R2', 'RMSE', 'MAE']
        available_metrics = [m for m in reg_metrics if m in df.columns]
        
        df_reg = df[available_metrics].copy()
        
        # Criar figura com dois subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Subplot 1: Heatmap das métricas por modelo (TRANSPOSTO: modelos no eixo X)
        sns.heatmap(df_reg.T, annot=True, fmt='.3f', cmap='RdYlGn',
                   cbar_kws={'label': 'Valor da Métrica'},
                   linewidths=0.5, ax=ax1)
        
        ax1.set_title('Métricas de Regressão por Modelo', 
                     fontsize=13, fontweight='bold', pad=15)
        ax1.set_xlabel('Modelo', fontweight='bold', fontsize=11)
        ax1.set_ylabel('Métrica', fontweight='bold', fontsize=11)
        ax1.tick_params(axis='x', rotation=45)
        ax1.tick_params(axis='y', rotation=0)
        
        # Subplot 2: Matriz de correlação entre as métricas
        # Calcular correlação entre as métricas (colunas)
        corr_matrix = df_reg.corr()
        
        # Criar máscara para triângulo superior
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, vmin=-1, vmax=1, mask=mask,
                   cbar_kws={'label': 'Correlação'},
                   linewidths=0.5, ax=ax2, square=True)
        
        ax2.set_title('Correlação entre Métricas de Regressão', 
                     fontsize=13, fontweight='bold', pad=15)
        ax2.set_xlabel('', fontweight='bold')
        ax2.set_ylabel('', fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.tick_params(axis='y', rotation=0)
        
        # Adicionar texto explicativo
        explanation = "Esquerda: Métricas por modelo | Direita: Correlação entre métricas (ex: Pearson_R vs RMSE)"
        fig.text(0.5, 0.02, explanation, ha='center', fontsize=9, style='italic', alpha=0.7)
        
        plt.tight_layout(rect=[0, 0.03, 1, 1])
        
        output_path = self.output_dir / 'heatmap_regression_correlation.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Limpar registry
        from matplotlib import units
        units.registry.clear()
        
        return output_path
    
    def create_distribution_boxplot(self, individual_results: Dict[str, Dict],
                                   metric_type: str = 'classification') -> Path:
        """
        Cria boxplots para distribuição de métricas entre modelos.
        
        Args:
            individual_results: Resultados individuais por modelo
            metric_type: 'classification' ou 'regression'
            
        Returns:
            Caminho do arquivo salvo
        """
        # Limpar registry antes
        from matplotlib import units
        units.registry.clear()
        plt.close('all')
        
        # Preparar dados
        data_list = []
        
        for model_name, model_data in individual_results.items():
            if metric_type == 'classification':
                # Suportar tanto 'classifier' quanto 'classification' (backward compatibility)
                results = model_data.get('classifier', model_data.get('classification', {})).get('individual_results', {})
                metrics_to_plot = ['F1', 'ROC_AUC', 'MCC']
            else:  # regression
                results = model_data.get('regression', {}).get('individual_results', {})
                metrics_to_plot = ['Pearson_R', 'RMSE']
            
            for algo_name, algo_results in results.items():
                for metric in metrics_to_plot:
                    if metric in algo_results:
                        value = algo_results[metric]
                        # Filtrar valores inválidos
                        if value is None or value == 'None' or value == '':
                            continue
                        try:
                            value = float(value)
                        except (ValueError, TypeError):
                            continue
                        
                        data_list.append({
                            'Protein_Model': model_name,  # Modelo de proteína (esm2, boltz2, etc.)
                            'Algorithm': algo_name,       # Algoritmo (RandomForest, SVM, etc.)
                            'Metric': metric,
                            'Value': value
                        })
        
        if not data_list:
            return None
        
        df = pd.DataFrame(data_list)
        
        # Criar gráfico
        metrics = df['Metric'].unique()
        n_metrics = len(metrics)
        
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6))
        if n_metrics == 1:
            axes = [axes]
        
        fig.suptitle(f'Distribuição de Métricas - {metric_type.capitalize()}',
                    fontsize=14, fontweight='bold')
        
        for idx, metric in enumerate(metrics):
            # Limpar registry antes de cada subplot
            from matplotlib import units
            units.registry.clear()
            
            df_metric = df[df['Metric'] == metric].copy()
            
            # Garantir tipos corretos
            df_metric['Value'] = pd.to_numeric(df_metric['Value'], errors='coerce')
            df_metric = df_metric.dropna(subset=['Value'])
            
            # Usar matplotlib direto ao invés de seaborn para evitar converter issues
            algorithms = df_metric['Algorithm'].unique()
            positions = np.arange(len(algorithms))
            
            # Calcular estatísticas para boxplot manual
            box_data = []
            for algo in algorithms:
                values = df_metric[df_metric['Algorithm'] == algo]['Value'].values
                box_data.append(values)
            
            bp = axes[idx].boxplot(box_data, positions=positions, 
                                   widths=0.6, patch_artist=True,
                                   boxprops=dict(facecolor='lightblue', alpha=0.7),
                                   medianprops=dict(color='red', linewidth=2))
            
            # Adicionar pontos individuais
            for i, algo in enumerate(algorithms):
                values = df_metric[df_metric['Algorithm'] == algo]['Value'].values
                y = values
                x = np.random.normal(positions[i], 0.04, size=len(y))
                axes[idx].plot(x, y, 'r.', alpha=0.5, markersize=8)
            
            axes[idx].set_title(metric, fontweight='bold')
            axes[idx].set_xlabel('Algorithm')
            axes[idx].set_xticks(positions)
            axes[idx].set_xticklabels(algorithms, rotation=45, ha='right')
            axes[idx].set_ylabel('Value')
            axes[idx].grid(axis='y', alpha=0.3)
            
            # Adicionar linhas de referência
            if metric == 'Pearson_R':
                axes[idx].axhline(y=0.7, color='green', linestyle='--', alpha=0.3)
                axes[idx].axhline(y=0.5, color='orange', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        
        output_path = self.output_dir / f'distribution_{metric_type}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Limpar registry
        from matplotlib import units
        units.registry.clear()
        
        return output_path
