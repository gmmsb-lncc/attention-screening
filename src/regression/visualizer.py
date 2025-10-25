#!/usr/bin/env python3
"""
Visualizador de Regressão - DockTKinase
========================================

Gera gráficos e visualizações para análise de modelos de regressão.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class RegressionVisualizer:
    """
    Cria visualizações para análise de regressão.
    
    Gera scatter plots, residual plots, comparações de modelos, etc.
    """
    
    @staticmethod
    def plot_predictions_vs_actual(
        y_true, 
        y_pred, 
        model_name='Model',
        save_path=None,
        figsize=(10, 8)
    ):
        """
        Scatter plot: Valores Preditos vs Valores Reais.
        
        Args:
            y_true: Valores reais
            y_pred: Valores preditos
            model_name: Nome do modelo
            save_path: Path para salvar figura
            figsize: Tamanho da figura
        """
        try:
            # Configurar estilo
            sns.set_style("whitegrid")
            plt.rcParams['figure.facecolor'] = 'white'
            
            fig, ax = plt.subplots(figsize=figsize)
            
            # Scatter plot
            ax.scatter(y_true, y_pred, alpha=0.5, s=30, edgecolors='k', linewidths=0.5)
            
            # Linha de identidade (predição perfeita)
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 
                   'r--', lw=2, label='Predição Perfeita', alpha=0.8)
            
            # Calcular R²
            from sklearn.metrics import r2_score
            r2 = r2_score(y_true, y_pred)
            
            # Labels e título
            ax.set_xlabel('Valor Real (nM)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Valor Predito (nM)', fontsize=14, fontweight='bold')
            ax.set_title(f'Predições vs Valores Reais - {model_name}\nR² = {r2:.4f}', 
                        fontsize=16, fontweight='bold')
            ax.legend(loc='upper left', fontsize=12)
            ax.grid(alpha=0.3)
            
            # Equal aspect ratio
            ax.set_aspect('equal', adjustable='box')
            
            plt.tight_layout()
            
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
                
        except Exception as e:
            print(f'⚠️  Erro ao gerar plot predições vs real: {e}')
            plt.close()
    
    @staticmethod
    def plot_residuals(
        y_true, 
        y_pred, 
        model_name='Model',
        save_path=None,
        figsize=(12, 5)
    ):
        """
        Plot de resíduos (erros).
        
        Args:
            y_true: Valores reais
            y_pred: Valores preditos
            model_name: Nome do modelo
            save_path: Path para salvar figura
            figsize: Tamanho da figura
        """
        try:
            residuals = y_true - y_pred
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            
            # ===== SUBPLOT 1: Residuals vs Predicted =====
            ax1.scatter(y_pred, residuals, alpha=0.5, s=30, edgecolors='k', linewidths=0.5)
            ax1.axhline(y=0, color='r', linestyle='--', lw=2, label='Zero Residual')
            ax1.set_xlabel('Valor Predito (nM)', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Resíduo (nM)', fontsize=12, fontweight='bold')
            ax1.set_title(f'Resíduos vs Predições - {model_name}', 
                         fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(alpha=0.3)
            
            # ===== SUBPLOT 2: Histogram of Residuals =====
            ax2.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
            ax2.axvline(x=0, color='r', linestyle='--', lw=2, label='Zero')
            ax2.set_xlabel('Resíduo (nM)', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Frequência', fontsize=12, fontweight='bold')
            ax2.set_title(f'Distribuição dos Resíduos - {model_name}', 
                         fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
                
        except Exception as e:
            print(f'⚠️  Erro ao gerar plot de resíduos: {e}')
            plt.close()
    
    @staticmethod
    def plot_models_comparison(
        results_dict, 
        metric='MAE',
        save_path=None,
        figsize=(14, 8)
    ):
        """
        Bar chart comparando todos os modelos.
        
        Args:
            results_dict: Dict {model_name: metrics_dict}
            metric: Métrica para comparar (MAE, RMSE, R2, etc)
            save_path: Path para salvar figura
            figsize: Tamanho da figura
        """
        try:
            # Extrair dados
            models = []
            values = []
            
            for model_name, metrics in results_dict.items():
                if metric in metrics:
                    models.append(model_name)
                    values.append(metrics[metric])
            
            if not models:
                print(f'⚠️  Métrica "{metric}" não encontrada nos resultados')
                return
            
            # Ordenar
            sorted_pairs = sorted(zip(values, models))
            values, models = zip(*sorted_pairs)
            
            # Criar figura
            fig, ax = plt.subplots(figsize=figsize)
            
            # Bar chart
            colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
            bars = ax.barh(models, values, color=colors, edgecolor='black', linewidth=1.5)
            
            # Adicionar valores nas barras
            for i, (bar, value) in enumerate(zip(bars, values)):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2, 
                       f'{value:.2f}',
                       ha='left', va='center', fontsize=11, fontweight='bold')
            
            # Labels
            ax.set_xlabel(f'{metric}', fontsize=14, fontweight='bold')
            ax.set_ylabel('Modelo', fontsize=14, fontweight='bold')
            ax.set_title(f'Comparação de Modelos - {metric}', 
                        fontsize=16, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
                
        except Exception as e:
            print(f'⚠️  Erro ao gerar comparação de modelos: {e}')
            plt.close()
    
    @staticmethod
    def plot_error_distribution(
        y_true, 
        y_pred, 
        model_name='Model',
        save_path=None,
        figsize=(12, 5)
    ):
        """
        Distribuição dos erros (absoluto e relativo).
        
        Args:
            y_true: Valores reais
            y_pred: Valores preditos
            model_name: Nome do modelo
            save_path: Path para salvar figura
            figsize: Tamanho da figura
        """
        try:
            absolute_errors = np.abs(y_true - y_pred)
            relative_errors = (absolute_errors / y_true) * 100
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            
            # ===== SUBPLOT 1: Absolute Errors =====
            ax1.hist(absolute_errors, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
            ax1.axvline(x=np.median(absolute_errors), color='r', linestyle='--', 
                       lw=2, label=f'Mediana: {np.median(absolute_errors):.2f} nM')
            ax1.set_xlabel('Erro Absoluto (nM)', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Frequência', fontsize=12, fontweight='bold')
            ax1.set_title(f'Distribuição do Erro Absoluto - {model_name}', 
                         fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(alpha=0.3)
            
            # ===== SUBPLOT 2: Relative Errors =====
            ax2.hist(relative_errors, bins=50, edgecolor='black', alpha=0.7, color='lightcoral')
            ax2.axvline(x=np.median(relative_errors), color='r', linestyle='--', 
                       lw=2, label=f'Mediana: {np.median(relative_errors):.2f}%')
            ax2.set_xlabel('Erro Relativo (%)', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Frequência', fontsize=12, fontweight='bold')
            ax2.set_title(f'Distribuição do Erro Relativo - {model_name}', 
                         fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
                
        except Exception as e:
            print(f'⚠️  Erro ao gerar distribuição de erros: {e}')
            plt.close()
    
    @staticmethod
    def plot_feature_importance(
        model, 
        feature_names=None,
        model_name='Model',
        save_path=None,
        top_n=20,
        figsize=(10, 8)
    ):
        """
        Plot de importância de features (se o modelo suportar).
        
        Args:
            model: Modelo treinado
            feature_names: Nomes das features
            model_name: Nome do modelo
            save_path: Path para salvar figura
            top_n: Número de top features para mostrar
            figsize: Tamanho da figura
        """
        try:
            # Verificar se modelo tem feature_importances_
            if not hasattr(model, 'feature_importances_'):
                print(f'⚠️  Modelo {model_name} não suporta feature importance')
                return
            
            importances = model.feature_importances_
            
            # Criar nomes de features se não fornecidos
            if feature_names is None:
                feature_names = [f'Feature {i}' for i in range(len(importances))]
            
            # Selecionar top N
            indices = np.argsort(importances)[::-1][:top_n]
            top_importances = importances[indices]
            top_names = [feature_names[i] for i in indices]
            
            # Plot
            fig, ax = plt.subplots(figsize=figsize)
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(top_names)))
            bars = ax.barh(range(len(top_names)), top_importances, color=colors, 
                          edgecolor='black', linewidth=1)
            
            ax.set_yticks(range(len(top_names)))
            ax.set_yticklabels(top_names)
            ax.set_xlabel('Importância', fontsize=12, fontweight='bold')
            ax.set_title(f'Top {top_n} Features - {model_name}', 
                        fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
                
        except Exception as e:
            print(f'⚠️  Erro ao gerar feature importance: {e}')
            plt.close()


if __name__ == '__main__':
    # Test
    np.random.seed(42)
    y_true = np.random.uniform(100, 1000, 1000)
    y_pred = y_true + np.random.normal(0, 50, 1000)
    
    RegressionVisualizer.plot_predictions_vs_actual(y_true, y_pred, 'TestModel')
    RegressionVisualizer.plot_residuals(y_true, y_pred, 'TestModel')
    RegressionVisualizer.plot_error_distribution(y_true, y_pred, 'TestModel')
