#!/usr/bin/env python3
"""
Avaliador de Modelos de Regressão - DockTKinase
================================================

Calcula métricas e compara performance dos modelos de regressão.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
    mean_absolute_percentage_error
)

# Import utilitários centralizados - sempre adiciona src ao path
import sys
from pathlib import Path
src_path = Path(__file__).parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from utils.data_utils import safe_get, safe_get_numeric


class RegressionEvaluator:
    """
    Avalia performance de modelos de regressão.
    
    Calcula métricas completas e permite comparação entre modelos.
    """
    
    @staticmethod
    def calculate_metrics(y_true, y_pred, model_name='Model'):
        """
        Calcula todas as métricas de regressão.
        
        Args:
            y_true: Valores reais
            y_pred: Valores preditos
            model_name: Nome do modelo
            
        Returns:
            Dict com todas as métricas
        """
        # Garantir arrays numpy
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Calcular métricas
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        median_ae = median_absolute_error(y_true, y_pred)
        
        # MAPE (cuidado com divisão por zero)
        try:
            mape = mean_absolute_percentage_error(y_true, y_pred)
        except (ValueError, ZeroDivisionError):
            # Se y_true tem zeros, calcular manualmente excluindo-os
            mask = y_true != 0
            if mask.sum() > 0:
                mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            else:
                mape = np.nan
        
        # Métricas adicionais
        residuals = y_true - y_pred
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        max_error = np.max(np.abs(residuals))
        
        # Percentis de erro
        percentile_errors = {
            'p25': np.percentile(np.abs(residuals), 25),
            'p50': np.percentile(np.abs(residuals), 50),
            'p75': np.percentile(np.abs(residuals), 75),
            'p90': np.percentile(np.abs(residuals), 90),
            'p95': np.percentile(np.abs(residuals), 95),
            'p99': np.percentile(np.abs(residuals), 99)
        }
        
        metrics = {
            'model_name': model_name,
            'n_samples': len(y_true),
            
            # Métricas principais
            'MAE': float(mae),
            'MSE': float(mse),
            'RMSE': float(rmse),
            'R2': float(r2),
            'MedianAE': float(median_ae),
            'MAPE': float(mape) if not np.isnan(mape) else None,
            
            # Estatísticas dos resíduos
            'mean_residual': float(mean_residual),
            'std_residual': float(std_residual),
            'max_error': float(max_error),
            
            # Percentis
            **{f'error_{k}': float(v) for k, v in percentile_errors.items()}
        }
        
        return metrics
    
    @staticmethod
    def compare_models(results_dict, metric='MAE', ascending=True):
        """
        Compara todos os modelos baseado em uma métrica.
        
        Args:
            results_dict: Dict {model_name: metrics_dict}
            metric: Métrica para ranking (MAE, RMSE, R2, etc)
            ascending: True para menor melhor (MAE), False para maior melhor (R2)
            
        Returns:
            DataFrame com ranking de modelos
        """
        # Converter para DataFrame
        df = pd.DataFrame(results_dict).T
        
        # Ordenar pela métrica escolhida
        if metric in df.columns:
            df = df.sort_values(by=metric, ascending=ascending)
        
        # Adicionar ranking
        df.insert(0, 'rank', range(1, len(df) + 1))
        
        return df
    
    @staticmethod
    def get_best_model(results_dict, metric='MAE', ascending=True):
        """
        Retorna o nome do melhor modelo baseado na métrica.
        
        Args:
            results_dict: Dict {model_name: metrics_dict}
            metric: Métrica para seleção
            ascending: True para menor melhor, False para maior melhor
            
        Returns:
            str: Nome do melhor modelo
        """
        df = RegressionEvaluator.compare_models(results_dict, metric, ascending)
        return df.index[0]
    
    @staticmethod
    def save_predictions_csv(
        y_true, 
        y_pred, 
        output_path,
        model_name='Model',
        df_original=None, 
        indices=None,
        dataset_name='test'
    ):
        """
        Salva predições detalhadas em CSV.
        
        Args:
            y_true: Valores reais
            y_pred: Valores preditos
            output_path: Path para salvar CSV
            model_name: Nome do modelo
            df_original: DataFrame original com informações dos compostos (opcional)
            indices: Índices das amostras no DataFrame original (opcional)
            dataset_name: Nome do conjunto (train/val/test)
        """
        try:
            # Calcular erro absoluto e relativo
            errors = np.abs(y_true - y_pred)
            relative_errors = (errors / y_true) * 100
            
            # Preparar dados
            predictions_data = []
            
            # Se df_original e indices forem fornecidos, adicionar informações detalhadas
            if df_original is not None and indices is not None:
                for idx, (i, yt, yp, err, rel_err) in enumerate(
                    zip(indices, y_true, y_pred, errors, relative_errors)
                ):
                    # FIX #38: Usar .loc para indexação por label do índice, não posição
                    row_data = df_original.loc[i].to_dict()
                    
                    prediction_row = {
                        'dataset': dataset_name,
                        'model': model_name,
                        'sample_index': int(i),
                        'molregno': safe_get(row_data, 'molregno'),
                        'seq_id': safe_get(row_data, 'seq_id'),
                        'target_kinase': safe_get(row_data, 'target_kinase'),
                        'canonical_smiles': safe_get(row_data, 'canonical_smiles'),
                        'aminoacid_sequence': safe_get(row_data, 'seq'),
                        
                        # Valores de atividade
                        'true_value_nM': float(yt),
                        'predicted_value_nM': float(yp),
                        'absolute_error_nM': float(err),
                        'relative_error_percent': float(rel_err),
                        
                        # Informações originais
                        'standard_type': safe_get(row_data, 'standard_type'),
                        'standard_value': safe_get(row_data, 'standard_value'),
                        'pchembl_value': safe_get(row_data, 'pchembl_value'),
                        'compound_name': safe_get(row_data, 'compound_name'),
                        'chembl_id': safe_get(row_data, 'chembl_id'),
                        'organism': safe_get(row_data, 'organism')
                    }
                    predictions_data.append(prediction_row)
            else:
                # Versão simplificada sem metadados
                for idx, (yt, yp, err, rel_err) in enumerate(
                    zip(y_true, y_pred, errors, relative_errors)
                ):
                    prediction_row = {
                        'dataset': dataset_name,
                        'model': model_name,
                        'sample_index': idx,
                        'true_value_nM': float(yt),
                        'predicted_value_nM': float(yp),
                        'absolute_error_nM': float(err),
                        'relative_error_percent': float(rel_err)
                    }
                    predictions_data.append(prediction_row)
            
            # Criar DataFrame
            df_predictions = pd.DataFrame(predictions_data)
            
            # Ordenar por erro absoluto (maiores erros primeiro para análise)
            df_predictions = df_predictions.sort_values(
                'absolute_error_nM', 
                ascending=False
            )
            
            # Salvar CSV
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df_predictions.to_csv(output_path, index=False)
            
            return df_predictions
            
        except Exception as e:
            print(f'⚠️  Erro ao salvar CSV de predições: {e}')
            import traceback
            traceback.print_exc()
            return None
    
    @staticmethod
    def print_metrics_summary(metrics, title='Métricas de Regressão'):
        """
        Imprime resumo formatado das métricas.
        
        Args:
            metrics: Dict com métricas
            title: Título do resumo
        """
        print('=' * 60)
        print(f' {title}')
        print('=' * 60)
        
        print(f"\nModelo: {metrics.get('model_name', 'N/A')}")
        print(f"Amostras: {metrics.get('n_samples', 0):,}")
        
        print('\n📊 Métricas Principais:')
        print(f"  MAE:       {metrics.get('MAE', 0):.4f} nM")
        print(f"  RMSE:      {metrics.get('RMSE', 0):.4f} nM")
        print(f"  MedianAE:  {metrics.get('MedianAE', 0):.4f} nM")
        print(f"  R²:        {metrics.get('R2', 0):.4f}")
        if metrics.get('MAPE') is not None:
            print(f"  MAPE:      {metrics.get('MAPE', 0):.2f}%")
        
        print('\n📈 Estatísticas dos Erros:')
        print(f"  Erro Médio:   {metrics.get('mean_residual', 0):.4f} nM")
        print(f"  Desvio Padrão: {metrics.get('std_residual', 0):.4f} nM")
        print(f"  Erro Máximo:   {metrics.get('max_error', 0):.4f} nM")
        
        print('\n📊 Percentis do Erro Absoluto:')
        print(f"  25%: {metrics.get('error_p25', 0):.4f} nM")
        print(f"  50%: {metrics.get('error_p50', 0):.4f} nM")
        print(f"  75%: {metrics.get('error_p75', 0):.4f} nM")
        print(f"  90%: {metrics.get('error_p90', 0):.4f} nM")
        print(f"  95%: {metrics.get('error_p95', 0):.4f} nM")
        print(f"  99%: {metrics.get('error_p99', 0):.4f} nM")
        
        print('=' * 60)


if __name__ == '__main__':
    # Test
    y_true = np.array([100, 200, 300, 400, 500])
    y_pred = np.array([110, 190, 320, 380, 510])
    
    metrics = RegressionEvaluator.calculate_metrics(y_true, y_pred, 'TestModel')
    RegressionEvaluator.print_metrics_summary(metrics)
