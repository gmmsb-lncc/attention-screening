#!/usr/bin/env python3
"""
Treinador Multi-Modelo Sklearn - Classificação
==============================================

Gerencia treinamento de múltiplos modelos sklearn de classificação.
Equivalente ao trainer de regressão, mas para classificação binária.
"""

import time
import numpy as np
from typing import Dict, Any, Optional, List
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    fbeta_score,
    matthews_corrcoef,
    average_precision_score,
    brier_score_loss
)


class ClassificationMetricsCalculator:
    """Calcula todas as métricas de classificação."""
    
    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, 
                              y_pred: np.ndarray,
                              y_pred_proba: Optional[np.ndarray] = None,
                              model_name: str = '') -> Dict[str, Any]:
        """
        Calcula todas as métricas de classificação.
        
        Args:
            y_true: Labels verdadeiros
            y_pred: Predições (0/1)
            y_pred_proba: Probabilidades (para ROC-AUC, etc)
            model_name: Nome do modelo (para logging)
            
        Returns:
            Dict com todas as métricas
        """
        metrics = {}
        
        # Métricas básicas
        metrics['Accuracy'] = accuracy_score(y_true, y_pred)
        metrics['Precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['Recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['F1'] = f1_score(y_true, y_pred, zero_division=0)
        
        # Confusion Matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['True_Negatives'] = int(tn)
        metrics['False_Positives'] = int(fp)
        metrics['False_Negatives'] = int(fn)
        metrics['True_Positives'] = int(tp)
        
        # Specificity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['Specificity'] = specificity
        
        # F-beta scores
        metrics['Fbeta_0.5'] = fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)
        metrics['Fbeta_2'] = fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)
        
        # Matthews Correlation Coefficient
        metrics['MCC'] = matthews_corrcoef(y_true, y_pred)
        
        # Métricas baseadas em probabilidades
        if y_pred_proba is not None:
            try:
                metrics['ROC_AUC'] = roc_auc_score(y_true, y_pred_proba)
                metrics['Average_Precision'] = average_precision_score(y_true, y_pred_proba)
                metrics['Brier_Score'] = brier_score_loss(y_true, y_pred_proba)
            except Exception as e:
                # Se houver apenas uma classe, algumas métricas podem falhar
                metrics['ROC_AUC'] = 0.0
                metrics['Average_Precision'] = 0.0
                metrics['Brier_Score'] = 0.0
        else:
            metrics['ROC_AUC'] = 0.0
            metrics['Average_Precision'] = 0.0
            metrics['Brier_Score'] = 0.0
        
        return metrics


class SklearnClassificationTrainer:
    """
    Treinador de múltiplos modelos sklearn de classificação.
    Equivalente ao RegressionTrainer mas para classificação.
    """
    
    def __init__(self, 
                 models_dict: Dict[str, Any],
                 verbose: bool = True,
                 random_state: int = 42):
        """
        Inicializar trainer.
        
        Args:
            models_dict: Dicionário {nome: modelo}
            verbose: Mostrar progresso
            random_state: Seed para reprodutibilidade
        """
        self.models_dict = models_dict
        self.verbose = verbose
        self.random_state = random_state
        
        self.trained_models = {}
        self.train_results = {}
        self.val_results = {}
        
        self.metrics_calculator = ClassificationMetricsCalculator()
    
    def train_single_model(self, 
                          model_name: str,
                          model: Any,
                          X_train: np.ndarray,
                          y_train: np.ndarray,
                          X_val: Optional[np.ndarray] = None,
                          y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Treinar um único modelo.
        
        Args:
            model_name: Nome do modelo
            model: Instância do modelo
            X_train: Features de treino
            y_train: Labels de treino
            X_val: Features de validação (opcional)
            y_val: Labels de validação (opcional)
            
        Returns:
            Dict com métricas de validação (ou treino se val=None)
        """
        start_time = time.time()
        
        if self.verbose:
            print(f'\n   🔧 Treinando {model_name}...')
        
        # Treinar
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Avaliar em treino
        y_train_pred = model.predict(X_train)
        
        # Obter probabilidades se disponível
        if hasattr(model, 'predict_proba'):
            y_train_proba = model.predict_proba(X_train)[:, 1]
        else:
            y_train_proba = None
        
        train_metrics = self.metrics_calculator.calculate_all_metrics(
            y_train, y_train_pred, y_train_proba, model_name
        )
        train_metrics['training_time'] = train_time
        
        # Avaliar em validação se disponível
        val_metrics = None
        if X_val is not None and y_val is not None:
            y_val_pred = model.predict(X_val)
            
            if hasattr(model, 'predict_proba'):
                y_val_proba = model.predict_proba(X_val)[:, 1]
            else:
                y_val_proba = None
            
            val_metrics = self.metrics_calculator.calculate_all_metrics(
                y_val, y_val_pred, y_val_proba, model_name
            )
        
        # Armazenar
        self.trained_models[model_name] = model
        self.train_results[model_name] = train_metrics
        if val_metrics:
            self.val_results[model_name] = val_metrics
        
        if self.verbose:
            print(f'      ✅ Treino: Acc={train_metrics["Accuracy"]:.4f} | '
                  f'F1={train_metrics["F1"]:.4f} | '
                  f'ROC-AUC={train_metrics["ROC_AUC"]:.4f} | '
                  f'Tempo={train_time:.2f}s')
            
            if val_metrics:
                print(f'      ✅ Valid: Acc={val_metrics["Accuracy"]:.4f} | '
                      f'F1={val_metrics["F1"]:.4f} | '
                      f'ROC-AUC={val_metrics["ROC_AUC"]:.4f}')
        
        return val_metrics if val_metrics else train_metrics
    
    def train_all(self,
                  X_train: np.ndarray,
                  y_train: np.ndarray,
                  X_val: Optional[np.ndarray] = None,
                  y_val: Optional[np.ndarray] = None) -> Dict[str, Dict[str, Any]]:
        """
        Treinar todos os modelos.
        
        Args:
            X_train: Features de treino
            y_train: Labels de treino
            X_val: Features de validação (opcional)
            y_val: Labels de validação (opcional)
            
        Returns:
            Dict com métricas de validação de todos os modelos
        """
        if self.verbose:
            print(f'\n   📊 Treinando {len(self.models_dict)} modelos...')
        
        for model_name, model in self.models_dict.items():
            try:
                self.train_single_model(
                    model_name, model,
                    X_train, y_train,
                    X_val, y_val
                )
            except Exception as e:
                if self.verbose:
                    print(f'      ❌ Erro ao treinar {model_name}: {str(e)}')
                continue
        
        return self.val_results if self.val_results else self.train_results
    
    def get_best_model(self, metric: str = 'ROC_AUC') -> tuple:
        """
        Retorna o melhor modelo baseado em uma métrica.
        
        Args:
            metric: Métrica para comparação (default: ROC_AUC)
            
        Returns:
            Tuple (nome_modelo, modelo, métrica_valor)
        """
        results = self.val_results if self.val_results else self.train_results
        
        if not results:
            return None, None, None
        
        # Encontrar melhor
        best_name = max(results.items(), key=lambda x: x[1][metric])[0]
        best_model = self.trained_models[best_name]
        best_value = results[best_name][metric]
        
        return best_name, best_model, best_value
    
    def get_results_summary(self) -> Dict[str, Any]:
        """
        Retorna resumo dos resultados de todos os modelos.
        
        Returns:
            Dict com resumo organizado
        """
        results = self.val_results if self.val_results else self.train_results
        
        summary = {
            'n_models': len(results),
            'models': {},
            'rankings': {}
        }
        
        # Adicionar resultados de cada modelo
        for name, metrics in results.items():
            summary['models'][name] = metrics
        
        # Rankings por métrica
        for metric in ['ROC_AUC', 'F1', 'Accuracy', 'Precision', 'Recall']:
            if metric in list(results.values())[0]:
                sorted_models = sorted(
                    results.items(),
                    key=lambda x: x[1][metric],
                    reverse=True
                )
                summary['rankings'][metric] = [
                    {'model': name, 'value': metrics[metric]}
                    for name, metrics in sorted_models
                ]
        
        return summary
    
    def print_summary(self, top_n: int = None, use_test: bool = True):
        """
        Imprime resumo dos resultados (deprecated - use print_results_summary no pipeline).
        
        Args:
            top_n: Número de top modelos a mostrar (None = todos)
            use_test: Se True, usa resultados de teste (default). Se False, usa validação.
        """
        # Por padrão, usar resultados de teste se disponíveis
        if use_test and self.test_results:
            results = self.test_results
            result_type = "Teste"
        elif self.val_results:
            results = self.val_results
            result_type = "Validação"
        else:
            results = self.train_results
            result_type = "Treino"
        
        if not results:
            print('⚠️  Nenhum modelo treinado')
            return
        
        print('\n' + '=' * 80)
        print(f'📊 RESUMO DOS RESULTADOS (Conjunto de {result_type})')
        print('=' * 80)
        
        # Ordenar por ROC-AUC
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1]['ROC_AUC'],
            reverse=True
        )
        
        # Limitar se necessário
        if top_n:
            sorted_results = sorted_results[:top_n]
        
        # Cabeçalho
        header = f"{'Modelo':<20} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'ROC-AUC':>10} {'Tempo':>8}"
        print(header)
        print('-' * 80)
        
        # Modelos
        for i, (model_name, metrics) in enumerate(sorted_results):
            train_time = self.train_results[model_name].get('training_time', 0)
            row = (
                f"{model_name:<20} "
                f"{metrics['Accuracy']:>8.4f} "
                f"{metrics['Precision']:>8.4f} "
                f"{metrics['Recall']:>8.4f} "
                f"{metrics['F1']:>8.4f} "
                f"{metrics['ROC_AUC']:>10.4f} "
                f"{train_time:>7.2f}s"
            )
            
            # Destacar o melhor
            if i == 0:
                print(f'🏆 {row}')
            else:
                print(f'   {row}')
        
        print('=' * 80)
        
        # Melhor modelo
        best_name = sorted_results[0][0]
        best_metrics = sorted_results[0][1]
        
        print(f'\n🏆 MELHOR MODELO: {best_name}')
        print(f'   ROC-AUC: {best_metrics["ROC_AUC"]:.4f}')
        print(f'   F1-Score: {best_metrics["F1"]:.4f}')
        print(f'   Accuracy: {best_metrics["Accuracy"]:.4f}')
        print(f'   Precision: {best_metrics["Precision"]:.4f}')
        print(f'   Recall: {best_metrics["Recall"]:.4f}')
        print()


if __name__ == '__main__':
    print('SklearnClassificationTrainer - Treinador Multi-Modelo Sklearn')
    print('=' * 70)
    print('\nMódulo de treinamento para múltiplos classificadores sklearn.')
    print('Use em conjunto com ClassificationModels.')
