"""
Sistema de métricas para avaliação de classificadores binários.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, fbeta_score, matthews_corrcoef,
    average_precision_score, brier_score_loss, roc_curve, precision_recall_curve
)
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ClassificationMetrics:
    """Contêiner para métricas de classificação binária."""
    
    # Métricas básicas
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    
    # Métricas avançadas
    roc_auc: float
    average_precision: float
    brier_score: float
    matthews_corrcoef: float
    
    # Matriz de confusão
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    
    # Métricas derivadas
    specificity: float
    sensitivity: float  # = recall
    fbeta_05: float  # Privilegia precision
    fbeta_2: float   # Privilegia recall
    
    # Meta-informações
    sample_count: int
    positive_rate: float
    
    def to_dict(self) -> Dict[str, float]:
        """Converte métricas para dicionário."""
        return {
            "Loss": self.loss,
            "Accuracy": self.accuracy,
            "Precision": self.precision,
            "Recall": self.recall,
            "F1": self.f1,
            "ROC_AUC": self.roc_auc,
            "Average_Precision": self.average_precision,
            "Brier_Score": self.brier_score,
            "MCC": self.matthews_corrcoef,
            "True_Positives": self.true_positives,
            "True_Negatives": self.true_negatives,
            "False_Positives": self.false_positives,
            "False_Negatives": self.false_negatives,
            "Specificity": self.specificity,
            "Sensitivity": self.sensitivity,
            "Fbeta_0.5": self.fbeta_05,
            "Fbeta_2": self.fbeta_2,
            "Sample_Count": self.sample_count,
            "Positive_Rate": self.positive_rate
        }


class MetricsCalculator:
    """Calculadora de métricas para classificação binária."""
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cpu")
    
    def evaluate_model(
        self, 
        model: nn.Module, 
        data_loader: DataLoader,
        criterion: Optional[nn.Module] = None,
        amp_enabled: bool = False,
        amp_dtype: Optional[torch.dtype] = None
    ) -> ClassificationMetrics:
        """
        Avalia modelo em um DataLoader e retorna métricas completas.
        
        Args:
            model: Modelo PyTorch
            data_loader: DataLoader com dados de teste
            criterion: Função de loss (padrão: BCEWithLogitsLoss)
            amp_enabled: Se usar automatic mixed precision
            amp_dtype: Dtype para autocast
            
        Returns:
            ClassificationMetrics com todas as métricas
        """
        if criterion is None:
            criterion = nn.BCEWithLogitsLoss()
        
        model.eval()
        
        # Coletores de resultados
        all_losses = []
        all_logits = []
        all_probs = []
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass com AMP opcional
                with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
                    logits = model(batch_x)
                    # Garante dimensões corretas para BCE loss
                    if logits.dim() > 1 and logits.size(1) == 1:
                        logits = logits.squeeze(1)
                    loss = criterion(logits, batch_y)
                
                # Conversões para CPU
                logits_cpu = logits.float().cpu()
                probs_cpu = torch.sigmoid(logits_cpu)
                preds_cpu = (probs_cpu >= 0.5).float()
                labels_cpu = batch_y.float().cpu()
                
                # Coleta resultados
                all_losses.append(loss.item())
                all_logits.extend(logits_cpu.numpy().ravel())
                all_probs.extend(probs_cpu.numpy().ravel())
                all_preds.extend(preds_cpu.numpy().ravel())
                all_labels.extend(labels_cpu.numpy().ravel())
        
        # Conversão para arrays NumPy
        labels = np.asarray(all_labels, dtype=np.float32)
        probs = np.asarray(all_probs, dtype=np.float32)
        preds = np.asarray(all_preds, dtype=np.float32)
        
        # Cálculo de métricas
        return self.calculate_metrics(
            y_true=labels,
            y_prob=probs,
            y_pred=preds,
            loss=np.mean(all_losses)
        )
    
    def calculate_metrics(
        self, 
        y_true: np.ndarray, 
        y_prob: np.ndarray, 
        y_pred: np.ndarray,
        loss: float = 0.0
    ) -> ClassificationMetrics:
        """
        Calcula todas as métricas a partir de arrays NumPy.
        
        Args:
            y_true: Labels verdadeiros (0/1)
            y_prob: Probabilidades preditas [0,1]
            y_pred: Predições binárias (0/1)
            loss: Loss médio
            
        Returns:
            ClassificationMetrics completas
        """
        n_samples = len(y_true)
        unique_labels = np.unique(y_true)
        
        # Métricas básicas
        accuracy = np.mean(y_pred == y_true)
        
        # Precisão e recall com divisão segura
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-6, precision + recall)
        
        # Especificidade
        specificity = tn / max(1, tn + fp)
        
        # Métricas avançadas (apenas se temos ambas as classes)
        if len(unique_labels) > 1:
            try:
                roc_auc = roc_auc_score(y_true, y_prob)
            except ValueError:
                roc_auc = 0.5
            
            try:
                avg_precision = average_precision_score(y_true, y_prob)
            except ValueError:
                avg_precision = np.mean(y_true)
            
            try:
                mcc = matthews_corrcoef(y_true, y_pred)
            except ValueError:
                mcc = 0.0
        else:
            # Todas as amostras da mesma classe
            roc_auc = 0.5
            avg_precision = float(unique_labels[0])
            mcc = 0.0
        
        # Brier score (sempre calculável)
        brier_score = brier_score_loss(y_true, y_prob)
        
        # F-beta scores
        fbeta_05 = fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)
        fbeta_2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
        
        # Taxa de positivos
        positive_rate = np.mean(y_true)
        
        return ClassificationMetrics(
            loss=loss,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            roc_auc=roc_auc,
            average_precision=avg_precision,
            brier_score=brier_score,
            matthews_corrcoef=mcc,
            true_positives=int(tp),
            true_negatives=int(tn),
            false_positives=int(fp),
            false_negatives=int(fn),
            specificity=specificity,
            sensitivity=recall,  # Sensitivity = Recall
            fbeta_05=fbeta_05,
            fbeta_2=fbeta_2,
            sample_count=n_samples,
            positive_rate=positive_rate
        )


class MetricsAggregator:
    """Agregador de métricas para cross-validation."""
    
    def __init__(self):
        self.fold_metrics: List[ClassificationMetrics] = []
    
    def add_fold_metrics(self, metrics: ClassificationMetrics):
        """Adiciona métricas de um fold."""
        self.fold_metrics.append(metrics)
    
    def get_summary_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Retorna estatísticas resumo (média, desvio padrão, min, max) das métricas.
        """
        if not self.fold_metrics:
            return {}
        
        # Extrai todas as métricas
        metrics_dict = {}
        metric_names = [
            'loss', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc',
            'average_precision', 'brier_score', 'matthews_corrcoef', 'specificity'
        ]
        
        for metric_name in metric_names:
            values = [getattr(fold_metrics, metric_name) for fold_metrics in self.fold_metrics]
            metrics_dict[metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values))
            }
        
        return metrics_dict
    
    def get_best_fold(self, metric_name: str = 'roc_auc', maximize: bool = True) -> Tuple[int, ClassificationMetrics]:
        """
        Retorna o índice e métricas do melhor fold baseado em uma métrica.
        """
        if not self.fold_metrics:
            raise ValueError("Nenhuma métrica de fold foi adicionada")
        
        values = [getattr(fold_metrics, metric_name) for fold_metrics in self.fold_metrics]
        
        if maximize:
            best_idx = np.argmax(values)
        else:
            best_idx = np.argmin(values)
        
        return best_idx, self.fold_metrics[best_idx]
    
    def get_confidence_interval(self, metric_name: str, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calcula intervalo de confiança para uma métrica usando bootstrap.
        """
        if not self.fold_metrics:
            raise ValueError("Nenhuma métrica de fold foi adicionada")
        
        values = [getattr(fold_metrics, metric_name) for fold_metrics in self.fold_metrics]
        
        # Bootstrap simples
        n_bootstrap = 1000
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(values, size=len(values), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        # Percentis para intervalo de confiança
        alpha = 1 - confidence
        lower_percentile = 100 * (alpha / 2)
        upper_percentile = 100 * (1 - alpha / 2)
        
        ci_lower = np.percentile(bootstrap_means, lower_percentile)
        ci_upper = np.percentile(bootstrap_means, upper_percentile)
        
        return float(ci_lower), float(ci_upper)


def calculate_threshold_metrics(y_true: np.ndarray, y_prob: np.ndarray, 
                              thresholds: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """
    Calcula métricas para diferentes thresholds de classificação.
    
    Útil para encontrar o threshold ótimo para o problema específico.
    """
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)
    
    precisions = []
    recalls = []
    f1_scores = []
    accuracies = []
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-6, precision + recall)
        accuracy = np.mean(y_pred == y_true)
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        accuracies.append(accuracy)
    
    return {
        'thresholds': thresholds,
        'precisions': np.array(precisions),
        'recalls': np.array(recalls),
        'f1_scores': np.array(f1_scores),
        'accuracies': np.array(accuracies)
    }


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray, 
                          metric: str = 'f1') -> Tuple[float, float]:
    """
    Encontra threshold ótimo baseado em uma métrica.
    
    Returns:
        (optimal_threshold, metric_value)
    """
    threshold_metrics = calculate_threshold_metrics(y_true, y_prob)
    
    if metric == 'f1':
        values = threshold_metrics['f1_scores']
    elif metric == 'accuracy':
        values = threshold_metrics['accuracies']
    elif metric == 'precision':
        values = threshold_metrics['precisions']
    elif metric == 'recall':
        values = threshold_metrics['recalls']
    else:
        raise ValueError(f"Métrica não suportada: {metric}")
    
    optimal_idx = np.argmax(values)
    optimal_threshold = threshold_metrics['thresholds'][optimal_idx]
    optimal_value = values[optimal_idx]
    
    return float(optimal_threshold), float(optimal_value)
