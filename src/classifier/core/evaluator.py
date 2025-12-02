"""
Sistema de avaliação e métricas para classificação molecular.

Implementa exatamente as mesmas métricas e lógica de avaliação
do classifier.py original.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any, Union
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, fbeta_score, 
    matthews_corrcoef, average_precision_score, brier_score_loss
)


class ModelEvaluator:
    """
    Avaliador de modelo que implementa exatamente a mesma lógica
    do método evaluate() do classifier.py original.
    """
    
    def __init__(self, device: torch.device):
        """
        Args:
            device: Device para computações (CPU/CUDA)
        """
        self.device = device
        self.criterion = nn.BCELoss()
    
    def evaluate(self, model: nn.Module, dataloader: torch.utils.data.DataLoader) -> Dict[str, Union[float, int]]:
        """
        Avalia o modelo e retorna um dicionário com as métricas calculadas.
        
        IMPLEMENTAÇÃO IDÊNTICA ao classifier.py original.
        
        Args:
            model: Modelo PyTorch para avaliar
            dataloader: DataLoader com dados para avaliação
            
        Returns:
            Dicionário com todas as métricas calculadas
        """
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = model(X_batch)
                loss = self.criterion(outputs, y_batch)
                total_loss += loss.item()
                preds = (outputs >= 0.5).float()
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
                all_probs.extend(outputs.cpu().numpy().flatten())
                all_preds.extend(preds.cpu().numpy().flatten())
                all_labels.extend(y_batch.cpu().numpy().flatten())
        
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        
        # Handle empty dataloader case
        if total == 0:
            self.logger.warning("Empty dataloader - returning zero metrics")
            return {
                'loss': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0,
                'f1': 0.0, 'roc_auc': 0.5, 'specificity': 0.0, 'mcc': 0.0,
                'fbeta_0.5': 0.0, 'fbeta_2': 0.0, 'avg_precision': 0.0,
                'confusion_matrix': {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
            }
        
        # Cálculo das métricas - EXATAMENTE como no original
        acc = correct / total
        precision = np.sum((all_preds == 1) & (all_labels == 1)) / max(1, np.sum(all_preds == 1))
        recall = np.sum((all_preds == 1) & (all_labels == 1)) / max(1, np.sum(all_labels == 1))
        f1 = 2 * (precision * recall) / max(1e-6, precision + recall)
        roc_auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5
        
        # Matriz de confusão - EXATAMENTE como no original
        if len(np.unique(all_labels)) > 1:
            tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
        else:
            if all_labels[0] == 0:
                tn, fp, fn, tp = len(all_labels), 0, 0, 0
            else:
                tn, fp, fn, tp = 0, 0, 0, len(all_labels)
        
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        fbeta_0_5 = fbeta_score(all_labels, all_preds, beta=0.5, zero_division=0)
        fbeta_2 = fbeta_score(all_labels, all_preds, beta=2, zero_division=0)
        mcc = matthews_corrcoef(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0
        avg_precision = average_precision_score(all_labels, all_probs)
        brier = brier_score_loss(all_labels, all_probs)
        
        # Retorno IDÊNTICO ao original
        metrics = {
            "Loss": total_loss / len(dataloader),
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "ROC_AUC": roc_auc,
            "True_Negatives": int(tn),
            "False_Positives": int(fp),
            "False_Negatives": int(fn),
            "True_Positives": int(tp),
            "Specificity": specificity,
            "Fbeta_0.5": fbeta_0_5,
            "Fbeta_2": fbeta_2,
            "MCC": mcc,
            "Average_Precision": avg_precision,
            "Brier_Score": brier
        }
        
        return metrics


class DataTypeConverter:
    """
    Conversor de tipos de dados para serialização JSON.
    
    Implementa exatamente a mesma lógica do método convert_to_native()
    do classifier.py original.
    """
    
    @staticmethod
    def convert_to_native(data: Any) -> Any:
        """
        Converte recursivamente dicionários e listas para usar tipos nativos do Python.
        
        IMPLEMENTAÇÃO IDÊNTICA ao classifier.py original.
        
        Args:
            data: Dados para converter
            
        Returns:
            Dados convertidos para tipos nativos do Python
        """
        if isinstance(data, dict):
            return {k: DataTypeConverter.convert_to_native(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [DataTypeConverter.convert_to_native(v) for v in data]
        elif isinstance(data, (np.float64, np.float32)):
            return float(data)
        elif isinstance(data, (np.int64, np.int32)):
            return int(data)
        else:
            return data


# Funções de conveniência para compatibilidade com código existente
def evaluate_model(model: nn.Module, 
                  dataloader: torch.utils.data.DataLoader,
                  device: torch.device) -> Dict[str, Union[float, int]]:
    """
    Função de conveniência para avaliar modelo.
    
    Args:
        model: Modelo para avaliar
        dataloader: DataLoader com dados
        device: Device de computação
        
    Returns:
        Dicionário com métricas
    """
    evaluator = ModelEvaluator(device)
    return evaluator.evaluate(model, dataloader)


def convert_to_native_types(data: Any) -> Any:
    """
    Função de conveniência para converter tipos.
    
    Args:
        data: Dados para converter
        
    Returns:
        Dados convertidos para tipos nativos
    """
    return DataTypeConverter.convert_to_native(data)


if __name__ == "__main__":
    # Teste básico do sistema de métricas
    print("🧪 Testando sistema de métricas...")
    
    # Simulação de dados
    import torch.utils.data as data_utils
    
    # Dados sintéticos
    n_samples = 100
    n_features = 64
    
    X = torch.randn(n_samples, n_features)
    y = torch.randint(0, 2, (n_samples, 1)).float()
    
    dataset = data_utils.TensorDataset(X, y)
    dataloader = data_utils.DataLoader(dataset, batch_size=16, shuffle=False)
    
    # Modelo simples para teste
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(n_features, 1)
            
        def forward(self, x):
            return torch.sigmoid(self.fc(x))
    
    device = torch.device("cpu")
    model = SimpleModel().to(device)
    
    # Testar avaliação
    evaluator = ModelEvaluator(device)
    metrics = evaluator.evaluate(model, dataloader)
    
    print("✅ Métricas calculadas:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")
    
    # Testar conversão de tipos
    converted = convert_to_native_types(metrics)
    print(f"✅ Conversão de tipos: {type(converted['Loss'])}")
    
    print("🎯 Sistema de métricas funcionando perfeitamente!")
