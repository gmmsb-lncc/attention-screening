"""
Módulo para extração e processamento de métricas.

Princípios aplicados:
- Single Responsibility: apenas extrai métricas
- Open/Closed: fácil adicionar novos tipos de métricas
- Clean Code: métodos pequenos com responsabilidades claras
"""

from typing import Dict, List


class MetricsExtractor:
    """Extrai métricas de classificação e regressão dos resultados."""
    
    # Métricas de classificação disponíveis
    CLASSIFICATION_METRICS = ['Accuracy', 'F1', 'ROC_AUC', 'MCC', 'Precision', 'Recall']
    
    # Métricas de regressão prioritárias
    REGRESSION_METRICS = ['Pearson_R', 'Pearson_P', 'RMSE', 'R2', 'MAE']
    
    def __init__(self, results: Dict[str, Dict]):
        """
        Inicializa o extrator de métricas.
        
        Args:
            results: Dicionário com resultados dos modelos
        """
        self.results = results
    
    def extract_classification_metrics(self) -> List[Dict]:
        """
        Extrai métricas de classificação de todos os modelos.
        
        Returns:
            Lista de dicts com métricas por modelo
        """
        data = []
        
        for model_name, model_data in self.results.items():
            metrics = self._get_classification_metrics(model_data)
            if metrics:
                metrics['Model'] = model_name
                data.append(metrics)
        
        return data
    
    def extract_regression_metrics(self) -> List[Dict]:
        """
        Extrai métricas de regressão de todos os modelos.
        
        Returns:
            Lista de dicts com métricas por modelo
        """
        data = []
        
        for model_name, model_data in self.results.items():
            metrics = self._get_regression_metrics(model_data)
            if metrics:
                metrics['Model'] = model_name
                data.append(metrics)
        
        return data
    
    def extract_embedding_info(self) -> List[Dict]:
        """
        Extrai informações sobre dimensões de embeddings.
        
        Returns:
            Lista de dicts com informações de embedding por modelo
        """
        data = []
        
        for model_name, model_data in self.results.items():
            info = self._get_embedding_dimensions(model_data)
            if info:
                info['Model'] = model_name
                data.append(info)
        
        return data
    
    def _get_classification_metrics(self, model_data: Dict) -> Dict:
        """
        Extrai métricas de classificação de um modelo.
        
        Args:
            model_data: Dados do modelo
            
        Returns:
            Dict com métricas extraídas
        """
        # Suportar tanto 'classifier' quanto 'classification'
        class_data = model_data.get('classifier') or model_data.get('classification')
        if not class_data:
            return {}
        
        # class_data já é o objeto correto
        best_model = class_data.get('best_model', '')
        
        metrics = {
            'Best_Classifier': best_model,
            'Num_Models': len(class_data.get('individual_results', {}))
        }
        
        # Extrair métricas do melhor modelo
        # Usar best_metrics ou buscar em individual_results
        best_metrics = class_data.get('best_metrics', {})
        if best_metrics:
            for metric in self.CLASSIFICATION_METRICS:
                metrics[metric] = best_metrics.get(metric, 0)
        elif best_model in class_data.get('individual_results', {}):
            model_results = class_data['individual_results'][best_model]
            for metric in self.CLASSIFICATION_METRICS:
                metrics[metric] = model_results.get(metric, 0)
        
        return metrics
    
    def _get_regression_metrics(self, model_data: Dict) -> Dict:
        """
        Extrai métricas de regressão de um modelo.
        
        Args:
            model_data: Dados do modelo
            
        Returns:
            Dict com métricas extraídas
        """
        if 'regression' not in model_data:
            return {}
        
        reg_data = model_data['regression']
        best_model = reg_data.get('best_model', '')
        
        metrics = {
            'Best_Regressor': best_model,
            'Num_Models': len(reg_data.get('individual_results', {}))
        }
        
        # Extrair métricas do melhor modelo
        # Usar best_metrics ou buscar em individual_results
        best_metrics = reg_data.get('best_metrics', {})
        if best_metrics:
            for metric in self.REGRESSION_METRICS:
                value = best_metrics.get(metric, 0)
                # Garantir que é numérico
                if value is None or value == 'None':
                    value = 0.0
                metrics[metric] = float(value)
        elif best_model in reg_data.get('individual_results', {}):
            model_results = reg_data['individual_results'][best_model]
            for metric in self.REGRESSION_METRICS:
                value = model_results.get(metric, 0)
                # Garantir que é numérico  
                if value is None or value == 'None':
                    value = 0.0
                metrics[metric] = float(value)
        
        return metrics
    
    def _get_embedding_dimensions(self, model_data: Dict) -> Dict:
        """
        Extrai informações sobre dimensões de embeddings.
        
        Args:
            model_data: Dados do modelo
            
        Returns:
            Dict com informações de dimensões
        """
        config = model_data.get('config', {})
        
        protein_dim = config.get('protein_embedding_dim', 0)
        ligand_dim = config.get('ligand_embedding_dim', 0)
        total_dim = protein_dim + ligand_dim
        
        return {
            'Protein_Dim': protein_dim,
            'Ligand_Dim': ligand_dim,
            'Total_Dim': total_dim
        }


def calculate_overall_score(classification_metrics: Dict, 
                           regression_metrics: Dict,
                           weights: Dict = None) -> float:
    """
    Calcula score geral combinando métricas de classificação e regressão.
    
    Args:
        classification_metrics: Métricas de classificação
        regression_metrics: Métricas de regressão
        weights: Pesos para cada métrica (opcional)
        
    Returns:
        Score geral normalizado (0-100)
    """
    if weights is None:
        weights = {
            'classification': 0.5,
            'regression': 0.5,
            'f1_weight': 0.3,
            'roc_auc_weight': 0.4,
            'mcc_weight': 0.3,
            'pearson_r_weight': 0.5,
            'rmse_weight': 0.3,
            'r2_weight': 0.2
        }
    
    # Score de classificação (média ponderada)
    class_score = (
        classification_metrics.get('F1', 0) * weights['f1_weight'] +
        classification_metrics.get('ROC_AUC', 0) * weights['roc_auc_weight'] +
        (classification_metrics.get('MCC', 0) + 1) / 2 * weights['mcc_weight']  # MCC: -1 a 1
    )
    
    # Score de regressão (média ponderada, normalizando RMSE)
    # IMPORTANTE: Pearson e R² negativos são VÁLIDOS! Não usar abs() ou max(0)
    pearson_r = regression_metrics.get('Pearson_R', 0)
    # Transformar Pearson de [-1, 1] para [0, 1] para scoring
    pearson_normalized = (pearson_r + 1) / 2
    
    rmse = regression_metrics.get('RMSE', 1)
    rmse_normalized = max(0, 1 - (rmse / 10))  # Assumindo RMSE máximo ~10
    
    # R² pode ser muito negativo, limitar a -1 para scoring
    r2 = regression_metrics.get('R2', 0)
    r2_normalized = (max(-1, r2) + 1) / 2  # Transformar [-1, 1] para [0, 1]
    
    reg_score = (
        pearson_normalized * weights['pearson_r_weight'] +
        rmse_normalized * weights['rmse_weight'] +
        r2_normalized * weights['r2_weight']
    )
    
    # Score final
    final_score = (
        class_score * weights['classification'] +
        reg_score * weights['regression']
    ) * 100
    
    return round(final_score, 2)
