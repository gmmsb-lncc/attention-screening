#!/usr/bin/env python3
"""
Modelos de Classificação - DockTKinase
======================================

Define todos os algoritmos de classificação a serem testados.
Equivalente ao models.py de regressão, mas para classificação binária.
"""

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

# Modelos opcionais (podem não estar instalados)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


class ClassificationModels:
    """
    Factory para criação de modelos de classificação binária.
    
    Suporta 11+ algoritmos diferentes de classificação.
    Equivalente ao RegressionModels, mas para classificação.
    """
    
    @staticmethod
    def get_all_models(random_state=42, verbose=False):
        """
        Retorna dicionário com todos os modelos disponíveis.
        
        Args:
            random_state: Seed para reprodutibilidade
            verbose: Mostrar progresso (onde aplicável)
            
        Returns:
            Dict[str, Classifier]: Dicionário {nome: modelo}
        """
        models = {}
        
        # 1. Random Forest Classifier
        models['RandomForest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',  # Para lidar com desbalanceamento
            n_jobs=-1,
            random_state=random_state,
            verbose=0
        )
        
        # 2. Gradient Boosting Classifier
        models['GradientBoosting'] = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=random_state,
            verbose=0
        )
        
        # 3. Logistic Regression (baseline linear)
        models['LogisticRegression'] = LogisticRegression(
            C=1.0,
            penalty='l2',
            max_iter=1000,
            class_weight='balanced',
            n_jobs=-1,
            random_state=random_state
        )
        
        # 4. Support Vector Classifier
        models['SVC'] = SVC(
            kernel='rbf',
            C=1.0,
            probability=True,  # Necessário para ROC-AUC
            class_weight='balanced',
            cache_size=1000,
            random_state=random_state
        )
        
        # 5. K-Nearest Neighbors Classifier
        models['KNN'] = KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            n_jobs=-1
        )
        
        # 6. Multi-Layer Perceptron Classifier (sklearn)
        models['MLP'] = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=random_state,
            verbose=False
        )
        
        # 7. Naive Bayes
        models['NaiveBayes'] = GaussianNB()
        
        # 8. XGBoost Classifier (se disponível)
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=1,  # Ajustar se houver desbalanceamento
                n_jobs=-1,
                random_state=random_state,
                verbosity=0,
                eval_metric='logloss'
            )
        
        # 9. LightGBM Classifier (se disponível)
        if LIGHTGBM_AVAILABLE:
            models['LightGBM'] = LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight='balanced',
                n_jobs=-1,
                random_state=random_state,
                verbose=-1
            )
        
        # 10. CatBoost Classifier (se disponível)
        if CATBOOST_AVAILABLE:
            models['CatBoost'] = CatBoostClassifier(
                iterations=100,
                depth=6,
                learning_rate=0.1,
                auto_class_weights='Balanced',
                random_state=random_state,
                verbose=False
            )
        
        return models
    
    @staticmethod
    def get_model(name, random_state=42, **kwargs):
        """
        Retorna modelo específico com parâmetros customizados.
        
        Args:
            name: Nome do modelo
            random_state: Seed para reprodutibilidade
            **kwargs: Parâmetros adicionais do modelo
            
        Returns:
            Modelo configurado
        """
        all_models = ClassificationModels.get_all_models(random_state=random_state)
        
        if name not in all_models:
            available = ', '.join(all_models.keys())
            raise ValueError(
                f"Modelo '{name}' não disponível. "
                f"Modelos disponíveis: {available}"
            )
        
        model = all_models[name]
        
        # Atualizar parâmetros se fornecidos
        if kwargs:
            model.set_params(**kwargs)
        
        return model
    
    @staticmethod
    def get_available_models():
        """
        Retorna lista de modelos disponíveis no sistema.
        
        Returns:
            List[str]: Lista de nomes dos modelos
        """
        models = ClassificationModels.get_all_models()
        return list(models.keys())
    
    @staticmethod
    def print_available_models():
        """Imprime lista de modelos disponíveis com status."""
        print('Modelos de Classificação Disponíveis:')
        print('=' * 50)
        
        base_models = [
            'RandomForest', 'GradientBoosting', 'LogisticRegression',
            'SVC', 'KNN', 'MLP', 'NaiveBayes'
        ]
        
        for model in base_models:
            print(f'  ✅ {model}')
        
        # Opcionais
        if XGBOOST_AVAILABLE:
            print(f'  ✅ XGBoost')
        else:
            print(f'  ⚠️  XGBoost (não instalado)')
        
        if LIGHTGBM_AVAILABLE:
            print(f'  ✅ LightGBM')
        else:
            print(f'  ⚠️  LightGBM (não instalado)')
        
        if CATBOOST_AVAILABLE:
            print(f'  ✅ CatBoost')
        else:
            print(f'  ⚠️  CatBoost (não instalado)')
        
        print('=' * 50)
    
    @staticmethod
    def get_model_info():
        """
        Retorna informações sobre cada modelo.
        
        Returns:
            Dict com informações dos modelos
        """
        return {
            'RandomForest': {
                'type': 'ensemble',
                'description': 'Random Forest com balanceamento de classes',
                'strengths': 'Robusto, não requer normalização, lida bem com não-linearidade',
                'weaknesses': 'Pode ser lento em datasets muito grandes'
            },
            'GradientBoosting': {
                'type': 'ensemble',
                'description': 'Gradient Boosting sequencial',
                'strengths': 'Alta acurácia, bom para relações complexas',
                'weaknesses': 'Mais lento que RF, sensível a overfitting'
            },
            'LogisticRegression': {
                'type': 'linear',
                'description': 'Regressão logística (baseline linear)',
                'strengths': 'Rápido, interpretável, bom baseline',
                'weaknesses': 'Assume relações lineares'
            },
            'SVC': {
                'type': 'kernel',
                'description': 'Support Vector Classifier com kernel RBF',
                'strengths': 'Bom para espaços de alta dimensão',
                'weaknesses': 'Lento em datasets grandes, requer normalização'
            },
            'KNN': {
                'type': 'instance-based',
                'description': 'K-Nearest Neighbors com distância ponderada',
                'strengths': 'Simples, não paramétrico',
                'weaknesses': 'Lento na predição, sensível à escala'
            },
            'MLP': {
                'type': 'neural_network',
                'description': 'Multi-Layer Perceptron (sklearn)',
                'strengths': 'Captura não-linearidades complexas',
                'weaknesses': 'Requer tuning, pode overfit'
            },
            'NaiveBayes': {
                'type': 'probabilistic',
                'description': 'Gaussian Naive Bayes',
                'strengths': 'Muito rápido, bom baseline',
                'weaknesses': 'Assume independência entre features'
            },
            'XGBoost': {
                'type': 'ensemble',
                'description': 'Extreme Gradient Boosting',
                'strengths': 'State-of-art, muito rápido, excelente performance',
                'weaknesses': 'Muitos hiperparâmetros para tunar'
            },
            'LightGBM': {
                'type': 'ensemble',
                'description': 'Light Gradient Boosting Machine',
                'strengths': 'Muito rápido, eficiente em memória',
                'weaknesses': 'Pode overfit em datasets pequenos'
            },
            'CatBoost': {
                'type': 'ensemble',
                'description': 'Categorical Boosting',
                'strengths': 'Lida bem com categóricas, robusto',
                'weaknesses': 'Mais lento que LightGBM'
            }
        }


if __name__ == '__main__':
    # Test
    print('🧪 Testando ClassificationModels...\n')
    
    ClassificationModels.print_available_models()
    
    print('\n📊 Informações dos Modelos:\n')
    for name, info in ClassificationModels.get_model_info().items():
        print(f'{name}:')
        print(f'  Tipo: {info["type"]}')
        print(f'  Descrição: {info["description"]}')
        print(f'  Forças: {info["strengths"]}')
        print(f'  Fraquezas: {info["weaknesses"]}')
        print()
