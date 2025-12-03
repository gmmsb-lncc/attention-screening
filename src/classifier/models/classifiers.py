#!/usr/bin/env python3
"""
Modelos de Classificação - DockTKinase
======================================

Define todos os algoritmos de classificação a serem testados.
Equivalente ao models.py de regressão, mas para classificação binária.
"""

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


class ClassificationModels:
    """
    Factory para criação de modelos de classificação binária.
    
    Suporta 12 algoritmos de classificação (ordenados do mais rápido ao mais lento):
    1. NaiveBayes (~2s) - Muito rápido, baseline probabilístico
    2. DecisionTree (~5s) - Rápido, interpretável
    3. LogisticRegression (~10s) - Baseline linear
    4. LinearSVC (~15s) - SVM linear, escalável
    5. LightGBM (~20s) - Gradient boosting otimizado
    6. XGBoost (~25s) - State-of-art gradient boosting
    7. ExtraTrees (~40s) - Ensemble rápido
    8. RandomForest (~60s) - Ensemble robusto
    9. AdaBoost (~80s) - Boosting clássico
    10. KNN (~120s) - Instance-based
    11. GradientBoosting (~180s) - Sklearn boosting
    12. MLP (~300s) - Rede neural
    
    Equivalente ao RegressionModels, mas para classificação.
    """
    
    @staticmethod
    def get_all_models(random_state=42, verbose=False):
        """
        Retorna dicionário com todos os modelos disponíveis.
        Ordenados do mais rápido ao mais lento.
        
        Args:
            random_state: Seed para reprodutibilidade
            verbose: Mostrar progresso (onde aplicável)
            
        Returns:
            Dict[str, Classifier]: Dicionário {nome: modelo}
        """
        models = {}
        
        # 1. Naive Bayes (~2s) - O mais rápido
        models['NaiveBayes'] = GaussianNB()
        
        # 2. Decision Tree (~5s) - Muito rápido
        models['DecisionTree'] = DecisionTreeClassifier(
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=random_state
        )
        
        # 3. Logistic Regression (~10s) - Baseline linear rápido
        models['LogisticRegression'] = LogisticRegression(
            C=1.0,
            penalty='l2',
            max_iter=1000,
            class_weight='balanced',
            n_jobs=-1,
            random_state=random_state
        )
        
        # 4. Linear SVC (~15s) - SVM linear escalável
        models['LinearSVC'] = LinearSVC(
            C=1.0,
            max_iter=2000,
            class_weight='balanced',
            dual='auto',
            random_state=random_state
        )
        
        # 5. LightGBM (~20s) - Gradient boosting muito otimizado
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
        
        # 6. XGBoost (~25s) - State-of-art gradient boosting
        models['XGBoost'] = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1,
            n_jobs=-1,
            random_state=random_state,
            verbosity=0,
            eval_metric='logloss'
        )
        
        # 7. Extra Trees (~40s) - Ensemble rápido
        models['ExtraTrees'] = ExtraTreesClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            n_jobs=-1,
            random_state=random_state,
            verbose=0
        )
        
        # 8. Random Forest (~60s) - Ensemble robusto
        models['RandomForest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            n_jobs=-1,
            random_state=random_state,
            verbose=0
        )
        
        # 9. AdaBoost (~80s) - Boosting clássico
        models['AdaBoost'] = AdaBoostClassifier(
            n_estimators=100,
            learning_rate=0.5,
            random_state=random_state,
            algorithm='SAMME'
        )
        
        # 10. KNN (~120s) - Instance-based, lento na predição
        models['KNN'] = KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            n_jobs=-1
        )
        
        # 11. Gradient Boosting (~180s) - Sklearn boosting sequencial
        models['GradientBoosting'] = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=random_state,
            verbose=0
        )
        
        # 12. MLP (~300s) - Rede neural, o mais lento
        models['MLP'] = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            max_iter=50,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
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
        """Imprime lista de modelos disponíveis com status (ordenados por velocidade)."""
        print('Modelos de Classificação Disponíveis (mais rápido → mais lento):')
        print('=' * 60)
        
        # Ordenados do mais rápido ao mais lento
        models_ordered = [
            ('NaiveBayes', '~2s'),
            ('DecisionTree', '~5s'),
            ('LogisticRegression', '~10s'),
            ('LinearSVC', '~15s'),
            ('LightGBM', '~20s'),
            ('XGBoost', '~25s'),
            ('ExtraTrees', '~40s'),
            ('RandomForest', '~60s'),
            ('AdaBoost', '~80s'),
            ('KNN', '~120s'),
            ('GradientBoosting', '~180s'),
            ('MLP', '~300s')
        ]
        
        for i, (model, time) in enumerate(models_ordered, 1):
            print(f'  {i:2d}. ✅ {model:<20} {time}')
        
        print('=' * 60)
        print(f'Total: 12 modelos')
    
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
            'LinearSVC': {
                'type': 'linear',
                'description': 'Linear Support Vector Classifier (100-1000x mais rápido que SVC-RBF)',
                'strengths': 'Muito rápido, bom para alta dimensão, escalável',
                'weaknesses': 'Não captura não-linearidades complexas'
            },
            'ExtraTrees': {
                'type': 'ensemble',
                'description': 'Extremely Randomized Trees (mais rápido que Random Forest)',
                'strengths': 'Muito rápido, reduz variância, robusto',
                'weaknesses': 'Pode ter acurácia ligeiramente menor que RF'
            },
            'DecisionTree': {
                'type': 'tree',
                'description': 'Árvore de decisão única (baseline simples)',
                'strengths': 'Muito interpretável, rápido, não requer normalização',
                'weaknesses': 'Tende a overfit, instável'
            },
            'AdaBoost': {
                'type': 'ensemble',
                'description': 'Adaptive Boosting (boosting clássico)',
                'strengths': 'Simples, robusto, bom com weak learners',
                'weaknesses': 'Sensível a outliers, mais lento que alguns ensemble'
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
