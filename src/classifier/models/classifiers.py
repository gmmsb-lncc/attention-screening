#!/usr/bin/env python3
"""
Classification Models - DockTKinase
====================================

Defines all classification algorithms to be tested.
All models include StandardScaler preprocessing via sklearn Pipeline
to ensure proper scaling of embeddings.
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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


def _make_pipeline(model, use_scaler=True):
    """
    Wrap a model in a Pipeline with StandardScaler.
    
    Args:
        model: sklearn-compatible classifier
        use_scaler: Whether to include StandardScaler (default True)
        
    Returns:
        Pipeline with scaler + model, or just the model if use_scaler=False
    """
    if use_scaler:
        return Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
    return model


class ClassificationModels:
    """
    Factory for creating binary classification models.
    
    All models are wrapped in sklearn Pipelines with StandardScaler
    to ensure embeddings are properly normalized. This benefits:
    - Linear models (LogisticRegression, LinearSVC): faster convergence
    - KNN: fair distance calculations
    - MLP: stable gradient updates
    - NaiveBayes: proper probability estimation
    
    Tree-based models (RF, XGB, LGBM, etc) don't need scaling,
    but including it doesn't hurt performance.
    
    Supports 12 classification algorithms (ordered from fastest to slowest):
    1. NaiveBayes (~2s) - Very fast, probabilistic baseline
    2. DecisionTree (~5s) - Fast, interpretable
    3. LogisticRegression (~10s) - Linear baseline
    4. LinearSVC (~15s) - Scalable linear SVM
    5. LightGBM (~20s) - Optimized gradient boosting
    6. XGBoost (~25s) - State-of-art gradient boosting
    7. ExtraTrees (~40s) - Fast ensemble
    8. RandomForest (~60s) - Robust ensemble
    9. AdaBoost (~80s) - Classic boosting
    10. KNN (~120s) - Instance-based
    11. GradientBoosting (~180s) - Sklearn boosting
    12. MLP (~300s) - Neural network
    """
    
    @staticmethod
    def get_all_models(random_state=42, verbose=False, use_scaler=True):
        """
        Returns dictionary with all available models.
        Ordered from fastest to slowest.
        
        All models are wrapped in sklearn Pipeline with StandardScaler
        for proper feature normalization.
        
        Args:
            random_state: Seed for reproducibility
            verbose: Show progress (where applicable)
            use_scaler: Include StandardScaler in pipeline (default True)
            
        Returns:
            Dict[str, Pipeline]: Dictionary {name: pipeline(scaler + model)}
        """
        models = {}
        
        # 1. Naive Bayes (~2s) - The fastest (BENEFITS from scaling)
        models['NaiveBayes'] = _make_pipeline(GaussianNB(), use_scaler)
        
        # 2. Decision Tree (~5s) - Very fast (scale invariant)
        models['DecisionTree'] = _make_pipeline(DecisionTreeClassifier(
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=random_state
        ), use_scaler)
        
        # 3. Logistic Regression (~10s) - Fast linear baseline (BENEFITS from scaling)
        # solver='lbfgs' with higher tol avoids ill-conditioned matrix warnings
        models['LogisticRegression'] = _make_pipeline(LogisticRegression(
            C=1.0,
            penalty='l2',
            solver='lbfgs',
            tol=1e-4,
            max_iter=1000,
            class_weight='balanced',
            n_jobs=-1,
            random_state=random_state
        ), use_scaler)
        
        # 4. Linear SVC (~15s) - Scalable linear SVM (BENEFITS from scaling)
        models['LinearSVC'] = _make_pipeline(LinearSVC(
            C=1.0,
            max_iter=2000,
            class_weight='balanced',
            dual='auto',
            random_state=random_state
        ), use_scaler)
        
        # 5. LightGBM (~20s) - Highly optimized gradient boosting (scale invariant)
        models['LightGBM'] = _make_pipeline(LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight='balanced',
            n_jobs=-1,
            random_state=random_state,
            verbose=-1
        ), use_scaler)
        
        # 6. XGBoost (~25s) - State-of-art gradient boosting (scale invariant)
        models['XGBoost'] = _make_pipeline(XGBClassifier(
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
        ), use_scaler)
        
        # 7. Extra Trees (~40s) - Fast ensemble (scale invariant)
        models['ExtraTrees'] = _make_pipeline(ExtraTreesClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            n_jobs=-1,
            random_state=random_state,
            verbose=0
        ), use_scaler)
        
        # 8. Random Forest (~60s) - Robust ensemble (scale invariant)
        models['RandomForest'] = _make_pipeline(RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            n_jobs=-1,
            random_state=random_state,
            verbose=0
        ), use_scaler)
        
        # 9. AdaBoost (~80s) - Classic boosting (scale invariant)
        models['AdaBoost'] = _make_pipeline(AdaBoostClassifier(
            n_estimators=100,
            learning_rate=0.5,
            random_state=random_state
        ), use_scaler)
        
        # 10. KNN (~120s) - Instance-based, slow on prediction (BENEFITS from scaling)
        models['KNN'] = _make_pipeline(KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            n_jobs=-1
        ), use_scaler)
        
        # 11. Gradient Boosting (~180s) - Sklearn sequential boosting (scale invariant)
        models['GradientBoosting'] = _make_pipeline(GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=random_state,
            verbose=0
        ), use_scaler)
        
        # 12. MLP (~300s) - Neural network, slowest (BENEFITS from scaling)
        models['MLP'] = _make_pipeline(MLPClassifier(
            hidden_layer_sizes=(512,),
            activation='relu',
            solver='adam',
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=random_state,
            verbose=False
        ), use_scaler)
        
        return models
    
    @staticmethod
    def get_model(name, random_state=42, use_scaler=True, **kwargs):
        """
        Returns specific model with custom parameters.
        
        Args:
            name: Model name
            random_state: Seed for reproducibility
            use_scaler: Include StandardScaler in pipeline (default True)
            **kwargs: Additional model parameters (prefixed with 'model__' for pipeline)
            
        Returns:
            Configured model pipeline
        """
        all_models = ClassificationModels.get_all_models(
            random_state=random_state, 
            use_scaler=use_scaler
        )
        
        if name not in all_models:
            available = ', '.join(all_models.keys())
            raise ValueError(
                f"Model '{name}' not available. "
                f"Available models: {available}"
            )
        
        model = all_models[name]
        
        # Update parameters if provided
        # Note: For pipeline, use 'model__param' to set inner model params
        if kwargs:
            model.set_params(**kwargs)
        
        return model
    
    @staticmethod
    def get_available_models():
        """
        Returns list of available models in the system.
        
        Returns:
            List[str]: List of model names
        """
        models = ClassificationModels.get_all_models()
        return list(models.keys())
    
    @staticmethod
    def print_available_models():
        """Prints list of available models with status (ordered by speed)."""
        print('Available Classification Models (fastest → slowest):')
        print('=' * 60)
        
        # Ordered from fastest to slowest
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
        print(f'Total: 12 models (all with StandardScaler)')
    
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
