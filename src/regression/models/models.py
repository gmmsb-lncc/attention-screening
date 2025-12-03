#!/usr/bin/env python3
"""
Regression Models - DockTKinase
================================

Defines all regression algorithms to be tested.
"""

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    AdaBoostRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


class RegressionModels:
    """
    Factory for creating regression models.
    
    Supports 12 regression algorithms (ordered from fastest to slowest):
    1. Ridge (~2s) - L2 linear regression
    2. Lasso (~3s) - L1 linear regression
    3. ElasticNet (~3s) - L1 + L2
    4. DecisionTree (~5s) - Single tree
    5. LinearSVR (~15s) - Linear SVM
    6. LightGBM (~20s) - Optimized gradient boosting
    7. XGBoost (~25s) - State-of-art boosting
    8. ExtraTrees (~40s) - Fast ensemble
    9. RandomForest (~60s) - Robust ensemble
    10. KNN (~120s) - Instance-based
    11. GradientBoosting (~180s) - Sklearn boosting
    12. MLP (~300s) - Neural network
    """
    
    @staticmethod
    def get_all_models(random_state=42, verbose=False):
        """
        Returns dictionary with all available models.
        Ordered from fastest to slowest.
        
        Args:
            random_state: Seed for reproducibility
            verbose: Show progress (where applicable)
            
        Returns:
            Dict[str, Regressor]: Dictionary {name: model}
        """
        models = {}
        
        # 1. Ridge (~2s) - The fastest
        models['Ridge'] = Ridge(
            alpha=1.0,
            random_state=random_state
        )
        
        # 2. Lasso (~3s) - Very fast
        models['Lasso'] = Lasso(
            alpha=1.0,
            max_iter=2000,
            random_state=random_state
        )
        
        # 3. ElasticNet (~3s) - Fast
        models['ElasticNet'] = ElasticNet(
            alpha=1.0,
            l1_ratio=0.5,
            max_iter=2000,
            random_state=random_state
        )
        
        # 4. Decision Tree (~5s) - Very fast
        models['DecisionTree'] = DecisionTreeRegressor(
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state
        )
        
        # 5. LinearSVR (~15s) - Scalable linear SVM
        models['LinearSVR'] = LinearSVR(
            C=1.0,
            epsilon=0.1,
            max_iter=2000,
            dual='auto',
            random_state=random_state
        )
        
        # 6. LightGBM (~20s) - Highly optimized gradient boosting
        models['LightGBM'] = LGBMRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=random_state,
            verbose=-1
        )
        
        # 7. XGBoost (~25s) - State-of-art gradient boosting
        models['XGBoost'] = XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=random_state,
            verbosity=0
        )
        
        # 8. Extra Trees (~40s) - Fast ensemble
        models['ExtraTrees'] = ExtraTreesRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=random_state,
            verbose=0
        )
        
        # 9. Random Forest (~60s) - Robust ensemble
        models['RandomForest'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=random_state,
            verbose=0
        )
        
        # 10. KNN (~120s) - Instance-based, slow on prediction
        models['KNN'] = KNeighborsRegressor(
            n_neighbors=5,
            weights='distance',
            algorithm='auto',
            n_jobs=-1
        )
        
        # 11. Gradient Boosting (~180s) - Sklearn sequential boosting
        models['GradientBoosting'] = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=random_state,
            verbose=0
        )
        
        # 12. MLP - Neural network with early stopping
        models['MLP'] = MLPRegressor(
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
        Returns specific model with custom parameters.
        
        Args:
            name: Model name
            random_state: Seed for reproducibility
            **kwargs: Additional model parameters
            
        Returns:
            Configured model
        """
        all_models = RegressionModels.get_all_models(random_state=random_state)
        
        if name not in all_models:
            available = ', '.join(all_models.keys())
            raise ValueError(
                f"Model '{name}' not available. "
                f"Available models: {available}"
            )
        
        model = all_models[name]
        
        # Update parameters if provided
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
        models = RegressionModels.get_all_models()
        return list(models.keys())
    
    @staticmethod
    def print_available_models():
        """Prints list of available models with status (ordered by speed)."""
        print('Available Regression Models (fastest → slowest):')
        print('=' * 60)
        
        # Ordenados do mais rápido ao mais lento
        models_ordered = [
            ('Ridge', '~2s'),
            ('Lasso', '~3s'),
            ('ElasticNet', '~3s'),
            ('DecisionTree', '~5s'),
            ('LinearSVR', '~15s'),
            ('LightGBM', '~20s'),
            ('XGBoost', '~25s'),
            ('ExtraTrees', '~40s'),
            ('RandomForest', '~60s'),
            ('KNN', '~120s'),
            ('GradientBoosting', '~180s'),
            ('MLP', '~300s')
        ]
        
        for i, (model, time) in enumerate(models_ordered, 1):
            print(f'  {i:2d}. ✅ {model:<20} {time}')
        
        print('=' * 60)
        print(f'Total: 12 models')


if __name__ == '__main__':
    # Test
    RegressionModels.print_available_models()
