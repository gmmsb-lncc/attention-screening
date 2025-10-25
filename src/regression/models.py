#!/usr/bin/env python3
"""
Modelos de Regressão - DockTKinase
===================================

Define todos os algoritmos de regressão a serem testados.
"""

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

# Modelos opcionais (podem não estar instalados)
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


class RegressionModels:
    """
    Factory para criação de modelos de regressão.
    
    Suporta 11+ algoritmos diferentes de regressão.
    """
    
    @staticmethod
    def get_all_models(random_state=42, verbose=False):
        """
        Retorna dicionário com todos os modelos disponíveis.
        
        Args:
            random_state: Seed para reprodutibilidade
            verbose: Mostrar progresso (onde aplicável)
            
        Returns:
            Dict[str, Regressor]: Dicionário {nome: modelo}
        """
        models = {}
        
        # 1. Random Forest Regressor
        models['RandomForest'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=random_state,
            verbose=0
        )
        
        # 2. Gradient Boosting Regressor
        models['GradientBoosting'] = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=random_state,
            verbose=0
        )
        
        # 3. Ridge Regression (L2 regularization)
        models['Ridge'] = Ridge(
            alpha=1.0,
            random_state=random_state
        )
        
        # 4. Lasso Regression (L1 regularization)
        models['Lasso'] = Lasso(
            alpha=1.0,
            max_iter=10000,  # Aumentado para evitar warnings de convergência
            random_state=random_state
        )
        
        # 5. ElasticNet (L1 + L2 regularization)
        models['ElasticNet'] = ElasticNet(
            alpha=1.0,
            l1_ratio=0.5,
            max_iter=2000,
            random_state=random_state
        )
        
        # 6. Support Vector Regressor
        models['SVR'] = SVR(
            kernel='rbf',
            C=1.0,
            epsilon=0.1,
            cache_size=1000
        )
        
        # 7. K-Nearest Neighbors Regressor
        models['KNN'] = KNeighborsRegressor(
            n_neighbors=5,
            weights='distance',
            n_jobs=-1
        )
        
        # 8. Multi-Layer Perceptron Regressor
        models['MLP'] = MLPRegressor(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=random_state,
            verbose=False
        )
        
        # 9. XGBoost Regressor (se disponível)
        if XGBOOST_AVAILABLE:
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
        
        # 10. LightGBM Regressor (se disponível)
        if LIGHTGBM_AVAILABLE:
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
        
        # 11. CatBoost Regressor (se disponível)
        if CATBOOST_AVAILABLE:
            models['CatBoost'] = CatBoostRegressor(
                iterations=100,
                depth=6,
                learning_rate=0.1,
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
        all_models = RegressionModels.get_all_models(random_state=random_state)
        
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
        models = RegressionModels.get_all_models()
        return list(models.keys())
    
    @staticmethod
    def print_available_models():
        """Imprime lista de modelos disponíveis com status."""
        print('Modelos de Regressão Disponíveis:')
        print('=' * 50)
        
        base_models = [
            'RandomForest', 'GradientBoosting', 'Ridge', 
            'Lasso', 'ElasticNet', 'SVR', 'KNN', 'MLP'
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


if __name__ == '__main__':
    # Test
    RegressionModels.print_available_models()
