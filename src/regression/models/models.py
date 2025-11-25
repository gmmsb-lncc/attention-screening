#!/usr/bin/env python3
"""
Modelos de Regressão - DockTKinase
===================================

Define todos os algoritmos de regressão a serem testados.
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
    Factory para criação de modelos de regressão.
    
    Suporta 12 algoritmos de regressão (ordenados do mais rápido ao mais lento):
    1. Ridge (~2s) - Regressão linear L2
    2. Lasso (~3s) - Regressão linear L1
    3. ElasticNet (~3s) - L1 + L2
    4. DecisionTree (~5s) - Árvore única
    5. LinearSVR (~15s) - SVM linear
    6. LightGBM (~20s) - Gradient boosting otimizado
    7. XGBoost (~25s) - State-of-art boosting
    8. ExtraTrees (~40s) - Ensemble rápido
    9. RandomForest (~60s) - Ensemble robusto
    10. KNN (~120s) - Instance-based
    11. GradientBoosting (~180s) - Sklearn boosting
    12. MLP (~300s) - Rede neural
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
            Dict[str, Regressor]: Dicionário {nome: modelo}
        """
        models = {}
        
        # 1. Ridge (~2s) - O mais rápido
        models['Ridge'] = Ridge(
            alpha=1.0,
            random_state=random_state
        )
        
        # 2. Lasso (~3s) - Muito rápido
        models['Lasso'] = Lasso(
            alpha=1.0,
            max_iter=2000,
            random_state=random_state
        )
        
        # 3. ElasticNet (~3s) - Rápido
        models['ElasticNet'] = ElasticNet(
            alpha=1.0,
            l1_ratio=0.5,
            max_iter=2000,
            random_state=random_state
        )
        
        # 4. Decision Tree (~5s) - Muito rápido
        models['DecisionTree'] = DecisionTreeRegressor(
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state
        )
        
        # 5. LinearSVR (~15s) - SVM linear escalável
        models['LinearSVR'] = LinearSVR(
            C=1.0,
            epsilon=0.1,
            max_iter=2000,
            dual='auto',
            random_state=random_state
        )
        
        # 6. LightGBM (~20s) - Gradient boosting muito otimizado
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
        
        # 8. Extra Trees (~40s) - Ensemble rápido
        models['ExtraTrees'] = ExtraTreesRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=random_state,
            verbose=0
        )
        
        # 9. Random Forest (~60s) - Ensemble robusto
        models['RandomForest'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=random_state,
            verbose=0
        )
        
        # 10. KNN (~120s) - Instance-based, lento na predição
        models['KNN'] = KNeighborsRegressor(
            n_neighbors=5,
            weights='distance',
            algorithm='auto',
            n_jobs=-1
        )
        
        # 11. Gradient Boosting (~180s) - Sklearn boosting sequencial
        models['GradientBoosting'] = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=random_state,
            verbose=0
        )
        
        # 12. MLP (~300s) - Rede neural, o mais lento
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
        """Imprime lista de modelos disponíveis com status (ordenados por velocidade)."""
        print('Modelos de Regressão Disponíveis (mais rápido → mais lento):')
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
        print(f'Total: 12 modelos')


if __name__ == '__main__':
    # Test
    RegressionModels.print_available_models()
