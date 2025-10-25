#!/usr/bin/env python3
"""
Validação de Dados - Regressão DockTKinase
===========================================

Funções para validação robusta de dados e parâmetros.
"""

import numpy as np
import warnings
from typing import Tuple, Optional, List


def validate_regression_data(X, y, feature_names: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Valida dados de entrada para regressão com verificações extensivas.
    
    Args:
        X: Features (array-like)
        y: Target values (array-like)
        feature_names: Nomes das features (opcional)
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: X e y validados
        
    Raises:
        ValueError: Se dados forem inválidos
        TypeError: Se tipos forem incompatíveis
    """
    # 1. Validação de tipo e conversão
    if X is None or y is None:
        raise ValueError('X e y não podem ser None')
    
    try:
        X = np.asarray(X, dtype=np.float64)
    except (ValueError, TypeError) as e:
        raise TypeError(f'Não foi possível converter X para array numérico: {e}')
    
    try:
        y = np.asarray(y, dtype=np.float64)
    except (ValueError, TypeError) as e:
        raise TypeError(f'Não foi possível converter y para array numérico: {e}')
    
    # 2. Validação de dimensões
    if X.ndim != 2:
        raise ValueError(f'X deve ser 2D (samples, features), recebido {X.ndim}D com shape {X.shape}')
    
    if y.ndim not in (1, 2):
        raise ValueError(f'y deve ser 1D ou 2D, recebido {y.ndim}D com shape {y.shape}')
    
    # Achatar y se for coluna única
    if y.ndim == 2:
        if y.shape[1] == 1:
            y = y.ravel()
        else:
            raise ValueError(f'y deve ter apenas 1 coluna, recebido {y.shape[1]} colunas')
    
    # 3. Validação de compatibilidade de tamanhos
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f'Número de amostras incompatível:\n'
            f'  X: {X.shape[0]} amostras\n'
            f'  y: {y.shape[0]} amostras'
        )
    
    # 4. Validação de tamanho mínimo
    if X.shape[0] == 0:
        raise ValueError('Dataset está vazio (0 amostras)')
    
    if X.shape[0] < 5:
        warnings.warn(
            f'Dataset muito pequeno: apenas {X.shape[0]} amostras. '
            f'Recomendado: pelo menos 10 amostras.',
            UserWarning,
            stacklevel=2
        )
    
    if X.shape[1] == 0:
        raise ValueError('Nenhuma feature presente (0 colunas)')
    
    # 5. Verificação de valores inválidos em X
    n_nan_x = np.sum(np.isnan(X))
    n_inf_x = np.sum(np.isinf(X))
    
    if n_nan_x > 0 or n_inf_x > 0:
        raise ValueError(
            f'X contém valores inválidos:\n'
            f'  NaN: {n_nan_x}\n'
            f'  Inf: {n_inf_x}\n'
            f'Aplique imputação ou remova linhas com valores faltantes.'
        )
    
    # 6. Verificação de valores inválidos em y
    n_nan_y = np.sum(np.isnan(y))
    n_inf_y = np.sum(np.isinf(y))
    
    if n_nan_y > 0 or n_inf_y > 0:
        raise ValueError(
            f'y contém valores inválidos:\n'
            f'  NaN: {n_nan_y}\n'
            f'  Inf: {n_inf_y}\n'
            f'Remova ou impute valores de target faltantes.'
        )
    
    # 7. Verificação de variância (features constantes)
    if X.shape[1] > 0:
        variances = np.var(X, axis=0)
        zero_var_features = np.sum(variances == 0)
        
        if zero_var_features > 0:
            warnings.warn(
                f'{zero_var_features}/{X.shape[1]} features têm variância zero (constantes).\n'
                f'Considere removê-las pois não contribuem para o modelo.',
                UserWarning,
                stacklevel=2
            )
            
            # Se temos nomes, mostrar quais features
            if feature_names is not None and len(feature_names) == X.shape[1]:
                const_features = [feature_names[i] for i in range(len(variances)) if variances[i] == 0]
                if len(const_features) <= 10:  # Mostrar até 10
                    warnings.warn(
                        f'Features constantes: {const_features}',
                        UserWarning,
                        stacklevel=2
                    )
    
    # 8. Verificação de variância em y
    y_var = np.var(y)
    if y_var == 0:
        raise ValueError('Target (y) tem variância zero. Todos os valores são iguais.')
    
    if y_var < 1e-10:
        warnings.warn(
            f'Target (y) tem variância muito baixa ({y_var:.2e}). '
            f'Verifique se os dados estão corretos.',
            UserWarning,
            stacklevel=2
        )
    
    # 9. Estatísticas descritivas (warnings)
    y_min, y_max = np.min(y), np.max(y)
    y_range = y_max - y_min
    
    if y_range < 1e-6:
        warnings.warn(
            f'Range de y muito pequeno ({y_range:.2e}). '
            f'Isso pode dificultar a previsão.',
            UserWarning,
            stacklevel=2
        )
    
    # 10. Verificar outliers extremos em y (mais de 10 desvios padrão)
    y_std = np.std(y)
    y_mean = np.mean(y)
    
    if y_std > 0:
        outliers = np.abs(y - y_mean) > 10 * y_std
        n_outliers = np.sum(outliers)
        
        if n_outliers > 0:
            warnings.warn(
                f'{n_outliers} valores em y são outliers extremos (>10 std).\n'
                f'Considere investigar ou remover esses pontos.',
                UserWarning,
                stacklevel=2
            )
    
    return X, y


def validate_train_test_split(X_train, y_train, X_test, y_test, min_samples: int = 5):
    """
    Valida splits de treino/teste.
    
    Args:
        X_train, y_train: Dados de treino
        X_test, y_test: Dados de teste
        min_samples: Número mínimo de amostras por split
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            X_train, y_train, X_test, y_test validados
    
    Raises:
        ValueError: Se splits forem inválidos
    """
    # Validar cada conjunto separadamente
    X_train, y_train = validate_regression_data(X_train, y_train)
    X_test, y_test = validate_regression_data(X_test, y_test)
    
    # Verificar tamanho mínimo
    if X_train.shape[0] < min_samples:
        raise ValueError(f'Conjunto de treino muito pequeno: {X_train.shape[0]} < {min_samples}')
    
    if X_test.shape[0] < min_samples:
        raise ValueError(f'Conjunto de teste muito pequeno: {X_test.shape[0]} < {min_samples}')
    
    # Verificar compatibilidade de features
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            f'Número de features incompatível:\n'
            f'  Treino: {X_train.shape[1]} features\n'
            f'  Teste: {X_test.shape[1]} features'
        )
    
    # Avisar sobre distribuições muito diferentes
    y_train_mean = np.mean(y_train)
    y_test_mean = np.mean(y_test)
    
    if y_train_mean > 0 and y_test_mean > 0:  # Evitar divisão por zero
        ratio = y_train_mean / y_test_mean
        if ratio > 2 or ratio < 0.5:
            warnings.warn(
                f'Médias de y muito diferentes entre treino e teste:\n'
                f'  Treino: {y_train_mean:.2f}\n'
                f'  Teste: {y_test_mean:.2f}\n'
                f'  Ratio: {ratio:.2f}x\n'
                f'Isso pode indicar data leakage ou splits não representativos.',
                UserWarning,
                stacklevel=2
            )
    
    # Retornar dados validados
    return X_train, y_train, X_test, y_test


def validate_model_params(params: dict, model_type: str):
    """
    Valida parâmetros de um modelo.
    
    Args:
        params: Dicionário com parâmetros
        model_type: Tipo do modelo (ex: 'RandomForest', 'Ridge')
    
    Raises:
        ValueError: Se parâmetros forem inválidos
    """
    if not isinstance(params, dict):
        raise TypeError(f'params deve ser dict, recebido {type(params)}')
    
    # Validações específicas por tipo
    if model_type in ['RandomForest', 'GradientBoosting']:
        if 'n_estimators' in params:
            n_est = params['n_estimators']
            if not isinstance(n_est, int) or n_est < 1:
                raise ValueError(f'n_estimators deve ser int >= 1, recebido {n_est}')
            if n_est > 1000:
                warnings.warn(
                    f'n_estimators muito alto ({n_est}). Treinamento pode ser lento.',
                    UserWarning,
                    stacklevel=2
                )
    
    if model_type in ['Ridge', 'Lasso', 'ElasticNet']:
        if 'alpha' in params:
            alpha = params['alpha']
            if alpha < 0:
                raise ValueError(f'alpha deve ser >= 0, recebido {alpha}')
    
    if 'random_state' in params:
        rs = params['random_state']
        if rs is not None and (not isinstance(rs, int) or rs < 0):
            raise ValueError(f'random_state deve ser None ou int >= 0, recebido {rs}')
