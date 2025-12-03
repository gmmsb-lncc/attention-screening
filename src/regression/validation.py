#!/usr/bin/env python3
"""
Data Validation - DockTKinase Regression
==========================================

Functions for robust data and parameter validation.
"""

import numpy as np
import warnings
from typing import Tuple, Optional, List


def validate_regression_data(X, y, feature_names: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validates input data for regression with extensive checks.
    
    Args:
        X: Features (array-like)
        y: Target values (array-like)
        feature_names: Feature names (optional)
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Validated X and y
        
    Raises:
        ValueError: If data is invalid
        TypeError: If types are incompatible
    """
    # 1. Type validation and conversion
    if X is None or y is None:
        raise ValueError('X and y cannot be None')
    
    try:
        X = np.asarray(X, dtype=np.float64)
    except (ValueError, TypeError) as e:
        raise TypeError(f'Could not convert X to numeric array: {e}')
    
    try:
        y = np.asarray(y, dtype=np.float64)
    except (ValueError, TypeError) as e:
        raise TypeError(f'Could not convert y to numeric array: {e}')
    
    # 2. Dimension validation
    if X.ndim != 2:
        raise ValueError(f'X must be 2D (samples, features), received {X.ndim}D with shape {X.shape}')
    
    if y.ndim not in (1, 2):
        raise ValueError(f'y must be 1D or 2D, received {y.ndim}D with shape {y.shape}')
    
    # Flatten y if it's a single column
    if y.ndim == 2:
        if y.shape[1] == 1:
            y = y.ravel()
        else:
            raise ValueError(f'y must have only 1 column, received {y.shape[1]} columns')
    
    # 3. Size compatibility validation
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f'Incompatible number of samples:\n'
            f'  X: {X.shape[0]} samples\n'
            f'  y: {y.shape[0]} samples'
        )
    
    # 4. Minimum size validation
    if X.shape[0] == 0:
        raise ValueError('Dataset is empty (0 samples)')
    
    if X.shape[0] < 5:
        warnings.warn(
            f'Dataset too small: only {X.shape[0]} samples. '
            f'Recommended: at least 10 samples.',
            UserWarning,
            stacklevel=2
        )
    
    if X.shape[1] == 0:
        raise ValueError('No features present (0 columns)')
    
    # 5. Check for invalid values in X
    n_nan_x = np.sum(np.isnan(X))
    n_inf_x = np.sum(np.isinf(X))
    
    if n_nan_x > 0 or n_inf_x > 0:
        raise ValueError(
            f'X contains invalid values:\n'
            f'  NaN: {n_nan_x}\n'
            f'  Inf: {n_inf_x}\n'
            f'Apply imputation or remove rows with missing values.'
        )
    
    # 6. Check for invalid values in y
    n_nan_y = np.sum(np.isnan(y))
    n_inf_y = np.sum(np.isinf(y))
    
    if n_nan_y > 0 or n_inf_y > 0:
        raise ValueError(
            f'y contains invalid values:\n'
            f'  NaN: {n_nan_y}\n'
            f'  Inf: {n_inf_y}\n'
            f'Remove or impute missing target values.'
        )
    
    # 7. Variance check (constant features)
    if X.shape[1] > 0:
        variances = np.var(X, axis=0)
        zero_var_features = np.sum(variances == 0)
        
        if zero_var_features > 0:
            warnings.warn(
                f'{zero_var_features}/{X.shape[1]} features have zero variance (constant).\n'
                f'Consider removing them as they do not contribute to the model.',
                UserWarning,
                stacklevel=2
            )
            
            # If we have names, show which features
            if feature_names is not None and len(feature_names) == X.shape[1]:
                const_features = [feature_names[i] for i in range(len(variances)) if variances[i] == 0]
                if len(const_features) <= 10:  # Show up to 10
                    warnings.warn(
                        f'Features constantes: {const_features}',
                        UserWarning,
                        stacklevel=2
                    )
    
    # 8. Variance check in y
    y_var = np.var(y)
    if y_var == 0:
        raise ValueError('Target (y) has zero variance. All values are equal.')
    
    if y_var < 1e-10:
        warnings.warn(
            f'Target (y) has very low variance ({y_var:.2e}). '
            f'Check if the data is correct.',
            UserWarning,
            stacklevel=2
        )
    
    # 9. Descriptive statistics (warnings)
    y_min, y_max = np.min(y), np.max(y)
    y_range = y_max - y_min
    
    if y_range < 1e-6:
        warnings.warn(
            f'Very small y range ({y_range:.2e}). '
            f'This may make prediction difficult.',
            UserWarning,
            stacklevel=2
        )
    
    # 10. Check extreme outliers in y (more than 10 standard deviations)
    y_std = np.std(y)
    y_mean = np.mean(y)
    
    if y_std > 0:
        outliers = np.abs(y - y_mean) > 10 * y_std
        n_outliers = np.sum(outliers)
        
        if n_outliers > 0:
            warnings.warn(
                f'{n_outliers} values in y are extreme outliers (>10 std).\n'
                f'Consider investigating or removing these points.',
                UserWarning,
                stacklevel=2
            )
    
    return X, y


def validate_train_test_split(X_train, y_train, X_test, y_test, min_samples: int = 5):
    """
    Validates train/test splits.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        min_samples: Minimum samples per split
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            Validated X_train, y_train, X_test, y_test
    
    Raises:
        ValueError: If splits are invalid
    """
    # Validate each set separately
    X_train, y_train = validate_regression_data(X_train, y_train)
    X_test, y_test = validate_regression_data(X_test, y_test)
    
    # Check minimum size
    if X_train.shape[0] < min_samples:
        raise ValueError(f'Training set too small: {X_train.shape[0]} < {min_samples}')
    
    if X_test.shape[0] < min_samples:
        raise ValueError(f'Test set too small: {X_test.shape[0]} < {min_samples}')
    
    # Check feature compatibility
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            f'Incompatible number of features:\n'
            f'  Train: {X_train.shape[1]} features\n'
            f'  Test: {X_test.shape[1]} features'
        )
    
    # Warn about very different distributions
    y_train_mean = np.mean(y_train)
    y_test_mean = np.mean(y_test)
    
    if y_train_mean > 0 and y_test_mean > 0:  # Avoid division by zero
        ratio = y_train_mean / y_test_mean
        if ratio > 2 or ratio < 0.5:
            warnings.warn(
                f'Very different y means between train and test:\n'
                f'  Train: {y_train_mean:.2f}\n'
                f'  Test: {y_test_mean:.2f}\n'
                f'  Ratio: {ratio:.2f}x\n'
                f'This may indicate data leakage or non-representative splits.',
                UserWarning,
                stacklevel=2
            )
    
    # Return validated data
    return X_train, y_train, X_test, y_test


def validate_model_params(params: dict, model_type: str):
    """
    Validates model parameters.
    
    Args:
        params: Dictionary with parameters
        model_type: Model type (e.g.: 'RandomForest', 'Ridge')
    
    Raises:
        ValueError: If parameters are invalid
    """
    if not isinstance(params, dict):
        raise TypeError(f'params must be dict, received {type(params)}')
    
    # Type-specific validations
    if model_type in ['RandomForest', 'GradientBoosting']:
        if 'n_estimators' in params:
            n_est = params['n_estimators']
            if not isinstance(n_est, int) or n_est < 1:
                raise ValueError(f'n_estimators must be int >= 1, received {n_est}')
            if n_est > 1000:
                warnings.warn(
                    f'n_estimators too high ({n_est}). Training may be slow.',
                    UserWarning,
                    stacklevel=2
                )
    
    if model_type in ['Ridge', 'Lasso', 'ElasticNet']:
        if 'alpha' in params:
            alpha = params['alpha']
            if alpha < 0:
                raise ValueError(f'alpha must be >= 0, received {alpha}')
    
    if 'random_state' in params:
        rs = params['random_state']
        if rs is not None and (not isinstance(rs, int) or rs < 0):
            raise ValueError(f'random_state must be None or int >= 0, received {rs}')
