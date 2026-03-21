#!/usr/bin/env python3
"""
Módulo para carregar métricas de classificação e regressão.
"""

import json
from pathlib import Path
from typing import Dict, Tuple


def load_classification_metrics(base_path: Path) -> Tuple[Dict, Dict]:
    """
    Carrega métricas de classificação (validação e teste).
    
    Args:
        base_path: Caminho base do modelo de proteína
        
    Returns:
        Tuple com (validation_metrics, test_metrics)
    """
    clf_val_path = base_path / "classifier/metrics/validation_metrics.json"
    clf_test_path = base_path / "classifier/metrics/test_metrics.json"
    
    clf_val = {}
    clf_test = {}
    
    if clf_val_path.exists():
        with open(clf_val_path) as f:
            clf_val = json.load(f)
    
    if clf_test_path.exists():
        with open(clf_test_path) as f:
            clf_test = json.load(f)
    
    return clf_val, clf_test


def load_regression_metrics(base_path: Path) -> Tuple[Dict, Dict]:
    """
    Carrega métricas de regressão (validação e teste).
    
    Args:
        base_path: Caminho base do modelo de proteína
        
    Returns:
        Tuple com (validation_metrics, test_metrics)
    """
    reg_val_path = base_path / "regression/metrics/validation_metrics.json"
    reg_test_path = base_path / "regression/metrics/test_metrics.json"
    
    reg_val = {}
    reg_test = {}
    
    if reg_val_path.exists():
        with open(reg_val_path) as f:
            reg_val = json.load(f)
    
    if reg_test_path.exists():
        with open(reg_test_path) as f:
            reg_test = json.load(f)
    
    return reg_val, reg_test


def load_all_metrics(model_name: str, base_path: Path) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Carrega todas as métricas (classificação e regressão, validação e teste).
    
    Args:
        model_name: Nome do modelo de proteína
        base_path: Caminho base do modelo
        
    Returns:
        Tuple com (clf_val, clf_test, reg_val, reg_test)
    """
    clf_val, clf_test = load_classification_metrics(base_path)
    reg_val, reg_test = load_regression_metrics(base_path)
    
    return clf_val, clf_test, reg_val, reg_test
