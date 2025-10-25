#!/usr/bin/env python3
"""
Utilitários de Dados - DockTKinase
==================================

Funções utilitárias compartilhadas para manipulação de dados.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, Optional


def safe_get(row_dict: Dict[str, Any], key: str, default: Any = 'N/A') -> Any:
    """
    Obter valor do dicionário tratando NaN e None de forma segura.
    
    Esta função é usada para extrair valores de dicionários derivados de
    DataFrames do Pandas, onde valores podem ser NaN, None ou ausentes.
    
    Args:
        row_dict: Dicionário contendo os dados (geralmente de df.to_dict())
        key: Chave a buscar no dicionário
        default: Valor padrão a retornar se key não existir ou valor for NaN/None
        
    Returns:
        Valor associado à chave, ou default se não existir/for NaN/None
        
    Examples:
        >>> row = {'name': 'ATP', 'value': 10.5, 'missing': np.nan}
        >>> safe_get(row, 'name')
        'ATP'
        >>> safe_get(row, 'missing')
        'N/A'
        >>> safe_get(row, 'nonexistent', default=0)
        0
    """
    value = row_dict.get(key, default)
    
    # Verificar se é NaN (funciona para np.nan e float('nan'))
    if pd.isna(value):
        return default
    
    return value


def safe_get_numeric(row_dict: Dict[str, Any], key: str, default: float = 0.0) -> float:
    """
    Obter valor numérico do dicionário tratando NaN e None.
    
    Similar a safe_get, mas garante retorno de valor numérico (float).
    
    Args:
        row_dict: Dicionário contendo os dados
        key: Chave a buscar no dicionário
        default: Valor numérico padrão (default: 0.0)
        
    Returns:
        Valor numérico associado à chave, ou default
        
    Examples:
        >>> row = {'ki': 5.5, 'kd': np.nan, 'ic50': None}
        >>> safe_get_numeric(row, 'ki')
        5.5
        >>> safe_get_numeric(row, 'kd')
        0.0
        >>> safe_get_numeric(row, 'missing', default=-1.0)
        -1.0
    """
    value = row_dict.get(key, default)
    
    if pd.isna(value):
        return default
    
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_get_int(row_dict: Dict[str, Any], key: str, default: int = 0) -> int:
    """
    Obter valor inteiro do dicionário tratando NaN e None.
    
    Similar a safe_get, mas garante retorno de valor inteiro.
    
    Args:
        row_dict: Dicionário contendo os dados
        key: Chave a buscar no dicionário
        default: Valor inteiro padrão (default: 0)
        
    Returns:
        Valor inteiro associado à chave, ou default
        
    Examples:
        >>> row = {'count': 10, 'missing': np.nan}
        >>> safe_get_int(row, 'count')
        10
        >>> safe_get_int(row, 'missing')
        0
    """
    value = row_dict.get(key, default)
    
    if pd.isna(value):
        return default
    
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_get_str(row_dict: Dict[str, Any], key: str, default: str = 'N/A') -> str:
    """
    Obter valor string do dicionário tratando NaN e None.
    
    Similar a safe_get, mas garante retorno de string.
    
    Args:
        row_dict: Dicionário contendo os dados
        key: Chave a buscar no dicionário
        default: Valor string padrão (default: 'N/A')
        
    Returns:
        Valor string associado à chave, ou default
        
    Examples:
        >>> row = {'name': 'ATP', 'desc': np.nan}
        >>> safe_get_str(row, 'name')
        'ATP'
        >>> safe_get_str(row, 'desc')
        'N/A'
    """
    value = row_dict.get(key, default)
    
    if pd.isna(value):
        return default
    
    return str(value)


# Aliases para compatibilidade com código existente
get_safe = safe_get  # Alias alternativo
