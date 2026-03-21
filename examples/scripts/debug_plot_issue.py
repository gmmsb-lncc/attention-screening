#!/usr/bin/env python3
"""
Script de debug para investigar problema com matplotlib.
"""

import sys
sys.path.insert(0, 'scripts')

from visualization.data_loader import load_results_from_files
from visualization.metrics_extractor import MetricsExtractor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

print("=" * 60)
print("DEBUG: Investigando problema com matplotlib")
print("=" * 60)

# 1. Carregar dados
print("\n1. Carregando dados...")
results = load_results_from_files(['results/integrated_results_esm2_t36_3B.json'])
extractor = MetricsExtractor(results)

classification_data = extractor.extract_classification_metrics()
regression_data = extractor.extract_regression_metrics()

print(f"   Classification data: {len(classification_data)} items")
print(f"   Regression data: {len(regression_data)} items")

# 2. Criar DataFrames
print("\n2. Criando DataFrames...")
df_class = pd.DataFrame(classification_data)
df_reg = pd.DataFrame(regression_data)

print(f"   Classification DataFrame shape: {df_class.shape}")
print(f"   Regression DataFrame shape: {df_reg.shape}")

# 3. Verificar tipos de dados
print("\n3. Tipos de dados:")
print("\n   Classification dtypes:")
for col in df_class.columns:
    print(f"      {col}: {df_class[col].dtype}")

print("\n   Regression dtypes:")
for col in df_reg.columns:
    print(f"      {col}: {df_reg[col].dtype}")

# 4. Preparar dados para plot de classificação
print("\n4. Preparando dados de classificação...")
metrics = ['Accuracy', 'F1', 'ROC_AUC', 'Precision', 'Recall']
df_plot_class = df_class[['Model'] + [m for m in metrics if m in df_class.columns]]

print(f"   Columns: {df_plot_class.columns.tolist()}")
print(f"   Values for F1: {df_plot_class['F1'].values}")
print(f"   Type of F1 values: {type(df_plot_class['F1'].values[0])}")

# 5. Testar plot de classificação
print("\n5. Testando plot de classificação...")
try:
    plt.figure(figsize=(8, 6))
    x_pos = np.arange(len(df_plot_class))
    plt.bar(x_pos, df_plot_class['F1'].values)
    plt.xticks(x_pos, df_plot_class['Model'].values, rotation=45)
    plt.title("Test Classification Plot")
    plt.savefig('debug_classification.png', bbox_inches='tight')
    plt.close()
    print("   ✅ Plot de classificação funcionou!")
except Exception as e:
    print(f"   ❌ Erro no plot de classificação: {e}")

# 6. Preparar dados para plot de regressão
print("\n6. Preparando dados de regressão...")
print(f"   Pearson_R values: {df_reg['Pearson_R'].values}")
print(f"   Pearson_R dtype: {df_reg['Pearson_R'].dtype}")
print(f"   Type of first value: {type(df_reg['Pearson_R'].values[0])}")

# 7. Verificar conversores de unidades
print("\n7. Verificando conversores matplotlib...")
from matplotlib import units
print(f"   Units registry: {units.registry}")

# 8. Testar plot de regressão ANTES de limpar registry
print("\n8. Testando plot de regressão SEM limpar registry...")
try:
    fig, ax = plt.subplots(figsize=(8, 6))
    x_pos = np.arange(len(df_reg))
    print(f"   x_pos: {x_pos}")
    print(f"   x_pos dtype: {x_pos.dtype}")
    print(f"   y values: {df_reg['Pearson_R'].values}")
    print(f"   y dtype: {df_reg['Pearson_R'].dtype}")
    
    # Verificar converters nos eixos
    print(f"   ax.xaxis converter: {ax.xaxis.converter}")
    print(f"   ax.yaxis converter: {ax.yaxis.converter}")
    
    bars = ax.bar(x_pos, df_reg['Pearson_R'].values)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df_reg['Model'].values, rotation=45)
    ax.set_title("Test Regression Plot")
    plt.savefig('debug_regression_before.png', bbox_inches='tight')
    plt.close()
    print("   ✅ Plot de regressão funcionou!")
except Exception as e:
    print(f"   ❌ Erro no plot de regressão: {type(e).__name__}: {e}")
    import traceback
    print("\n   Stack trace completo:")
    traceback.print_exc()

# 9. Limpar registry e tentar novamente
print("\n9. Limpando registry e tentando novamente...")
from matplotlib import units
units.registry.clear()
print(f"   Registry após limpeza: {units.registry}")

try:
    fig, ax = plt.subplots(figsize=(8, 6))
    x_pos = np.arange(len(df_reg))
    
    # Verificar converters nos eixos
    print(f"   ax.xaxis converter: {ax.xaxis.converter}")
    print(f"   ax.yaxis converter: {ax.yaxis.converter}")
    
    bars = ax.bar(x_pos, df_reg['Pearson_R'].values)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df_reg['Model'].values, rotation=45)
    ax.set_title("Test Regression Plot - After Clear")
    plt.savefig('debug_regression_after.png', bbox_inches='tight')
    plt.close()
    print("   ✅ Plot de regressão funcionou após limpar registry!")
except Exception as e:
    print(f"   ❌ Ainda falhou: {type(e).__name__}: {e}")
    import traceback
    print("\n   Stack trace completo:")
    traceback.print_exc()

# 10. Testar com tolist() explícito
print("\n10. Testando com conversão explícita para list...")
try:
    fig, ax = plt.subplots(figsize=(8, 6))
    x_pos = list(range(len(df_reg)))
    y_values = df_reg['Pearson_R'].tolist()
    
    print(f"   x_pos type: {type(x_pos)}, first element: {type(x_pos[0])}")
    print(f"   y_values type: {type(y_values)}, first element: {type(y_values[0])}")
    
    bars = ax.bar(x_pos, y_values)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df_reg['Model'].tolist(), rotation=45)
    ax.set_title("Test Regression Plot - With tolist()")
    plt.savefig('debug_regression_tolist.png', bbox_inches='tight')
    plt.close()
    print("   ✅ Plot com tolist() funcionou!")
except Exception as e:
    print(f"   ❌ Falhou com tolist(): {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("DEBUG concluído. Verifique os arquivos debug_*.png")
print("=" * 60)
