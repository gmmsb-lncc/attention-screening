# Changelog - Regression Ablation Study

## 2026-01-17 - Melhorias de Robustez e Métricas

### ✅ Novas Funcionalidades

1. **CCC (Concordance Correlation Coefficient)**
   - Adicionada métrica equivalente ao MCC para regressão
   - Fórmula de Lin: `CCC = 2·cov / (var_true + var_pred + (mean_true - mean_pred)²)`
   - Varia de -1 a 1 (1 = concordância perfeita)
   - Mais rigorosa que R² pois penaliza viés sistemático

2. **Salvamento Incremental de Checkpoints**
   - Resultados salvos após cada seed completado
   - Formato: `regression_results_{model}_{seed}.json`
   - Formato: `regression_summary_{model}_{seed}.csv`
   - Previne perda de dados em caso de interrupção

3. **Logging em Tempo Real**
   - `sys.stdout.flush()` após cada operação importante
   - Logs aparecem imediatamente (sem buffering)
   - Melhor monitoramento do progresso

### 📊 Métricas Calculadas

**Regressão:**
- R² (Coefficient of Determination)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- Pearson Correlation
- Spearman Correlation
- **CCC (Concordance Correlation Coefficient)** ⭐ NOVO

### 📁 Arquivos Gerados

**Durante execução (checkpoints):**
- `results/regression_results_{model}_seed{seed}.json`
- `results/regression_summary_{model}_seed{seed}.csv`

**Final:**
- `results/regression_results.json` (detalhado)
- `results/regression_summary.csv` (tabela resumo)

**Visualizações:**
- `figures/regression_metrics_comparison.png`
- `figures/regression_r2_focus.png`
- `figures/regression_correlation_comparison.png`
- `figures/regression_error_metrics.png`
- `figures/regression_heatmap_summary.png`
- `figures/regression_summary_table.csv`

### 🔧 Arquivos Modificados

1. **`02_run_regression.py`**
   - ✅ Adicionado cálculo de CCC em `calculate_regression_metrics()`
   - ✅ Adicionada função `save_intermediate_results()`
   - ✅ Adicionado `sys.stdout.flush()` em pontos críticos
   - ✅ Salvamento automático após cada seed

2. **`03_visualize_regression_results.py`**
   - ✅ MSE substituído por CCC no grid 2x3
   - ✅ CCC adicionado ao heatmap de correlações

3. **`README.md`**
   - ✅ Documentada métrica CCC com fórmula
   - ✅ Atualizada lista de métricas

### 🚀 Como Usar

```bash
# Ativar ambiente
source env/bin/activate

# Rodar experimentos (com checkpoints automáticos)
nohup python -u ablation/regression/scripts/02_run_regression.py > ablation/regression/regression.log 2>&1 &

# Monitorar progresso em tempo real
tail -f ablation/regression/regression.log

# Verificar checkpoints salvos
ls -lh ablation/regression/results/regression_*_seed*.json

# Gerar visualizações após conclusão
python ablation/regression/scripts/03_visualize_regression_results.py
```

### 📈 Resultados Esperados

**ESM-2 8M (Preliminar):**
- KNN R²: ~0.73-0.75
- MLP R²: ~0.73-0.77
- CCC: ~0.70-0.75 (esperado)

**ESM-2 150M/3B:**
- Resultados melhores esperados (R² > 0.80)

### ⚠️ Notas Importantes

1. **Processo atual em execução NÃO tem CCC** (iniciado antes da atualização)
2. **Próxima execução terá todas as métricas** incluindo CCC
3. **Checkpoints evitam perda de dados** em caso de interrupção
4. **Logs em tempo real** facilitam monitoramento
