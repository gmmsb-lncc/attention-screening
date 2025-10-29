# Guia de Visualizações do DockTKinase# Guia de Visualizações do DockTKinase



**Data**: 28 de Outubro de 2025  ## 📊 Visualizações Automáticas

**Branch**: regression  

**Sistema**: Dual Pipeline (Classification + Regression)Todos os scripts agora geram visualizações automaticamente para tornar as estatísticas mais didáticas e acessíveis.



## 📊 Visualizações Automáticas### 📁 Localização



Ambos os pipelines agora geram visualizações automaticamente para tornar as estatísticas mais didáticas e acessíveis.Todas as visualizações são salvas em:

```

### 📁 Localizaçãotests/pipeline_output/visualizations/

tests/comparison_output/visualizations/

Todas as visualizações são salvas em:```

```

results/classification/visualizations/    # Classification pipelineFormato: **PNG de alta resolução (300 DPI)** - pronto para publicações!

results/regression/visualizations/        # Regression pipeline ⭐ NOVO!

```---



Formato: **PNG de alta resolução (300 DPI)** - pronto para publicações!## 🎨 Visualizações do Pipeline (`run_complete_pipeline.py`)



---### 1. `stratification_analysis.png`



## 🎨 Visualizações do Classification Pipeline**Descrição**: Análise completa da estratificação em 3 gráficos



### 1. `stratification_analysis.png`#### Subplot 1: Distribuição de Classes

- **Tipo**: Gráfico de barras agrupadas

**Descrição**: Análise completa da estratificação em 3 gráficos- **Mostra**: Contagem absoluta de cada classe (INATIVO/ATIVO) em cada conjunto

- **Útil para**: Verificar se o número de amostras está balanceado

#### Subplot 1: Distribuição de Classes- **Valores**: Contagens exatas sobre cada barra

- **Tipo**: Gráfico de barras agrupadas

- **Mostra**: Contagem absoluta de cada classe (INATIVO/ATIVO) em cada conjunto#### Subplot 2: Proporções de Classes

- **Útil para**: Verificar se o número de amostras está balanceado- **Tipo**: Gráfico de barras agrupadas (%)

- **Valores**: Contagens exatas sobre cada barra- **Mostra**: Porcentagem de cada classe em cada conjunto

- **Útil para**: Confirmar que a estratificação mantém as proporções originais

#### Subplot 2: Proporções de Classes- **Critério de sucesso**: Proporções iguais em Train/Val/Test

- **Tipo**: Gráfico de barras agrupadas (%)- **Valores**: Porcentagens sobre cada barra

- **Mostra**: Porcentagem de cada classe em cada conjunto

- **Útil para**: Confirmar que a estratificação mantém as proporções originais#### Subplot 3: Distribuição de Amostras

- **Critério de sucesso**: Proporções iguais em Train/Val/Test- **Tipo**: Gráfico de pizza

- **Valores**: Porcentagens sobre cada barra- **Mostra**: Distribuição do total de amostras por conjunto

- **Útil para**: Visualizar o split 80/10/10

#### Subplot 3: Distribuição de Amostras- **Cores**: 

- **Tipo**: Gráfico de pizza  - Original: Cinza

- **Mostra**: Distribuição do total de amostras por conjunto  - Train: Verde

- **Útil para**: Visualizar o split 80/10/10  - Val: Laranja

- **Cores**:   - Test: Roxo

  - Original: Cinza

  - Train: Verde**Exemplo de Interpretação:**

  - Val: Laranja```

  - Test: Roxo✅ BOM: Proporções iguais em todos os subplots 2

⚠️  ATENÇÃO: Proporções diferentes indicam problema na estratificação

**Exemplo de Interpretação:**```

```

✅ BOM: Proporções iguais em todos os subplots 2---

⚠️  ATENÇÃO: Proporções diferentes indicam problema na estratificação

```### 2. `evaluation_validation.png` e `evaluation_test.png`



---**Descrição**: Avaliação do modelo em 2 gráficos



### 2. `evaluation_validation.png` e `evaluation_test.png`#### Subplot 1: Matriz de Confusão

- **Tipo**: Heatmap

**Descrição**: Avaliação do modelo em 2 gráficos- **Mostra**: Predições corretas e incorretas

- **Layout**:

#### Subplot 1: Matriz de Confusão  ```

- **Tipo**: Heatmap  Verdadeiro\Predito  INATIVO(0)  ATIVO(1)

- **Mostra**: Predições corretas e incorretas  INATIVO(0)          TN          FP

- **Layout**:  ATIVO(1)            FN          TP

  ```  ```

  Verdadeiro\Predito  INATIVO(0)  ATIVO(1)- **Valores**: Contagens absolutas + porcentagens

  INATIVO(0)          TN          FP- **Cores**: Azul (mais escuro = mais amostras)

  ATIVO(1)            FN          TP

  ```**Interpretação:**

- **Valores**: Contagens absolutas + porcentagens- **Diagonal principal** (TN e TP): Predições corretas

- **Cores**: Azul (mais escuro = mais amostras)- **Diagonal secundária** (FP e FN): Erros do modelo

- **FP (Falso Positivo)**: Classificou INATIVO como ATIVO

**Interpretação:**- **FN (Falso Negativo)**: Classificou ATIVO como INATIVO

- **Diagonal principal** (TN e TP): Predições corretas

- **Diagonal secundária** (FP e FN): Erros do modelo#### Subplot 2: Curva ROC

- **FP (Falso Positivo)**: Classificou INATIVO como ATIVO- **Tipo**: Linha

- **FN (Falso Negativo)**: Classificou ATIVO como INATIVO- **Mostra**: Taxa de VP vs Taxa de FP

- **Baseline**: Linha tracejada diagonal (random)

#### Subplot 2: Curva ROC- **Métrica**: AUC (Area Under Curve)

- **Tipo**: Linha- **Interpretação**:

- **Mostra**: Taxa de VP vs Taxa de FP  - AUC = 1.0: Classificador perfeito

- **Baseline**: Linha tracejada diagonal (random)  - AUC = 0.9-1.0: Excelente

- **Métrica**: AUC (Area Under Curve)  - AUC = 0.8-0.9: Bom

- **Interpretação**:  - AUC = 0.7-0.8: Razoável

  - AUC = 1.0: Classificador perfeito  - AUC = 0.5: Random (inútil)

  - AUC = 0.9-1.0: Excelente

  - AUC = 0.8-0.9: Bom---

  - AUC = 0.7-0.8: Razoável

  - AUC = 0.5: Random (inútil)## 🏆 Visualizações da Comparação (`compare_classifiers.py`)



---### 3. `comparison_metrics.png`



## 🏆 Visualizações do Classifier Comparison**Descrição**: Comparação de todas as métricas em 6 subplots



### 3. `comparison_metrics.png`#### Subplots 1-5: Métricas de Performance

- **F1-Score**: Média harmônica de Precision e Recall

**Descrição**: Comparação de todas as métricas em 6 subplots- **Acurácia**: Proporção de predições corretas

- **Precisão**: Proporção de positivos preditos que são realmente positivos

#### Subplots 1-5: Métricas de Performance- **Recall**: Proporção de positivos reais que foram identificados

- **F1-Score**: Média harmônica de Precision e Recall- **ROC-AUC**: Area under ROC curve

- **Acurácia**: Proporção de predições corretas

- **Precisão**: Proporção de positivos preditos que são realmente positivos**Cores das Barras:**

- **Recall**: Proporção de positivos reais que foram identificados- 🔵 Azul: Train (deve ser alto)

- **ROC-AUC**: Area under ROC curve- 🟠 Laranja: Validation (critério de seleção)

- 🔴 Vermelho: Test (critério final)

**Cores das Barras:**

- 🔵 Azul: Train (deve ser alto)#### Subplot 6: Tempo de Treinamento

- 🟠 Laranja: Validation (critério de seleção)- **Tipo**: Barras horizontais

- 🔴 Vermelho: Test (critério final)- **Mostra**: Tempo em segundos para treinar cada modelo

- **Útil para**: Avaliar trade-off performance vs. velocidade

#### Subplot 6: Tempo de Treinamento

- **Tipo**: Barras horizontais**Análise de Overfitting:**

- **Mostra**: Tempo em segundos para treinar cada modelo```

- **Útil para**: Avaliar trade-off performance vs. velocidade✅ BOM: Train ≈ Val ≈ Test (modelo generaliza bem)

⚠️  ATENÇÃO: Train >> Val > Test (overfitting)

**Análise de Overfitting:**```

```

✅ BOM: Train ≈ Val ≈ Test (modelo generaliza bem)---

⚠️  ATENÇÃO: Train >> Val > Test (overfitting)

```### 4. `comparison_ranking.png`



---**Descrição**: Ranking por F1-Score com medalhas



### 4. `comparison_ranking.png`**Características:**

- Barras horizontais para fácil comparação

**Descrição**: Ranking por F1-Score com medalhas- 🥇🥈🥉 Medalhas para top 3

- Comparação lado-a-lado: Validation vs Test

**Características:**- Ordenado por F1 de Validação (critério de seleção)

- Barras horizontais para fácil comparação

- 🥇🥈🥉 Medalhas para top 3**Interpretação:**

- Comparação lado-a-lado: Validation vs Test- Modelos no topo: Melhor F1-Score

- Ordenado por F1 de Validação (critério de seleção)- Gap pequeno Val→Test: Boa generalização

- Gap grande Val→Test: Overfitting

**Interpretação:**

- Modelos no topo: Melhor F1-Score---

- Gap pequeno Val→Test: Boa generalização

- Gap grande Val→Test: Overfitting### 5. `comparison_overfitting.png`



---**Descrição**: Análise de overfitting e generalização em 2 scatter plots



### 5. `comparison_overfitting.png`#### Subplot 1: Train vs Validation

- **Eixo X**: F1-Score no treino

**Descrição**: Análise de overfitting e generalização em 2 scatter plots- **Eixo Y**: F1-Score na validação

- **Linha vermelha**: Ideal (sem overfitting)

#### Subplot 1: Train vs Validation- **Análise**:

- **Eixo X**: F1-Score no treino  - Pontos **na linha**: Sem overfitting

- **Eixo Y**: F1-Score na validação  - Pontos **abaixo da linha**: Overfitting

- **Linha vermelha**: Ideal (sem overfitting)  - Pontos **acima da linha**: Impossível (erro)

- **Análise**:

  - Pontos **na linha**: Sem overfitting#### Subplot 2: Validation vs Test

  - Pontos **abaixo da linha**: Overfitting- **Eixo X**: F1-Score na validação

  - Pontos **acima da linha**: Impossível (erro)- **Eixo Y**: F1-Score no teste

- **Linha vermelha**: Ideal (boa generalização)

#### Subplot 2: Validation vs Test- **Análise**:

- **Eixo X**: F1-Score na validação  - Pontos **na linha**: Generaliza perfeitamente

- **Eixo Y**: F1-Score no teste  - Pontos **abaixo da linha**: Não generaliza bem

- **Linha vermelha**: Ideal (boa generalização)  - Pontos **acima da linha**: Sorte no teste (raro)

- **Análise**:

  - Pontos **na linha**: Generaliza perfeitamente**Modelos Ideais:**

  - Pontos **abaixo da linha**: Não generaliza bem```

  - Pontos **acima da linha**: Sorte no teste (raro)1. Próximo da linha vermelha em ambos os plots

2. Alto F1 em ambos os eixos

**Modelos Ideais:**3. Consistente entre Train/Val/Test

``````

1. Próximo da linha vermelha em ambos os plots

2. Alto F1 em ambos os eixos---

3. Consistente entre Train/Val/Test

```## 🎯 Como Interpretar os Resultados



---### Checklist de Validação



## 🆕 Visualizações do Regression Pipeline ⭐ **NOVO!**#### ✅ Estratificação Correta

1. `stratification_analysis.png`:

### 6. `predictions_vs_actual_{model_name}.png`   - [ ] Proporções iguais em Train/Val/Test (subplot 2)

   - [ ] Distribuição 80/10/10 (subplot 3)

**Descrição**: Scatter plot de predições vs valores reais   - [ ] Diferença < 2% entre conjuntos



**Características:**#### ✅ Modelo Bem Treinado

- **Eixo X**: Valor Real (nM)2. `evaluation_*.png`:

- **Eixo Y**: Valor Predito (nM)   - [ ] Matriz de confusão: Diagonal principal dominante

- **Linha vermelha**: Predição perfeita (y=x)   - [ ] ROC AUC > 0.8 (bom) ou > 0.9 (excelente)

- **Métrica**: R² score no título   - [ ] Poucos falsos positivos e negativos

- **Alpha**: 0.5 para ver sobreposições

- **Aspect ratio**: Equal (45° = perfeito)#### ✅ Melhor Modelo Selecionado

3. `comparison_ranking.png`:

**Interpretação:**   - [ ] F1 de validação alto (critério principal)

```   - [ ] Gap pequeno entre Val e Test (generalização)

✅ BOM: Pontos próximos da linha vermelha   - [ ] Tempo de treino aceitável

✅ R² > 0.8: Excelente predição

⚠️  R² < 0.5: Modelo fraco#### ✅ Sem Overfitting

```4. `comparison_overfitting.png`:

   - [ ] Pontos próximos da linha vermelha

---   - [ ] Train ≈ Val ≈ Test

   - [ ] Generalização consistente

### 7. `residuals_{model_name}.png`

---

**Descrição**: Análise de resíduos em 2 subplots

## 💡 Dicas de Uso

#### Subplot 1: Resíduos vs Valores Preditos

- **Tipo**: Scatter plot### Para Apresentações

- **Eixo X**: Valor Predito (nM)1. Use `stratification_analysis.png` para mostrar que o split é justo

- **Eixo Y**: Resíduo (Real - Predito)2. Use `comparison_ranking.png` para mostrar os melhores modelos

- **Linha vermelha**: Resíduo = 0 (ideal)3. Use `evaluation_test.png` para mostrar performance final

- **Útil para**: Detectar padrões nos erros

### Para Publicações

**Interpretação:**- Todas as imagens são 300 DPI (alta resolução)

```- Fontes em negrito para legibilidade

✅ BOM: Pontos aleatórios ao redor de y=0- Cores consistentes com padrões científicos

⚠️  RUIM: Padrão sistemático (ex: curva)- Legendas completas e autoexplicativas

```

### Para Debugging

#### Subplot 2: Distribuição de Resíduos1. `comparison_overfitting.png`: Identificar overfitting

- **Tipo**: Histograma2. `comparison_metrics.png`: Ver qual métrica está problemática

- **Mostra**: Frequência de cada magnitude de erro3. `evaluation_*.png`: Ver onde o modelo está errando

- **Ideal**: Distribuição normal centrada em 0

- **Cores**: Gradiente baseado em frequência---



**Interpretação:**## 🔧 Customização

```

✅ BOM: Distribuição simétrica, centrada em 0### Desabilitar Visualizações

⚠️  RUIM: Distribuição assimétrica ou bimodalSe quiser rodar sem gerar visualizações (mais rápido):

```

```python

---# Em run_complete_pipeline.py ou compare_classifiers.py

# Comente as linhas:

### 8. `model_comparison.png` ⭐# self.plot_stratification(...)

# self.plot_evaluation(...)

**Descrição**: Comparação de múltiplos modelos regression# self.plot_comparison(...)

```

**Subplots:**

1. **RMSE** (Root Mean Squared Error) - Menor é melhor### Alterar Resolução

2. **MAE** (Mean Absolute Error) - Menor é melhor```python

3. **R²** (Coeficiente de determinação) - Maior é melhor (max 1.0)# Trocar dpi=300 por outro valor:

4. **Pearson r** (Correlação de Pearson) - Maior é melhor (max 1.0)plt.savefig(file, dpi=150)  # Menor resolução, arquivo menor

5. **Spearman ρ** (Correlação de Spearman) - Maior é melhor (max 1.0)plt.savefig(file, dpi=600)  # Maior resolução, arquivo maior

```

**Cores:**

- 🟢 Verde: Validation set### Alterar Cores

- 🔵 Azul: Test set```python

# Usar outros colormaps do seaborn:

**Interpretação:**sns.set_palette("husl")  # Cores vibrantes

```sns.set_palette("muted")  # Cores suaves

✅ MELHOR MODELO: Menor RMSE/MAE + Maior R²/Pearson/Spearmansns.set_palette("deep")   # Cores intensas (padrão)

✅ BOA GENERALIZAÇÃO: Val ≈ Test em todas as métricas```

⚠️  OVERFITTING: Val muito melhor que Test

```---



---## 📚 Referências de Visualização



## 🎯 Como Interpretar os Resultados- **Matplotlib**: https://matplotlib.org/stable/gallery/

- **Seaborn**: https://seaborn.pydata.org/examples/

### Checklist de Validação- **Curva ROC**: Fawcett (2006) - "An introduction to ROC analysis"

- **Matriz de Confusão**: Stehman (1997) - "Selecting and interpreting measures of thematic classification accuracy"

#### ✅ Classification Pipeline

---

1. **Estratificação Correta** (`stratification_analysis.png`):

   - [ ] Proporções iguais em Train/Val/Test (subplot 2)**Gerado automaticamente pelo pipeline DockTKinase** 🚀  

   - [ ] Distribuição 80/10/10 (subplot 3)**Data**: 22 de Outubro de 2025  

   - [ ] Diferença < 2% entre conjuntos**Versão**: 1.0


2. **Modelo Bem Treinado** (`evaluation_*.png`):
   - [ ] Matriz de confusão: Diagonal principal dominante
   - [ ] ROC AUC > 0.8 (bom) ou > 0.9 (excelente)
   - [ ] Poucos falsos positivos e negativos

3. **Melhor Modelo Selecionado** (`comparison_ranking.png`):
   - [ ] F1 de validação alto (critério principal)
   - [ ] Gap pequeno entre Val e Test (generalização)
   - [ ] Tempo de treino aceitável

4. **Sem Overfitting** (`comparison_overfitting.png`):
   - [ ] Pontos próximos da linha vermelha
   - [ ] Train ≈ Val ≈ Test
   - [ ] Generalização consistente

#### ✅ Regression Pipeline ⭐ **NOVO!**

1. **Predições Acuradas** (`predictions_vs_actual_*.png`):
   - [ ] Pontos próximos da linha vermelha (y=x)
   - [ ] R² > 0.7 (bom) ou > 0.8 (excelente)
   - [ ] Sem outliers extremos

2. **Resíduos Bem Comportados** (`residuals_*.png`):
   - [ ] Resíduos aleatórios ao redor de 0
   - [ ] Sem padrões sistemáticos
   - [ ] Distribuição aproximadamente normal

3. **Melhor Modelo Regression** (`model_comparison.png`):
   - [ ] Menor RMSE e MAE
   - [ ] Maior R², Pearson r e Spearman ρ
   - [ ] Val ≈ Test (boa generalização)

---

## 💡 Dicas de Uso

### Para Apresentações
1. Use `stratification_analysis.png` para mostrar que o split é justo
2. Use `comparison_ranking.png` (classification) ou `model_comparison.png` (regression) para mostrar os melhores modelos
3. Use `evaluation_test.png` ou `predictions_vs_actual_*.png` para mostrar performance final

### Para Publicações
- Todas as imagens são 300 DPI (alta resolução)
- Fontes em negrito para legibilidade
- Cores consistentes com padrões científicos
- Legendas completas e autoexplicativas

### Para Debugging
1. **Classification**: `comparison_overfitting.png` - Identificar overfitting
2. **Regression**: `residuals_*.png` - Ver padrões nos erros
3. **Ambos**: `*_comparison.png` - Comparar métricas

---

## 🔧 Customização

### Desabilitar Visualizações
Se quiser rodar sem gerar visualizações (mais rápido):

```python
# Em run_complete_pipeline.py ou run_regression_pipeline.py
# Comentar as linhas:
# self.plot_stratification(...)
# self.plot_evaluation(...)
# self.plot_comparison(...)
```

### Alterar Resolução
```python
# Trocar dpi=300 por outro valor:
plt.savefig(file, dpi=150)  # Menor resolução, arquivo menor
plt.savefig(file, dpi=600)  # Maior resolução, arquivo maior
```

### Alterar Cores
```python
# Usar outros colormaps do seaborn:
sns.set_palette("husl")  # Cores vibrantes
sns.set_palette("muted")  # Cores suaves
sns.set_palette("deep")   # Cores intensas (padrão)
```

---

## 📚 Referências de Visualização

- **Matplotlib**: https://matplotlib.org/stable/gallery/
- **Seaborn**: https://seaborn.pydata.org/examples/
- **Curva ROC**: Fawcett (2006) - "An introduction to ROC analysis"
- **Matriz de Confusão**: Stehman (1997) - "Selecting and interpreting measures of thematic classification accuracy"
- **Residual Plots**: Regression diagnostics for statistical modeling

---

## 🎉 Resumo

**Classification Pipeline** (6 visualizações):
1. ✅ Stratification analysis
2. ✅ Evaluation validation (confusion matrix + ROC)
3. ✅ Evaluation test (confusion matrix + ROC)
4. ✅ Comparison metrics (6 subplots)
5. ✅ Comparison ranking (medals)
6. ✅ Comparison overfitting (2 scatter plots)

**Regression Pipeline** (3 visualizações por modelo + 1 comparação): ⭐ **NOVO!**
1. ✅ Predictions vs Actual (scatter + R²)
2. ✅ Residuals analysis (2 subplots)
3. ✅ Model comparison (5 métricas)

**Total**: 6 (classification) + 3N+1 (regression, N=número de modelos) visualizações automáticas!

---

**Gerado automaticamente pelo pipeline DockTKinase** 🚀  
**Data**: 28 de Outubro de 2025  
**Branch**: regression  
**Versão**: 2.0 - Dual Pipeline System  
**Sistema**: Classification (6 modelos) + Regression (11 modelos)
