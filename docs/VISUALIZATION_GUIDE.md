# Guia de Visualizações do DockTKinase

## 📊 Visualizações Automáticas

Todos os scripts agora geram visualizações automaticamente para tornar as estatísticas mais didáticas e acessíveis.

### 📁 Localização

Todas as visualizações são salvas em:
```
tests/pipeline_output/visualizations/
tests/comparison_output/visualizations/
```

Formato: **PNG de alta resolução (300 DPI)** - pronto para publicações!

---

## 🎨 Visualizações do Pipeline (`run_complete_pipeline.py`)

### 1. `stratification_analysis.png`

**Descrição**: Análise completa da estratificação em 3 gráficos

#### Subplot 1: Distribuição de Classes
- **Tipo**: Gráfico de barras agrupadas
- **Mostra**: Contagem absoluta de cada classe (INATIVO/ATIVO) em cada conjunto
- **Útil para**: Verificar se o número de amostras está balanceado
- **Valores**: Contagens exatas sobre cada barra

#### Subplot 2: Proporções de Classes
- **Tipo**: Gráfico de barras agrupadas (%)
- **Mostra**: Porcentagem de cada classe em cada conjunto
- **Útil para**: Confirmar que a estratificação mantém as proporções originais
- **Critério de sucesso**: Proporções iguais em Train/Val/Test
- **Valores**: Porcentagens sobre cada barra

#### Subplot 3: Distribuição de Amostras
- **Tipo**: Gráfico de pizza
- **Mostra**: Distribuição do total de amostras por conjunto
- **Útil para**: Visualizar o split 80/10/10
- **Cores**: 
  - Original: Cinza
  - Train: Verde
  - Val: Laranja
  - Test: Roxo

**Exemplo de Interpretação:**
```
✅ BOM: Proporções iguais em todos os subplots 2
⚠️  ATENÇÃO: Proporções diferentes indicam problema na estratificação
```

---

### 2. `evaluation_validation.png` e `evaluation_test.png`

**Descrição**: Avaliação do modelo em 2 gráficos

#### Subplot 1: Matriz de Confusão
- **Tipo**: Heatmap
- **Mostra**: Predições corretas e incorretas
- **Layout**:
  ```
  Verdadeiro\Predito  INATIVO(0)  ATIVO(1)
  INATIVO(0)          TN          FP
  ATIVO(1)            FN          TP
  ```
- **Valores**: Contagens absolutas + porcentagens
- **Cores**: Azul (mais escuro = mais amostras)

**Interpretação:**
- **Diagonal principal** (TN e TP): Predições corretas
- **Diagonal secundária** (FP e FN): Erros do modelo
- **FP (Falso Positivo)**: Classificou INATIVO como ATIVO
- **FN (Falso Negativo)**: Classificou ATIVO como INATIVO

#### Subplot 2: Curva ROC
- **Tipo**: Linha
- **Mostra**: Taxa de VP vs Taxa de FP
- **Baseline**: Linha tracejada diagonal (random)
- **Métrica**: AUC (Area Under Curve)
- **Interpretação**:
  - AUC = 1.0: Classificador perfeito
  - AUC = 0.9-1.0: Excelente
  - AUC = 0.8-0.9: Bom
  - AUC = 0.7-0.8: Razoável
  - AUC = 0.5: Random (inútil)

---

## 🏆 Visualizações da Comparação (`compare_classifiers.py`)

### 3. `comparison_metrics.png`

**Descrição**: Comparação de todas as métricas em 6 subplots

#### Subplots 1-5: Métricas de Performance
- **F1-Score**: Média harmônica de Precision e Recall
- **Acurácia**: Proporção de predições corretas
- **Precisão**: Proporção de positivos preditos que são realmente positivos
- **Recall**: Proporção de positivos reais que foram identificados
- **ROC-AUC**: Area under ROC curve

**Cores das Barras:**
- 🔵 Azul: Train (deve ser alto)
- 🟠 Laranja: Validation (critério de seleção)
- 🔴 Vermelho: Test (critério final)

#### Subplot 6: Tempo de Treinamento
- **Tipo**: Barras horizontais
- **Mostra**: Tempo em segundos para treinar cada modelo
- **Útil para**: Avaliar trade-off performance vs. velocidade

**Análise de Overfitting:**
```
✅ BOM: Train ≈ Val ≈ Test (modelo generaliza bem)
⚠️  ATENÇÃO: Train >> Val > Test (overfitting)
```

---

### 4. `comparison_ranking.png`

**Descrição**: Ranking por F1-Score com medalhas

**Características:**
- Barras horizontais para fácil comparação
- 🥇🥈🥉 Medalhas para top 3
- Comparação lado-a-lado: Validation vs Test
- Ordenado por F1 de Validação (critério de seleção)

**Interpretação:**
- Modelos no topo: Melhor F1-Score
- Gap pequeno Val→Test: Boa generalização
- Gap grande Val→Test: Overfitting

---

### 5. `comparison_overfitting.png`

**Descrição**: Análise de overfitting e generalização em 2 scatter plots

#### Subplot 1: Train vs Validation
- **Eixo X**: F1-Score no treino
- **Eixo Y**: F1-Score na validação
- **Linha vermelha**: Ideal (sem overfitting)
- **Análise**:
  - Pontos **na linha**: Sem overfitting
  - Pontos **abaixo da linha**: Overfitting
  - Pontos **acima da linha**: Impossível (erro)

#### Subplot 2: Validation vs Test
- **Eixo X**: F1-Score na validação
- **Eixo Y**: F1-Score no teste
- **Linha vermelha**: Ideal (boa generalização)
- **Análise**:
  - Pontos **na linha**: Generaliza perfeitamente
  - Pontos **abaixo da linha**: Não generaliza bem
  - Pontos **acima da linha**: Sorte no teste (raro)

**Modelos Ideais:**
```
1. Próximo da linha vermelha em ambos os plots
2. Alto F1 em ambos os eixos
3. Consistente entre Train/Val/Test
```

---

## 🎯 Como Interpretar os Resultados

### Checklist de Validação

#### ✅ Estratificação Correta
1. `stratification_analysis.png`:
   - [ ] Proporções iguais em Train/Val/Test (subplot 2)
   - [ ] Distribuição 80/10/10 (subplot 3)
   - [ ] Diferença < 2% entre conjuntos

#### ✅ Modelo Bem Treinado
2. `evaluation_*.png`:
   - [ ] Matriz de confusão: Diagonal principal dominante
   - [ ] ROC AUC > 0.8 (bom) ou > 0.9 (excelente)
   - [ ] Poucos falsos positivos e negativos

#### ✅ Melhor Modelo Selecionado
3. `comparison_ranking.png`:
   - [ ] F1 de validação alto (critério principal)
   - [ ] Gap pequeno entre Val e Test (generalização)
   - [ ] Tempo de treino aceitável

#### ✅ Sem Overfitting
4. `comparison_overfitting.png`:
   - [ ] Pontos próximos da linha vermelha
   - [ ] Train ≈ Val ≈ Test
   - [ ] Generalização consistente

---

## 💡 Dicas de Uso

### Para Apresentações
1. Use `stratification_analysis.png` para mostrar que o split é justo
2. Use `comparison_ranking.png` para mostrar os melhores modelos
3. Use `evaluation_test.png` para mostrar performance final

### Para Publicações
- Todas as imagens são 300 DPI (alta resolução)
- Fontes em negrito para legibilidade
- Cores consistentes com padrões científicos
- Legendas completas e autoexplicativas

### Para Debugging
1. `comparison_overfitting.png`: Identificar overfitting
2. `comparison_metrics.png`: Ver qual métrica está problemática
3. `evaluation_*.png`: Ver onde o modelo está errando

---

## 🔧 Customização

### Desabilitar Visualizações
Se quiser rodar sem gerar visualizações (mais rápido):

```python
# Em run_complete_pipeline.py ou compare_classifiers.py
# Comente as linhas:
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

---

**Gerado automaticamente pelo pipeline DockTKinase** 🚀  
**Data**: 22 de Outubro de 2025  
**Versão**: 1.0
