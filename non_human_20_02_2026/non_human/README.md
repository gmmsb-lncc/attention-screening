# README - Logica Das Imagens (Split Comparison Analysis)

Este arquivo documenta **o significado e a logica de calculo** de cada imagem gerada nesta pasta.

## Configuracao Da Execucao
- `dataset`: `non_human`
- `model_seed`: `42`
- `n_folds`: `10`
- `scenarios`: `['new_compound_new_kinase', 'scaffold', 'compound', 'kinase', 'random']`
- `split_protocol_version`: `single_split_v1`
- `threshold_pchembl`: `6.0`

## Arquivos De Imagem
- `01`: non_human_20_02_2026/non_human/01_leakage_analysis.png
- `02`: non_human_20_02_2026/non_human/02_baseline_comparison.png
- `03`: non_human_20_02_2026/non_human/03_kinase_imbalance.png
- `04`: non_human_20_02_2026/non_human/04_compound_consistency.png
- `05`: non_human_20_02_2026/non_human/05_similarity_analysis.png
- `06`: non_human_20_02_2026/non_human/06_split_comparison.png
- `07`: non_human_20_02_2026/non_human/07_inflated_vs_real_performance.png

## Significado De Cada Imagem

### 01_leakage_analysis.png
- O que mostra: proporcao de linhas de teste com composto ja visto no treino e proporcao de duplicatas exatas (composto+quinase).
- Logica: split estratificado; interseccao de `chembl_id` entre treino e teste; interseccao de pares `(chembl_id, target_kinase)`.
- Interpretacao: valores altos indicam vazamento/memorizacao facilitada.

### 02_baseline_comparison.png
- O que mostra: comparacao entre baselines de lookup e KNN original (Accuracy e MCC).
- Logica:
  - Lookup por composto: classe majoritaria por `chembl_id` no treino.
  - Lookup por quinase: classe majoritaria por `target_kinase` no treino.
  - Lookup composto+quinase: cascata par -> composto -> quinase -> classe global.
  - KNN original: Morgan FP + one-hot de quinase.
- Interpretacao: se lookup chega perto do KNN, ha forte sinal de memorizacao no dataset.

### 03_kinase_imbalance.png
- O que mostra: distribuicao da proporcao de ativos por quinase e classes de balanceamento.
- Logica: para cada quinase, calcula `prop_active`; classifica em desbalanceada/moderada/balanceada.
- Interpretacao: muitas quinases desbalanceadas tornam a predicao mais facil por vies de classe.

### 04_compound_consistency.png
- O que mostra: consistencia do comportamento dos compostos em diferentes quinases.
- Logica: por `chembl_id`, calcula `prop_active` e numero de quinases testadas; marca consistencia perfeita quando `prop_active`=0 ou 1.
- Interpretacao: compostos muito consistentes carregam grande parte do sinal preditivo.

### 05_similarity_analysis.png
- O que mostra: similaridade quimica (Tanimoto) de compostos novos de teste vs treino.
- Logica: Morgan fingerprint; para cada composto novo no teste, calcula a similaridade maxima com o treino.
- Interpretacao: similaridade alta indica generalizacao quimica curta (teste muito proximo do treino).
- Observacao: pode nao ser gerada quando nao ha compostos novos suficientes no teste.

### 06_split_comparison.png
- O que mostra: desempenho de KNN/MLP por cenario de split (Accuracy e MCC).
- Logica: k-fold com cenarios `random`, `scaffold`, `compound`, `kinase`, `new_compound_new_kinase`; agrega media e desvio-padrao por fold.
- Interpretacao: quantifica queda de performance conforme o split remove vazamento e aumenta dificuldade de generalizacao.

### 07_inflated_vs_real_performance.png
- O que mostra: comparacao entre performance inflada (`Random Split`) e performance real de generalizacao (`New Compound + New Kinase`).
- Logica: barras de MCC e Accuracy para KNN e MLP + percentual de queda.
- Interpretacao: mede o tamanho da superestimacao quando o protocolo de avaliacao permite vazamento.
- Observacao: so e gerada quando ambos os cenarios necessarios existem.

## Resumo Da Leitura Recomendada
1. Comece em `01` para entender vazamento direto.
2. Use `02-05` para diagnosticar fontes de inflacao (lookup, desbalanceamento, consistencia e similaridade).
3. Feche com `06-07` para ver o impacto final nos modelos.
