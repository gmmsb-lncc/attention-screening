# Split Protocol for `split_comparison_analysis.py`

## Objetivo

Documentar o protocolo de particionamento usado no baseline `KNN/MLP` para:

- evitar vazamento entre treino e teste;
- reduzir viés de particionamento;
- selecionar splits mais representativos via critério quantitativo;
- reportar incerteza por múltiplas rodadas.

Protocolo atual: `80_20_balanced_tanimoto_candidates_v2`.

## Cenários

### 1) `random`

- Split estratificado por `label` em `80/20` (treino/teste).
- Permite vazamento de compostos e proteínas (baseline superior).

### 2) `compound`

- Split por `chembl_id` em `80/20` de compostos únicos.
- Critério duro: `compostos_teste ∩ compostos_treino = ∅`.
- Proteínas podem sobrepor entre treino e teste.

### 3) `new_compound_new_kinase`

- Define conjuntos de compostos e quinases de teste.
- `teste`: pares cujo composto está no conjunto de compostos de teste **e** cuja quinase está no conjunto de quinases de teste.
- `treino`: pares cujo composto e quinase estão fora dos conjuntos de teste.
- `val`: pares restantes (intermediários), não usados no treino deste script.
- Critérios duros:
- `compostos_teste ∩ compostos_treino = ∅`
- `quinases_teste ∩ quinases_treino = ∅`

## Rodadas e candidatos

- O modelo usa seed fixa (`--seed`, default `42`).
- A variabilidade vem dos particionamentos: `--n_rounds` (default `5`).
- Em cada rodada, o script testa `--n_split_candidates` candidatos (default `25`) e escolhe o melhor pelo menor score.

## Score de qualidade do split

Cada candidato válido recebe score (menor é melhor), combinando:

- desvio da fração alvo de teste (`|test_fraction - target|`);
- desvio da proporção de `label` em teste e treino vs global;
- distância de distribuição de grupos de proteína (`L1`) em teste e treino vs global;
- desvio de diversidade química interna (Tanimoto) em teste e treino vs global.

### Diversidade química (Tanimoto)

- Calculada em compostos únicos com fingerprint de Morgan (`2048 bits`).
- Métrica: `ID(S) = 1 - mean(Tanimoto(i,j))`.
- Para escala, usa amostragem de pares aleatórios (`all-vs-all` estimado).
- Amostragem é reprodutível por seed.

## Validações duras por candidato

Um candidato é rejeitado se:

- treino ou teste ficarem vazios;
- teste tiver menos que o mínimo (`MIN_TEST_SAMPLES`);
- teste tiver apenas uma classe de `label`;
- teste tiver menos de 2 quinases;
- houver sobreposição indevida de compostos/quinases nos cenários que exigem disjunção.

## Saídas

O JSON final inclui:

- `split_protocol_version`;
- `model_seed`, `n_rounds`, `split_seeds`;
- `target_test_fraction`, `n_split_candidates`;
- resultados agregados por cenário/modelo (média e desvio padrão);
- resultados por rodada, incluindo:
- `round_seed`, `candidate_seed`;
- métricas de qualidade do split;
- diagnósticos e metadados do particionamento.

## Uso recomendado

Execução padrão:

```bash
python split_comparison_analysis.py --dataset non_human --seed 42 --n_rounds 5
```

Execução mais rigorosa (mais candidatos):

```bash
python split_comparison_analysis.py --dataset non_human --seed 42 --n_rounds 5 --n_split_candidates 40 --test_fraction 0.20
```

## Observações

- Este protocolo foi desenhado para o baseline `KNN/MLP`.
- Para modelos que exigem validação externa explícita para early stopping/model selection, o protocolo pode precisar ajuste.
