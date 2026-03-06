# Protocolo Atual de Splits por Scaffold e Cenários (S1, S2, S3, S4, Sc)

Este documento descreve a estratégia implementada no `scaffold_split.py`.

## Objetivo

Comparar os cenários `S1`, `S2`, `S3`, `S4` e `Sc` no **mesmo conjunto de teste**,
com remoção prévia de casos monotônicos e controle explícito da distribuição de classes.

## 1) Pré-processamento obrigatório: remoção de monotônicos

Antes de qualquer split, removemos:

1. **Kinases monotônicas**: `target_kinase` com taxa de classe exatamente `0` ou `1`.
2. **Compostos monotônicos**: `chembl_id` testado em >=2 kinases com taxa exatamente `0` ou `1`.

Isso evita casos triviais (pan-ativo/pan-inativo) que inflacionam métricas.

## 2) Notação

Para cada domínio `d in {H, N}`:

- `D_d`: conjunto de linhas após limpeza
- `C_d`: compostos únicos
- `U_d = |C_d|`
- `phi(c)`: scaffold Murcko do composto `c`

Teste fixo:

- alvo `alpha = 0.10` (em compostos únicos)
- `T_d`: linhas de teste (scaffolds selecionados)
- `R_d = D_d \ T_d`: pool restante para `train/val` dos cenários

## 3) Construção do teste fixo

Modo padrão (`shared_scaffold`):

- mesmo conjunto de scaffolds define teste nos dois domínios
  (protocolo universal estrito).

Modo opcional (`per_dataset`):

- seleciona scaffolds de teste separadamente para humano e não-humano,
- cada um próximo de `10%` de compostos únicos.

Arquivo universal:

- `universal_test.tsv = human_test.tsv union non_human_test.tsv` (concatenação com `dataset_source`).

## 4) Construção de train/val por cenário (com teste congelado)

Com `R_d` fixo, geramos `train/val` para cada cenário:

1. `S1` random: split estratificado por classe.
2. `S2` compound: grupos por `chembl_id` (disjunção de compostos entre train/val).
3. `S3` kinase: grupos por `target_kinase` (disjunção de kinases).
4. `Sc` scaffold: grupos por scaffold (disjunção de scaffold).
5. `S4` new_compound_new_kinase: disjunção simultânea de compostos e kinases; linhas órfãs (quadrantes cruzados) são descartadas.

Assim, todos os cenários usam o mesmo teste; muda apenas o protocolo de `train/val`.

## 5) Controle de proporcionalidade de classes

Para cenários em grupos (`S2`, `S3`, `Sc`) a seleção de validação minimiza:

- erro de tamanho da validação,
- erro de taxa de classe em train/val vs pool,
- penalidade quando falta alguma classe.

Forma resumida:

- `L = |f_val - f_target| + w * (|p_train - p_pool| + |p_val - p_pool|) + lambda * I(classe_faltante)`

onde:

- `p_pool`: taxa de positivos no pool,
- `p_train`, `p_val`: taxas em train/val,
- `f_val`: fração efetiva da validação.

Para `S4`, adiciona-se penalidade por descarte de órfãos.

## 6) Relatório de distribuição por split

O script imprime, para cada cenário e domínio:

- `% positivos` e `% negativos` em `train`, `val`, `test`,
- número de linhas descartadas (`S4`).

Exemplo:

- `[human][S2] Train: +41.99% / -58.01% | Val: +41.81% / -58.19% | Test: +52.50% / -47.50% | dropped=0`

Também salva:

- `split_class_distribution_summary.tsv`
- `split_class_distribution_report.txt`

## 7) Saídas

Diretório `scaffolds_splits/output/`:

- `human_test.tsv`, `non_human_test.tsv`, `universal_test.tsv`
- `test_scaffolds_universal.json`
- `manifest.json`
- `split_class_distribution_summary.tsv`
- `split_class_distribution_report.txt`
- `scenarios/S1/`, `scenarios/S2/`, `scenarios/S3/`, `scenarios/S4/`, `scenarios/Sc/`

Cada pasta de cenário contém `*_train.tsv`, `*_val.tsv` e opcionalmente `*_dropped.tsv`.

## 8) Execução

```bash
source env/bin/activate
python scripts/scaffold_split.py --output-dir scaffolds_splits/output
```

## 9) Observação importante

No `S4`, é normal haver descarte considerável de linhas devido à disjunção dupla (composto + kinase).
Isso não é erro; é consequência da definição do cenário.
