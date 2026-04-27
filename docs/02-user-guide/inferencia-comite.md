# Manual de Inferência por Comitê Multi-Modelo

Este manual descreve como utilizar o pipeline de inferência distribuído junto à tese para predizer interações kinase-ligante a partir de SMILES e/ou sequências FASTA.

## Visão geral

O pipeline submete cada par proteína-ligante a quatro modelos treinados de forma independente (DT-Kinase-LEGACY, DrugBAN, GraphBAN, ConPLex), agrega as quatro probabilidades calibradas em um consenso ordenado, e emite mapas de atenção interpretáveis para os pares de alta confiança. A motivação para a arquitetura por consenso é a observação documentada no Anexo A da tese, segundo a qual cada um dos quatro modelos exibe perfil de erro distinto sob *distribution shift cross-corpus*, de modo que a concordância entre paradigmas heterogêneos é um sinal mais robusto que a confiança em qualquer modelo isoladamente.

## Pré-requisitos

A execução completa exige quatro ambientes Conda separados (um por modelo, devido a conflitos de versões entre os *repositories* originais), cada qual com seu próprio conjunto de dependências:

| Ambiente | Modelo | Dependências principais |
|---|---|---|
| `attention-screening` | DT-Kinase-LEGACY | PyTorch 2.x, RDKit, ESM-2 8M, MoLFormer |
| `drugban` | DrugBAN | PyTorch 1.x, DGL, RDKit |
| `graphban` | GraphBAN | PyTorch + DGL + ESM-1b + ChemBERTa |
| `conplex` | ConPLex | PyTorch + ProtBERT |

Os `checkpoints` treinados, as bibliotecas de referência e os arquivos de calibração são distribuídos junto ao repositório e ficam organizados em diretórios fixos descritos na seção *Estrutura de arquivos*.

## Modos de uso

O orquestrador `committee.py` aceita quatro modos de entrada distintos, despachados automaticamente a partir da combinação de *flags* fornecida na linha de comando.

### Modo 1: triagem ligante contra kinome

Forneça apenas um SMILES e o pipeline expande a entrada combinatoriamente contra todas as quinases do kinome de referência (humanas ou todas), produzindo um *ranking* ordenado por probabilidade média do comitê.

```bash
python scripts/inference/committee.py \
    --smiles "CC1=C(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc(Nc2nccc(-c3cccnc3)n2)cc1" \
    --organism human \
    --out results/inference/run_001 \
    --top-k 20 \
    --parallel
```

Para o kinome humano (483 quinases), a triagem completa executa em torno de 5 a 10 minutos quando os caches de *embeddings* ESM-2 e MoLFormer já estão construídos (situação típica após a primeira execução).

### Modo 2: triagem proteína contra biblioteca de ligantes

Forneça apenas a sequência de uma quinase em formato FASTA e o pipeline expande contra toda a biblioteca curada de ligantes ChEMBL (110.963 compostos após desduplicação por *scaffold*).

```bash
python scripts/inference/committee.py \
    --fasta minha_quinase.fa \
    --out results/inference/run_002 \
    --top-k 50
```

A operação é da ordem de uma a duas horas por modelo, mesmo sob *cache* populado. Recomenda-se utilizá-la apenas quando o objetivo é um *screening* sistemático contra a biblioteca completa.

### Modo 3: par único explícito

Forneça simultaneamente SMILES e FASTA quando o usuário tem em mente o par específico que deseja avaliar. O pipeline gera uma única predição.

```bash
python scripts/inference/committee.py \
    --smiles "CC(=O)Oc1ccccc1C(=O)O" \
    --fasta abl1.fa \
    --out results/inference/run_003
```

### Modo 4: lote arbitrário

Forneça um arquivo TSV com colunas `uniprot, sequence, chembl_id, smiles` e o pipeline processa todas as linhas como pares explícitos.

```bash
python scripts/inference/committee.py \
    --pairs meus_pares.tsv \
    --out results/inference/run_004 \
    --top-k 100
```

## Critério de decisão consensual

Cada par recebe uma classificação categórica baseada no número de modelos que predizem ligação sob seu próprio limiar nativo. A regra de atribuição depende do tamanho do comitê efetivamente presente na rodada (ver Anexo B Tabelas B.2 e B.6).

| *Agreement* sobre 4 modelos | *Tier* | Interpretação operacional |
|---|---|---|
| 4/4 | STRONG | alta confiança, inspeção experimental prioritária |
| 3/4 | LIKELY | confiança moderada, inspeção em segunda passada |
| 2/4 | UNCERTAIN | inconclusivo, requer evidência adicional |
| ≤1/4 | UNLIKELY | predição negativa por consenso |

Quando algum modelo não está disponível, o pipeline opera em comitê parcial e re-escala os limiares automaticamente: para três modelos, STRONG corresponde a 3/3 e LIKELY a 2/3; para dois modelos, STRONG é 2/2 e LIKELY é 1/2.

A decisão operacional final é a conjunção:

```
binder consensual ⟺ (tier ∈ {STRONG, LIKELY}) ∧ (prob_mean > 0,5)
```

Pares satisfazendo essa conjunção são candidatos prioritários para extração de mapas de atenção e validação experimental dirigida.

## Estrutura da saída

Cada execução gera um diretório auto-contido cujo conteúdo é completo e suficiente para reprodução posterior:

```
results/inference/<run_id>/
├── pairs.tsv                       (entrada expandida)
├── scores_dtkinase.csv             (1 linha por par: prob, pred, threshold)
├── scores_drugban.csv
├── scores_graphban.csv
├── scores_conplex.csv
├── consensus.csv                   (ranking ordenado por prob_mean)
├── consensus.top.csv               (subset top-K, se solicitado)
├── attention/                      (apenas pares STRONG/LIKELY)
│   └── <pair_id>/
│       ├── dtkinase_Mk.npz         (mapa de interação 16 cabeças × resíduos × tokens)
│       ├── dtkinase_hierpool.npz   (pesos de atenção por resíduo + por token)
│       ├── drugban_BAN.npz         (matriz de atenção bilinear DrugBAN)
│       ├── graphban_BAN.npz        (matriz de atenção bilinear GraphBAN)
│       └── consensus_heatmap.pdf   (diagrama composto 2×2 para revisão visual)
└── config_snapshot.yaml            (revisões git + hashes dos checkpoints)
```

## Esquema do arquivo `consensus.csv`

| Coluna | Significado |
|---|---|
| `pair_id` | identificador composto `{uniprot}__{chembl_id}` |
| `uniprot, chembl_id` | chaves primárias do par |
| `prob_<modelo>` | probabilidade calibrada por cada modelo |
| `pred_<modelo>` | predição binária por cada modelo (sob seu limiar nativo) |
| `thr_<modelo>` | limiar usado por cada modelo |
| `prob_mean` | média soft das probabilidades do comitê |
| `prob_std` | dispersão entre modelos |
| `confidence` | $1 - \sigma_p$, alta indica consenso |
| `agreement_count` | número de modelos que predizem *binder* |
| `tier` | STRONG / LIKELY / UNCERTAIN / UNLIKELY |
| `rank_fusion` | soma de *ranks* via Borda *count* (menor é melhor) |

A ordenação padrão é por `prob_mean` decrescente, com critérios secundários `agreement_count` decrescente e `confidence` decrescente para desempate.

## Mapas de atenção

Para os pares de *tier* STRONG ou LIKELY, o pipeline extrai três níveis de atenção do DT-Kinase via *forward hooks* no PyTorch (Anexo B Figura B.4):

1. **Nível 1 (M_k pré-CNN)**: tensor de forma `[16, sp, sl]` capturando a distribuição local da interação resíduo × *token* antes da integração espacial pela CNN 2D. Equivale conceitualmente à *BAN attention* de DrugBAN e GraphBAN.

2. **Nível 2 (HierPool eixo ligante)**: pesos de forma `[sp, sl]` indicando, para cada posição protéica, quais *tokens* de ligante o modelo considera relevantes naquela região.

3. **Nível 3 (HierPool eixo proteína)**: pesos de forma `[sp]` correspondendo ao ranqueamento de relevância dos resíduos protéicos para a predição final.

Os arquivos `dtkinase_Mk.npz` e `dtkinase_hierpool.npz` contêm tanto os tensores brutos quanto agregações estatísticas (média sobre cabeças, somatórios por eixo, intensidade absoluta por cabeça). O arquivo `consensus_heatmap.pdf` apresenta um diagrama composto 2×2 com o *heatmap* da média sobre cabeças, as duas distribuições de *HierPool* como gráficos de barras, e a intensidade por cabeça como diagrama auxiliar.

DrugBAN e GraphBAN expõem a matriz de atenção bilinear $A \in \mathbb{R}^{s_p \times s_l}$ computada por seus cabeçotes BAN. ConPLex, por usar exclusivamente um esquema contrastivo no espaço métrico, não dispõe de matriz de atenção interpretável no sentido posicional, sendo omitido da saída de atenção.

## Limitações operacionais

A inferência é restrita ao domínio de quinases. Todos os quatro modelos foram treinados exclusivamente em pares quinase-ligante, e a aplicação a proteínas fora dessa família é extrapolação não validada. O pipeline emite *warning* (sem bloquear a execução) quando a sequência fornecida tem similaridade de cosseno inferior a 0,7 contra qualquer entrada do kinome de referência.

A saída é um *ranking* probabilístico, não uma afirmação determinística sobre afinidade absoluta. Os modelos foram calibrados sob a tarefa de classificação binária (*binder* versus *non-binder*), e a probabilidade reportada deve ser interpretada como pontuação de relevância para inspeção subsequente, não como predição de constante de inibição $K_i$ ou $\mathrm{IC}_{50}$. O uso do *ranking* para guiar campanhas de *lead optimization* deve ser acompanhado por validação experimental dirigida nos pares de *tier* STRONG, e nunca substituí-la.

## Reprodutibilidade

Cada execução grava em `config_snapshot.yaml` o conjunto exato de *checkpoints*, calibrações e *git revision* utilizado, viabilizando reprodução posterior a partir do diretório isolado, sem dependência de estados temporários externos. O *checkpoint* de cada modelo é identificado pelo seu *hash* SHA-256 e a calibração persiste os parâmetros de *Platt scaling* (coeficientes $a$ e $b$ da regressão logística do DT-Kinase) e o limiar selecionado, todos derivados do conjunto de validação do corpus de treinamento e fixados antes da inferência.

## Referências cruzadas

A documentação acadêmica completa está no Anexo B da tese, com a discussão metodológica em `~/PhD/tex/anexoB.tex`. A motivação para a arquitetura por consenso, fundamentada na matriz cross-dataset 3×3 dos quatro modelos, está no Anexo A. As decisões metodológicas individuais (calibração nativa preservada, escolha do *checkpoint* canônico vanilla v7, refutação do v7+F em validação cinco-sementes) estão documentadas no Apêndice F (Lições 16, 24).
