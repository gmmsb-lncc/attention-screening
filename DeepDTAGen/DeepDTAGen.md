# DeepDTAGen: Reprodução e Avaliação Comparativa no Contexto de Predição de Interação Droga-Alvo em Quinases

---

## 1. Introdução e Motivação

O modelo DeepDTAGen (Kalemati *et al.*, Nature Communications, 2025) representa uma abordagem multitarefa para predição de afinidade droga-alvo (DTA) e geração condicional de moléculas bioativas. Diferentemente de arquiteturas contemporâneas que dependem de modelos fundacionais pré-treinados — como ESM-2 para codificação proteica ou MoLFormer para representação molecular —, o DeepDTAGen treina a totalidade de seus 34,6 milhões de parâmetros a partir do zero (*from scratch*), sem qualquer transferência de aprendizado.

Esta característica torna o DeepDTAGen um contraponto metodológico significativo para a análise comparativa conduzida nesta tese. Enquanto o DT-Kinase emprega embeddings contextuais de modelos fundacionais (ESM-2 com 650M parâmetros para proteínas, MoLFormer com 47M parâmetros para ligantes), o DeepDTAGen constrói representações internas exclusivamente a partir dos dados de treinamento, utilizando redes convolucionais de grafos (GCN) para moléculas e redes convolucionais com mecanismo de *gating* (Gated-CNN) para sequências proteicas.

A reprodução do DeepDTAGen neste trabalho visa dois objetivos complementares:

1. **Avaliação do modelo pré-treinado**: Testar os pesos publicados pelos autores diretamente nos conjuntos-teste universais de quinases (não-humanas, humanas e combinado), sem qualquer retreino, para avaliar a capacidade de generalização *zero-shot* do modelo.

2. **Treinamento com dados universais**: Treinar o DeepDTAGen do zero nos conjuntos de treinamento universais (partição *scaffold-split*) e avaliar exclusivamente nos respectivos conjuntos-teste, permitindo comparação direta com DT-Kinase, DrugBAN e GraphBAN sob condições idênticas de dados.

---

## 2. Arquitetura do Modelo

### 2.1 Visão Geral

O DeepDTAGen é composto por quatro módulos principais organizados em uma arquitetura multitarefa com dois ramos de saída: predição de afinidade e geração molecular condicional.

```
                      ┌─────────────────────────────┐
                      │     SMILES (Ligante)         │
                      └──────────┬──────────────────┘
                                 │
                          ┌──────▼──────┐
                          │  RDKit →    │
                          │  Grafo      │
                          │  Molecular  │
                          └──────┬──────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Encoder (GCN + VAE)   │
                    │   3 camadas GCNConv     │
                    │   Reparametrização      │
                    │   z ~ N(μ, σ²)          │
                    └────────┬───────┬────────┘
                             │       │
                    ┌────────▼──┐  ┌─▼──────────────┐
                    │  PMVO     │  │  AMVO (latent)  │
                    │  (Drug    │  │  + Sequência    │
                    │  Features)│  │  do Grafo       │
                    └─────┬────┘  └──────┬──────────┘
                          │              │
                          │     ┌────────▼────────┐
                          │     │  Transformer    │
                          │     │  Encoder (8L)   │
                          │     │  (dencoder)     │
                          │     └────────┬────────┘
                          │              │
┌──────────────┐          │     ┌────────▼────────┐
│ Sequência    │          │     │  Transformer    │     ┌──────────────┐
│ Proteica     │          │     │  Decoder (8L)   │────▶│ Geração de   │
│ (FASTA)      │          │     │  (decoder)      │     │ SMILES       │
└──────┬───────┘          │     └─────────────────┘     └──────────────┘
       │                  │
┌──────▼───────┐    ┌─────▼─────┐
│ Gated-CNN    │    │           │
│ 3 camadas    │    │  FC Head  │
│ (conv+gate)  │    │  4 camadas│──────▶ Afinidade (ŷ)
│              │    │  (1024→   │
└──────┬───────┘    │   512→    │
       │            │   256→1)  │
       └────────────┘           │
       (Concatenação)           │
                                │
```

### 2.2 Codificação de Ligantes: GCN + VAE

O módulo `Encoder` processa moléculas como grafos moleculares, onde cada nó representa um átomo e cada aresta uma ligação química.

**Featurização atômica** (94 dimensões por átomo):
- Símbolo atômico (44 tipos + desconhecido): one-hot encoding
- Grau de adjacência (0–10): one-hot encoding
- Número de hidrogênios totais (0–10): one-hot encoding
- Valência implícita (0–10): one-hot encoding
- Carga formal (−2, −1, 0, +1, +2): one-hot encoding
- Hibridização (SP, SP2, SP3, SP3D, SP3D2): one-hot encoding
- Aromaticidade: binário
- Pertencimento a anel: binário

**Featurização de ligações** (5 dimensões por aresta):
- Tipo de ligação (simples, dupla, tripla, aromática): one-hot encoding
- Ordem da ligação: valor contínuo

A sequência de três camadas `GCNConv` (94→188→282→376) transforma os *features* atômicos em representações latentes de dimensionalidade fixa (376). As representações aprendidas passam por um módulo VAE com reparametrização (*reparameterization trick*), produzindo um espaço latente contínuo (AMVO) condicionado pela representação proteica:

$$z = \mu + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

$$\mathcal{L}_{KL} = -\frac{1}{2} \sum_{i=1}^{d} \left(1 + \log\sigma^2_i - \mu^2_i - \sigma^2_i\right)$$

### 2.3 Codificação de Proteínas: Gated-CNN

O módulo `GatedCNN` processa sequências proteicas representadas como inteiros (25 aminoácidos + token de desconhecido) truncadas em 1000 resíduos.

A arquitetura emprega três camadas convolucionais 1D com mecanismo de *gating* (Dauphin *et al.*, 2017):

$$h_l = \text{Conv}_{1D}(x_l) \odot \sigma\left(\text{Gate}_{1D}(x_l)\right)$$

onde $\odot$ denota o produto elemento-a-elemento e $\sigma$ é a função sigmóide. Os filtros crescem progressivamente (32→64→96) com kernel de tamanho 8. A saída é achatada e projetada para a dimensão final via camada FC.

### 2.4 Predição de Afinidade

As representações de droga (PMVO, obtida via *global max pooling* do GCN) e proteína são concatenadas e processadas por um *head* de quatro camadas fully-connected:

$$\hat{y} = \text{FC}_{256 \to 1}\left(\text{FC}_{512 \to 256}\left(\text{FC}_{1024 \to 512}\left(\text{FC}_{256 \to 1024}([d \| p])\right)\right)\right)$$

com ativações ReLU e *dropout* (p=0.3) entre camadas.

### 2.5 Geração Molecular Condicional

O ramo generativo utiliza um Transformer-Decoder de 8 camadas (8 *attention heads*, FFN=1024, dim=376) que gera SMILES autoregressivamente, condicionado pela representação latente do encoder. A sequência latente do grafo molecular passa primeiro por um Transformer-Encoder de 8 camadas (`dencoder`) antes de servir como memória para o decoder.

A perda de modelagem de linguagem é calculada como *cross-entropy* sobre o vocabulário SMILES:

$$\mathcal{L}_{LM} = -\sum_{t=1}^{T} \log P(w_t | w_{<t}, z)$$

### 2.6 Otimização Multitarefa: FetterGrad

O treinamento conjunto das tarefas de predição e geração utiliza o otimizador FetterGrad, que resolve conflitos de gradiente entre objetivos. Para cada par de gradientes $g_i, g_j$ correspondentes a diferentes tarefas, FetterGrad calcula a distância euclidiana normalizada:

$$\alpha_{ij} = \frac{1}{1 + \|g_i - g_j\|_2}$$

Quando $\alpha_{ij} < 0.5$ (gradientes divergentes), o gradiente $g_i$ é corrigido adicionando $\alpha_{ij} \cdot g_j$, favorecendo direções de descida compatíveis entre tarefas.

A perda total combina três componentes:

$$\mathcal{L} = \mathcal{L}_{MSE} + \mathcal{L}_{LM} + 0.001 \cdot \mathcal{L}_{KL}$$

### 2.7 Hiperparâmetros do Modelo

| Componente | Parâmetro | Valor |
|---|---|---|
| Encoder (GCN) | Camadas | 3 |
| | Dimensões | 94→188→282→376 |
| | Dropout | 0.2 |
| Gated-CNN | Camadas | 3 |
| | Filtros | 32→64→96 |
| | Kernel | 8 |
| | Embedding proteico | 128 dim |
| Transformer Encoder | Camadas | 8 |
| | Attention heads | 8 |
| | FFN dim | 1024 |
| | Dim modelo | 376 |
| Transformer Decoder | Camadas | 8 |
| | Attention heads | 8 |
| | FFN dim | 1024 |
| FC Head | Camadas | 4 |
| | Dimensões | 256→1024→512→256→1 |
| | Dropout | 0.3 |
| Treinamento | Optimizer | Adam (lr=2×10⁻⁴) |
| | Batch size | 32 |
| | Épocas | 500 (original) / 200 (universal) |
| | Seed | 4221 |
| **Total de parâmetros** | | **34.666.731** |

---

## 3. Protocolo de Reprodução

### 3.1 Ambiente Computacional

O ambiente de execução foi configurado como um **ambiente Conda isolado** (`deepdtagen`, Python 3.10), seguindo o padrão estabelecido para DrugBAN e GraphBAN neste trabalho:

| Dependência | Versão | Função |
|---|---|---|
| Python | 3.10 | Runtime |
| PyTorch | 2.4.1 | Framework de deep learning |
| PyTorch-Geometric | 2.7.0 | Redes neurais em grafos (GCNConv) |
| torch-scatter/sparse | Compatível | Operações esparsas para PyG |
| RDKit | ≥2023.09 | Processamento molecular (SMILES→grafos) |
| einops | ≥0.7 | Operações tensoriais (Rearrange) |
| scikit-learn | ≥1.3 | Métricas de avaliação |

### 3.2 Shim Local para fairseq

O DeepDTAGen utiliza componentes do framework fairseq (Meta AI) para as camadas Transformer: `TransformerEncoderLayer`, `TransformerDecoderLayer` e `FairseqIncrementalDecoder`. A instalação completa do fairseq apresenta falhas de compilação C++ em plataformas modernas (macOS ARM, CUDA ≥12). Para contornar este obstáculo sem comprometer a compatibilidade com os *checkpoints* pré-treinados, foi implementado um **shim local** (`fairseq/`) com as seguintes características:

- **Parametrização idêntica ao fairseq original**: Projeções Q, K, V separadas (`k_proj`, `v_proj`, `q_proj`, `out_proj`) como instâncias de `nn.Linear`, em vez do `nn.MultiheadAttention` monolítico do PyTorch (que usa `in_proj_weight`). Esta decisão foi mandatória para carregar corretamente os pesos pré-treinados.

- **Funcionalidade equivalente**: As camadas reimplementadas produzem computação de *scaled dot-product attention* idêntica, incluindo máscaras causais e *padding masks*.

- **Estrutura de arquivos**:
  ```
  fairseq/
  ├── __init__.py                    # Pacote raiz
  ├── models/__init__.py             # FairseqIncrementalDecoder (base class)
  └── modules/__init__.py            # TransformerEncoderLayer, TransformerDecoderLayer,
                                     # MultiheadAttentionSeparateProj
  ```

### 3.3 Adaptação dos Dados Universais

Os conjuntos de dados universais de quinases foram originalmente preparados para avaliação de DrugBAN e GraphBAN em formato CSV com três colunas: `SMILES`, `Protein`, `Y` (rótulo binário: 0=não-ligante, 1=ligante), empregando partição baseada em *scaffold* molecular (Bemis-Murcko) para garantir que compostos com esqueletos químicos semelhantes não apareçam simultaneamente nos conjuntos de treino e teste.

A conversão para o formato DeepDTAGen requer as seguintes transformações:

| Campo DeepDTAGen | Origem | Transformação |
|---|---|---|
| `compound_iso_smiles` | `SMILES` | Mapeamento direto |
| `target_smiles` | `SMILES` | Cópia do SMILES do composto (MTS) |
| `target_sequence` | `Protein` | Mapeamento direto |
| `affinity` | `Y` | Conversão float (0.0 ou 1.0) |

Adicionalmente, cada molécula é convertida para representação de grafo molecular (featurização atômica de 94 dimensões + featurização de ligações de 5 dimensões) e cada sequência proteica é codificada como vetor inteiro de comprimento fixo (1000 posições). Moléculas com um único átomo (grafos sem arestas) são filtradas pois geram erros na transposição da matriz de adjacência.

**Conjuntos de dados processados:**

| Dataset | Train | Test | Tokenizer (vocab) |
|---|---|---|---|
| non_human | 7.624 | 1.702 | 79 tokens |
| human | 269.715 | 40.470 | 105 tokens |
| all | 278.258 | 41.669 | 105 tokens |

### 3.4 Modelos Pré-treinados

Três modelos pré-treinados foram obtidos do repositório dos autores:

| Modelo | Dados de Treino | Parâmetros | Vocabulário |
|---|---|---|---|
| `deepdtagen_model_bindingdb.pth` | BindingDB (54K pares) | 34.666.731 | 107 tokens |
| `deepdtagen_model_davis.pth` | Davis (30K pares) | 34.666.731 | 58 tokens |
| `deepdtagen_model_kiba.pth` | KIBA (118K pares) | 34.666.731 | 69 tokens |

Os modelos Davis e KIBA possuem vocabulários SMILES mais restritos (58 e 69 tokens, respectivamente) do que o requerido pelos dados universais de quinases, impedindo sua avaliação direta. O modelo BindingDB, com vocabulário de 107 tokens, abrange a totalidade dos caracteres SMILES presentes nos dados universais.

---

## 4. Validação de Reprodutibilidade

### 4.1 Verificação de Integridade

Antes da avaliação nos dados universais, a integridade do pipeline foi verificada através de:

1. **Forward pass**: O modelo carrega os dados processados, executa a passagem direta e produz predições de afinidade com valores no intervalo esperado (escala pKd/pKi para pretrained, 0–1 para dados binários).

2. **Backward pass**: O otimizador FetterGrad computa gradientes para as duas tarefas (MSE + LM loss) sem erros de dimensionalidade.

3. **Geração condicional**: O Transformer-Decoder gera sequências SMILES autoregressivamente a partir da representação latente.

4. **Compatibilidade de pesos**: O shim local carrega corretamente os 34,6M parâmetros dos checkpoints originais sem chaves faltantes ou inesperadas.

### 4.2 Avaliação Preliminar: Pretrained BindingDB → non_human

O modelo pré-treinado no BindingDB foi avaliado diretamente no conjunto-teste `non_human` (1.702 amostras, scaffold-split):

| Métrica | Valor |
|---|---|
| AUROC | 0.6143 |
| AUPRC | 0.6494 |
| MCC | 0.1823 |
| Threshold ótimo | 6.7852 |
| Faixa de predição | [3.70, 9.55] |
| Taxa de positivos | 0.5317 |

**Interpretação**: O modelo prediz afinidades contínuas na escala pKd (valores tipicamente entre 3 e 10), enquanto os rótulos dos dados universais são binários (0 ou 1). O AUROC de 0.61 reflete uma transferência *zero-shot* — o modelo não foi treinado para classificação binária de quinases nem exposto a este conjunto de dados durante o treinamento. Este resultado serve como *baseline* inferior; o treinamento com dados universais deverá produzir métricas significativamente superiores.

---

## 5. Protocolo de Execução

### 5.1 Estrutura de Scripts

O pipeline de execução é composto por três scripts shell e três scripts Python:

| Script | Finalidade |
|---|---|
| `setup_env.sh` | Criação do ambiente Conda com todas as dependências |
| `run_universal.sh` | Pipeline orquestrador: conversão → avaliação pretrained → treinamento |
| `convert_universal_data.py` | Conversão de CSVs universais para formato .pt (PyTorch Geometric) |
| `eval_pretrained_universal.py` | Avaliação dos modelos pré-treinados nos conjuntos-teste universais |
| `train_universal.py` | Treinamento do DeepDTAGen do zero nos dados universais |

### 5.2 Execução na Máquina com GPU

```bash
# Na máquina de execução (RTX 4090)
cd DeepDTAGen

# Passo 1: Preparar ambiente
bash setup_env.sh

# Passo 2: Pipeline completo
bash run_universal.sh

# Alternativas:
bash run_universal.sh --eval-only     # Apenas avaliar pré-treinados
bash run_universal.sh --train-only    # Apenas treinar do zero
```

O pipeline `run_universal.sh` executa sequencialmente:

1. **Conversão** (`convert_universal_data.py`): Gera os arquivos `.pt` para `human` e `all` (apenas `non_human` foi processado localmente).

2. **Avaliação pretrained** (`eval_pretrained_universal.py`): Testa os três modelos pré-treinados (BindingDB, Davis, KIBA) nos três conjuntos-teste universais, com verificação automática de compatibilidade de vocabulário.

3. **Treinamento** (`train_universal.py`): Treina o DeepDTAGen do zero em cada um dos três conjuntos de treinamento (200 épocas, avaliação a cada 10 épocas), salvando o melhor modelo por AUROC.

### 5.3 Métricas de Avaliação

Para manter consistência com a análise comparativa da tese, as seguintes métricas são computadas:

- **AUROC** (*Area Under the ROC Curve*): Capacidade discriminativa global
- **AUPRC** (*Area Under the Precision-Recall Curve*): Desempenho em cenários com desbalanceamento de classes
- **MCC** (*Matthews Correlation Coefficient*): Métrica de classificação balanceada (threshold otimizado por busca em grade)

As predições contínuas de afinidade do DeepDTAGen são utilizadas diretamente como *scores* para AUROC e AUPRC, e binarizadas com threshold ótimo para MCC.

---

## 6. Posicionamento Comparativo

### 6.1 DeepDTAGen vs. DT-Kinase

| Aspecto | DeepDTAGen | DT-Kinase |
|---|---|---|
| Tipo de modelo | Treinado do zero | Transfer learning |
| Codificação molecular | GCN (grafos) | MoLFormer (SMILES) |
| Codificação proteica | Gated-CNN (seq. inteiro) | ESM-2 (embeddings contextuais) |
| Interação droga-alvo | Concatenação + FC | Bilinear Attention Network |
| Parâmetros totais | 34,6M | ~700M (incluindo fundacionais) |
| Parâmetros treináveis | 34,6M (100%) | ~5M (fine-tuning) |
| Tarefa auxiliar | Geração de SMILES (VAE+Decoder) | Nenhuma |
| Otimização | FetterGrad (multi-task) | Adam (single-task) |

### 6.2 DeepDTAGen vs. DrugBAN / GraphBAN

| Aspecto | DeepDTAGen | DrugBAN | GraphBAN |
|---|---|---|---|
| Representação molecular | GCN (grafo) | GCN (grafo) | GCN (grafo) |
| Representação proteica | Gated-CNN (sequência) | CNN (fc_after) | CNN (sequência) |
| Mecanismo de interação | Concatenação + FC | Bilinear Attention | Bipartite Graph + BAT |
| Pré-treinamento | Nenhum | Nenhum | Nenhum |
| Capacidade generativa | Sim (Transformer-Decoder) | Não | Não |
| Total de parâmetros | 34,6M | ~1,4M | ~2,1M |

---

## 7. Limitações e Considerações

1. **Mismatch de tarefa**: O DeepDTAGen foi concebido para regressão de afinidade contínua (pKd/pKi), enquanto os dados universais de quinases empregam classificação binária. A adaptação é feita interpretando os *scores* de afinidade como *ranking scores* para métricas de classificação.

2. **Vocabulário SMILES**: Os modelos pré-treinados Davis e KIBA possuem vocabulários restritos e não podem avaliar diretamente os dados universais. Apenas o modelo BindingDB tem vocabulário suficiente.

3. **Coluna `target_smiles`**: O DeepDTAGen original utiliza uma representação SMILES "modificada do alvo" (MTS) para o ramo generativo. Na adaptação para dados universais, esta coluna recebe o SMILES do composto, pois a representação MTS não é disponível nos dados de entrada. Isto pode afetar o desempenho do ramo generativo mas não impacta a predição de afinidade.

4. **Escala de dados**: Os conjuntos `human` (270K) e `all` (278K) são substancialmente maiores que os dados originais (Davis: 30K, BindingDB: 54K), o que pode requerer ajuste de hiperparâmetros.

---

## Referências

- Kalemati, M., Zamani Emani, M., & Koohi, S. (2025). DeepDTAGen: a multitask deep learning framework for drug-target affinity prediction and target-aware drugs generation. *Nature Communications*, 16, 5117. https://doi.org/10.1038/s41467-025-59917-6

- Dauphin, Y. N., Fan, A., Auli, M., & Grangier, D. (2017). Language modeling with gated convolutional networks. *International Conference on Machine Learning (ICML)*.

- Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. *International Conference on Learning Representations (ICLR)*.

- Kingma, D. P., & Welling, M. (2014). Auto-encoding variational Bayes. *International Conference on Learning Representations (ICLR)*.
