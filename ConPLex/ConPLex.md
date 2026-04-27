# ConPLex — Contextualização com a Tese DT-Kinase

> **Referência**: Singh, R., Sledzieski, S., Bryson, B., Cowen, L. & Berger, B. *Contrastive learning in protein language space predicts interactions between drugs and protein targets.* PNAS **120**(24), e2220778120 (2023). DOI: [10.1073/pnas.2220778120](https://doi.org/10.1073/pnas.2220778120)

---

## 1. Por Que ConPLex na Tese?

O ConPLex é o modelo DTI **structure-free** (SMILES + FASTA) de **maior impacto** publicado entre 2023–2025 (PNAS, 741+ citações). Ele compartilha o mesmo paradigma de entrada do DT-Kinase — dispensando estruturas 3D — mas emprega uma filosofia arquitetural fundamentalmente distinta: **co-embedding contrastivo** versus **modelagem explícita de interação bimodal**.

Essa complementaridade o torna o baseline ideal para contextualizar as contribuições do DT-Kinase:

| Aspecto | DT-Kinase | ConPLex |
|---------|-----------|---------|
| **Inputs** | SMILES + FASTA | SMILES + FASTA |
| **Escopo** | Kinases (EC 2.7.x) | Proteoma completo |
| **Representação proteica** | ProtBERT (mean pooling) | ProtBert / ESM-1b (mean pooling) |
| **Representação molecular** | Morgan FP 2048-bit | Morgan FP 2048-bit |
| **Interação** | Cross-attention + CNN bimodal | Projeção linear → cosseno |
| **Treinamento** | BCE supervisionado | BCE + contrastivo com decoys (DUD-E) |
| **Métricas target** | AUROC, AUPRC, MCC | AUPR, AUROC |
| **Validação experimental** | — | 12/19 hits (KD < 100 nM) |
| **Publicação** | (tese, 2024–2025) | PNAS 2023 |

---

## 2. Arquitetura do ConPLex

### 2.1 Pipeline de Featurização

```
      SMILES ────→ Morgan FP (2048-bit) ────→ drug_emb ∈ ℝ²⁰⁴⁸
      FASTA  ────→ ProtBert (mean pool)  ────→ target_emb ∈ ℝ¹⁰²⁴
```

As featurizações são **pré-computadas e cacheadas em disco** (HDF5). A primeira execução computa e salva; execuções subsequentes carregam diretamente.

**Featurizadores alternativos** (implementados no código, selecionáveis via config):
- **Proteínas**: ESM-1b (1280d), ProtT5-XL (1024d), D-SCRIPT (100d), ProSE (6165d), BindPredict21 (128d), FoldSeek (22d)
- **Drogas**: Mol2Vec (300d), MolR/GNN (1024d)

### 2.2 Modelo de Predição

```
drug_emb ──→ [Linear(2048, 1024) → ReLU] ──→ drug_proj ∈ ℝ¹⁰²⁴
                                                    ↓
                                              CosineSimilarity ──→ p̂ ∈ [0, 1]
                                                    ↑
target_emb ──→ [Linear(1024, 1024) → ReLU] ──→ target_proj ∈ ℝ¹⁰²⁴
```

O modelo usado no paper (`SimpleCoembeddingNoSigmoid` = `SimpleCoembedding` com `classify=True`) consiste em apenas **duas camadas lineares + ReLU** que projetam para o mesmo espaço latente de d=1024, seguidas por similaridade cosseno. A predição é diretamente o cosseno ∈ [-1, 1] (sem sigmoid).

> **Contraste com DT-Kinase**: O DT-Kinase modela a interação via cross-attention explícita entre tokens do drug e do target, produzindo um mapa de interação que é processado por CNN 2D. Isso captura **quais regiões** da proteína interagem com quais subestruturas da droga. O ConPLex, por design, colapsa essa informação em um único escalar de similaridade — mais eficiente, mas sem interpretabilidade local.

### 2.3 Treinamento Dual-Objetivo

O diferencial central do ConPLex é o **treinamento alternado** entre dois objetivos em cada época:

**Passo 1 — Classificação Binária (BCE)**:
```python
# Sobre dataset de baixa cobertura (BindingDB/BIOSNAP/DAVIS)
pred = model(drug_emb, target_emb)           # cosseno
loss = BCELoss(pred, label)                   # label ∈ {0, 1}
```

**Passo 2 — Contrastivo (Triplet Margin Loss) sobre DUD-E**:
```python
# Triplets (âncora=target, positivo=droga, negativo=decoy)
anchor_proj  = model.target_projector(target_emb)
pos_proj     = model.drug_projector(drug_emb)
neg_proj     = model.drug_projector(decoy_emb)

L_TRM = (1/N) * Σ max(D(a,p) - D(a,n) + m, 0)
# onde D(u,v) = 1 - CosineSimilarity(u,v)
```

**Margin Annealing** (tanh decay com warm restarts):
```
m(i) = M_max × (1 − tanh(2 × (i mod E_max) / E_max))
```
Com `M_max = 0.25`, `E_max = 10`, por 50 épocas contrastivas.

> **Insight para a tese**: Sem treinamento contrastivo, o ConPLex **não diferencia** drogas de decoys (p = 0.999 no teste pareado). O contrastivo aumenta o Cohen's d mediano de 0.730 para 4.716 — é este componente que fundamenta a capacidade de generalização.

### 2.4 Hiperparâmetros de Referência

| Parâmetro | Valor |
|-----------|-------|
| Latent dimension (d) | 1024 |
| BCE learning rate | 1e-4 |
| Contrastive learning rate | 1e-5 |
| LR scheduler | CosineAnnealing com WarmRestarts (T₀=10) |
| Margin máximo | 0.25 |
| Batch size (BCE) | 32 |
| Batch size (contrastivo) | 256 |
| Optimizer | AdamW |
| Épocas | 50 |
| Init pesos | Xavier Normal |
| Negativas por positivo (DUD-E) | k = 50 |

---

## 3. Resultados Reportados (Paper)

### 3.1 Benchmarks DTI (AUPR, 5 replicatas)

| Dataset | ConPLex | EnzPred-CPI | MolTrans | GNN-CPI | DeepConv-DTI |
|---------|---------|-------------|----------|---------|-------------|
| **BIOSNAP** | **0.897 ± 0.001** | 0.866 ± 0.003 | 0.885 ± 0.005 | 0.890 ± 0.004 | 0.889 ± 0.005 |
| **BindingDB** | **0.628 ± 0.012** | 0.602 ± 0.006 | 0.598 ± 0.013 | 0.578 ± 0.015 | 0.611 ± 0.015 |
| **DAVIS** | **0.458 ± 0.016** | 0.277 ± 0.009 | 0.335 ± 0.017 | 0.269 ± 0.020 | 0.299 ± 0.039 |
| Unseen Drugs | **0.874 ± 0.002** | 0.844 ± 0.005 | 0.863 ± 0.005 | — | 0.847 ± 0.009 |
| Unseen Targets | **0.842 ± 0.006** | 0.795 ± 0.004 | 0.668 ± 0.045 | — | 0.766 ± 0.022 |

### 3.2 Viés para Kinases no DUD-E

A avaliação com decoys do DUD-E (31 alvos no test set, agrupados por família) mostra que o ConPLex separa drogas reais de decoys com eficácia **dependente da família proteica**:

- **Kinases e proteínas nucleares**: maior Cohen's d, p-values mais baixos → separação mais forte
- **GPCRs**: separação mais modesta

Isso ocorre porque o **DUD-E contém vários alvos kinase** no treinamento contrastivo. A Fig. 4F do paper confirma: domínios Pfam de kinase (PF00069 "Pkinase", PF07714, PF01404, PF14575) são os que mais melhoram em relação ao baseline ProtBert puro.

> **Implicação para a tese**: O ConPLex tem um **viés implícito** para kinases herdado do DUD-E. O DT-Kinase tem um **viés explícito** via curadoria do BindingDB (filtro EC 2.7.x + PFAM/InterPro). A comparação no domínio kinase é justa, mas a origem do viés é distinta.

### 3.3 Validação Experimental

Os autores testaram 19 interações quinase-droga via ensaio KdELECT (DiscoveryX):
- **12/19 (63%)** confirmadas com KD < 100 nM
- **4 com afinidade sub-nanomolar**: AG-1478→EGFR (0.33 nM), Gefitinib→EGFR (0.60 nM), Nintedanib→FLT3 (0.17 nM), Linifanib→FLT3 (0.72 nM)
- **Descoberta nova**: **EPHB1 + PD-166326** (KD = 1.30 nM), não previamente caracterizada na literatura

---

## 4. Comparação Arquitetural: ConPLex vs DT-Kinase

### 4.1 Paradigma de Interação

```
┌─────────────────────────────────────────────────────────────────┐
│ ConPLex: Co-Embedding Contrastivo                               │
│                                                                 │
│  drug ──→ Linear+ReLU ──→ proj_d ─┐                            │
│                                    ├─ cos(proj_d, proj_t) → p̂   │
│  target ──→ Linear+ReLU ──→ proj_t┘                            │
│                                                                 │
│  ✓ Eficiente: O(d) por par                                     │
│  ✓ Genome-wide scan viável (~24h para 15K prot × 1.5M drugs)   │
│  ✗ Sem interpretabilidade local (quais resíduos interagem?)     │
│  ✗ Projeção destroi informação posicional da sequência          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ DT-Kinase: Cross-Attention Bimodal + CNN                        │
│                                                                 │
│  drug tokens ──→ ProtBERT ──→ H_d ─┐                           │
│                                     ├─ CrossAttention(H_d, H_t) │
│  target tokens ──→ ProtBERT ──→ H_t┘      ↓                    │
│                                     Interaction Map (L_d × L_t)  │
│                                            ↓                    │
│                                     CNN 2D → σ → p̂              │
│                                                                 │
│  ✓ Interpretabilidade: mapa de interação resíduo × subestrutura │
│  ✓ Captura relações posicionais                                 │
│  ✗ Custo O(L_d × L_t × d) por par                              │
│  ✗ Genome-wide scan impraticável sem subsampling                │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Complementaridade

| Característica | ConPLex | DT-Kinase |
|---------------|---------|-----------|
| Escala | Proteoma inteiro | Domínio kinase |
| Throughput | ~24h para 24B pares | Horas para ~40K pares |
| Generalização zero-shot | Forte (PLM + contrastivo) | Moderada (cross-domain) |
| Especificidade intra-classe | Via DUD-E contrastivo | Via curadoria + partição scaffold |
| Interpretabilidade | Espacial (clustering UMAP) | Estrutural (mapas de interação) |

---

## 5. Processo de Implementação

### 5.1 Ambiente Conda

O script `setup_env.sh` cria o ambiente `conplex` com todas as dependências. **6 bugs** foram identificados e corrigidos durante testes locais (Mac M1):

| Bug | Causa | Fix no `setup_env.sh` |
|-----|-------|----------------------|
| `No module 'dgl'` | Dep implícita | `pip install dgl` |
| `FileNotFoundError: libgraphbolt_pytorch_2.4.1` | DGL graphbolt ABI mismatch | Patch `__init__.py` |
| `No module 'deepchem'` | Dep implícita em `molecule.py` | `pip install deepchem` |
| `No module 'mol2vec'` | Dep implícita em `molecule.py` | `pip install mol2vec@git+...` |
| `AttributeError: 'get_adjustment'` | pandas 2.3 ↔ rdkit | `pandas>=1.5,<2.1` |
| `No module 'pkg_resources'` | setuptools 72+ | `setuptools>=65,<71` |

Adicionalmente, corrigido caminho hardcoded do MIT CSAIL em `molecule.py`:
```diff
-MODEL_CACHE_DIR = Path("/afs/csail.mit.edu/u/s/samsl/Work/Adapting_PLM_DTI/models")
+MODEL_CACHE_DIR = Path(os.environ.get("CONPLEX_MODEL_DIR",
+    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models")))
```

### 5.2 Instalação na Máquina GPU

```bash
cd /path/to/attention-screening/ConPLex
bash setup_env.sh        # auto-detecta CUDA, instala tudo
conda activate conplex
```

### 5.3 Execução dos Benchmarks

```bash
# DAVIS (default do paper)
WANDB_MODE=disabled python train_DTI.py \
  --exp-id conplex_davis_rep0 \
  --config configs/default_config.yaml \
  --task davis --d 0

# BIOSNAP
WANDB_MODE=disabled python train_DTI.py \
  --exp-id conplex_biosnap_rep0 \
  --config configs/default_config.yaml \
  --task biosnap --d 0

# BindingDB
WANDB_MODE=disabled python train_DTI.py \
  --exp-id conplex_bindingdb_rep0 \
  --config configs/default_config.yaml \
  --task bindingdb --d 0
```

**Replicatas**: Para replicar os resultados do paper (5 seeds):
```bash
for rep in 0 1 2 3 4; do
  WANDB_MODE=disabled python train_DTI.py \
    --exp-id conplex_davis_rep${rep} \
    --config configs/default_config.yaml \
    --task davis --r ${rep} --d 0
done
```

### 5.4 Dados Disponíveis

| Dataset | Train | Val | Test | Drogas | Targets | Pos Ratio |
|---------|-------|-----|------|--------|---------|-----------|
| DAVIS | 2,086 | 3,006 | 6,011 | 68 | 372 | 0.136 |
| BIOSNAP | 19,238 | 2,748 | 5,497 | 4,400 | 2,176 | 0.503 |
| BindingDB | 12,668 | 6,644 | 13,289 | 3,814 | 1,026 | 0.281 |

### 5.5 Testes de Validação Local

8/8 testes passaram localmente (Mac M1, CPU):

1. ✅ Imports de todos os módulos (`architectures`, `data`, `margin`, `utils`, `featurizers`)
2. ✅ Forward pass nos 4 modelos (`SimpleCoembedding`, `Sigmoid`, `GoldmanCPI`, `SimpleCosine`)
3. ✅ MorganFeaturizer em SMILES reais
4. ✅ Resolução de caminhos dos 3 datasets
5. ✅ Carregamento do `default_config.yaml`
6. ✅ Loop de treino sintético (forward + backward + optimizer step)
7. ✅ Contrastive loss (`MarginScheduledLossFunction`)
8. ✅ Métricas (AUROC/AUPR via `torchmetrics`)

### 5.6 Primeira Execução (notas)

Na **primeira execução**, o sistema fará downloads automáticos:
- **ProtBert** (~1.68 GB) — modelo HuggingFace `Rostlab/prot_bert`
- Após download, as features são computadas e salvas em `dataset/{TASK}/ProtBert_features.h5`
- Execuções subsequentes carregam as features do cache (rápido)

---

## 6. O Que "Excela em Kinases" Significa

O paper avalia a capacidade do modelo em distinguir drogas reais de decoys (moléculas estruturalmente similares mas inativas) via o benchmark DUD-E. Os resultados são estratificados por família proteica:

- **Kinases**: Cohen's d **mais alto**, indicando que as predições do ConPLex para pares droga-kinase são significativamente diferentes das predições para pares decoy-kinase
- **GPCRs**: Cohen's d mais modesto, separação menor

**A razão é composicional, não arquitetural**: o DUD-E contém múltiplos alvos kinase no treinamento contrastivo, criando um viés de amostragem. O modelo "aprendeu mais" sobre kinases porque viu mais exemplos contrastivos desse domínio.

Para a tese, isso significa:
1. A comparação ConPLex vs DT-Kinase **no domínio kinase** é justa — ambos possuem viés (explícito via curadoria no DT-Kinase, implícito via DUD-E no ConPLex)
2. Se o DT-Kinase superar o ConPLex nos benchmarks kinase-específicos, o argumento de que modelagem explícita de interação (cross-attention) é um **fator determinante** para especificidade intra-domínio se fortalece
3. Se o ConPLex superar, o argumento se inverte: representações contrastivas generalistas já capturam especificidade kinase suficiente, sem necessidade de modelagem explícita

---

## 7. Integração com a Narrativa da Tese

### Capítulo 5 — Contextualização com ConPLex

A inclusão do ConPLex como baseline comparativo fortalece a tese em três eixos:

**Eixo 1 — Validação do paradigma structure-free**: Ambos os modelos demonstram que representações de linguagem (SMILES + FASTA) são suficientes para DTI, dispensando docking molecular ou estruturas experimentais. O ConPLex confirma essa premissa em escala proteômica.

**Eixo 2 — Justificativa da modelagem explícita**: O ConPLex atinge SOTA com arquitetura **extremamente simples** (2 camadas lineares + cosseno). O DT-Kinase adiciona complexidade (cross-attention + CNN). A comparação direta evidencia se essa complexidade adicional se traduz em ganho mensurável, particularmente no MCC (métrica mais exigente que AUROC/AUPR para datasets desbalanceados).

**Eixo 3 — Trade-off generalização vs especialização**: O ConPLex visa generalização proteômica; o DT-Kinase visa especialização em kinases. Demonstrar que a especialização (curadoria + partição scaffold) supera a generalização no domínio-alvo justifica a abordagem de "painel computacional" proposta na tese.
