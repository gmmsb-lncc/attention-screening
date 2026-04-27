# Pipeline de Inferência: Comitê Multi-Modelo

Pipeline de inferência por consenso de quatro modelos (DT-Kinase-LEGACY, DrugBAN, GraphBAN, ConPLex) sobre pares proteína-ligante (kinase-inhibitor).

Documentação acadêmica completa no Anexo B da tese (`~/PhD/tex/anexoB.tex`).

## Arquitetura

```
input → expand_pairs.py → pairs.tsv
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
     dtkinase_score.py  drugban_score.py  graphban_score.py  conplex_score.py
        (env: ss)         (env: drugban)    (env: graphban)   (env: conplex)
              │               │               │               │
              └───────────────┴───────┬───────┴───────────────┘
                                      ▼
                          aggregate.py (soft mean + Borda + tier)
                                      │
                                      ▼
                          attention.py (top-K STRONG/LIKELY)
                                      │
                                      ▼
                  results/inference/<run_id>/{consensus.csv, attention/}
```

Orquestração end-to-end: `committee.py`.

## Modos de uso

### Triagem ligante → kinome humano

```bash
python committee.py \
    --smiles "CC1=C(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc(Nc2nccc(-c3cccnc3)n2)cc1" \
    --organism human \
    --out results/inference/imatinib_vs_human \
    --top-k 20 \
    --parallel
```

Expande SMILES contra `data/reference/kinome_human.fasta` (483 quinases), gera 483 pares, scora c/ comitê, retorna ranking top-20 + mapas atenção.

### Triagem proteína → biblioteca ligantes

```bash
python committee.py \
    --fasta my_kinase.fa \
    --out results/inference/abl1_vs_lib \
    --top-k 50
```

Expande proteína contra `data/reference/ligand_library.tsv` (110.963 compostos ChEMBL kinase), gera ~111k pares, retorna top-50.

### Par único explícito

```bash
python committee.py \
    --smiles "CC1=C(...)" --fasta abl1.fa \
    --out results/inference/abl1_imatinib
```

### Lote arbitrário

```bash
# pairs.tsv: uniprot, sequence, chembl_id, smiles
python committee.py \
    --pairs my_pairs.tsv \
    --out results/inference/batch_001 \
    --top-k 100
```

### Subset de modelos

```bash
# 3 modelos (sem ConPLex)
python committee.py \
    --pairs pairs.tsv \
    --models dtkinase,drugban,graphban \
    --out results/inference/run_no_conplex
```

Tier rescala automaticamente (Tab B.6 Anexo B): STRONG=3/3, LIKELY=2/3.

### Dry-run (sem executar)

```bash
python committee.py --pairs pairs.tsv --out /tmp/test --dry-run
```

Imprime comandos `conda run -n {env}` que seriam executados.

## Outputs por execução

```
results/inference/<run_id>/
├── pairs.tsv                        # input expandido
├── scores_dtkinase.csv              # uniprot, chembl_id, prob, pred, threshold
├── scores_drugban.csv
├── scores_graphban.csv
├── scores_conplex.csv
├── consensus.csv                    # ranking ordenado por prob_mean DESC
├── consensus.top.csv                # subset top-K (se --top-k > 0)
└── attention/                       # apenas pares STRONG/LIKELY
    └── <pair_id>/
        ├── dtkinase_Mk.npz          # tensor [16, sp, sl] pré-CNN + agregados
        ├── dtkinase_hierpool.npz    # pesos [sp] + [sl]
        ├── drugban_BAN.npz          # BAN attention DrugBAN
        ├── graphban_BAN.npz         # BAN attention GraphBAN
        └── consensus_heatmap.pdf    # diagrama composto 2×2
```

## Esquema `consensus.csv`

| Coluna | Descrição |
|---|---|
| `pair_id` | `{uniprot}__{chembl_id}` |
| `uniprot, chembl_id` | chaves identificadoras |
| `prob_{model}, pred_{model}, thr_{model}` | output bruto por modelo |
| `prob_mean` | $\overline{p}$ = média soft das 4 probas calibradas |
| `prob_std` | $\sigma_p$ = desvio entre modelos |
| `confidence` | $1 - \sigma_p$ — alta = consenso |
| `agreement_count` | nº de modelos que predizem binder (0..4) |
| `tier` | STRONG / LIKELY / UNCERTAIN / UNLIKELY |
| `rank_fusion` | Borda count (soma de ranks por modelo, lower = better) |

## Decisão consensual

```
binder consensual ⟺ (tier ∈ {STRONG, LIKELY}) AND (prob_mean > 0.5)
```

| Agreement | Tier | Ação operacional |
|---|---|---|
| 4/4 | STRONG | inspeção experimental prioritária |
| 3/4 | LIKELY | inspeção em segunda passada |
| 2/4 | UNCERTAIN | requer evidência adicional |
| ≤1/4 | UNLIKELY | predição negativa por consenso |

Comitê parcial (3 modelos): rescala 3→3=STRONG, 2→3=LIKELY, 1→3=UNCERTAIN.

## Requisitos

### 3 opções de instalação

**A. Per-model conda envs** (default, mais robusto):
```
semantic-screening:  DT-Kinase, ESM-2 8M, MoLFormer, RDKit
drugban:             DrugBAN + DGL + RDKit
graphban:            GraphBAN + ESM-1b + ChemBERTa + DGL
conplex:             ConPLex + ProtBERT + Morgan FP
```
Cada baseline executa em subprocess `conda run -n {env}`; conflitos de versão isolados. ~15 GB total.

**B. Conda env unificado** (`baseline`):
```bash
bash scripts/inference/setup_baseline_env.sh
python kinase_profiling.py "..." --single-env baseline
```
1 env de ~3-4 GB hospeda todas as deps. Pin: Python 3.10, PyTorch 2.4.1+cu121, DGL cu121, transformers 4.39.3.

**C. Python venv via pip** (`requirements-baseline.txt`):
```bash
bash scripts/inference/setup_baseline_venv.sh                 # auto-detect CUDA
bash scripts/inference/setup_baseline_venv.sh --cpu           # force CPU build
# OU manual:
python -m venv env_baseline && source env_baseline/bin/activate
pip install torch==2.4.1 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121
pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html
pip install -r requirements-baseline.txt
pip install --no-deps dscript
```
Sem conda. ~3 GB venv. Compatível Linux+CUDA, macOS+CPU.

### Checkpoints (todos locais)

```
DT-Kinase:  results/{benchmark_*}/test/level4_cnn_8M/{corpus}/seed_42/level4_cnn_model.pt
DrugBAN:    DrugBAN/results/{corpus}/seed_42/best_model_epoch_*.pth
GraphBAN:   GraphBAN/results/{corpus}/seed_42/best_model_epoch_*.pth
ConPLex:    ConPLex/best_models/trained_{corpus}_rep{0..4}/trained_{corpus}_rep{rep}_best_model.pt
            (mapping seed→rep: 42→0, 123→1, 456→2, 789→3, 1024→4)
```

### Calibração (sidecars JSON, 60 arquivos)

Construídos por `build_calibration.py` ou pré-computados de `raw_predictions.npz`:

```
DT-Kinase:  results/.../seed_*/level4_cnn_calibration.json
            (Platt a/b + threshold MCC-óptimo)
DrugBAN:    DrugBAN/results_universal/results_universal/{corpus}/seed_*/drugban_calibration.json
            (threshold F1-óptimo, paper nativo)
GraphBAN:   GraphBAN/results_universal/{corpus}/seed_*/graphban_calibration.json
            (threshold F1-óptimo, paper nativo)
ConPLex:    ConPLex/results_universal/{corpus}/seed_*/conplex_calibration.json
            (threshold MCC-óptimo, adaptado de cosine sim)
```

### Reference libs

```
data/reference/kinome_human.fasta     (483 quinases UniProt)
data/reference/kinome_full.fasta      (660 quinases: 483 humanas + 177 NH)
data/reference/ligand_library.tsv     (110.963 ligantes ChEMBL kinase)
data/reference/cache/                 (embeddings sha1-keyed; populado on-the-fly)
```

### Cache embeddings (cold start = ~10 min p/ kinome × 1 SMILES)

```
results/protein_model_benchmark_{human,non_human}_v2/8M/build/protein_matrices/
results/protein_model_benchmark_{human,non_human}_v2/8M/build/molformer_matrix/
```

Pré-cacheados aceleram inferência: triagem SMILES vs kinome humano completa em ~5 min (warm) vs ~30 min (cold).

## Scripts auxiliares

| Script | Função |
|---|---|
| `expand_pairs.py` | input → pairs.tsv (4 modos: SMILES, FASTA, ambos, batch) |
| `encoders.py` | ESM-2 8M + MoLFormer batch encoders, cache sha1-keyed |
| `build_calibration.py` | Wrapper `eval_checkpoint_on_dataset.py` p/ extrair Platt+thr |
| `aggregate.py` | merge 4 CSVs → consensus.csv (dedupe + soft mean + Borda + tier) |
| `attention.py` | hooks 3-níveis DT-Kinase + plot 2×2 PDF |
| `committee.py` | orquestrador end-to-end |
| `models/{model}_score.py` | adapters por baseline; aceita `--pairs` + `--corpus` + `--out` |

## Smoke test

Validação end-to-end via lookup nas `raw_predictions.npz` existentes (sem GPU):

```bash
# Já validado:
#   - 4 CSVs lidos OK
#   - Dedupe by (uniprot, chembl_id): 50 input rows → 27 unique → 27 output
#   - Tiers atribuídos: STRONG/LIKELY/UNCERTAIN/UNLIKELY corretos
#   - pair_id 100% unique no output
```

## Limitações documentadas (Anexo B §7)

1. **Domínio kinase only**: extrapolação a outras famílias proteicas é não validada. Pipeline emite warning quando similaridade cosseno < 0.7 contra kinome ref.
2. **Custo cold-cache**: triagem FASTA vs ligand_library (~111k pares) em horas por modelo.
3. **Output é ranking probabilístico**, não predição de Ki/IC50. Validação experimental dirigida nos pares STRONG é obrigatória para uso em campanhas de drug discovery.

## Troubleshooting

| Sintoma | Causa provável | Fix |
|---|---|---|
| `RuntimeError: need at least 2 model score files` | Subprocesses falharam | Rode c/ `--models dtkinase` p/ debug isolado |
| `checkpoint not found` | Path mismatch | Verifique `CKPT_DIR_BY_CORPUS` no adapter |
| `Missing/Unexpected key(s) in state_dict` | Ckpt antigo c/ schema legacy | Loader auto-renomeia `query` → `queries` (lição §6.5 reformulação) |
| `consensus.csv` c/ 1000+ rows p/ 50 input | Dedupe falhou | Confirme aggregate.py linha 32 ativa `groupby` |
| ConPLex prob constante | Modelo não convergiu p/ esse subset | Verifique `raw_predictions.npz` test_y_prob |

## Referências cruzadas

- Anexo A: matriz cross-dataset 3×3 (motivação consenso)
- Anexo B: arquitetura comitê + agregação + atenção
- Apêndice F Lição 16: matched objective threshold/selection
- Apêndice F Lição 24: v7+F refutado em 5-seed; vanilla v7 canônico
- `CLAUDE.md`: estado operacional repo + path conventions
