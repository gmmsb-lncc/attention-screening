# Experiments Log — DT-Kinase Optimization Track

Registro completo de cada execução de benchmark para a otimização do
DT-Kinase v7 sobre o corpus *non-human* (NH) e — quando aplicável —
sobre os corpora *human* e *all*. Cada linha da tabela canônica é
extraída do `benchmark_comparison.json` correspondente, que reside no
host de execução em `results/benchmark_<variant>_<corpus>_8M/test/`.
Como o diretório `results/` é gitignored, este documento é a única
forma persistente, versionada e portável dos valores observados;
qualquer comparação metodologicamente sólida entre execuções deve
referenciar este registro.

---

## Configurações canônicas (por nome de config)

| Config | Tiers | Variant | Backbone | Adapter | Heads/head_dim | Selection |
|---|---|---|---|---|---|---|
| `v7.yaml` | baseline | v7 | ESM-2 8M + MoLFormer | symmetric (256/512, 4 heads) | K=8 / d=32 | val_mcc |
| `v7_plus.yaml` | A + C | v7 | idem | symmetric (512/1024, 4 heads) | K=16 / d=64 | val_mcc |
| `v7_plus_E.yaml` | A + C + E | v7 | idem | symmetric | K=16 / d=64 | val_mcc |
| `v7_plus_F.yaml` | A + C + F | v7 | idem | symmetric | K=16 / d=64 | val_mcc |
| `v7_pro.yaml` | A + C + E + F | v7 | idem | symmetric | K=16 / d=64 | val_mcc |
| `v7_ban_F.yaml` | A + C + F + BAN (Xavier W_ban) | v8 | idem | symmetric | K=16 / W_ban full | val_mcc |
| `v7_asymF.yaml` | A + C + F + asymmetric adapter (pre-norm + LoRA gates) | v7 | idem | asymmetric (prot 1×4, lig 2×12) | K=16 / d=64 | val_mcc + composite (env λ=0.5) |
| `v7_plus_F_adapt.yaml` | A + C + F + §6.5 fixes (pre-norm + LoRA gates + zero-init self_attn) | v7 | idem | symmetric (512/1024, 4 heads) | K=16 / d=64 | val_F1 (THRESHOLD_METRIC=f1, matching DrugBAN/GraphBAN) |
| `v7_plus_F_adapt_v2.yaml` | A + C + F + §6.5 + THR=f1, SEL=mcc (decoupled) | v7 | idem | symmetric | K=16 / d=64 | DESCARTADA (lição 16) — objective mismatch, AUROC regrediu |

**Tier glossary** (também em `CLAUDE.md`):
- A: capacidade (num_heads=16, head_dim=64, mlp_head, adapter dim 512/1024, patience=15, lr_mult=2.0)
- C: contrastive aux loss (weight=0.3, cosine_feat=true)
- F: label smoothing (eps=0.05)
- E: Mixup (alpha=0.3) — REJEITADO
- D: SWA vanilla (swa_start=5) — REJEITADO
- B: multi-head HierPool (sem zero-init head_proj) — REJEITADO

---

## Resultados consolidados — corpus `non_human`

| # | Config + variante | Host | n | Seeds | Train MCC | Test MCC | σ_test | AUROC | Δ vs v7 baseline | Comentário |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | v7 baseline (referência tese) | — | 5 | canônicas | — | **0,506** | 0,020 | ~0,80 | — | valor de `CLAUDE.md` (pré-otimização) |
| 2 | v7 baseline seed 42 | d02 | 1 | 42 | 0,5208 | 0,4862 | — | — | −0,020 (single) | reprodução em `diamante-02` |
| 3 | v7+ Tier A não-tunado (patience=5, lr_mult=5) | d02 | 1 | 42 | 0,5115 | 0,4697 | — | — | −0,016 | underfit por early-stop prematuro |
| 4 | v7+ Tier A tunado (patience=15, lr_mult=2) | d02 | 1 | 42 | 0,5652 | 0,5004 | — | — | +0,014 | correção de hiperparâmetros |
| 5 | v7+ Tier A + Tier B (pool=4) | d02 | 1 | 42 | 0,5200 | 0,4560 | — | — | −0,030 | head_proj Xavier viola identity-init |
| 6 | v7+ A+C seed 42 | d02 | 1 | 42 | 0,5880 | 0,5167 | — | — | +0,031 | sinal preliminar (single) |
| 7 | **v7+ A+C 5-seed canônico** | d02 | **5** | 42,123,456,789,1024 | 0,576 ± 0,036 | **0,5143 ± 0,0079** | **0,008** | — | **+0,008** | **canônico validado multi-seed** |
| 8 | v7+ A+C+D (SWA vanilla) | d02 | 1 | 42 | 0,5088 | 0,4964 | — | — | −0,010 | regime mismatch SWA |
| 9 | v7+ A+C seed 42 | d01 | 1 | 42 | 0,5153 | 0,5256 | — | — | +0,020 | ~+0,009 vs d02 (cuDNN ON) |
| 10 | v7-pro (A+C+E+F) seed 42 | d01 | 1 | 42 | 0,5870 | 0,5320 | — | — | +0,046 | sinal preliminar (single) |
| 11 | v7-pro (A+C+E+F) 5-seed | d01 | 5 | 42,123,456,789,1024 | 0,576 | 0,4961 ± 0,0245 | 0,025 | 0,805 ± 0,008 | −0,010 | regrediu vs v7+; Mixup tóxico |
| 12 | v7+E (A+C + Mixup) 3-seed | d01 | 3 | 42,123,456 | — | 0,4988 ± 0,0254 | 0,025 | 0,802 ± 0,020 | −0,007 | Mixup deletério isolado |
| 13 | **v7+F (A+C + label smooth) 3-seed** | d01 | **3** | 42,123,456 | — | **0,5260 ± 0,0274** | 0,027 | 0,808 ± 0,017 | **+0,020** | **melhor candidato 3-seed** |
| 14 | v7_ban_F (A+C+F + BAN Xavier) 3-seed | d01 | 3 | 42,123,456 | 0,599 ± 0,034 | 0,5028 ± 0,0462 | 0,046 | — | −0,003 | regrediu; W_ban Xavier viola identity-init |
| 15 | v7_asymF (A+C+F + asym adapter + pre-norm) 3-seed | d01 | 3 | 42,123,456 | — | 0,4611 ± 0,0279 | 0,028 | 0,7951 | −0,045 | regrediu; pre-norm + LoRA + asym todos juntos sem ablação |
| 16 | v7_plus_F_adapt (A+C+F + §6.5 fixes + THR=f1, SEL=f1) 3-seed | d01 | 3 | 42,123,456 | — | 0,4929 ± 0,0160 | 0,016 | 0,8105 | −0,013 | F1=0,7868; selection acoplado a F1 (matched) |
| 17 | v7_plus_F_adapt_v2 (A+C+F + §6.5 + THR=f1, SEL=mcc) 3-seed | d01 | 3 | 42,123,456 | — | **0,4590 ± 0,0517** | **0,052** | 0,7778 | **−0,047** | DESCARTADO; AUROC+σ pioraram → lição 16: matched objective |
| 18 | v7+F + λ=0.5 composite (matched mcc/mcc) 3-seed | **d02** | 3 | 42,123,456 | — | **0,4606 ± 0,0518** | **0,052** | 0,7988 | **−0,056** (cross-host adj.) | suspeita §6.5 prejudicial em treino curto → lição 17 |
| 19 | **v7+F + ADAPTER_LEGACY=1 (isolation A)** 3-seed | **d01** | 3 | 42,123,456 | — | **0,5266 ± 0,0103** | **0,010** | 0,8105 | **+0,000** vs v7+F histórico | **HIPÓTESE 17 CONFIRMADA** — reproduz canônico exato |
| 20 | v7+F + ADAPTER_LEGACY=0 (isolation B, §6.5 ON) 3-seed | d02 | 3 | 42,123,456 | — | 0,4652 ± 0,0563 | 0,056 | 0,7780 | −0,053 vs A (cross-host adj) | §6.5 culpado: ΔAUROC=−0.033, σ×5.6 |
| 21 | v7+F + Direção A só (lig 2L/12h, LEGACY default) 3-seed | d01 | 3 | 42,123,456 | — | 0,4996 ± 0,0258 | 0,026 | 0,8025 | −0,027 | params extras no lig sem LR matching → undertrained, σ_recall=0.098 |
| 22 | v7+F + Direção D só (lig_lr=5x, LEGACY default) 3-seed | d02 | 3 | 42,123,456 | — | 0,4867 ± 0,0222 | 0,022 | ~0,496 (adj) | −0,031 | LR alta sem capacidade extra → overshoot/oscila |
| 23 | **v7+F + Direção A+D combo (lig 2L/12h + lr=5x) 3-seed** | d03 | 3 | 42,123,456 | — | **0,5215 ± 0,0041** | **0,004** | **~0,530 (adj)** | **+0,004** | empate; claim σ=0.004 **RETRATADO** após grid d02 mostrar σ=0.037 mesma config |
| 24 | A+D grid sweep cell (5,2) 3-seed | d02 | 3 | 42,123,456 | — | 0,5144 ± 0,0369 | 0,037 | ~0,523 (adj) | −0,003 | grid mostra σ alto (não 0.004 d03) |
| 25 | A+D grid sweep cell (3,2) 3-seed | d02 | 3 | 42,123,456 | — | 0,4796 ± 0,0056 | 0,006 | ~0,489 (adj) | −0,038 | menor σ do grid mas pior MCC |
| 26 | A+D grid sweep cell (8,3) 3-seed | d02 | 3 | 42,123,456 | — | **0,4064 ± 0,1494** | **0,149** | ~0,415 (adj) | −0,112 | falha catastrófica — lr alto + capacidade alta = divergência |
| 27 | **v7_ban_res (BAN-residual α-gate) 3-seed** | ? | 3 | 42,123,456 | — | **0,5081 ± 0,0449** | 0,045 | ~0,517 (adj se cuDNN OFF) | −0,010 a −0,018 | **REGREDIU** — capacidade extra W_k sem LR matching (Lição 19 reaplica) |
| 28 | **v7_rope (2D RoPE per-modality) 3-seed** | d02 | 3 | 42,123,456 | — | **0,4971 ± 0,0220** | 0,022 | **~0,506 (adj)** | **−0,020** | **REGREDIU — Lição 22**: RoPE em cross-modal map é category error (offset i−j sem semântica entre prot/lig) |
| 24 | v7+F + LoRA-MLM offline (rank=8, top-2L, 10 ep MLM) 3-seed | d03 | 3 | 42,123,456 | — | 0,4933 ± 0,0321 | 0,032 | ~0,502 (adj) | −0,025 | **REFUTADO lição 20**: MLM não-alinhado a downstream + corpus 5276 SMILES = overfit; AUROC −0,010 confirma piora real do encoder |
| 29 | **v7_ban_res_lr (BAN-residual + BAN_LR_MULT=5) 2-seed** | d03 | 2 | 42,123 | — | **0,5108 ± 0,0459** | 0,046 | 0,8081 ± 0,0037 | **−0,016** | **REFUTA correção §6.12**: LR-boost dedicado em $W_k$ não recupera; gargalo é acoplamento multiplicativo $\alpha_k \cdot W_k L^\top$ (gate zero $\Rightarrow$ grad zero), não magnitude de update. F1=0,7912 ± 0,0159; elapsed=1312s |
| 30 | **v7+F LEGACY 5-seed (RE-EXECUÇÃO)** | d01? | **5** | 42,123,456,789,1024 | — | **0,4923 ± 0,0250** | 0,025 | 0,8027 ± 0,0066 | **−0,034** vs 3-seed | **LIÇÃO 24**: 3-seed 0,5266 era cherry-pick upper-tail; seeds 789+1024 médio ~0,441. v7+F 5-seed = empate estatístico c/ vanilla v7 baseline (z=0,55σ). Tier F efeito empírico nulo sob 5-seed. F1=0,7829, recall=0,8716, precision=0,7114, elapsed=3047s |

---

## Métricas adicionais — corpus `non_human`

Cada *run* multi-seed também produz acurácia, F1, precisão e revocação.
Os valores extraídos diretamente do `benchmark_comparison.json` são
listados abaixo para preservação completa.

### v7-pro (A+C+E+F) 5-seed em `diamante-01`

```yaml
config: configs/v7_pro.yaml
host: diamante-01
corpus: non_human
embedding: 8M
seeds: [42, 123, 456, 789, 1024]
elapsed_seconds: 2438.3
results:
  level4_cnn_mlp:
    accuracy:  0.744066 ± 0.008295
    mcc:       0.496109 ± 0.024545
    f1:        0.781642 ± 0.016314
    precision: 0.715573 ± 0.018318
    recall:    0.864752 ± 0.063508
    auc:       0.804667 ± 0.007621
```

### v7+E (A+C+E) 3-seed em `diamante-01`

```yaml
config: configs/v7_plus_E.yaml
host: diamante-01
corpus: non_human
embedding: 8M
seeds: [42, 123, 456]
elapsed_seconds: 1443.4
results:
  level4_cnn_mlp:
    accuracy:  0.746377 ± 0.012770
    mcc:       0.498771 ± 0.025381
    f1:        0.786368 ± 0.009488
    precision: 0.712356 ± 0.013309
    recall:    0.877716 ± 0.012321
    auc:       0.801592 ± 0.019664
```

### v7+F (A+C+F) 3-seed em `diamante-01`

```yaml
config: configs/v7_plus_F.yaml
host: diamante-01
corpus: non_human
embedding: 8M
seeds: [42, 123, 456]
elapsed_seconds: 1868.4
results:
  level4_cnn_mlp:
    accuracy:  0.759694 ± 0.013666
    mcc:       0.525959 ± 0.027351
    f1:        0.796545 ± 0.011040
    precision: 0.724680 ± 0.015864
    recall:    0.884714 ± 0.022922
    auc:       0.808077 ± 0.016974
```

### v7_plus_F_adapt (A+C+F + §6.5 fixes + threshold F1) 3-seed em `diamante-01`

```yaml
config: configs/v7_plus_F_adapt.yaml
host: diamante-01
corpus: non_human
embedding: 8M
seeds: [42, 123, 456]
elapsed_seconds: 1390.5
env:
  BENCHMARK_LEVEL4CNN_THRESHOLD_METRIC: f1
  BENCHMARK_LEVEL4CNN_SELECTION_LAMBDA_LOSS: 0.0
results:
  level4_cnn_mlp:
    accuracy:  0.739522 ± 0.007977
    mcc:       0.492850 ± 0.015989
    f1:        0.786800 ± 0.005853
    precision: 0.696615 ± 0.007428
    recall:    0.903867 ± 0.007968
    auc:       0.810485 ± 0.005741
per_seed:
  42:  test_mcc: 0.5106  test_f1: 0.7935  thr: 0.498
  123: test_mcc: 0.4796  test_f1: 0.7824  thr: 0.529
  456: test_mcc: 0.4877  test_f1: 0.7846  thr: ~0.50
notes: |
  EmbeddingAdapter §6.5 fixes (pre-norm + LoRA gates + zero-init
  self_attn) hardcoded since 58805ca/abc4167. Adapter SIMÉTRICO
  (sem assimetria do v7_asymF). Threshold = F1-óptimo val (matching
  DrugBAN/GraphBAN native criterion).

  Regressão MCC vs v7+F (-0.033) explicada por acoplamento
  THRESHOLD↔SELECTION na mesma env var: ao trocar p/ F1, seleção
  de checkpoint passou também a otimizar F1. Modelo escolhe
  epoch com recall alto / precision baixo (ótimo p/ F1, subótimo
  p/ MCC). Lição 15 documentada em licoes_aprendidas.md §6.7.

  Recall=0.904 vs precision=0.697 confirma diagnóstico (vs v7+F
  típico recall~0.85, precision~0.73).

  Próxima iteração: desacoplar via SELECTION_METRIC=mcc separado.
```

### v7+F + LoRA-MLM offline — refutação direção §6.3 (lição 20) — 3-seed em `diamante-03`

```yaml
config: configs/v7_plus_F.yaml + LoRA cache override
host: diamante-03
cudnn: OFF (default LEGACY=1)
pipeline:
  stage1_lora_ft:
    target: MoLFormer-XL-both-10pct top-2 layers Q/K/V/O
    rank: 8
    alpha: 16
    objective: MLM (mask_prob=0.15)
    train_smiles: 5276 unique
    epochs: 10
    batch_size: 64
    lr: 5e-4
    schedule: cosine + 6% warmup
    precision: bf16
    trainable_params: 98_304 / 46_879_488 (0.21%)
  stage2_recache:
    smiles_recached: 5276 (train+val+test unique)
    cache_dir: results/lora/molformer_cache_v1/non_human/
  stage3_benchmark:
    config: configs/v7_plus_F.yaml (LEGACY default)
    seeds: [42, 123, 456]
    elapsed_seconds: 1761.7
results:
  level4_cnn_mlp:
    accuracy:  0.741285 ± 0.016705
    mcc:       0.493264 ± 0.032107   # cross-host adj: ~0.502
    f1:        0.785989 ± 0.011449
    precision: 0.702070 ± 0.017913
    recall:    0.893186 ± 0.019308
    auc:       0.801062 ± 0.008345   # -0.010 vs base, RANKING piorou
diagnosis: |
  Refuta hipótese §6.3 (LoRA esperava +0.020 a +0.040). Quatro causas
  convergentes (lição 20):
  1. MoLFormer já saturado em MLM (1.1B SMILES pretraining).
  2. Objetivo MLM ≠ tarefa discriminativa downstream.
  3. AUROC -0.010 confirma encoder literalmente pior, não threshold.
  4. Corpus pequeno (5276) → LoRA delta absorve overfit.

  Cache LoRA-FT-ed produz embeddings worse que pretrained baseline
  → adapter+CNN downstream herda degradação.

next_step: |
  ABANDONAR abordagem LoRA-MLM-offline-then-cache.
  Re-implementar LoRA end-to-end com loss DT-Kinase (BCE+focal+contrast).
  Custo: ~5× tempo treino atual (carregar PLM dentro pipeline).
  Alternativas: LoRA + multi-task RDKit properties; LoRA + contrastive.
```

### Direções A, D, A+D — assimetria capacity vs LR (lição 19) — 3-seed paralelo

```yaml
A_solo:
  config: configs/v7_plus_F.yaml + env asym
  host: diamante-01
  cudnn: ON (LEGACY=1 default)
  env:
    BENCHMARK_LEVEL4CNN_ADAPTER_LAYERS_LIG: 2
    BENCHMARK_LEVEL4CNN_ADAPTER_ATTN_HEADS_LIG: 12
  results:
    accuracy:  0.747356 ± 0.012055
    mcc:       0.499644 ± 0.025786   # -0.027 vs base 0.5266
    f1:        0.775158 ± 0.029286
    precision: 0.735913 ± 0.031972
    recall:    0.826519 ± 0.097614   # σ EXTREMO 0.098
    auc:       0.802528 ± 0.007902
  diagnosis: |
    Capacidade extra no lig adapter (3× params) sem LR matching
    → params under-trained → σ_recall 0.098 = 4× baseline.

D_solo:
  config: configs/v7_plus_F.yaml + env LR asym
  host: diamante-02
  cudnn: OFF (LEGACY=1 default)
  env:
    BENCHMARK_LEVEL4CNN_ADAPTER_LR_MULT_PROT: 2.0
    BENCHMARK_LEVEL4CNN_ADAPTER_LR_MULT_LIG: 5.0
  results:
    accuracy:  0.744027 ± 0.010581
    mcc:       0.486687 ± 0.022158   # cross-host adj: ~0.496, -0.031 vs base
    f1:        0.771166 ± 0.016997
    precision: 0.734591 ± 0.010081
    recall:    0.812891 ± 0.045617
    auc:       0.809263 ± 0.010351
  diagnosis: |
    LR 2.5× sem capacidade extra → overshoot/oscila no espaço pequeno
    de params do lig adapter base → σ_recall 0.046 (1.8× baseline).

AD_combo:
  config: configs/v7_plus_F.yaml + ambos envs
  host: diamante-03
  cudnn: OFF (LEGACY=1 default)
  env:
    BENCHMARK_LEVEL4CNN_ADAPTER_LAYERS_LIG: 2
    BENCHMARK_LEVEL4CNN_ADAPTER_ATTN_HEADS_LIG: 12
    BENCHMARK_LEVEL4CNN_ADAPTER_LR_MULT_PROT: 2.0
    BENCHMARK_LEVEL4CNN_ADAPTER_LR_MULT_LIG: 5.0
  results:
    accuracy:  0.758715 ± 0.002224   # σ_acc 5× baseline
    mcc:       0.521457 ± 0.004070   # cross-host adj: ~0.530, +0.004 vs base
    f1:        0.793829 ± 0.002257   # σ_F1 2× baseline
    precision: 0.727475 ± 0.006263
    recall:    0.873665 ± 0.012566   # σ_recall 0.013 = 2× baseline
    auc:       0.795413 ± 0.002543
  diagnosis: |
    A+D restaura razão "magnitude update por param" próxima do baseline.
    Capacidade extra absorve LR maior → trajetória determinística.
    σ_MCC 2.5× MELHOR que baseline (0.004 vs 0.010).
    Δ MCC marginal mas redução de variância é o sinal forte.

lesson_19: |
  Capacidade arquitetural e magnitude de otimização sobre o mesmo
  módulo são MUTUAMENTE DEPENDENTES em treino curto (~30 epochs).
  Aplicar uma sem a outra desbalanceia gradiente-per-param e regride.
  Combinar restaura balanço, possivelmente com bonus de σ.

  Implicação: testar A e D individualmente é metodologicamente
  enganoso. Devem ser tratados como uma única intervenção atômica.

next_step: |
  Validar A+D em 5-seed (reproduzir σ=0.004) + grid sweep
  lr_mult_lig ∈ {3, 5, 8} × layers_lig ∈ {2, 3} para encontrar
  ponto ótimo da curva média-variância.
```

### Isolation §6.5 — runs A (LEGACY=1) e B (LEGACY=0) — 3-seed paralelo

```yaml
run_A:
  config: configs/v7_plus_F.yaml
  host: diamante-01
  cudnn: ON
  env: { BENCHMARK_LEVEL4CNN_ADAPTER_LEGACY: 1 }
  seeds: [42, 123, 456]
  elapsed_seconds: 1581.0
  results:
    accuracy:  0.760086 ± 0.006030
    mcc:       0.526604 ± 0.010304   # reproduz histórico v7+F=0.5260
    f1:        0.796503 ± 0.005120
    precision: 0.725728 ± 0.013009
    recall:    0.883241 ± 0.025446
    auc:       0.810529 ± 0.016751

run_B:
  config: configs/v7_plus_F.yaml
  host: diamante-02
  cudnn: OFF
  env: { (defaults — LEGACY=0, §6.5 ativo) }
  seeds: [42, 123, 456]
  elapsed_seconds: 1463.5
  results:
    accuracy:  0.728946 ± 0.023253
    mcc:       0.465198 ± 0.056343   # cross-host adj p/ d01: ~0.474
    f1:        0.768451 ± 0.034411
    precision: 0.702148 ± 0.005081
    recall:    0.852302 ± 0.086840
    auc:       0.778032 ± 0.022755

delta_attribution_to_§6.5:
  ΔMCC:    -0.053  (B_adjusted vs A)
  ΔAUROC:  -0.033  # modelo literalmente pior, não threshold
  σ_MCC:   ×5.6    (0.010 → 0.056)
  ΔF1:     -0.028
  Δprecision: -0.023
  Δrecall:    -0.031

decision: |
  Hipótese 17 confirmada. §6.5 (pre-norm + LoRA gates +
  zero-init self_attn) prejudica capacidade efetiva do adapter
  no regime de treino curto (~30-50 epochs com patience=15).
  §6.5 deve virar opt-in, default = LEGACY (lição 18).
  v7+F volta a ser canonicamente 0.5266 com código novo.
```

### v7_rope (2D RoPE per-modality) 3-seed em `diamante-02`

```yaml
config: configs/v7_rope.yaml
host: diamante-02
cudnn: OFF (DISABLE_CUDNN=1)
corpus: non_human
seeds: [42, 123, 456]
elapsed_seconds: 1047.2  # ~17 min, +5% sobre v7+F base
env:
  BENCHMARK_LEVEL4CNN_USE_ROPE: 1
arch_changes:
  formula: M_k = (rope(P W_p^k))_i · (rope(L W_l^k))_j / sqrt(d_h)
  trainable_params_added: 0  (RoPE é determinístico)
  identity_init_at_t0: NÃO (rotaciona desde t=0)
results:
  level4_cnn_mlp:
    accuracy:  0.749314 ± 0.010450
    mcc:       0.497130 ± 0.021970   # cross-host adj cuDNN OFF→ON: ~0.506
    f1:        0.778315 ± 0.011457
    precision: 0.734393 ± 0.006314
    recall:    0.827993 ± 0.021549
    auc:       0.809231 ± 0.008930   # AUROC quase IDÊNTICO a base (0.811)
diagnosis: |
  REGREDIU −0.020 MCC vs v7+F base. AUROC essencialmente preservada
  (Δ = −0.002) → ranking similar mas threshold/calibração quebraram.

  CATEGORY ERROR cross-modal: RoPE codifica offset relativo (i-j)
  via rotação. M_k[prot_pos, lig_pos] tem dois eixos de entidades
  moleculares distintas em escalas diferentes — i (resíduo proteico)
  e j (token SMILES BPE) NÃO são comparáveis. Termo (i-j) injeta
  estrutura posicional espúria que CNN tenta acomodar em prejuízo
  do sinal real.

  CNN sem RoPE funciona bem porque convolução local é coordenada-
  livre (translation invariant), não assume comparabilidade entre
  posições prot e lig.
lesson_22: |
  Positional encoding 1D intra-modal (RoPE) NÃO transfere para
  interaction maps cross-modais onde os dois eixos correspondem
  a entidades em escalas diferentes. Termo (i-j) assume
  comparabilidade semântica que não existe no nosso caso.

  Heurística operacional:
  - NÃO aplicar RoPE direto em ambos eixos cross-modais
  - Alternativas válidas (não testadas): positional encoding
    independente por eixo (sem dot product), bias B[i,j]
    aprendido separado de M_k, ou manter CNN local sem PE
  - Axial attention seria correto: q,k separados por axis
plateau_status: |
  14ª modificação incremental sobre v7+F testada, 14ª regressão.
  Lição 21 (plateau) ainda mais robustamente confirmada.
```

### v7_ban_res (BAN-residual α-gate identity-init) 3-seed

```yaml
config: configs/v7_ban_res.yaml
host: ? (cuDNN status indeterminado, elapsed 1511s sugere cuDNN OFF)
corpus: non_human
embedding: 8M
seeds: [42, 123, 456]
elapsed_seconds: 1511.0
env:
  BENCHMARK_LEVEL4CNN_BAN_RESIDUAL: 1
arch_changes:
  caminho_extra: M_k = M_k_dot + α_k · (P W_k L^T)
  alphas_init: zero (ParameterList of 16 scalars)
  W_k_init: Xavier uniform com escala (d_p · d_l)^-0.25
  trainable_extra: 3.93M (16 W_k matrices [320 × 768])
results:
  level4_cnn_mlp:
    accuracy:  0.752252 ± 0.019996
    mcc:       0.508133 ± 0.044878   # cross-host adj cuDNN OFF→ON: ~0.517
    f1:        0.786284 ± 0.023056
    precision: 0.725595 ± 0.012760
    recall:    0.859300 ± 0.051996   # σ alto = sub-treinado
    auc:       0.803388 ± 0.009551
diagnosis: |
  REGREDIU vs v7+F base (0.5266). Padrão Lição 19 reaplica:
  capacidade extra (3.93M params W_k) sem LR matching dedicado →
  params sub-treinados em ~30 epochs.

  Identity-init em t=0 confirmado (sanity test diff 0.0000e+00).
  Anti-cascade verificado (∂L/∂α ≠ 0 porque W_k Xavier não-zero).
  Mecanismo correto, mas treino curto não permite ativação útil.

  Sintoma diagnóstico: σ_recall = 0.052 (vs 0.025 base) → seeds
  discordam = sinal de undertraining.

next_step: |
  Implementar BENCHMARK_LEVEL4CNN_BAN_LR_MULT (3-5x) para que
  W_k receba updates compatíveis com sua quantidade de params.
  Aplicação direta princípio Lição 19 atomicamente: capacidade
  extra + LR boost juntos.
```

### v7_ban_res_lr (BAN-residual + BAN_LR_MULT=5) 2-seed em `diamante-03`

```yaml
config: configs/v7_ban_res_lr.yaml
host: diamante-03  (cuDNN ON; results/benchmark_banres_lr_non_human_8M)
corpus: non_human
embedding: 8M
seeds: [42, 123]
elapsed_seconds: 1312.3
env:
  BENCHMARK_LEVEL4CNN_BAN_RESIDUAL: 1
  BENCHMARK_LEVEL4CNN_BAN_LR_MULT:  5.0
arch_changes:
  param_group_dedicado: W_k + α_k receberam lr * 5.0
  baseline_share: caminho dot-product + adapter inalterados
results:
  level4_cnn_mlp:
    accuracy:  0.75235  ± 0.024512
    mcc:       0.51081  ± 0.045921   # vs BAN-res sem boost: +0.003 (ruído)
    f1:        0.791156 ± 0.015928
    precision: 0.718074 ± 0.027784
    recall:    0.881215 ± 0.002344
    auc:       0.808115 ± 0.003707
diagnosis: |
  REFUTA a hipótese corretiva proposta em §6.12. LR-boost
  dedicado (×5) NÃO recuperou a regressão de BAN-residual.
  Δ MCC vs BAN-res vanilla = +0.003 (indistinguível de ruído σ).
  Δ MCC vs v7+F baseline   = -0.016 (regressão preservada).
  σ_test continua inflado ~4.5× vs baseline.

  Diagnóstico mecanístico atualizado: gargalo NÃO é magnitude
  de update. É acoplamento multiplicativo do gate:
    ∂L/∂W_k = α_k · (P L^T) · ∂L/∂out
  Com α_k(t=0)=0, gradiente de W_k é estritamente zero na
  primeira iteração. Mesmo com LR×5 em W_k, sem sinal não
  há aprendizado. O sinal só emerge quando α_k se afasta de
  zero, mas α_k depende de W_k L^T já útil — circular.

  LR-boost local resolve sub-treinamento por capacidade
  (Lição 19 original com adapter), NÃO resolve sub-treinamento
  por acoplamento multiplicativo (BAN-residual; também §6.5).

next_step: |
  BAN-residual está empiricamente refutado em todas as
  parametrizações testadas. Para reabrir: warm-up assimétrico
  com α_k(t=0) ≠ 0 pequeno (e.g. 0.01) ou schedule dedicado
  desacoplando crescimento de α_k do crescimento de W_k.
  Não é prioridade no plateau atual (Lição 21).
```

### v7+F LEGACY 5-seed (RE-EXECUÇÃO — Lição 24)

```yaml
config: configs/v7_plus_F.yaml
host: diamante-?? (provável d01 — AUROC 0.803 alinhado com d01 cuDNN ON)
corpus: non_human
embedding: 8M
seeds: [42, 123, 456, 789, 1024]
elapsed_seconds: 3046.9
env:
  BENCHMARK_LEVEL4CNN_ADAPTER_LEGACY: 1  # default desde commit de2ef0e
results:
  level4_cnn_mlp:
    accuracy:  0.743008 ± 0.013036
    mcc:       0.492251 ± 0.024974
    f1:        0.782862 ± 0.010172
    precision: 0.711402 ± 0.018410
    recall:    0.871602 ± 0.032950
    auc:       0.802718 ± 0.006567
comparison:
  vs_3seed_v7+F_LEGACY: -0.034 MCC (3-seed cherry-pick upper-tail)
  vs_vanilla_v7_5seed:  -0.014 MCC, z=0.55σ → empate estatístico
  vs_AUROC_v7+F_LEGACY: -0.005 (ranking quality preservado)
seed_decomposition:
  seeds_42_123_456_avg:  ~0.5266 (3-seed antigo)
  seeds_789_1024_avg:    ~0.441  (back-out: (5×0.4923 - 3×0.5266)/2)
diagnosis: |
  3-seed (42,123,456) era amostragem upper-tail. 5-seed canônico
  expõe true mean ≈ 0.49, indistinguível de vanilla v7 (0.506 ± 0.006).
  Tier F (label_smooth=0.05) NÃO sobrevive validação 5-seed.
  Aplicação direta da Lição 3 ao próprio caso v7+F.
canonical_decision: |
  configs/v7.yaml (vanilla v7) é o ckpt operacional canônico da tese
  (já adotado em Cap 5). v7+F é tratado como tentativa promissora em
  3-seed que regrediu para empate em 5-seed (Lição 24).
```

### A+D Grid Sweep (6 cells, lr_mult_lig × layers_lig) 3-seed em `diamante-02`

```yaml
config: configs/v7_plus_F.yaml + per-cell envs
host: diamante-02
cudnn: OFF (DISABLE_CUDNN=1)
corpus: non_human
fixed_envs:
  BENCHMARK_LEVEL4CNN_ADAPTER_ATTN_HEADS_LIG: 12
  BENCHMARK_LEVEL4CNN_ADAPTER_LR_MULT_PROT: 2.0
seeds: [42, 123, 456]
results_per_cell:
  # ordenado por MCC desc
  lr5_L2: { mcc: 0.5144 ± 0.0369, auc: 0.7906, f1: 0.7914 }
  lr5_L3: { mcc: 0.5068 ± 0.0310, auc: 0.7983, f1: 0.7869 }
  lr3_L3: { mcc: 0.5019 ± 0.0215, auc: 0.7987, f1: 0.7792 }
  lr8_L2: { mcc: 0.5014 ± 0.0125, auc: 0.8057, f1: 0.7862 }
  lr3_L2: { mcc: 0.4796 ± 0.0056, auc: 0.7884, f1: 0.7706 }
  lr8_L3: { mcc: 0.4064 ± 0.1494, auc: 0.7418, f1: 0.6514 }   # divergiu
key_findings:
  - Nenhum cell bate v7+F base (0.5266) cross-host adjusted.
  - Best cell (5,2) raw 0.5144 → adj d01 0.5234 → empate, não ganho.
  - σ ALTO em quase todas células (>0.02) — variabilidade significativa.
  - lr_mult_lig=8 + layers_lig=3 = falha catastrófica (σ=0.149).
  - Pareto front difuso (5 de 6 cells na frente) — sem ponto dominante.
retraction: |
  Resultado d03 anterior (lr=5, L=2 single replication: 0.5215 ± 0.0041)
  NÃO REPLICA em d02. Mesma config produziu σ=0.037, nove vezes maior.
  Conclusão: σ=0.004 d03 foi acidente fortuito, não propriedade
  reproducível. Claim variância da Lição 19 RETRATADO.
```

### v7+F + λ=0.5 composite criterion (matched mcc/mcc) 3-seed em `diamante-02`

```yaml
config: configs/v7_plus_F.yaml
host: diamante-02
corpus: non_human
embedding: 8M
seeds: [42, 123, 456]
elapsed_seconds: 1352.5
env:
  BENCHMARK_LEVEL4CNN_SELECTION_LAMBDA_LOSS: 0.5
  BENCHMARK_LEVEL4CNN_DISABLE_CUDNN: 1
  (THRESHOLD_METRIC, SELECTION_METRIC: ambos default = mcc)
results:
  level4_cnn_mlp:
    accuracy:  0.724638 ± 0.024595
    mcc:       0.460639 ± 0.051773
    f1:        0.774815 ± 0.018915
    precision: 0.685694 ± 0.019463
    recall:    0.890608 ± 0.017783
    auc:       0.798839 ± 0.019926
notes: |
  Configuração metodologicamente "limpa": critérios matched (mcc/mcc),
  apenas λ=0.5 composite isolado (lição 14). Esperava-se neutro ou
  marginalmente positivo. Regrediu -0.056 MCC vs v7+F histórico
  (ajustando cross-host drift +0.009).

  Padrão notável: TODOS os 4 experimentos recentes sobre v7+F
  regrediram (asymF, adapt, adapt_v2, λ=0.5). Interseção comum:
  todos rodam código atual com §6.5 fixes hardcoded. v7+F histórico
  (0.5260) foi pre-§6.5.

  Hipótese (lição 17): §6.5 (pre-norm + LoRA gates + zero-init
  self_attn) torna adapter NO-OP demais em t=0; em treino curto
  (~30-50 epochs com early stop), gates não acordam → capacidade
  efetiva subutilizada.

  Próximo passo crítico: isolar §6.5 via flag legacy, comparar
  v7+F com adapter antigo vs adapter atual no mesmo host.
```

### v7_plus_F_adapt_v2 (A+C+F + §6.5 + THR=f1, SEL=mcc) 3-seed em `diamante-01`

```yaml
config: configs/v7_plus_F_adapt_v2.yaml
host: diamante-01
corpus: non_human
embedding: 8M
seeds: [42, 123, 456]
elapsed_seconds: 1346.1
env:
  BENCHMARK_LEVEL4CNN_THRESHOLD_METRIC: f1
  BENCHMARK_LEVEL4CNN_SELECTION_METRIC: mcc
  BENCHMARK_LEVEL4CNN_SELECTION_LAMBDA_LOSS: 0.0
results:
  level4_cnn_mlp:
    accuracy:  0.724442 ± 0.019654
    mcc:       0.458957 ± 0.051650
    f1:        0.766686 ± 0.033292
    precision: 0.695760 ± 0.009765
    recall:    0.858563 ± 0.093021
    auc:       0.777797 ± 0.022950
notes: |
  REFUTAÇÃO da hipótese §6.7. AUROC caiu 0.811 → 0.778 (modelo
  literalmente pior, não só threshold mismatch). σ MCC triplicou
  (0.016 → 0.052). Lição 16 documentada em §6.8: critério de
  selection deve casar com critério de threshold.

  Mismatch: epoch escolhido por melhor val_MCC at MCC-opt threshold,
  mas test eval aplica F1-opt threshold ao mesmo modelo →
  modelo opera em ponto subótimo na superfície de decisão.

  Variância alta (recall σ=0.093, precision σ=0.010) confirma
  instabilidade do critério val_MCC para selection vs val_F1.

  CONFIG DESCARTADA. Manter v7+F (matched MCC) e v7+F_adapt
  (matched F1) como canônicas.
```

### v7_ban_F (A+C+F + BAN) 3-seed em `diamante-01`

```yaml
config: configs/v7_ban_F.yaml
host: diamante-01
corpus: non_human
embedding: 8M
seeds: [42, 123, 456]
elapsed_seconds: 2678 (≈ 44m38s)
results:
  level4_cnn_mlp:
    train_mcc: 0.5992 ± 0.0339
    test_mcc:  0.5028 ± 0.0462
    notes: |
      W_ban inicializado por nn.init.xavier_uniform_; viola
      principio identity-init do v7. σ inflada vs v7+F (+70%).
      Configuração descartada em favor de BAN-residual com
      α-gate identity-init (pendente implementação).
```

### v7-pro single seed em `diamante-01` (semente 42)

```yaml
config: configs/v7_pro.yaml
host: diamante-01
corpus: non_human
embedding: 8M
seeds: [42]
elapsed_seconds: 289 (≈ 4m49s)
results:
  level4_cnn_mlp:
    val_mcc:  0.5870
    test_mcc: 0.5320
    notes: |
      Single-seed isolado superestimou em +0.036 vs média 5-seed
      subsequente (0.4961). Confirmação empírica da terceira lição
      metodológica (licoes_aprendidas.md §7).
```

---

## Cross-dataset matrix (off-diagonal médio, 5 sementes)

Resultados completos da matriz 3×3 com baselines DrugBAN, GraphBAN,
ConPLex contra DT-Kinase v7. Não otimizada — usa configurações nativas
de cada modelo. Valores em `results/cross_matrix/summary/cross_matrix.json`
de `diamante-02`.

| Modelo | Off-diag MCC médio | Posição |
|---|---:|:-:|
| DrugBAN | 0,348 | 1 |
| GraphBAN | 0,342 | 2 |
| **DT-Kinase v7** | **0,298** | **3** |
| ConPLex | 0,209 | 4 |

Diagonais não foram agregadas (parâmetros `--diagonal-*` não passados
ao `aggregate.py`). Off-diagonais por par disponíveis em
`scripts/thesis_followups/cross_dataset_matrix/analyze.py` (output
detalhado em `slides/cross_matrix/cross_matrix.pdf`).

---

## Observação metodológica sobre comparabilidade entre hosts

A configuração v7+ A+C — exatamente os mesmos *seeds*, mesma config,
mesmo protocolo — produziu MCC teste de $0{,}5167$ em `diamante-02`
(cuDNN OFF) e $0{,}5256$ em `diamante-01` (cuDNN ON). A diferença
$+0{,}009$ MCC é da mesma ordem do desvio-padrão entre sementes.
**Comparações quantitativas só são válidas dentro do mesmo host**.
Para a tese, fixar `diamante-01` como host de validação multi-semente
final.

---

## Atualização

Este documento deve ser atualizado a cada execução multi-semente
nova. Procedimento:

1. Após `bash scripts/v8/run_v7_yaml.sh` terminar, ler
   `results/benchmark_<variant>_<corpus>_8M/test/benchmark_comparison.json`
2. Adicionar nova linha à tabela "Resultados consolidados — corpus
   `<corpus>`"
3. Adicionar bloco YAML completo à seção "Métricas adicionais"
4. Commit com mensagem `experiments_log: <variant> <corpus> <n>-seed result`
5. Push

Configurações *single-seed* podem ser registradas inline no campo
"Comentário" da tabela, sem bloco YAML separado.

---

## Validação reprodutível futura

Para reproduzir qualquer linha desta tabela em um host diferente:

1. `git checkout <commit_hash>` (campo "commit" da linha — pendente
   adicionar)
2. `cd <repo_root>`
3. Reproduzir condições do host:
   - cuDNN: `BENCHMARK_LEVEL4CNN_DISABLE_CUDNN=0` em `diamante-01`,
     `=1` em `diamante-02` (ou outros com cuDNN 9.x quebrado)
4. Executar:
   ```bash
   SEEDS="<seeds_list>" V7_CONFIG=configs/<config>.yaml CORPUS=<corpus> \
     bash scripts/v8/run_v7_yaml.sh
   ```
5. Diferença esperada vs valores deste log: dentro de $\pm 0{,}01$ MCC
   no mesmo host (variabilidade numérica + tempo de máquina); até
   $\pm 0{,}009$ MCC entre hosts (efeito cuDNN).
