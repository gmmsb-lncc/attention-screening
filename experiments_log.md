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
| 15 | v7_asymF (A+C+F + asym adapter + pre-norm) 3-seed | d01 | 3 | 42,123,456 | em execução | em execução | — | — | — | inclui pre-norm + LoRA gates + composite criterion |

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
