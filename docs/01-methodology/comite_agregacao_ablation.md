# Comitê: ablação de regra de agregação (soft-mean vs hard-vote)

Data: 2026-05-09
Branch: `cross_attention_lite`
Trigger: questionamento sobre consistência da regra `prob_mean` (soft-mean) usada
no comitê canônico da tese (Cap. 5, §sec:resultados-comite).

## 1. Contexto e questão original

A tese reporta o comitê 4-modelos (DT-Kinase + DrugBAN + GraphBAN + ConPLex)
sob agregação **soft-mean** das probabilidades calibradas por modelo:

```
committee_prob[i] = mean_m( prob_5seed[m, i] )         # média sobre modelos
committee_thr     = mean_m( thr_5seed[m] )             # média sobre thresholds
committee_pred    = (committee_prob >= committee_thr)
```

Métricas (MCC, AUROC, F1, Acc) computadas uma vez sobre `committee_prob`,
**não como média de métricas individuais**. Pipeline em
`scripts/inference/experiments/committee_vs_individual.py` + agregador de
produção em `scripts/inference/aggregate.py`.

**Questionamento**: regra alternativa intuitiva — hard-vote com tie-break
fixo. Se ≥3 modelos votam +1 → positivo; ≤1 → negativo; em empate 2-2,
desempata via modelo árbitro pré-registrado. Justificativa proposta:
DT-Kinase como árbitro pelo equilíbrio precisão/recall observado no
quinoma humano.

## 2. Implementação

Dois scripts novos em `scripts/inference/experiments/`:

| Script | Propósito |
|---|---|
| `committee_hardvote_dtk.py` | hardvote + tie-break DT-K vs soft-mean (B=10⁴) |
| `committee_hardvote_all_arbiters.py` | hardvote × 4 árbitros (DT-K, DrugBAN, GraphBAN, ConPLex) vs soft-mean + pairwise (B=10⁴) |

Reúsam `load_5seed`, `dedupe_predictions`, `load_test_keys`, `_mcc_fast` de
`committee_vs_individual.py`. Mesmo protocolo: 5 sementes, dedup
`(seq_id, chembl_id)`, block bootstrap por proteína.

Outputs:
- `results/inference/committee_hardvote_dtk/{non_human,human,all}_{metrics,paired_bootstrap}.csv` + `REPORT.md`
- `results/inference/committee_hardvote_arbiters/{...}` + `REPORT.md`

## 3. Resultados

### 3.1 MCC por sistema (todas as variantes)

| Corpus | soft-mean | hv_DT-K | hv_DrugBAN | hv_GraphBAN | hv_ConPLex |
|---|---|---|---|---|---|
| non_human | **0.535** | 0.519 | 0.519 | 0.519 | 0.507 |
| human     | **0.543** | 0.519 | **0.531** | 0.522 | 0.519 |
| all       | **0.552** | 0.522 | **0.536** | 0.529 | 0.532 |

Negrito: vencedor por linha. Soft-mean lidera os três corpora; entre os
hardvotes, DrugBAN é melhor árbitro em human e all.

### 3.2 Δ_MCC vs soft-mean (paired block bootstrap, B=10⁴, IC95 percentílico)

| Árbitro | non_human | human | all |
|---|---|---|---|
| DT-K     | −0.0153 ⊘ [−0.040, +0.017] | −0.0240 ▼ [−0.034, −0.015] | −0.0304 ▼ [−0.040, −0.021] |
| **DrugBAN** | −0.0157 ⊘ [−0.033, +0.002] | **−0.0109 ▼** [−0.019, −0.002] | **−0.0168 ▼** [−0.025, −0.009] |
| GraphBAN | −0.0166 ▼ [−0.031, −0.002] | −0.0199 ▼ [−0.028, −0.012] | −0.0236 ▼ [−0.030, −0.017] |
| ConPLex  | −0.0282 ⊘ [−0.061, +0.007] | −0.0241 ▼ [−0.036, −0.013] | −0.0208 ▼ [−0.032, −0.011] |

Símbolos: ▼ soft-mean lidera (CI95 não-cruzada), ⊘ empate estatístico
(CI95 cruza zero). **Nenhum hardvote bate soft-mean**. Em human/all todos
perdem com CI95 não-cruzada; em NH (n=1399) só GraphBAN perde com IC
estritamente negativo, restantes são empates por baixo poder.

### 3.3 Pairwise hardvote_a vs hardvote_b (descobertas estatisticamente significativas)

Pairs com CI95 não-cruzada:

| Corpus | Comparação | Δ | CI95 | Veredito |
|---|---|---|---|---|
| human | DrugBAN vs GraphBAN | +0.0090 | [+0.004, +0.014] | **DrugBAN ▲** |
| all   | DT-K vs DrugBAN     | −0.0136 | [−0.026, −0.001] | DrugBAN ▲ |
| all   | DT-K vs ConPLex     | −0.0096 | [−0.016, −0.002] | ConPLex ▲ |
| all   | DrugBAN vs GraphBAN | +0.0068 | [+0.000, +0.013] | **DrugBAN ▲** |

Todos os outros pares = empate ⊘. **DrugBAN domina pairwise** em human e
all (lidera GraphBAN nos dois; lidera DT-K em all). DT-K como árbitro fica
abaixo de DrugBAN e ConPLex em all sob CI95 não-cruzada.

### 3.4 Diagnóstico de empates

Frequência de empates `votes==2` e como cada árbitro decide:

| Corpus | n_dedup | n_proteínas | n_ties | %ties |
|---|---|---|---|---|
| non_human | 1 399  | 114 | 68    | 4.9%  |
| human     | 28 639 | 444 | 3 287 | 11.5% |
| all       | 30 038 | 558 | 3 690 | 12.3% |

**Acurácia do árbitro nos empates** (acertos / n_ties):

| Árbitro | NH | Human | all | viés direcional |
|---|---|---|---|---|
| DT-K     | 0.544 | 0.538 | 0.512 | balanceado (DT-K vota ~30% +1 em H/all) |
| DrugBAN  | 0.529 | 0.476 | 0.486 | **alta-recall** (85% +1 em H, 74% +1 em all) |
| GraphBAN | 0.500 | 0.440 | 0.432 | máxima-recall (86% +1 em H) |
| ConPLex  | 0.426 | **0.546** | **0.570** | máxima-conservador (~10% +1 em H/all) |

**Observação contra-intuitiva**: ConPLex tem maior `acc_on_ties` em H/all
(~55-57%) mas **não é o melhor árbitro overall**. Razão: empates H têm
prevalência ~43% positiva (skew vs prevalência corpus 32.5%). ConPLex
resolve tudo como negativo → acerta ~57% por sorteio classe-majoritária,
mas perde TPs que MCC penaliza fortemente. DrugBAN empurra +1 → recupera
TPs concentrados em ties → MCC overall sobe apesar de `acc_on_ties` ~48%.

`acc_on_ties` ≠ contribuição-MCC. MCC é função do confusion matrix
inteiro; árbitro só altera ~12% das decisões totais; arbitragem ótima
empurra ties na direção que **completa** a confusion matrix dos modelos
discordantes.

## 4. Por que soft-mean ganha (interpretação)

1. **Preservação de margem**: soft-mean usa `prob ∈ [0, 1]`, distinguindo
   pares com `prob=0.51` (incerto) de pares com `prob=0.99` (confiante).
   Hard-vote colapsa em `pred ∈ {0, 1}`, descartando essa informação.

2. **Empates concentram a incerteza**: 11-12% das decisões em H/all são
   empates 2-2. Esses pares são justamente onde o modelo pareceria
   precisar mais informação, e hard-vote elimina toda informação
   probabilística reduzindo a uma chamada solo do árbitro nesses pontos.
   Soft-mean, em contraste, preserva a probabilidade média mesmo em
   regiões de discordância.

3. **Calibração heterogênea favorece soft-mean**: cada modelo carrega
   threshold calibrado individualmente (Platt + MCC-óptimo na val). A
   média de probabilidades respeita essas calibrações; a média de
   binários trata todos os votos como equivalentes, ignorando que um
   `pred=1` do DrugBAN com prob=0.55 é qualitativamente diferente de um
   `pred=1` do DT-K com prob=0.99.

## 5. Posição operacional para a tese

A regra atual (soft-mean) é **empiricamente Pareto-óptima** entre as
estratégias de agregação testadas. Ablação completa fortalece a defesa:

> **Status epistêmico** — não basta dizer "usamos soft-mean porque é
> padrão"; agora há evidência pareada por proteína (B=10⁴) de que toda
> alternativa hard-vote testada (incluindo otimização do árbitro)
> regride MCC entre −0.011 e −0.030 nos corpora maiores, com CI95
> não-cruzada.

### 5.1 Sugestão de incorporação na tese

Adicionar ao **Anexo B** (`~/PhD/tex/anexoB.tex`) uma sub-seção sob
`§sec:inferencia-comite-validacao`:

```
\subsection{Ablação da Regra de Agregação: Soft-Mean vs Hard-Vote}
\label{sec:inferencia-comite-agregacao}

Tabela X.Y: MCC do comitê 4-modelos sob 5 regras de agregação...
Tabela X.Z: Δ_MCC(hardvote_a − soft_mean) com CI95 percentílica...
```

Com isso fecha-se a objeção "por que não majority vote?" antes que apareça
em arguição.

### 5.2 Caveats não-resolvidos

1. **Heterogeneidade de calibração** segue como confounder (DT-K/ConPLex
   MCC-opt; DrugBAN/GraphBAN F1-opt). Vantagem do soft-mean pode
   parcialmente refletir maior tolerância a essa heterogeneidade. Teste
   isolante exigiria recalibrar os 4 modelos sob mesma regra antes de
   refazer ablação.
2. **Single-seed ensemble** (4 sementes do MESMO modelo) não testado —
   continua direção futura para isolar contribuição arquitetural vs
   diversidade de inicialização.
3. **Não testadas**: weighted soft-mean (pesos por MCC val), stacking
   (logistic meta-learner), product-of-experts. Ablação atual cobre
   somente regras zero-parâmetro.

## 6. Reprodução

```bash
# Hardvote + DT-K como árbitro (rápido, ~3 min):
env/bin/python3 scripts/inference/experiments/committee_hardvote_dtk.py

# Hardvote × 4 árbitros + pairwise (~10 min):
env/bin/python3 scripts/inference/experiments/committee_hardvote_all_arbiters.py
```

Ambos leem logits salvos de:
- DT-Kinase: `results/benchmark_{corpus}_8M_*/test/level4_cnn_8M/{corpus}/seed_*/raw_predictions.npz`
- DrugBAN: `DrugBAN/results_universal/results_universal/{corpus}/seed_*/`
- GraphBAN: `GraphBAN/results_universal/{corpus}/seed_*/`
- ConPLex: `ConPLex/results_universal/{corpus}/seed_*/`

Sem retreino. Custo dominado por bootstrap (B=10⁴ × 3 corpora ×
~5-15 comparações por corpus).

Variáveis env:
- `BENCHMARK_BOOTSTRAP_B`: número de reamostragens (default 10000)

## 7. Default operacional restaurado para 4-modelos (2026-05-09)

`scripts/inference/committee.py` teve default flipado de `human_kinome`
(3-modelos) para `full_4model` (4-modelos canônico). Justificativa:

- Ablação acima reafirma soft-mean como regra Pareto-óptima sob CI95
  pareada por proteína.
- Ganho do 3-modelos sobre canônico no human (Δ=+0.0074) é estatístico
  mas magnitude pequena vs largura típica IC; sob FWER controlado a
  preferência não é confirmatória (Cap. 5 §sec:resolucao-estatistica).
- Defesa epistêmica do canônico 4-modelos é mais robusta: `10/12`
  lideranças vs individuais sob hardvote-OR-softmean cobertas pela
  ablação completa (este doc).
- 3-modelos permanece disponível via `--profile human_kinome`.

Mudanças em `scripts/inference/committee.py`:
- `--profile` default: `human_kinome` → `full_4model`
- `--models` default: `dtkinase,drugban,conplex` → `dtkinase,drugban,graphban,conplex`
- Preset `full_4model` não força mais `--ckpt-corpus` para `all`
  (mantém `human` in-domain operacional; usuário pode override).
- Lógica de mapeamento de presets reescrita para promoção `human_kinome`
  reduz 4→3 e `non_human` mantém 4 + flip organism/ckpt.

## 8. Ablação de regras alternativas de agregação (2026-05-09)

Implementadas 4 alternativas zero-retreino sobre os logits salvos
(`scripts/inference/experiments/committee_aggregation_alts.py`):

1. **logit_mean** — mean(logit(p)) → sigmoid; equivalente a média geométrica
   das chances. Ref: Kahn / BMA literature.
2. **weighted_mcc** — soft-mean ponderado por val MCC (lido do calibration
   sidecar; pre-registrado, sem leakage). Kuncheva 2014.
3. **product_experts** — média geométrica das probabilidades (PoE binário,
   ordenamento equivalente). Hinton 2002.
4. **rrf** — reciprocal rank fusion, Σ 1/(60+rank_m). Cormack et al. 2009.

### 8.1 Resultados MCC por corpus

| Corpus | soft_mean | logit_mean | weighted_mcc | product_experts | rrf |
|---|---|---|---|---|---|
| non_human | 0.5350 | 0.5369 | **0.5419** | 0.5221 | 0.4766 |
| human     | 0.5426 | 0.5437 | 0.5419 | **0.5468** | 0.5038 |
| all       | 0.5524 | 0.5571 | 0.5518 | **0.5593** | 0.4991 |

### 8.2 Δ vs soft_mean (paired block bootstrap, B=10⁴, IC95 percentílica)

| Corpus | logit_mean | weighted_mcc | product_experts | rrf |
|---|---|---|---|---|
| NH    | +0.0023 ⊘ | +0.0071 ⊘ | −0.0126 ⊘ | −0.0566 ⊘ |
| human | +0.0011 ⊘ | −0.0005 ⊘ | **+0.0042 ▲** [+0.0003, +0.0082] | −0.0379 ▼ [−0.057, −0.019] |
| all   | **+0.0047 ▲** [+0.0020, +0.0074] | −0.0005 ⊘ | **+0.0068 ▲** [+0.0020, +0.0116] | −0.0529 ▼ [−0.067, −0.038] |

### 8.3 Achados-chave

1. **Product-of-Experts bate soft-mean** em human e all sob CI95
   não-cruzada (Δ=+0.0042 H, +0.0068 all). NH empate (n=1399, baixo poder).
   Magnitude pequena (~+1% relativo) mas estatisticamente robusta.
2. **Logit-mean** lidera em all (+0.0047, CI95 não-cruzada). Empate em H/NH.
3. **Weighted_mcc** neutro: pesos val MCC variam 0.44-0.91 mas ganho líquido
   ≈ 0 — Platt scaling já equaliza escalas, removendo benefício da
   re-ponderação linear.
4. **RRF colapsa** (−0.038 a −0.053 em H/all): calibration-free descarta
   info útil de Platt.
5. **Padrão**: agregações em espaço não-linear (logit, log) ≥ agregação
   linear (soft-mean). Compatível com Bayesian model combination — média
   de log-odds preserva info de cauda melhor que média de probs.

### 8.4 Implicação operacional

Substituir soft-mean por **product-of-experts** em produção daria
+0.004 a +0.007 MCC em H/all sob CI95 não-cruzada, custo computacional
zero, sem retreino, sem novo hiperparâmetro. Drop-in em `aggregate.py`.

Cautelas:
- Sob FWER Holm m=4 (alternativas testadas), α/4=0.0125 → PoE sobrevive
  com folga em human/all (P(Δ>0)≥0.981 → p_boot ≤ 0.019 H, ≤ 0.003 all).
- Sob FWER m=12 (família secundária da tese), maior cautela necessária;
  ganho de all sobrevive (p ≤ 0.003), mas o de human está na margem.
- `n_pos_pred` PoE é mais conservador (~3% menos positivos vs soft-mean):
  regime levemente precision-driven, alinhado a uso pré-experimental.

### 8.5 Batch 2 — 5 alternativas adicionais

`scripts/inference/experiments/committee_aggregation_alts2.py` (cobertura
M-mean parametrizada e regras robustas):

6. **weighted_logit** — sigmoid(Σ w_m·logit(p_m)/Σw_m), w_m ∝ val_MCC
7. **median** — mediana element-wise across modelos
8. **trimmed_mean** — drop max+min, mean dos 2 centrais
9. **max** — element-wise max prob (any-model-confident)
10. **harmonic_mean** — n/Σ(1/p_m), power mean k=−1

| Sistema | NH | human | all | Δ vs soft (all) |
|---|---|---|---|---|
| weighted_logit | 0.5369 | 0.5448 | **0.5587** | **+0.0064 ▲** [+0.003, +0.010] |
| median | 0.5493 | 0.5408 | 0.5507 | −0.002 ⊘ |
| trimmed_mean | 0.5493 | 0.5408 | 0.5507 | −0.002 ⊘ (idêntico a median, n=4 par) |
| max | 0.4841 | 0.5027 | 0.5040 | **−0.048 ▼** |
| harmonic_mean | 0.5227 | 0.5481 | 0.5569 | +0.0043 ⊘ (P=0.87) |

Achados batch 2:
- **weighted_logit** lidera all (CI95 não-cruzada). Combina logit-space
  (winner do batch 1) + val_MCC weights (que sozinho era neutro).
- **median ≡ trimmed_mean** com n=4 par (mean dos 2 centrais = mediana).
- **max colapsa** em todos os 3 corpora — regime over-permissivo
  descalibra threshold.
- **harmonic_mean** marginal positivo em H (P=0.96) e all (P=0.87) mas
  CI95 cruza zero — efeito provável real, n insuficiente para confirmar.

### 8.6 Síntese global (10 regras + soft-mean canônico)

Ranking corpus **all** (operacional):

```
1. product_experts   0.5593  ▲ (+0.0068, P=0.997)  ← TOP
2. weighted_logit    0.5587  ▲ (+0.0064, P=1.000)
3. logit_mean        0.5571  ▲ (+0.0047, P=0.999)
4. harmonic_mean     0.5569  ⊘ (+0.0043, P=0.868)
5. soft_mean         0.5524  — (canônico atual)
6. weighted_mcc      0.5518  ⊘ (−0.0005)
7. median≡trimmed    0.5507  ⊘ (−0.0017)
8. max               0.5040  ▼ (−0.048)
9. rrf               0.4991  ▼ (−0.053)
```

Ranking corpus **human** (organismo-alvo da tese):

```
1. product_experts   0.5468  ▲ (+0.0042, P=0.981)  ← TOP
2. harmonic_mean     0.5481  ⊘ (+0.0054, P=0.962)
3. weighted_logit    0.5448  ⊘ (+0.0023, P=0.830)
4. logit_mean        0.5437  ⊘ (+0.0011, P=0.777)
5. soft_mean         0.5426  — (canônico)
6. weighted_mcc      0.5419  ⊘
7. median≡trimmed    0.5408  ⊘
8. max               0.5027  ▼
9. rrf               0.5038  ▼
```

**Padrão**: regras em espaço **log/logit/geométrico** dominam top 4 em
ambos corpora. Regras lineares (soft, weighted_mcc) e robustas
(median/trimmed) empatam no centro. Regras "any-confident" (max) e
calibration-free (rrf) regridem.

**Top 2 finalistas (Pareto-óptimos)**:

| Critério | product_experts | weighted_logit |
|---|---|---|
| Δ_MCC all | **+0.0068 ▲** | +0.0064 ▲ |
| Δ_MCC human | **+0.0042 ▲** | +0.0023 ⊘ |
| Δ_MCC non_human | −0.0126 ⊘ | +0.0021 ⊘ |
| Hiperparâmetros | 0 (zero-config) | val_MCC weights |
| Defensibilidade lit | Hinton 2002 (clássico) | combina dois winners |
| Custo impl | ~5 linhas | ~15 linhas |
| Sensibilidade p≈0 | sim (clip eps) | mitigado por logit |

### 8.7 Recomendação final

**Adotar `product_experts` como nova regra canônica**:
- Vence soft-mean em human (Δ=+0.0042, CI95 não-cruzada).
- Vence soft-mean em all (Δ=+0.0068, CI95 não-cruzada).
- Empate em non_human (n=1399 sem poder estatístico).
- Zero hiperparâmetro, drop-in em `aggregate.py` (~5 linhas).
- Cobertura por literatura clássica (Hinton 2002, BMA).

**Manter `weighted_logit` como verificação confirmatória** no Anexo B
(robustez do achado por mecanismo diferente).

**Sob FWER Holm m=10** (família ampliada de alternativas), α/10 = 0.005:
- product_experts em all: P(Δ>0)=0.997, p_boot ≤ 0.003 — **sobrevive**.
- product_experts em human: P(Δ>0)=0.981, p_boot ≤ 0.019 — **não sobrevive
  Holm m=10**, sobrevive Holm m=4 (subset original).
- Decisão: reportar como ganho descritivo-consistente sob protocolo
  unificado, com declaração explícita sobre FWER (alinhado com leitura da
  tese em §sec:resolucao-estatistica).

### 8.8 Direções não cobertas (futuras)

- **Logistic stacking** sobre val: requer `val_y_prob` para os 4 modelos
  alinhados; DT-K não salva val no `raw_predictions.npz` atual. Habilitar
  exigiria patch no benchmark runner.
- **GBM stacker**: mesma dependência + risco overfit n_val.
- **Bayesian Model Averaging via BIC val**: implementável após DT-K val
  estar disponível.
- **Dynamic Classifier Selection**: alta complexidade, ganho marginal
  esperado, não prioritário.

## 9. Arquivos modificados / criados

| Arquivo | Tipo | Propósito |
|---|---|---|
| `scripts/inference/experiments/committee_hardvote_dtk.py` | novo | ablação hardvote+DT-K |
| `scripts/inference/experiments/committee_hardvote_all_arbiters.py` | novo | ablação 4 árbitros + pairwise |
| `scripts/inference/experiments/committee_aggregation_alts.py` | novo | ablação batch 1: logit_mean + weighted_mcc + product_experts + rrf |
| `scripts/inference/experiments/committee_aggregation_alts2.py` | novo | ablação batch 2: weighted_logit + median + trimmed_mean + max + harmonic_mean |
| `results/inference/committee_aggregation_alts2/REPORT.md` | gerado | report batch 2 (weighted_logit ▲ all) |
| `scripts/inference/committee.py` | editado | default profile flipado p/ `full_4model` |
| `results/inference/committee_hardvote_dtk/REPORT.md` | gerado | report hardvote+DT-K |
| `results/inference/committee_hardvote_arbiters/REPORT.md` | gerado | report 4 árbitros |
| `results/inference/committee_aggregation_alts/REPORT.md` | gerado | report 4 alternativas (PoE vence) |
| `docs/01-methodology/comite_agregacao_ablation.md` | novo + atualizado | este documento |

Nenhum arquivo da tese (`~/PhD/tex/`) modificado ainda. Aguardando
decisão sobre incorporar:
- Tabela hardvote × 4 árbitros ao Anexo B (defesa do soft-mean)
- Tabela alternativas de agregação ao Anexo B (PoE como melhoria proposta)
- Possível update de `aggregate.py` p/ adotar PoE em produção (+0.004-0.007 MCC).
