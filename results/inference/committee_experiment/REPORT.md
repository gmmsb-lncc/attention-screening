# Committee vs Individual Models — Empirical Test

**Protocol**: 5-seed averaged probabilities and thresholds per model; committee = mean of 4 models. Bootstrap B=10000, per-pair resampling, IC95 = percentile [2.5, 97.5].

**Hypothesis under test**: 4-model consensus committee outperforms (or matches) each of the four individual models on the canonical in-domain test set, when each individual model is itself the 5-seed ensemble of its own canonical seeds {42, 123, 456, 789, 1024}.

## Corpus: non_human  (n=1702 test pairs)

### Metrics (5 systems)

| system | mcc | auroc | f1 | accuracy |
| --- | --- | --- | --- | --- |
| dtkinase | 0.5295 | 0.8092 | 0.8004 | 0.7573 |
| drugban | 0.5569 | 0.8632 | 0.8075 | 0.7767 |
| graphban | 0.5635 | 0.8468 | 0.8124 | 0.7767 |
| conplex | 0.4965 | 0.8457 | 0.7736 | 0.7497 |
| committee | 0.5748 | 0.8649 | 0.8171 | 0.7814 |

### Paired bootstrap: committee vs individual model

| comparison | delta_mean | ci_lo | ci_hi | frac_positive | verdict |
| --- | --- | --- | --- | --- | --- |
| committee - dtkinase | 0.0453 | 0.0213 | 0.0693 | 1.0000 | committee leads ▲ |
| committee - drugban | 0.0179 | -0.0035 | 0.0397 | 0.9521 | indistinguishable ⊘ |
| committee - graphban | 0.0111 | -0.0063 | 0.0293 | 0.8912 | indistinguishable ⊘ |
| committee - conplex | 0.0784 | 0.0446 | 0.1122 | 1.0000 | committee leads ▲ |

## Corpus: human  (n=39739 test pairs)

### Metrics (5 systems)

| system | mcc | auroc | f1 | accuracy |
| --- | --- | --- | --- | --- |
| dtkinase | 0.5052 | 0.8357 | 0.6797 | 0.7741 |
| drugban | 0.5435 | 0.8690 | 0.7183 | 0.7710 |
| graphban | 0.5088 | 0.8486 | 0.6966 | 0.7590 |
| conplex | 0.4285 | 0.8190 | 0.5693 | 0.7515 |
| committee | 0.5602 | 0.8709 | 0.7199 | 0.7960 |

### Paired bootstrap: committee vs individual model

| comparison | delta_mean | ci_lo | ci_hi | frac_positive | verdict |
| --- | --- | --- | --- | --- | --- |
| committee - dtkinase | 0.0550 | 0.0482 | 0.0620 | 1.0000 | committee leads ▲ |
| committee - drugban | 0.0167 | 0.0102 | 0.0230 | 1.0000 | committee leads ▲ |
| committee - graphban | 0.0514 | 0.0457 | 0.0572 | 1.0000 | committee leads ▲ |
| committee - conplex | 0.1317 | 0.1227 | 0.1408 | 1.0000 | committee leads ▲ |

## Corpus: all  (n=41441 test pairs)

### Metrics (5 systems)

| system | mcc | auroc | f1 | accuracy |
| --- | --- | --- | --- | --- |
| dtkinase | 0.4747 | 0.8216 | 0.6798 | 0.7421 |
| drugban | 0.5302 | 0.8613 | 0.7130 | 0.7676 |
| graphban | 0.5105 | 0.8534 | 0.7036 | 0.7502 |
| conplex | 0.4245 | 0.8130 | 0.5587 | 0.7467 |
| committee | 0.5639 | 0.8685 | 0.7298 | 0.7911 |

### Paired bootstrap: committee vs individual model

| comparison | delta_mean | ci_lo | ci_hi | frac_positive | verdict |
| --- | --- | --- | --- | --- | --- |
| committee - dtkinase | 0.0892 | 0.0824 | 0.0960 | 1.0000 | committee leads ▲ |
| committee - drugban | 0.0337 | 0.0277 | 0.0398 | 1.0000 | committee leads ▲ |
| committee - graphban | 0.0534 | 0.0476 | 0.0592 | 1.0000 | committee leads ▲ |
| committee - conplex | 0.1394 | 0.1303 | 0.1486 | 1.0000 | committee leads ▲ |

## Conclusão geral

Em **Human (n=39.739)** e **All (n=41.441)**, ambos corpora de alta resolução estatística, o comitê **lidera os quatro modelos individuais** com IC95% estritamente positivo (Δ_MCC ∈ [+0,017; +0,139], todos com P(Δ>0)=1,000). Em **Non-Human (n=1.702)**, corpus de menor volume e menor resolução estatística, o comitê lidera DT-Kinase e ConPLex (IC95% estritamente positivo) e empata estatisticamente com DrugBAN e GraphBAN (IC95% cruza zero).

**Veredito**: o comitê de 4 modelos é **empiricamente efetivo** — nunca pior que o melhor modelo individual em nenhum dos três corpora, e estatisticamente superior em 10 das 12 comparações pareadas (3 corpora × 4 modelos). As 2 comparações sem evidência de superioridade (NH × DrugBAN, NH × GraphBAN) refletem o limite intrínseco de resolução do bootstrap com n_test = 1.702 pares e Δ_MCC pontual de +0,011 a +0,018, dentro do erro-padrão típico inter-semente.