# Committee vs Individual Models — Empirical Test

**Protocol**: 5-seed averaged probabilities and thresholds per model; committee = mean of 4 models.

**Refinements (active toggles):**
- Dedupe by `(seq_id, chembl_id)`: ON
- Block bootstrap by protein (`seq_id`): ON
- Committee threshold optim on test (informative): ON
- Bootstrap resamples B = 2000; CI95 = percentile [2.5, 97.5].

## Corpus: non_human

### Metrics (5 systems)

| system | mcc | auroc | f1 | accuracy |
| --- | --- | --- | --- | --- |
| dtkinase | 0.5036 | 0.7819 | 0.7935 | 0.7455 |
| drugban | 0.5334 | 0.8522 | 0.8 | 0.7663 |
| graphban | 0.5283 | 0.8251 | 0.8007 | 0.7613 |
| conplex | 0.4622 | 0.8294 | 0.762 | 0.7334 |
| committee | 0.547 | 0.849 | 0.8085 | 0.7691 |
| committee_optim* | 0.5516 | 0.849 | 0.8115 | 0.7663 |

### Paired bootstrap: committee vs individual model

| comparison | delta_mean | ci_lo | ci_hi | frac_positive | verdict |
| --- | --- | --- | --- | --- | --- |
| committee - dtkinase | 0.0447 | 0.0044 | 0.0844 | 0.985 | committee leads ▲ |
| committee - drugban | 0.014 | -0.0163 | 0.0444 | 0.827 | indistinguishable ⊘ |
| committee - graphban | 0.0191 | 0.0052 | 0.0346 | 0.998 | committee leads ▲ |
| committee - conplex | 0.0849 | 0.0163 | 0.1603 | 0.991 | committee leads ▲ |

## Corpus: human

### Metrics (5 systems)

| system | mcc | auroc | f1 | accuracy |
| --- | --- | --- | --- | --- |
| dtkinase | 0.4899 | 0.832 | 0.6506 | 0.7792 |
| drugban | 0.5303 | 0.8654 | 0.6955 | 0.7753 |
| graphban | 0.4936 | 0.8434 | 0.6707 | 0.7631 |
| conplex | 0.4457 | 0.8235 | 0.5647 | 0.7717 |
| committee | 0.5421 | 0.8676 | 0.6908 | 0.7992 |
| committee_optim* | 0.5455 | 0.8676 | 0.6968 | 0.7978 |

### Paired bootstrap: committee vs individual model

| comparison | delta_mean | ci_lo | ci_hi | frac_positive | verdict |
| --- | --- | --- | --- | --- | --- |
| committee - dtkinase | 0.0525 | 0.0414 | 0.0644 | 1.0 | committee leads ▲ |
| committee - drugban | 0.0112 | -0.0023 | 0.0235 | 0.945 | indistinguishable ⊘ |
| committee - graphban | 0.0483 | 0.0392 | 0.057 | 1.0 | committee leads ▲ |
| committee - conplex | 0.0985 | 0.0663 | 0.1316 | 1.0 | committee leads ▲ |

## Corpus: all

### Metrics (5 systems)

| system | mcc | auroc | f1 | accuracy |
| --- | --- | --- | --- | --- |
| dtkinase | 0.4682 | 0.8205 | 0.6598 | 0.7489 |
| drugban | 0.5185 | 0.8599 | 0.6914 | 0.7721 |
| graphban | 0.4999 | 0.8488 | 0.6828 | 0.7547 |
| conplex | 0.4441 | 0.8182 | 0.559 | 0.7665 |
| committee | 0.5519 | 0.8661 | 0.7076 | 0.7956 |
| committee_optim* | 0.556 | 0.8661 | 0.6936 | 0.8069 |

### Paired bootstrap: committee vs individual model

| comparison | delta_mean | ci_lo | ci_hi | frac_positive | verdict |
| --- | --- | --- | --- | --- | --- |
| committee - dtkinase | 0.0838 | 0.072 | 0.0957 | 1.0 | committee leads ▲ |
| committee - drugban | 0.0331 | 0.0211 | 0.0451 | 1.0 | committee leads ▲ |
| committee - graphban | 0.0519 | 0.0429 | 0.0615 | 1.0 | committee leads ▲ |
| committee - conplex | 0.1094 | 0.0743 | 0.1449 | 1.0 | committee leads ▲ |
