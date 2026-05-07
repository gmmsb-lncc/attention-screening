# Statistical comparison protocol (Ash/Wognum 2025-aligned)

This document defines the statistical methodology used to compare the four
benchmark models (DT-Kinase, DrugBAN, GraphBAN, ConPLex) and the multi-model
committee. The standard adopted is Ash, Wognum *et al.* 2025
(`docs/01-methodology/references/ash_wognum_2025_jcim.pdf`,
DOI 10.1021/acs.jcim.5c01609), with explicit deviations declared below.

## 1. Adopted standard

Ash/Wognum 2025 propose four guidelines for ML method comparison in small
molecule property modeling:

1. **Performance sampling distribution** — sample model performance with both
   weight-initialization variance (random seeds) and data-resampling variance
   (splits). Recommended default for datasets in 500–100 000 samples: 5×5
   repeated cross-validation (n=25 samples).
2. **Statistical significance** — repeated-measures ANOVA per metric +
   Tukey HSD for pairwise comparisons (parametric default; Conover-Friedman +
   Holm-Bonferroni when assumptions are violated). Bonferroni correction at
   the ANOVA level when multiple metrics are tested.
3. **Practical significance** — go beyond p-values. Report (a) standardized
   effect size (Cohen's *d*; Hedges' *g* under small samples), (b) lower
   performance limit (null model = majority class), (c) upper performance
   limit (experimental variability of the assay), (d) downstream-relevant
   metrics through post-hoc classification (precision@recall, recall@precision,
   TNR@recall).
4. **Presentation** — simultaneous CI plot (statsmodels), MCSim heatmap
   (scikit-posthocs), CI of differences. Report multiple metrics holistically,
   include confusion matrices for classification, scatter plots for regression.

## 2. Deviations from the standard (declared, with justification)

The thesis was designed and partially executed before this standard was
adopted. The following deviations are kept by deliberate decision and
declared transparently as scope limitations.

### D1. Single scaffold split, 5 seeds (instead of 5×5 repeated CV, n=25)

**Status:** persistent deviation, not migrated.
**Justification:** retraining 4 models × 3 corpora × 25 samples = 300 runs is
~150 GPU-hours sequential. The four models were trained under a fixed scaffold
split + 5 seeds protocol that consumed substantial compute budget; refactoring
to 5×5 CV would require full retraining. The thesis adopts single-split + 5
seeds as the canonical protocol and declares this explicitly.
**Consequence:** sample-size for statistical inference is n=5 per cell, below
the n=25 recommended by Ash/Wognum 2025. Bootstrap CIs over the 5 seeds carry
empirical coverage approximately 80–87 % (subnominal vs. 95 % target), as
reported in Chapter 5, §`sec:resolucao-bootstrap-pareado`.
**Mitigation:** all inferential statements (TOST equivalence, paired bootstrap
CIs, etc.) are read as conservative estimates. Future-work statement explicitly
lists 5×5 CV migration as the next protocol upgrade.

### D2. Paired bootstrap (B=10⁴) instead of repeated-measures ANOVA + Tukey HSD

**Status:** primary inference is bootstrap; ANOVA + Tukey HSD added as
complementary verification.
**Justification:** bootstrap by protein partially mitigates the dependency
concern raised by Dietterich 1998; samples are blocked at the kinase level,
which is the relevant unit of biological independence. The thesis was designed
with this stack and the implementation lives in
`scripts/thesis_followups/bootstrap_ci.py`.
**Migration:** RM-ANOVA + Tukey HSD is added as a parallel inference layer.
Both are reported. Convergence between the two strengthens the reading; divergence
is acknowledged as a sign of underpowered comparison.

### D3. TOST equivalence band: SESOI-anchored (δ_eq = 0.05 MCC) instead of
Cohen-anchored band

**Status:** primary band is SESOI; Cohen-anchored bands added in sensitivity
analysis.
**Justification:** δ_eq = 0.05 MCC is anchored in operational SESOI for kinase
virtual screening (decisional resolution for compound advance under fixed
recall@k constraint). It is pre-declared retroactively (acknowledged in
Chapter 1, §`sec:rqs-escopo`).
**Migration:** sensitivity analysis reports the count of equivalent pairs
under multiple bands: δ ∈ {0.03, 0.05, 0.07} (absolute, SESOI variants) and
δ_eq ∈ {0.2·σ_pooled, 0.5·σ_pooled, 0.8·σ_pooled} (Lakens 2017
SESOI-via-Cohen's-d). The primary band remains 0.05 MCC; alternative bands
are reported in a sensitivity table.

**Footnote on Hedges' ν.** The small-sample correction factor for Cohen's
*d* is implemented under the Lakens 2013 / Borenstein approximation
J(ν) = 1 − 3/(4ν − 1). For our 5-seed paired design (same proteins, same
test set, different inits), the correct degrees of freedom is ν = n − 1 = 4,
giving J(4) = 0.8000 exactly (Cohen's d_z corrected). The unpaired
(two-sample) form ν = 2(n − 1) = 8 with J(8) ≈ 0.9032 is reported as
cross-check only. The paired form is the primary reported value.

## 3. Implementation register

Components of the protocol mapped to scripts and outputs.

| Guideline | Component | Implementation | Output |
|---|---|---|---|
| 1 | 5-seed single split | `run_from_config.py` + `configs/v7.yaml` (DT-Kinase); equivalent runners for baselines | `results/.../seed_{42,123,456,789,1024}/raw_predictions.npz` |
| 1 | (Future) 5×5 repeated CV | not implemented; declared as deviation D1 | — |
| 1 | Shared NPZ loader (DT-Kinase + 3 baselines) | `scripts/statistical_analysis/data_loader.py` | normalized in-memory dict + alignment guard |
| 2 | Paired bootstrap by protein, B=10⁴ | `scripts/thesis_followups/bootstrap_ci.py::paired_delta` | per-pair median Δ + IC95% percentile |
| 2 | RM-ANOVA per metric + Tukey HSD | `scripts/statistical_analysis/anova_tukey.py` | `results/statistical/{corpus}/anova_tukey.json` |
| 2 | Bonferroni inter-metric | applied at ANOVA level (m=4 metrics → α'=0.0125) | adjusted ANOVA p |
| 3 | Null model lower limit | `scripts/statistical_analysis/null_model.py` | `results/statistical/{corpus}/null_model.json` |
| 3 | Upper limit (experimental variability) | `scripts/statistical_analysis/upper_limit.py`; Brown 2009 / Kramer 2012 method | `results/statistical/{corpus}/upper_limit.json` |
| 3 | Cohen's d / Hedges' g (paired J(4) primary, unpaired J(8) cross-check) | `scripts/statistical_analysis/effect_size.py` | `results/statistical/{corpus}/effect_size.json` |
| 3 | Post-hoc classification (precision@recall=0.8, recall@precision=0.8, TNR@recall=0.9) | `scripts/statistical_analysis/posthoc_classification.py` | `results/statistical/{corpus}/posthoc.json` |
| 3 | TOST sensitivity over six δ_eq bands | `scripts/statistical_analysis/tost_sensitivity.py` | `results/statistical/{corpus}/tost.json` |
| 4 | Simultaneous CI plot per (corpus, metric) | `scripts/statistical_analysis/plot_simultaneous_ci.py` | `results/statistical/{corpus}/figures/sim_ci_{metric}.pdf` |
| 4 | MCSim heatmap (color = paired g, stars = Tukey p_adj) | `scripts/statistical_analysis/plot_mcsim_heatmap.py` | `results/statistical/{corpus}/figures/mcsim_{metric}.pdf` |
| 4 | Confusion matrices | already in `scripts/inference/committee.py` outputs | Anexo A figures |
| -- | Aggregator (panel JSON + LaTeX + checklist compliance) | `scripts/statistical_analysis/aggregate_panel.py` | `panel.json`, `panel.tex`, `checklist.md` |
| -- | Reproducibility entry-point (single corpus) | `scripts/statistical_analysis/run_full_stats.sh` | full `results/statistical/{corpus}/` tree |
| -- | Reproducibility entry-point (all corpora) | `scripts/statistical_analysis/run_all_corpora.sh` | full `results/statistical/{human,non_human,all}/` |
| -- | Test suite | `tests/test_statistical_protocol.py` | sanity (J(4)=0.8, null MCC=0, alignment, etc.) |

## 4. Reporting checklist

For any new comparison to be declared statistically grounded under this
protocol, the report must include:

- [ ] All four metrics: MCC (primary), AUROC, F1, AUPRC.
- [ ] Mean ± σ over 5 seeds per cell (model × corpus).
- [ ] Paired bootstrap by protein with IC95 % (Δ_MCC vs. comparator).
- [ ] RM-ANOVA p-value per metric + Bonferroni adjustment.
- [ ] Tukey HSD pairwise intervals.
- [ ] Hedges' *g* per pair (Cohen's *d* small-sample correction).
- [ ] Null-model MCC (majority class) and assay upper-limit MCC for the corpus.
- [ ] At least one downstream-relevant metric (precision@recall=0.8 or
      recall@precision=0.8) per model.
- [ ] TOST sensitivity table over at least three δ_eq bands.
- [ ] Either simultaneous CI plot, MCSim heatmap, or CI-of-differences plot.
- [ ] Confusion matrix per cell.
- [ ] Explicit declaration of D1–D3 deviations in the methods section.

## 5. References

- Ash, J. R.; Wognum, C.; Rodríguez-Pérez, R. *et al.* Practically Significant
  Method Comparison Protocols for Machine Learning in Small Molecule Drug
  Discovery. *J. Chem. Inf. Model.* 2025, 65, 9398–9411.
  DOI 10.1021/acs.jcim.5c01609.
  Local copy: `references/ash_wognum_2025_jcim.pdf`.
- Lakens, D. Equivalence Tests: A Practical Primer for *t* Tests, Correlations,
  and Meta-Analyses. *Soc. Psychol. Personal. Sci.* 2017, 8, 355–362
  (SESOI-via-Cohen's *d* anchoring).
- Brown, S. P.; Muchmore, S. W.; Hajduk, P. J. Healthy skepticism: assessing
  realistic model performance. *Drug Discov. Today* 2009, 14, 420–427
  (upper-limit estimation under assay variability).
- Kramer, C.; Kalliokoski, T.; Gedeck, P.; Vulpetti, A. The experimental
  uncertainty of heterogeneous public *K_i* data. *J. Med. Chem.* 2012, 55,
  5165–5173 (IC₅₀ noise calibration).
- Hedges, L. V. Distribution Theory for Glass's Estimator of Effect Size and
  Related Estimators. *J. Educ. Stat.* 1981, 6, 107–128 (small-sample bias
  correction for Cohen's *d*).
- Dietterich, T. G. Approximate Statistical Tests for Comparing Supervised
  Classification Learning Algorithms. *Neural Comput.* 1998, 10, 1895–1923
  (bootstrap dependency caveats).
- Schuirmann, D. J. A comparison of the two one-sided tests procedure and the
  power approach for assessing the equivalence of average bioavailability.
  *J. Pharmacokinet. Biopharm.* 1987, 15, 657–680 (TOST).
