# Statistical comparison protocol (Ash/Wognum 2025-aligned)

This document defines the statistical methodology used to compare the four
benchmark models (DT-Kinase, DrugBAN, GraphBAN, ConPLex) and the multi-model
committee. The standard adopted is Ash, Wognum *et al.* 2025
(`docs/01-methodology/references/ash_wognum_2025_jcim.pdf`,
DOI 10.1021/acs.jcim.5c01609), with explicit deviations declared below.

## Status of canonical regeneration (2026-05-08)

- **non_human**: **canonical B=10⁴ complete**. The 6 JSONs, 8 PDFs, 6
  LaTeX tables, panel and checklist were regenerated under
  `N_BOOTSTRAP=10000`; `run_full_stats.sh` automatically dropped the
  `--provisional` flag, so all emitted .tex files no longer carry the
  `% [PROVISIONAL B=2000]` disclaimer header. **Empirical confirmation
  of the disclaimer prediction**: qualitative characteristics (TOST
  counts 3/6 at δ=0.05 and 5/6 at δ=0.07, ANOVA p_bonf=0.21 NS in MCC,
  Tukey HSD reject pattern, ranking, checklist 10/10 PASS) are
  identical to the smoke B=2000 run; only the fourth decimal digit of
  bootstrap CIs moved (e.g., upper_limit MCC 0.8759 → 0.8763). Thesis
  (`~/PhD/tex/apendiceG.tex`) recompiled with canonical numbers, 388 p.
- **human, all**: still on smoke B=2000. Canonical regeneration pending
  via `bash scripts/statistical_analysis/run_full_stats.sh {human,all}`
  (default `N_BOOTSTRAP=10000`). Auto-generated tables under
  `results/statistical/{human,all}/tables/` continue to carry the
  `% [PROVISIONAL B=2000]` disclaimer until that regeneration runs.

The narrowing from B=2000 to B=10⁴ is approximately 5–10 % in CI width
(Davison & Hinkley 1997), insufficient to move equivalence verdicts;
this is now an empirical confirmation, not just a theoretical
prediction, validated on the non_human corpus.

## 0. Notation and uncertainty conventions (mandatory)

All numerical reports in this project and in the thesis follow these
conventions, declared once here to eliminate ambiguity:

- **`mean ± σ`** denotes mean ± **sample standard deviation** (ddof=1)
  computed over the n=5 seeds of the canonical multi-seed protocol. This
  is the descriptive form used in tables, panels, captions, and figure
  error bars. It captures the empirical inter-seed variability (init +
  numerical noise) observed in the experiment.
- **SE / SEM** (standard error of the mean, σ/√n = σ/2.236 for n=5) is
  only used when explicitly labeled as such (e.g., `mean (SE = 0.005)`).
  SE shrinks with √n and conflates sample size with intrinsic variability;
  σ is preferred for descriptive multi-seed reporting.
- **Bootstrap IC95%** is the formal uncertainty quantification for
  paired comparisons (Δ between models). Reported as
  `Δ_median [CI95%: lo, hi]` with B=10⁴ paired bootstrap iterations.
  When IC95% is reported, σ and SE are redundant and may be omitted.
- **Effect sizes** (Cohen's d_z, Hedges' g_paired) are unitless, derived
  from σ. They do not need a `±` modifier; their interpretive scale is
  Cohen 1988 / Lakens 2013 (small / medium / large at 0.2 / 0.5 / 0.8).
- Conversion: under n=5, **σ ≈ SE × 2.236**. To translate an SE-style
  number to σ-style: multiply by √5 ≈ 2.236. To translate σ to SE:
  divide by √5.

This convention applies prospectively to all new artifacts. Legacy
numbers in the thesis or older docs that use SE form are footnoted with
their σ equivalent at first occurrence rather than rewritten in place
(see CLAUDE.md item 6 for the conversion table).

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

**Status:** persistent deviation, **thesis-scope only**. Migration to 5×5
repeated CV is **scheduled for the journal-paper version** (post-defense),
not for the thesis.
**Scope decision (2026-05-08).** The thesis is considered statistically
robust enough under D1 because the complementary layers (null-model lower
limit, assay-noise upper limit, Hedges' g paired with J(4) correction,
post-hoc classification metrics, TOST sensitivity over six bands,
RM-ANOVA + Tukey HSD with parametric-assumption checks, simultaneous CI
plots, MCSim heatmaps) collectively provide a defensible reading at n=5
under the paper's own conclusion item 5 (transparency clause). The
journal-paper version will run the canonical 5×5 CV (n=25) on top of the
existing artifacts to bring D1 into compliance.
**Justification (original).** Retraining 4 models × 3 corpora × 25 samples
= 300 runs is approximately 150 GPU-hours sequential. The four models were
trained under a fixed scaffold split + 5 seeds protocol that consumed
substantial compute budget; refactoring to 5×5 CV requires full retraining.
**Consequence at n=5.** Sample-size for statistical inference is n=5 per
cell, below the n=25 recommended by Ash/Wognum 2025. Bootstrap CIs over
the 5 seeds carry empirical coverage approximately 80–87 % (subnominal
vs. 95 % target), as reported in Chapter 5,
§`sec:resolucao-bootstrap-pareado`.
**Mitigation in the thesis.** All inferential statements (TOST equivalence,
paired bootstrap CIs, RM-ANOVA + Tukey HSD) are read as conservative
estimates. The transparency clause (paper Section 5 conclusion item 5)
explicitly endorses transparent deviation with documented rationale.
**Closure path (publication).** When preparing the journal-paper version,
re-run all 4 models under 5×5 repeated CV per corpus, regenerate the full
panel via `bash scripts/statistical_analysis/run_all_corpora.sh` (the
toolkit code is already 5×5-ready since it iterates over `seeds`, only
the underlying training would change). All other components (Hedges' g,
TOST, plots, checklist) carry over unchanged.

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

### D3. TOST equivalence band — REFORMULATED (2026-05-08)

**Status (legacy, prior to 2026-05-08):** primary band δ_eq = 0.05 MCC,
SESOI-anchored. Cohen-anchored bands reported as sensitivity analysis.
**Status (current, post-audit):** **TOST relegated to robustness audit
only.** The primary criterion of H1 is now the **triple-convergence**
of (i) paired bootstrap by protein, (ii) RM-ANOVA + Tukey HSD with
Bonferroni inter-metric correction, and (iii) Hedges' g paired with J(4)
correction. TOST over six δ_eq bands is reported in the appendix as
robustness verification, not as decision criterion for H1.
**Reason for reformulation (Lição 25 §6.15.3 reformulated):** the
δ_eq = 0.05 MCC band was genealogically derived from the empirically
observed gap between DT-Kinase v7 and the leading baseline (DrugBAN, gap
of 0.050 MCC off-diagonal in the cross-dataset matrix). This is post-hoc
anchoring on observed magnitude, constituting retroactive pre-registration
whose confirmatory value requires explicit caveat. The triple-convergence
framework has independent statistical foundations (parametric NHST +
non-parametric resampling + standardized effect size) and does not depend
on any specific δ_eq choice.
**Migration:** Chapter 1, §`sec:rqs-escopo` (introducao.tex) and Chapter 4,
§`sec:ashwognum` (capitulo4.tex) declare the new triple-convergence
criterion; Chapter 5, §`sec:linguagem-estatistica` and §`sec:reformulacao-h1`
(capitulo5.tex) explain the reformulation; Appendix G, §`sec:auditoria-stats-tost-sensitivity`
reports the TOST sensitivity over six bands as robustness audit, with
the legacy 0.05 band marked as `(legacy primary)` in the table for
historical comparison; Appendix F, §`sec:licao-25` (Lição 25, apendiceF.tex)
documents the migration as a methodological lesson.

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
| 2 | Parametric assumption checks (Ash/Wognum 2025 conclusion item 2 — "we recommend always checking the parametric assumptions") | `scripts/statistical_analysis/anova_tukey.py::_check_assumptions` (Shapiro-Wilk per model + Levene across models + Levene on per-seed pairwise differences as sphericity proxy) | `assumptions` block per metric in `anova_tukey.json` |
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
