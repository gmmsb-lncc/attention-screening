# scripts/statistical_analysis/

Statistical comparison toolkit aligned with Ash/Wognum 2025
(*J. Chem. Inf. Model.* 65:9398–9411, DOI 10.1021/acs.jcim.5c01609).
Operates on `raw_predictions.npz` already on disk — no retraining.

Full protocol contract: `docs/01-methodology/statistical_protocol.md`.
Reporting checklist (mandatory): §4 of the same document.
Reference paper: `docs/01-methodology/references/ash_wognum_2025_jcim.pdf`.

## Quick start

```bash
# Single corpus, full panel:
bash scripts/statistical_analysis/run_full_stats.sh non_human

# All three corpora:
bash scripts/statistical_analysis/run_all_corpora.sh
```

Artifacts are written under `results/statistical/{corpus}/`:

| File | Source script | Content |
|---|---|---|
| `null_model.json` | `null_model.py` | Majority-class baseline (lower limit). |
| `upper_limit.json` | `upper_limit.py` | Assay-noise ceiling (Brown 2009 / Kramer 2012). |
| `posthoc.json` | `posthoc_classification.py` | precision@recall=0.8, recall@precision=0.8, TNR@recall=0.9. |
| `effect_size.json` | `effect_size.py` | Hedges' g paired (J(4) primary) + unpaired (J(8) cross-check). |
| `anova_tukey.json` | `anova_tukey.py` | RM-ANOVA + Tukey HSD per metric. |
| `tost.json` | `tost_sensitivity.py` | TOST sensitivity over 6 bands (D3 preserved). |
| `figures/sim_ci_{metric}.pdf` | `plot_simultaneous_ci.py` | Tukey HSD simultaneous CI plot. |
| `figures/mcsim_{metric}.pdf` | `plot_mcsim_heatmap.py` | 4×4 Hedges' g heatmap with significance stars. |
| `panel.json` / `panel.tex` / `checklist.md` | `aggregate_panel.py` | Full structured panel + LaTeX + checklist compliance. |

## Module-level conventions

- All scripts read paths via `data_loader.py` and operate on the canonical
  4 models × 3 corpora × 5 seeds matrix (see `__init__.MODELS`,
  `CORPORA`, `SEEDS`).
- All scripts export both JSON (machine-readable) and a stdout summary
  (human-readable).
- Bootstrap defaults to B = 10⁴ (`DEFAULT_BOOTSTRAP_B` in `__init__`).
- Hedges' nu defaults: paired ν=4 (J(4) = 0.8000 exactly under the
  Lakens 2013 / Borenstein approximation 1 − 3/(4ν − 1)) primary,
  unpaired ν=8 (J(8) ≈ 0.9032) cross-check
  (`HEDGES_NU_PAIRED`, `HEDGES_NU_UNPAIRED`).
- Three protocol deviations from Ash/Wognum 2025 are preserved as
  declared in `docs/01-methodology/statistical_protocol.md` §2:
  - D1: single scaffold split + 5 seeds (no 5×5 CV).
  - D2: paired bootstrap by protein remains primary; ANOVA + Tukey HSD
    is parallel verification, not replacement.
  - D3: TOST primary band δ_eq = 0.05 MCC (SESOI); Cohen-anchored bands
    reported as sensitivity only.

## Dependencies

Python 3.10+ with: numpy, scipy, scikit-learn, statsmodels, pandas,
matplotlib. All present in `env_baseline/`.

## Reusable utilities (do not duplicate)

- `scripts.thesis_followups.bootstrap_ci.paired_delta` — paired bootstrap
  + Wilcoxon (used by `tost_sensitivity.py`).
- `scripts.thesis_followups.bootstrap_ci.compute_metrics` — sklearn metric
  bundle (used by `null_model.py`).
- `scripts.inference.aggregate` — uncertainty decomposition reference.
