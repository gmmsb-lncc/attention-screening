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

## Monitoring and resuming a long run

The toolkit's expensive step is `tost_sensitivity.py`: with the default
`N_BOOTSTRAP=10000`, 12 ordered model pairs × 5 seeds × 7 metrics × B
iterations = on the order of 4 million metric evaluations, ~30–60 minutes
of single-CPU time per corpus on `non_human`, longer on `human` and
`all`. The full panel is launched via `run_full_stats.sh` or
`run_all_corpora.sh` and writes to `results/statistical/<corpus>/`.

### Where outputs land (in dependency order)

```
results/statistical/<corpus>/
    null_model.json        # written by null_model.py        (~ seconds)
    upper_limit.json       # written by upper_limit.py       (~ seconds)
    posthoc.json           # written by posthoc_classification.py (~ seconds)
    effect_size.json       # written by effect_size.py       (~ seconds)
    anova_tukey.json       # written by anova_tukey.py       (~ seconds)
    tost.json              # written by tost_sensitivity.py  (slowest step)
    figures/sim_ci_<metric>.pdf      # 4 files, plot_simultaneous_ci.py
    figures/mcsim_<metric>.pdf       # 4 files, plot_mcsim_heatmap.py
    panel.json             # written last by aggregate_panel.py
    panel.tex
    checklist.md
```

`panel.json` + `checklist.md` only exist after the entire pipeline
finishes successfully. Any intermediate JSON present indicates that step
completed; `tost.json` present means the slow step is done and the
remaining work (8 figures + aggregator) is just minutes.

### Checking status remotely or after restart

```bash
# Is the pipeline still running on this host?
pgrep -af "run_full_stats|scripts.statistical_analysis"

# How far did the run progress?
ls -la results/statistical/<corpus>/

# Is the slow step (tost) still grinding?
pgrep -af "tost_sensitivity"

# Has the aggregator finished?
test -f results/statistical/<corpus>/panel.json && echo "PANEL READY"

# Quick look at what passed / failed in the checklist:
cat results/statistical/<corpus>/checklist.md

# Open the simultaneous CI plot for MCC:
xdg-open results/statistical/<corpus>/figures/sim_ci_mcc.pdf  # Linux
# or scp/sshfs the figures back to a workstation.
```

### Resuming a partially completed run

The pipeline is idempotent: every script overwrites its single JSON
output and is independent except for the aggregator (which only reads
the others). To resume after a kill or crash:

1. Check which JSONs exist under `results/statistical/<corpus>/`.
2. Re-run only the missing per-step modules. Each script has its own
   `python -m scripts.statistical_analysis.<name> --corpus <corpus> --out <path>`
   CLI; see the per-file docstring.
3. Once `null_model.json`, `upper_limit.json`, `posthoc.json`,
   `effect_size.json`, `anova_tukey.json`, and `tost.json` are all
   present, regenerate figures and panel:

```bash
PY=env_baseline/bin/python
OUT=results/statistical/non_human
for M in mcc auroc f1 auprc; do
  $PY -m scripts.statistical_analysis.plot_simultaneous_ci \
    --corpus non_human --metric $M --out $OUT/figures/sim_ci_$M.pdf
  $PY -m scripts.statistical_analysis.plot_mcsim_heatmap \
    --corpus non_human --metric $M \
    --effect-source $OUT/effect_size.json \
    --pvalue-source $OUT/anova_tukey.json \
    --out $OUT/figures/mcsim_$M.pdf
done
$PY -m scripts.statistical_analysis.aggregate_panel \
    --corpus non_human --in-dir $OUT \
    --out-json $OUT/panel.json --out-tex $OUT/panel.tex \
    --out-checklist $OUT/checklist.md
```

If the smoke run on the work machine completed overnight, the panel and
checklist should already be written; just inspect them.

### Reducing runtime for iteration

Pass `N_BOOTSTRAP=2000` (or even 1000) when iterating; only the final
canonical run needs `N_BOOTSTRAP=10000` for protocol compliance. The
plots and aggregator are not affected.

### Reproducibility entry-points (cheat sheet)

```bash
# Single corpus, default B = 10000:
bash scripts/statistical_analysis/run_full_stats.sh non_human

# Single corpus, faster smoke (B = 2000):
N_BOOTSTRAP=2000 bash scripts/statistical_analysis/run_full_stats.sh non_human

# All three corpora sequentially:
bash scripts/statistical_analysis/run_all_corpora.sh

# Mirror figures to PhD/figures (when running with thesis builds):
MIRROR_TO=~/PhD/figures bash scripts/statistical_analysis/run_full_stats.sh non_human
```
