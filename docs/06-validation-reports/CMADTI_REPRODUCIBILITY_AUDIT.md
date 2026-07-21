# CMA-DTI reproducibility and test-leakage audit

- **Audit date:** 2026-07-21
- **Official repository:** `qinchi1/CMA-DTI`
- **Pinned commit:** `379bca54e20c5dbac9ba4658ce9a899cbd010824`
**Publication:** [Qin et al. (2026), DOI 10.3389/fbinf.2026.1861685](https://doi.org/10.3389/fbinf.2026.1861685)

## Finding

There is a confirmed contradiction between the peer-reviewed publication and
the pinned official implementation concerning decision-threshold selection.

- The publication states that the decision threshold is selected using the
  validation set and then applied to the test set. It does not state which
  metric is optimized to select that threshold.
- The official implementation calls `test(dataloader="test")`, constructs an
  ROC curve from the test labels and test probabilities, chooses the threshold
  that maximizes its F1-derived objective, and then reports threshold-dependent
  metrics on those same test samples. See the pinned
  [`trainer.py`](https://github.com/qinchi1/CMA-DTI/blob/379bca54e20c5dbac9ba4658ce9a899cbd010824/trainer.py#L144-L160)
  and its
  [threshold optimization block](https://github.com/qinchi1/CMA-DTI/blob/379bca54e20c5dbac9ba4658ce9a899cbd010824/trainer.py#L232-L270).

When the official code path is executed as shipped, this is test-set leakage:
the held-out test labels influence the operating threshold at which the same
test set is scored.

## Scope of the impact

| Component or metric | Affected by this threshold leakage? | Reason |
|---|---:|---|
| Model weights | No | Training uses the training loader. |
| Best checkpoint | No | Selection uses validation AUROC. |
| AUROC | No | It is threshold-independent. |
| AUPRC | No | It is threshold-independent. |
| F1, accuracy and precision | Yes | They are computed after selecting the threshold on test labels. |
| Sensitivity and specificity | Yes | They use the same test-optimized operating point. |

This audit does **not** establish that the numerical results in the publication
were produced with the leaking code path. The paper explicitly describes
validation-based threshold selection, and the original run logs, thresholds
and per-fold prediction artifacts needed to resolve the discrepancy are not
available in this repository. The defensible conclusion is therefore:

> Leakage is confirmed in the behavior of the pinned official code when run as
> provided; leakage in the experiments reported by the paper is undetermined.

## Canonical remediation

The canonical wrapper never uses test labels for model or threshold selection.
It selects the checkpoint by validation AUROC, chooses the committee operating
point by validation MCC, freezes that threshold, and only then scores the test
set. Validation MCC is a benchmark evaluation overlay; it is not claimed to be
the undocumented threshold objective used by the publication.

Every new `cmadti_results.json`, `cmadti_calibration.json` and multiseed
aggregate records this audit status so the provenance travels with exported
results.
