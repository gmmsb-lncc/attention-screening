# ChemGLaM reproducibility and data-leakage audit

- **Audit date:** 2026-07-21
- **Official repository:** `clinfo/ChemGLaM`
- **Pinned commit:** `3f09b907af3b53fde32e44c7e98b098c2a2c552c`
- **Publication:** [Hirakawa et al. (2026), DOI 10.1186/s13321-026-01155-z](https://doi.org/10.1186/s13321-026-01155-z)

## Finding

ChemGLaM does **not** contain the same confirmed test-threshold leakage found
in the official CMA-DTI implementation. The pinned ChemGLaM source reports
validation/test AUROC and AUPRC and does not search for a classification
threshold on either split. However, the audit found a separate data-leakage
problem in the official Davis artifacts: exact compound-protein pairs occur in
both training and validation in every fold.

The official trainer selects the best checkpoint by minimum validation loss.
Consequently, the Davis checkpoint is selected using a validation set that is
not independent of training. The held-out Davis test artifacts remain clean:
no test compound, protein or exact pair occurs in training or validation.

## Reproducible split audit

Run the checked audit against the pinned submodule:

```bash
conda run -n chemglam-cuda \
  python scripts/chemglam/audit_upstream_splits.py \
  --output results/chemglam/upstream_split_audit.json
```

An exact pair is defined as `(smiles, target_sequence)`. For the five official
Davis folds, the training-validation result is:

| Fold | Shared exact pairs | Same label | Conflicting labels |
|---:|---:|---:|---:|
| 0 | 327 | 255 | 72 |
| 1 | 312 | 249 | 63 |
| 2 | 380 | 297 | 83 |
| 3 | 239 | 181 | 58 |
| 4 | 202 | 169 | 33 |

The same audit produced the following test-boundary result:

| Official dataset | Folds | Train-test entity/pair overlap | Validation-test entity/pair overlap |
|---|---:|---:|---:|
| BindingDB | 5 | 0 | 0 |
| Davis | 5 | 0 | 0 |
| Metz | 5 | 0 | 0 |
| PDBbind | 5 | 0 PDB IDs | 0 PDB IDs |

For BindingDB, Davis and Metz, “entity” covers the shipped compound IDs,
SMILES, target IDs and target sequences. The PDBbind CSVs ship only PDB IDs,
so compound/protein disjointness cannot be independently checked from those
files.

## Threshold provenance

The publication reports threshold-dependent classification metrics including
F1, MCC and accuracy, but it does not document a decision threshold or a
threshold-selection procedure. The pinned shared implementation only computes
AUROC and AUPRC during validation and test; it contains no code that can
reproduce the paper's F1/MCC/accuracy operating point.

Therefore:

- no test-label threshold optimization was found in the shared ChemGLaM code;
- the threshold behind the published F1/MCC/accuracy remains undocumented and
  unverifiable without the authors' unshared analysis artifacts;
- it is not defensible to claim that the published threshold-dependent numbers
  leaked through test-threshold optimization;
- it is also not defensible to call the complete official evaluation clean,
  because Davis has confirmed training-validation pair leakage.

## Scope of the impact

| Component | Audit conclusion |
|---|---|
| Official Davis model selection | Affected: validation loss includes pairs already present in training. |
| Official Davis held-out test | No direct entity or pair contamination found. |
| Official BindingDB/Metz held-out test | No direct entity or pair contamination found. |
| Official PDBbind held-out test | PDB IDs are disjoint; compound/protein audit is unavailable. |
| Published AUROC/AUPRC | Not affected by an undocumented decision threshold, but Davis checkpoint selection can affect predictions. |
| Published F1/MCC/accuracy | Threshold provenance is unknown; values cannot be reproduced from the shared evaluation code alone. |

This audit does **not** prove that every reported ChemGLaM result is leaked.
It proves a narrower claim: the official Davis training/validation artifacts
are non-independent, while no direct test contamination or test-threshold
selection was found in the auditable official source.

## Canonical remediation and real results

The canonical ChemGLaM wrapper does not reuse the official Davis folds. It
uses the fixed universal Bemis-Murcko train/validation/test partitions for
`human`, `non_human` and `all`; the converter preserves those assignments and
never re-splits them. The best checkpoint follows the upstream minimum
validation-loss rule. The operating threshold is then selected by validation
MCC, frozen, and applied once to the held-out test set.

Validation-MCC threshold selection is an explicit comparison-protocol overlay,
not a claim about the undocumented procedure used in the ChemGLaM paper. The
result files retain train/validation/test probabilities, row identifiers,
checkpoint/config paths, the frozen calibration, and this audit status. Thus,
the canonical outputs are real held-out results for this project's common
benchmark; they must not be presented as a reproduction of the paper's
undocumented F1/MCC/accuracy calculation.

## Source evidence

- The publication describes five-fold zero-shot-like tests containing compounds
  and proteins unseen during training and gives the 10-epoch training setup:
  [full article](https://link.springer.com/article/10.1186/s13321-026-01155-z).
- The official trainer selects minimum `avg_val_loss`:
  [`train.py`](https://github.com/clinfo/ChemGLaM/blob/3f09b907af3b53fde32e44c7e98b098c2a2c552c/train.py#L35-L53).
- The official model computes only AUROC/AUPRC for classification validation
  and test:
  [`chemglam.py`](https://github.com/clinfo/ChemGLaM/blob/3f09b907af3b53fde32e44c7e98b098c2a2c552c/chemglam/model/chemglam.py#L176-L247).
- The official data module consumes the supplied split indices:
  [`datamodule.py`](https://github.com/clinfo/ChemGLaM/blob/3f09b907af3b53fde32e44c7e98b098c2a2c552c/chemglam/data/datamodule.py#L133-L170).
