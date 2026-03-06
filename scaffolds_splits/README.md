# Fixed-Test Scaffold Splitting with Scenario Train/Val (`scaffolds_splits`)

This pipeline creates a **single fixed test set** and then generates scenario-specific
`train/val` splits for:

- `S1` random
- `S2` compound-disjoint
- `S3` kinase-disjoint
- `S4` compound+kinase disjoint
- `Sc` scaffold-disjoint

for both datasets:

- `human`: `tests/datasets/kinase_human_compounds.tsv`
- `non_human`: `tests/datasets/kinase_non_human_compounds.tsv`

The main objective is to evaluate all scenarios on the **same test set**.

## 1. Core Protocol

1. Load and label data (`pChEMBL >= threshold` -> label `1`, else `0`).
2. Remove monotonic profiles **before any split**.
3. Build Murcko scaffolds per compound.
4. Select a fixed test scaffold set (`~10%` unique compounds by default).
5. Freeze that test set.
6. For each scenario `S1/S2/S3/S4/Sc`, split only the remaining pool into `train/val`.
7. Report class distribution (`%pos/%neg`) for `train`, `val`, and fixed `test`.

## 2. Monotonic Filtering (Mandatory by Default)

Two trivial-profile filters are applied by default:

1. **Monotonic kinases**:
   - kinase activity rate is exactly `0` or `1`.
2. **Monotonic compounds** (pan-inactive/pan-active):
   - compounds tested against at least 2 kinases,
   - activity rate exactly `0` or `1`.

This removes trivial signals that can inflate performance without real chemistry learning.

CLI flags:

- `--keep-monotonic-kinases` disables kinase filtering.
- `--keep-monotonic-compounds` disables compound filtering.

## 3. Mathematical Setup

For each dataset `d in {H, N}`:

- `D_d`: all rows after preprocessing
- `C_d`: unique compounds
- `U_d = |C_d|`
- `phi(c)`: Murcko scaffold of compound `c`

Fixed test selection uses scaffold subsets and targets:

- test target fraction `alpha = 0.10` (on unique compounds)

After fixed test is built:

- `T_d`: fixed test rows
- `R_d = D_d \ T_d`: remaining pool used for scenario `train/val`

## 4. Fixed Test Construction

Default mode (`--test-mode shared_scaffold`):

- one shared scaffold set for both domains (scientifically strict universal protocol).

Optional mode (`--test-mode per_dataset`):

- independent scaffold test sets `T_H_scaf` and `T_N_scaf` near 10% unique compounds.

`universal_test.tsv` is always written as:

- `concat(human_test.tsv, non_human_test.tsv)`

with `dataset_source` column.

### 4.1 Feasibility Bound in Shared-Scaffold Mode

Let:

- `S_H`: human scaffolds
- `S_N`: non-human scaffolds
- `S* = S_H ∩ S_N` (shared scaffolds)

Define shared-scaffold coverage for each domain:

- `gamma_H = (sum_{s in S*} n_{H,s}) / U_H`
- `gamma_N = (sum_{s in S*} n_{N,s}) / U_N`

with:

- `n_{d,s}` = number of unique compounds of domain `d` in scaffold `s`
- `U_d` = total unique compounds in domain `d`

Then the requested test target `alpha` is only feasible if:

- `alpha <= min(gamma_H, gamma_N)`

So the theoretical effective target bound becomes:

- `alpha_eff = min(alpha, gamma_H, gamma_N)`

`alpha_eff` is an upper bound, not an exact guarantee. Because scaffold selection is discrete and constrained by class support in both domains, the achieved fraction can be below `alpha_eff`.

This is why, in strict universal mode, the human test fraction may be lower than 10% if shared scaffold coverage is limited. This is expected and scientifically correct, not a bug.

## 5. Scenario Train/Val on the Same Fixed Test

Given `R_d` (test removed), each scenario creates `train/val` as follows:

1. `S1` (random): stratified row split.
2. `S2` (compound): validation groups selected by `chembl_id` (compound disjoint train/val).
3. `S3` (kinase): validation groups selected by `target_kinase` (kinase disjoint train/val).
4. `Sc` (scaffold): validation groups selected by `scaffold` (scaffold disjoint train/val).
5. `S4` (new compound + new kinase): train/val built from disjoint compound and kinase assignments; cross-quadrant rows are dropped as orphans.

So all scenarios share the same `test`, and only `train/val` protocol changes.

## 6. Class Distribution Preservation

For grouped scenarios, validation-group selection minimizes:

- split-size error (target validation fraction in remaining pool), plus
- class-rate deviation in both train and val, plus
- penalty if either split loses a class.

Conceptually:

- `L = |val_frac - target| + w * (|p_train - p_pool| + |p_val - p_pool|) + penalty * class_missing`

where:

- `p_pool`: positive rate in the scenario pool,
- `p_train`, `p_val`: positive rates in train/val.

For `S4`, an additional drop penalty is applied to reduce orphan loss.

## 7. Distribution Reports

The script prints lines like:

```text
[human][S2] Train: +41.99% / -58.01% | Val: +41.81% / -58.19% | Test: +52.50% / -47.50% | dropped=0
```

and writes:

- `split_class_distribution_summary.tsv`
- `split_class_distribution_report.txt`

These include class percentages per dataset, scenario, and split.

## 8. Output Structure

In `scaffolds_splits/output/`:

- `human_test.tsv`, `non_human_test.tsv`, `universal_test.tsv`
- `test_scaffolds_universal.json`
- `manifest.json`
- `split_class_distribution_summary.tsv`
- `split_class_distribution_report.txt`
- `scenarios/S1/...`, `scenarios/S2/...`, `scenarios/S3/...`, `scenarios/S4/...`, `scenarios/Sc/...`

Each scenario folder contains `{dataset}_train.tsv`, `{dataset}_val.tsv`, and optionally `{dataset}_dropped.tsv` (mainly S4).

Top-level `human_train.tsv`/`human_val.tsv` and `non_human_train.tsv`/`non_human_val.tsv`
are backward-compatible aliases of the canonical scenario (default: `Sc` if requested).

## 9. Run

```bash
source env/bin/activate
python scripts/scaffold_split.py --output-dir scaffolds_splits/output
```

Useful flags:

- `--scenarios S1,S2,S3,S4,Sc`
- `--target-test-frac 0.10`
- `--target-val-frac 0.10`
- `--class-penalty 10`
- `--class-rate-weight 2.0`
- `--scenario-restarts 16`
- `--s4-restarts 192`
- `--test-mode shared_scaffold|per_dataset` (default: `shared_scaffold`)

## 10. Practical Note

If `S4` drops many rows, that is expected behavior: enforcing simultaneous compound and kinase disjointness introduces cross-quadrant orphans.

Check `dropped_rows` in `manifest.json` and in the class-distribution reports.

## 11. Validation Checklist (Recommended)

After running:

```bash
source env/bin/activate
python scripts/scaffold_split.py --output-dir scaffolds_splits/output
```

validate:

1. Shared-scaffold test is active:
   - `manifest.json` -> `config.test_mode == "shared_scaffold"`.
2. Test was created before scenario splits:
   - scenario outputs exist under `scaffolds_splits/output/scenarios/*` and reuse the fixed `human_test.tsv` / `non_human_test.tsv`.
3. Universal test scaffolds are identical across domains:
   - scaffold sets in `human_test.tsv` and `non_human_test.tsv` are equal.
4. Feasibility metrics explain achieved test size:
   - `manifest.json` -> `test_selection.gamma_human`, `test_selection.gamma_non_human`,
     `test_selection.effective_target_test_fraction`.
5. Class distribution reporting exists:
   - `split_class_distribution_summary.tsv`
   - `split_class_distribution_report.txt`
6. Scenario-specific constraints hold:
   - `S2`: no compound overlap between train/val
   - `S3`: no kinase overlap between train/val
   - `Sc`: no scaffold overlap between train/val
   - `S4`: no compound and no kinase overlap between train/val
