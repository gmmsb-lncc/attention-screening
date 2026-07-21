# ChemGLaM integration

The upstream repository is pinned as the `ChemGLaM` submodule. It is not
modified locally. The canonical comparison uses the repository's ESM-2 3B +
MoLFormer configuration and the existing universal Bemis-Murcko split.

Initialize only this submodule after cloning or switching branches:

```bash
git submodule update --init ChemGLaM
```

Do not run an unscoped recursive update: the parent repository contains the
legacy gitlink `GraphBAN/upstream_data`, which has no matching URL entry in
`.gitmodules` and is unrelated to the ChemGLaM integration.

The runner applies an idempotent local patch that pins MoLFormer remote code to
revision `7b12d946...`. The current MoLFormer `main` revision imports an
unreleased Transformers masking API and fails with the ChemGLaM-pinned
Transformers 4.46.3. The pinned revision contains the safetensors weights and
the compatible model implementation.

Prepare data:

```bash
conda run -p .conda/envs/dti4-macos python scripts/chemglam/prepare_universal.py
```

The converter preserves every source row and writes explicit train/validation
indices. It never derives a new random split. Generated CSV files remain
ignored by Git; `manifest.json` records counts for auditing.

The official ChemGLaM artifacts have a documented reproducibility limitation:
all five Davis folds contain exact compound-protein pairs in both training and
validation, although no direct test contamination was found. The publication
also does not document the threshold behind F1/MCC/accuracy, and the shared
code only computes AUROC/AUPRC. See the
[ChemGLaM reproducibility audit](../../docs/06-validation-reports/CHEMGLAM_REPRODUCIBILITY_AUDIT.md).
Reproduce the upstream split check with:

```bash
conda run -n chemglam-cuda \
  python scripts/chemglam/audit_upstream_splits.py \
  --output results/chemglam/upstream_split_audit.json
```

Production environment (RTX 4090):

```bash
bash environments/install_chemglam_cuda.sh
conda activate chemglam-cuda
```

The upstream training/data code currently hardcodes CUDA. The production
runner will wrap that code; the Mac environment is intended for conversion,
unit tests and reduced-backbone smoke tests only.

Run the canonical five-seed protocol on the RTX 4090:

```bash
bash scripts/chemglam/run_canonical_cuda.sh
```

For a one-seed smoke run, use `CHEMGLAM_SEEDS=42`. Select training corpora with
`CHEMGLAM_CORPORA`; accepted values are `all`, `human`, and `non_human`. Each
corpus has independent data, caches, checkpoints, and results. The runner
chooses the MCC-optimal threshold on that corpus's validation split and freezes
it for its held-out test split. This threshold rule is the common benchmark
overlay, not an undocumented claim about the original ChemGLaM evaluation.

The runner reuses completed checkpoints by default, which makes it safe to
backfill the expanded artifacts after an interrupted or older run. Set
`CHEMGLAM_SKIP_TRAIN_IF_CHECKPOINT=0` only when intentional retraining is
required.

Prediction is memory-safe for the canonical configuration: when attention
weights are disabled, only the prediction tensor is returned from each batch.
The upstream implementation returned every full cross-attention matrix and
could exhaust host RAM on the large train split. Prediction CSVs are written
atomically and reused on restart. Set `CHEMGLAM_SKIP_PREDICT_IF_COMPLETE=0`
only to intentionally recompute existing prediction files.

For every seed, the run retains the fitting and prediction configs, the best
checkpoint, train/validation/test predictions, `raw_predictions.npz` with pair
identifiers, a validation-derived `chemglam_calibration.json`, and a
self-contained `chemglam_results.json`. After all requested seeds finish,
`chemglam_<corpus>_aggregate.json` records mean and population standard
deviation across seeds. Prediction caches are isolated by corpus and split.
New result, calibration and aggregate JSON files carry the methodology-audit
provenance; legacy per-seed JSON files must be regenerated from saved
predictions before aggregation.

Once all five checkpoints and calibration sidecars exist, enable ChemGLaM as
an experimental fifth committee member with:

```bash
python scripts/inference/committee.py \
  --profile full_5model \
  --pairs pairs.tsv \
  --ckpt-corpus human \
  --out results/inference/chemglam_committee
```

The validated `full_4model` profile remains the default until the five-model
panel has been evaluated under the same held-out protocol.
