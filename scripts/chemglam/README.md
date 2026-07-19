# ChemGLaM integration

The upstream repository is pinned as the `ChemGLaM` submodule. It is not
modified locally. The canonical comparison uses the repository's ESM-2 3B +
MoLFormer configuration and the existing universal Bemis-Murcko split.

Prepare data:

```bash
conda run -p .conda/envs/dti4-macos python scripts/chemglam/prepare_universal.py
```

The converter preserves every source row and writes explicit train/validation
indices. It never derives a new random split. Generated CSV files remain
ignored by Git; `manifest.json` records counts for auditing.

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

For a one-seed smoke run, use `CHEMGLAM_SEEDS=42`. The runner trains on the
explicit universal train/validation partition, predicts validation and test
separately, chooses the MCC-optimal threshold on validation, freezes it, and
reports held-out metrics for `all`, `human`, and `non_human`.
