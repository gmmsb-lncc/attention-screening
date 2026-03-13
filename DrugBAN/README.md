# DrugBAN Baseline Integration

Reproducible setup for running DrugBAN as an external baseline in this repository.

## What is included

- `setup_env.sh`: creates a dedicated conda environment and installs dependencies
- `prepare_data.py`: converts scaffold split TSV files into `SMILES,Protein,Y` CSV format
- `src/`: location where upstream DrugBAN source is cloned

## Quick Start

```bash
# 1) Setup environment and clone upstream source
cd DrugBAN
bash setup_env.sh

# 2) Activate environment
conda activate drugban

# 3) Prepare dataset CSVs from scaffold splits
python prepare_data.py --dataset non_human
```

Generated CSVs are saved to:

- `DrugBAN/datasets/kinase/non_human/scaffold/train.csv`
- `DrugBAN/datasets/kinase/non_human/scaffold/val.csv`
- `DrugBAN/datasets/kinase/non_human/scaffold/test.csv`

## Upstream source

By default, `setup_env.sh` clones:

- `https://github.com/peizhenbai/DrugBAN.git`

into `DrugBAN/src/`.

If you use a fork or pinned commit, edit these variables in `setup_env.sh`:

- `UPSTREAM_URL`: source repository URL
- `UPSTREAM_REF`: commit hash or tag to pin a deterministic version

## Reproducibility notes

- Use a dedicated environment (`drugban`) to avoid dependency conflicts.
- Keep `DrugBAN/setup_env.sh` versioned so other machines can reproduce the same stack.
- Keep `DrugBAN/prepare_data.py` versioned so split conversion stays consistent across runs.
- Keep `UPSTREAM_REF` pinned for deterministic source checkout across machines.
- `DrugBAN/.gitignore` ignores cloned source/data outputs to avoid polluting commits.

## Suggested next step

After environment + data setup, run training from the upstream `src/` entrypoint according to the upstream DrugBAN instructions, using the prepared `train/val/test` CSV files.
