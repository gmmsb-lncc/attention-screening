# Shared environment for the four additional DTI models

This directory defines one logical compatibility stack for CS-DTA, ChemGLaM,
BIND and Top-DTI. The upstream projects pin mutually incompatible releases, so
the shared stack uses PyTorch 2.5 and Transformers 4.46. CS-DTA may require
small source-level compatibility fixes when it is integrated.

## macOS Apple Silicon (development)

```bash
CONDA_PKGS_DIRS=/tmp/attention-screening-conda-pkgs \
  conda env create -p .conda/envs/dti4-macos -f environments/dti4-macos.yml
conda run -p .conda/envs/dti4-macos \
  python environments/verify_dti4.py --require-mps
```

PyTorch selects MPS at runtime. `giotto-tda` is omitted because the project
does not publish an osx-arm64 build; Top-DTI topological preprocessing must run
on the Linux production environment.

## Linux + RTX 4090 (canonical production environment)

```bash
bash environments/install_dti4_cuda.sh
conda activate dti4-cuda
```

The installer uses the official PyTorch CUDA 12.1 wheels and requires CUDA in
the final smoke test. The NVIDIA driver must support CUDA 12.1 (driver 525.60.13
or newer on Linux).

## Reproducibility boundary

This is an integration environment, not a claim that the four original
environments are bit-identical. Zero-shot BIND results must first be checked
against the upstream example output. Original manifests remain the reference
when reproducing a paper result exactly.

Top-DTI's published `requirements.txt` names `fair-esm==2.0.1`, a release that
does not exist on PyPI. The compatibility manifests use the latest published
release, `fair-esm==2.0.0`.
