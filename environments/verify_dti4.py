#!/usr/bin/env python3
"""Smoke-test the shared runtime for CS-DTA, ChemGLaM, BIND and Top-DTI."""

from __future__ import annotations

import argparse
import importlib
import platform


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--require-mps", action="store_true")
    parser.add_argument("--require-topology", action="store_true")
    args = parser.parse_args()

    modules = [
        "torch", "torchvision", "torch_geometric", "transformers", "lightning",
        "peft", "esm", "Bio", "networkx", "pysmiles", "pandas", "sklearn",
        "rdkit", "datasets",
    ]
    if args.require_topology:
        modules.append("gtda")

    versions: dict[str, str] = {}
    for name in modules:
        module = importlib.import_module(name)
        versions[name] = str(getattr(module, "__version__", "ok"))

    import torch

    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was required but torch.cuda.is_available() is False")
    if args.require_mps and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was required but torch.backends.mps.is_available() is False")

    accelerator = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"platform={platform.platform()}")
    print(f"accelerator={accelerator}")
    for name, version in versions.items():
        print(f"{name}={version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
