#!/usr/bin/env python3
"""One synthetic CUDA forward/backward through the trainable CMA-DTI stack."""

from __future__ import annotations

import sys
from pathlib import Path

import dgl
import pandas as pd
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "CMA-DTI"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataloader import DTIDataset  # type: ignore  # noqa: E402
from modeling import CachedCMA  # noqa: E402


def main() -> int:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    device = torch.device("cuda")
    config = yaml.safe_load((REPO_ROOT / "configs/cmadti_universal.yaml").read_text())
    frame = pd.DataFrame({
        "SMILES": ["CCO", "CCN"], "Protein": ["ACDE", "FGHI"], "Y": [0, 1]
    })
    dataset = DTIDataset(range(2), frame, max_drug_nodes=config["model"]["max_drug_nodes"])
    graphs = dgl.batch([dataset[0][0], dataset[1][0]]).to(device)
    chem = torch.randn(2, 8, config["model"]["chemberta_feature_dim"], device=device)
    protein = torch.randn(2, 12, config["model"]["protein_feature_dim"], device=device)
    chem_mask = torch.ones(2, 8, dtype=torch.bool, device=device)
    protein_mask = torch.ones(2, 12, dtype=torch.bool, device=device)
    model = CachedCMA(config, device).to(device).train()
    logits, attention = model(graphs, chem, chem_mask, protein, protein_mask)
    torch.nn.functional.binary_cross_entropy_with_logits(
        logits, torch.tensor([0.0, 1.0], device=device)
    ).backward()
    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()
    print(f"CMA-DTI CUDA smoke OK: logits={tuple(logits.shape)} attention={tuple(attention.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
