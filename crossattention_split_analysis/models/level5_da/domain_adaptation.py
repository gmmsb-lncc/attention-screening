"""Gradient Reversal Layer and scaffold clustering for Level 5-DA.

Components
----------
GradientReversalLayer : Reverses gradient sign during backward pass.
DomainDiscriminator   : MLP that predicts scaffold cluster membership.
build_scaffold_clusters : Butina clustering of scaffolds via Tanimoto
                         similarity on Morgan fingerprints.
lambda_schedule       : Progressive schedule for GRL strength.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function


# ---------------------------------------------------------------------------
# Gradient Reversal Layer (Ganin et al., JMLR 2016)
# ---------------------------------------------------------------------------

class _GradientReversal(Function):
    """Autograd function that reverses gradient by -lambda."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lam: float) -> torch.Tensor:  # noqa: D401
        ctx.lam = lam
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lam * grad_output, None


class GradientReversalLayer(nn.Module):
    """Module wrapper for the gradient reversal function.

    During the forward pass the input is passed through unchanged.
    During the backward pass the gradient is multiplied by ``-lam``.
    """

    def __init__(self, lam: float = 1.0) -> None:
        super().__init__()
        self.lam = lam

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GradientReversal.apply(x, self.lam)


# ---------------------------------------------------------------------------
# Domain discriminator
# ---------------------------------------------------------------------------

class DomainDiscriminator(nn.Module):
    """MLP that predicts scaffold-cluster membership.

    Architecture:  Linear → BN → ReLU → Dropout → Linear → BN → ReLU
                   → Dropout → Linear(num_domains)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_domains: int = 16,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_domains),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits [B, num_domains]."""
        return self.net(x)


# ---------------------------------------------------------------------------
# Lambda schedule
# ---------------------------------------------------------------------------

def lambda_schedule(progress: float, gamma: float = 10.0) -> float:
    """Progressive lambda schedule (Ganin et al.).

    Parameters
    ----------
    progress : float
        Training progress in [0, 1].
    gamma : float
        Steepness of the sigmoid.

    Returns
    -------
    float
        Lambda value in [0, 1].
    """
    return float(2.0 / (1.0 + math.exp(-gamma * progress)) - 1.0)


# ---------------------------------------------------------------------------
# Scaffold clustering via Butina / Tanimoto
# ---------------------------------------------------------------------------

def build_scaffold_clusters(
    scaffolds: Sequence[str],
    num_clusters: int = 16,
    cutoff: float = 0.4,
) -> Dict[str, int]:
    """Cluster scaffolds by Tanimoto similarity on Morgan fingerprints.

    Falls back to frequency-based binning when RDKit is unavailable.

    Parameters
    ----------
    scaffolds : Sequence[str]
        Scaffold SMILES strings (may contain duplicates for frequency).
    num_clusters : int
        Target number of clusters (used for fallback).
    cutoff : float
        Butina distance cutoff (1 - Tanimoto).

    Returns
    -------
    Dict[str, int]
        Mapping from scaffold SMILES to cluster id (0-indexed).
    """
    unique_scaffolds = list(dict.fromkeys(scaffolds))  # deduplicate, preserve order
    # Filter out NaN / non-string scaffold values that come from missing data
    unique_scaffolds = [s for s in unique_scaffolds if isinstance(s, str)]

    try:
        return _butina_cluster(unique_scaffolds, cutoff, num_clusters)
    except ImportError:
        return _frequency_cluster(scaffolds, unique_scaffolds, num_clusters)


def _butina_cluster(
    scaffolds: List[str],
    cutoff: float,
    max_clusters: int,
) -> Dict[str, int]:
    """Butina clustering using RDKit Morgan fingerprints."""
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    from rdkit.ML.Cluster import Butina

    gen = AllChem.GetMorganGenerator(radius=2, fpSize=1024)
    fps = []
    valid_scaffolds: List[str] = []
    for smi in scaffolds:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = gen.GetFingerprint(mol)
            fps.append(fp)
            valid_scaffolds.append(smi)

    if not fps:
        return _frequency_cluster(scaffolds, scaffolds, max_clusters)

    # Compute pairwise distance matrix (upper triangle)
    n = len(fps)
    dists = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1.0 - s for s in sims])

    clusters = Butina.ClusterData(dists, n, cutoff, isDistData=True)

    # Map scaffold -> cluster_id, cap at max_clusters
    scaffold_to_cluster: Dict[str, int] = {}
    for cid, members in enumerate(clusters):
        assigned_id = min(cid, max_clusters - 1)
        for idx in members:
            scaffold_to_cluster[valid_scaffolds[idx]] = assigned_id

    # Assign any invalid scaffolds to the last cluster
    for smi in scaffolds:
        if smi not in scaffold_to_cluster:
            scaffold_to_cluster[smi] = max_clusters - 1

    return scaffold_to_cluster


def _frequency_cluster(
    all_scaffolds: Sequence[str],
    unique_scaffolds: List[str],
    num_clusters: int,
) -> Dict[str, int]:
    """Fallback: assign scaffolds to clusters by frequency rank."""
    counts = Counter(all_scaffolds)
    sorted_scaffolds = sorted(unique_scaffolds, key=lambda s: -counts[s])

    scaffold_to_cluster: Dict[str, int] = {}
    for i, smi in enumerate(sorted_scaffolds):
        scaffold_to_cluster[smi] = min(i, num_clusters - 1)

    return scaffold_to_cluster
