"""Bilinear Attention Network (BAN) modules.

Adapted from DrugBAN (Bai et al., Briefings in Bioinformatics, 2023).
Provides low-rank bilinear pooling for protein-ligand interaction modeling.

The ``BANLayer`` computes pairwise bilinear attention scores between all
protein tokens and all ligand tokens, then performs attention-weighted
pooling to produce a single fused interaction vector.

Mathematical formulation
------------------------
Given protein features  v ∈ ℝ^{B × N_p × d_v}  and  ligand features
q ∈ ℝ^{B × N_l × d_q}:

1. Project:  ṽ = FC_v(v) ∈ ℝ^{B × N_p × (h·k)},  q̃ = FC_q(q) ∈ ℝ^{B × N_l × (h·k)}
2. Bilinear scoring (per head h):  A^{(h)}_{i,j} = Σ_k  W^{(h)}_k · ṽ_{i,k} · q̃_{j,k}
3. Attention pooling:  z = Σ_h Σ_{i,j}  ṽ_{i,:} · A^{(h)}_{i,j} · q̃_{j,:}

Output z ∈ ℝ^{B × h_dim} is the fused interaction representation.

Reference
---------
- Kim et al., "Bilinear Attention Networks", NeurIPS 2018
- Bai et al., "Interpretable bilinear attention network ...", Briefings Bioinf 2023
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm


class FCNet(nn.Module):
    """Non-linear fully-connected network with weight normalisation."""

    def __init__(
        self,
        dims: list[int],
        act: str = "ReLU",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(len(dims) - 2):
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(dims[i], dims[i + 1]), dim=None))
            if act:
                layers.append(getattr(nn, act)())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if act:
            layers.append(getattr(nn, act)())
        self.main = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


class BANLayer(nn.Module):
    """Bilinear Attention Network layer.

    Parameters
    ----------
    v_dim : int
        Input dimension of the first modality (protein).
    q_dim : int
        Input dimension of the second modality (ligand).
    h_dim : int
        Hidden dimension of the bilinear interaction.
    h_out : int
        Number of bilinear attention heads.
    act : str
        Activation function name (default: ``'ReLU'``).
    dropout : float
        Dropout rate in FCNet projections.
    k : int
        Rank of the low-rank factorisation.
    """

    def __init__(
        self,
        v_dim: int,
        q_dim: int,
        h_dim: int,
        h_out: int,
        act: str = "ReLU",
        dropout: float = 0.2,
        k: int = 3,
    ) -> None:
        super().__init__()
        self.c = 32
        self.k = k
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.h_dim = h_dim
        self.h_out = h_out

        self.v_net = FCNet([v_dim, h_dim * k], act=act, dropout=dropout)
        self.q_net = FCNet([q_dim, h_dim * k], act=act, dropout=dropout)

        if k > 1:
            self.p_net = nn.AvgPool1d(k, stride=k)

        if h_out <= self.c:
            self.h_mat = nn.Parameter(
                torch.Tensor(1, h_out, 1, h_dim * k).normal_()
            )
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * k, h_out), dim=None)

        self.bn = nn.BatchNorm1d(h_dim)

    def attention_pooling(
        self,
        v: torch.Tensor,
        q: torch.Tensor,
        att_map: torch.Tensor,
    ) -> torch.Tensor:
        """Bilinear attention-weighted pooling.

        Parameters
        ----------
        v : Tensor [B, v_num, h_dim*k]
        q : Tensor [B, q_num, h_dim*k]
        att_map : Tensor [B, v_num, q_num]

        Returns
        -------
        Tensor [B, h_dim]
        """
        fusion_logits = torch.einsum("bvk,bvq,bqk->bk", v, att_map, q)
        if self.k > 1:
            fusion_logits = fusion_logits.unsqueeze(1)  # [B, 1, h_dim*k]
            fusion_logits = self.p_net(fusion_logits).squeeze(1) * self.k
        return fusion_logits

    def forward(
        self,
        v: torch.Tensor,
        q: torch.Tensor,
        softmax: bool = True,
        v_mask: torch.Tensor | None = None,
        q_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute bilinear attention and pooled interaction vector.

        Parameters
        ----------
        v : Tensor [B, v_num, v_dim]
            First modality (protein per-token features).
        q : Tensor [B, q_num, q_dim]
            Second modality (ligand per-token features).
        softmax : bool
            If True, normalise attention maps with softmax.
        v_mask : Tensor [B, v_num] or None
            1 = real token, 0 = padding (for protein).
        q_mask : Tensor [B, q_num] or None
            1 = real token, 0 = padding (for ligand).

        Returns
        -------
        logits : Tensor [B, h_dim]
            Fused interaction vector (bilinear-pooled).
        att_maps : Tensor [B, h_out, v_num, q_num]
            Attention maps per head.
        """
        v_num = v.size(1)
        q_num = q.size(1)

        if self.h_out <= self.c:
            v_ = self.v_net(v)
            q_ = self.q_net(q)
            att_maps = (
                torch.einsum("xhyk,bvk,bqk->bhvq", self.h_mat, v_, q_)
                + self.h_bias
            )
        else:
            v_ = self.v_net(v).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)  # [B, h_dim*k, v, q]
            att_maps = self.h_net(d_.transpose(1, 2).transpose(2, 3))
            att_maps = att_maps.transpose(2, 3).transpose(1, 2)

        # Mask padding positions before softmax
        if v_mask is not None or q_mask is not None:
            mask_2d = torch.ones(v.size(0), v_num, q_num, device=v.device)
            if v_mask is not None:
                mask_2d = mask_2d * v_mask.unsqueeze(2)  # [B, v, 1]
            if q_mask is not None:
                mask_2d = mask_2d * q_mask.unsqueeze(1)  # [B, 1, q]
            att_maps = att_maps.masked_fill(
                mask_2d.unsqueeze(1) == 0, float("-inf")
            )

        if softmax:
            p = torch.softmax(
                att_maps.view(-1, self.h_out, v_num * q_num), dim=2
            )
            att_maps = p.view(-1, self.h_out, v_num, q_num)

        # Recompute projected features for pooling (use the same projections)
        if self.h_out > self.c:
            v_ = self.v_net(v)
            q_ = self.q_net(q)

        logits = self.attention_pooling(v_, q_, att_maps[:, 0, :, :])
        for i in range(1, self.h_out):
            logits = logits + self.attention_pooling(
                v_, q_, att_maps[:, i, :, :]
            )
        if logits.size(0) > 1 or not self.training:
            logits = self.bn(logits)

        return logits, att_maps
