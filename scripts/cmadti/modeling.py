"""CMA-DTI architecture operating on cached frozen-encoder token features."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
CMA_ROOT = REPO_ROOT / "CMA-DTI"
sys.path.insert(0, str(CMA_ROOT))

from attention import MultiHeadAttentionLayer  # type: ignore  # noqa: E402
from models import MLPDecoder, MolecularGCN  # type: ignore  # noqa: E402


def masked_mean_pooling(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(values.dtype).unsqueeze(-1)
    return (values * mask).sum(1) / mask.sum(1).clamp_min(1e-6)


class CachedCMA(nn.Module):
    """Official CMA fusion/prediction modules with frozen LM features supplied.

    The upstream graph padding mask is reconstructed from the virtual-node bit
    rather than padded graph node counts (which are always MAX_NODES).
    """

    def __init__(self, config: dict, device: torch.device):
        super().__init__()
        m = config["model"]
        self.device_hint = device
        self.max_drug_nodes = int(m["max_drug_nodes"])
        self.drug_extractor = MolecularGCN(
            in_feats=int(m["drug_node_in_feats"]),
            dim_embedding=int(m["drug_node_embedding"]), padding=True,
            hidden_feats=list(m["drug_hidden_layers"]), max_nodes=self.max_drug_nodes,
        )
        graph_dim = int(m["drug_hidden_layers"][-1])
        chem_dim = int(m["chemberta_feature_dim"])
        protein_dim = int(m["protein_feature_dim"])
        heads = int(m["attention_heads"])
        dropout = float(m["attention_dropout"])
        self.gcn_proj_for_cross_attn = nn.Linear(graph_dim, chem_dim)
        self.cross_attn_gc = MultiHeadAttentionLayer(chem_dim, heads, dropout, device)
        self.fused_nodes_proj_for_protein_attn = nn.Linear(chem_dim, protein_dim)
        self.bcn = MultiHeadAttentionLayer(protein_dim, heads, dropout, device)
        self.mlp_classifier = MLPDecoder(
            protein_dim, int(m["decoder_hidden_dim"]), int(m["decoder_out_dim"]), binary=1
        )

    def forward(self, graph, chem_features, chem_mask, protein_features, protein_mask):
        # Last atom feature is the virtual-node indicator added by DTIDataset.
        real_nodes = (~graph.ndata["h"][:, -1].bool()).view(-1, self.max_drug_nodes)
        graph_nodes = self.drug_extractor(graph)
        graph_proj = self.gcn_proj_for_cross_attn(graph_nodes)
        cross_mask = real_nodes[:, :, None] & chem_mask[:, None, :]
        fused, _ = self.cross_attn_gc(
            graph_proj, chem_features, chem_features, mask=cross_mask[:, None]
        )
        drug_nodes = self.fused_nodes_proj_for_protein_attn(fused)
        interaction_mask = real_nodes[:, :, None] & protein_mask[:, None, :]
        interaction, attention = self.bcn(
            drug_nodes, protein_features, protein_features,
            mask=interaction_mask[:, None],
        )
        pooled = masked_mean_pooling(interaction, real_nodes)
        return self.mlp_classifier(pooled).squeeze(-1), attention
