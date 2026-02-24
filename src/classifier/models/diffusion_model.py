"""
Diffusion-based affinity model for protein-ligand classification/regression.

Uses a lightweight denoising diffusion encoder per matrix and attention pooling
to preserve token-level information before classification.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseClassifier
from .cross_attention_model import CrossAttentionBlock


class AttentionPool(nn.Module):
    """Learned attention pooling over sequence dimension."""

    def __init__(self, hidden_dim: int, num_queries: int = 1):
        super().__init__()
        self.num_queries = num_queries
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.query = nn.Parameter(torch.randn(num_queries, hidden_dim))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, hidden_dim]
            mask: [batch, seq_len] with 1 for valid tokens
        Returns:
            pooled: [batch, hidden_dim]
        """
        scores = torch.matmul(self.proj(x), self.query.t())  # [batch, seq_len, num_queries]
        if mask is not None:
            scores = scores.masked_fill(~mask.bool().unsqueeze(-1), float("-inf"))
        attn = torch.softmax(scores, dim=1)  # [batch, seq_len, num_queries]
        pooled = torch.einsum("bsq,bsh->bqh", attn, x)  # [batch, num_queries, hidden_dim]
        return pooled.mean(dim=1)


class DiffusionDenoiser(nn.Module):
    """Transformer-based denoiser with time embeddings."""

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        num_timesteps: int,
    ):
        super().__init__()
        self.time_embed = nn.Embedding(num_timesteps, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        time_emb = self.time_embed(timesteps).unsqueeze(1)  # [batch, 1, hidden_dim]
        h = self.input_norm(x + time_emb)
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask.bool()  # True for padding
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        out = self.out(h)
        if mask is not None:
            out = out * mask.unsqueeze(-1).to(out.dtype)
        return out


def _sinusoidal_position_encoding(
    seq_len: int,
    hidden_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create sinusoidal positional encodings [1, seq_len, hidden_dim]."""
    if seq_len <= 0:
        return torch.zeros(1, 0, hidden_dim, device=device, dtype=dtype)
    position = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, hidden_dim, 2, device=device, dtype=dtype)
        * (-torch.log(torch.tensor(10000.0, device=device, dtype=dtype)) / hidden_dim)
    )
    pe = torch.zeros(seq_len, hidden_dim, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)


class MultiTaskHead(nn.Module):
    """Shared head for classification and regression."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        classification_only: bool = False,
    ):
        super().__init__()
        self.classification_only = classification_only
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        if not classification_only:
            self.regressor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )
        else:
            self.regressor = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shared = self.shared(x)
        cls = self.classifier(shared)
        if self.regressor is None:
            reg = torch.zeros_like(cls)
        else:
            reg = self.regressor(shared)
        return cls, reg


class DiffusionAffinityModel(BaseClassifier):
    """
    Diffusion-based model for protein-ligand affinity prediction.

    Uses a denoising diffusion encoder on protein and ligand matrices, followed by
    attention pooling and a multi-task head.
    """

    def __init__(
        self,
        protein_dim: int,
        ligand_dim: int,
        hidden_dim: int = 256,
        num_diffusion_layers: int = 4,
        num_cross_attn_layers: int = 1,
        num_heads: int = 8,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        pool_num_queries: int = 4,
        diffusion_steps: int = 200,
        diffusion_beta_start: float = 1e-4,
        diffusion_beta_end: float = 0.02,
        diffusion_loss_weight: float = 0.1,
        snr_sampling_gamma: float = 0.5,
        snr_sampling_mix: float = 0.2,
        classification_only: bool = False,
    ):
        super().__init__(input_dim=hidden_dim * 2)
        self.hidden_dim = hidden_dim
        self.diffusion_steps = diffusion_steps
        self.diffusion_loss_weight = diffusion_loss_weight
        self.classification_only = classification_only
        self.snr_sampling_gamma = snr_sampling_gamma
        self.snr_sampling_mix = snr_sampling_mix

        self.protein_proj = nn.Linear(protein_dim, hidden_dim)
        self.ligand_proj = nn.Linear(ligand_dim, hidden_dim)
        self.protein_norm = nn.LayerNorm(hidden_dim)
        self.ligand_norm = nn.LayerNorm(hidden_dim)
        self.pos_scale_protein = nn.Parameter(torch.tensor(1.0))
        self.pos_scale_ligand = nn.Parameter(torch.tensor(1.0))

        self.protein_denoiser = DiffusionDenoiser(
            hidden_dim=hidden_dim,
            num_layers=num_diffusion_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            num_timesteps=diffusion_steps,
        )
        self.ligand_denoiser = DiffusionDenoiser(
            hidden_dim=hidden_dim,
            num_layers=num_diffusion_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            num_timesteps=diffusion_steps,
        )

        self.protein_pool = AttentionPool(hidden_dim, num_queries=pool_num_queries)
        self.ligand_pool = AttentionPool(hidden_dim, num_queries=pool_num_queries)
        self.task_head = MultiTaskHead(
            hidden_dim * 2,
            hidden_dim,
            dropout,
            classification_only=classification_only,
        )
        self.cross_attn_blocks = nn.ModuleList(
            [
                CrossAttentionBlock(hidden_dim, num_heads, ff_dim, dropout)
                for _ in range(max(0, int(num_cross_attn_layers)))
            ]
        )

        betas = torch.linspace(diffusion_beta_start, diffusion_beta_end, diffusion_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))
        self.register_buffer("snr_sampling_probs", self._build_snr_sampling_probs())

    def _build_snr_sampling_probs(self) -> torch.Tensor:
        """Precompute SNR-biased timestep sampling probabilities."""
        snr = self.alpha_bars / (1.0 - self.alpha_bars).clamp_min(1e-6)
        probs = snr.pow(self.snr_sampling_gamma)
        probs = probs / probs.sum()
        uniform = torch.ones_like(probs) / probs.numel()
        mix = float(self.snr_sampling_mix)
        probs = (1.0 - mix) * probs + mix * uniform
        return probs

    def _q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_ab = self.sqrt_alpha_bars[t].view(-1, 1, 1)
        sqrt_omb = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1)
        return sqrt_ab * x0 + sqrt_omb * noise

    def _predict_x0(self, xt: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        sqrt_ab = self.sqrt_alpha_bars[t].view(-1, 1, 1)
        sqrt_omb = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1)
        return (xt - sqrt_omb * eps) / sqrt_ab.clamp_min(1e-6)

    def _snr_weight(self, t: torch.Tensor) -> torch.Tensor:
        """Compute per-sample SNR weight for diffusion loss."""
        alpha_bar = self.alpha_bars[t].view(-1, 1, 1)
        snr = alpha_bar / (1.0 - alpha_bar).clamp_min(1e-6)
        return torch.log1p(snr).clamp(max=10.0)

    @staticmethod
    def _masked_mse(
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor],
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is None:
            if weights is None:
                return F.mse_loss(pred, target)
            diff = (pred - target).pow(2)
            return (diff * weights).mean()
        mask = mask.unsqueeze(-1).to(pred.dtype)
        diff = (pred - target).pow(2)
        if weights is not None:
            diff = diff * weights
        diff = diff * mask
        denom = mask.sum().clamp_min(1.0)
        return diff.sum() / denom

    def forward(
        self,
        protein_matrix: torch.Tensor,
        ligand_matrix: torch.Tensor,
        protein_mask: Optional[torch.Tensor] = None,
        ligand_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        batch_size = protein_matrix.size(0)
        device = protein_matrix.device

        protein = self.protein_norm(self.protein_proj(protein_matrix))
        ligand = self.ligand_norm(self.ligand_proj(ligand_matrix))
        protein = protein + self.pos_scale_protein * _sinusoidal_position_encoding(
            protein.size(1), self.hidden_dim, protein.device, protein.dtype
        )
        ligand = ligand + self.pos_scale_ligand * _sinusoidal_position_encoding(
            ligand.size(1), self.hidden_dim, ligand.device, ligand.dtype
        )

        if self.training:
            timesteps = torch.multinomial(
                self.snr_sampling_probs,
                num_samples=batch_size,
                replacement=True,
            ).to(device)
            noise_p = torch.randn_like(protein)
            noise_l = torch.randn_like(ligand)
            protein_t = self._q_sample(protein, timesteps, noise_p)
            ligand_t = self._q_sample(ligand, timesteps, noise_l)

            eps_p = self.protein_denoiser(protein_t, protein_mask, timesteps)
            eps_l = self.ligand_denoiser(ligand_t, ligand_mask, timesteps)

            snr_weight = self._snr_weight(timesteps)
            diffusion_loss = self._masked_mse(eps_p, noise_p, protein_mask, snr_weight)
            diffusion_loss = diffusion_loss + self._masked_mse(eps_l, noise_l, ligand_mask, snr_weight)

            protein_hat = self._predict_x0(protein_t, timesteps, eps_p)
            ligand_hat = self._predict_x0(ligand_t, timesteps, eps_l)
        else:
            timesteps = torch.zeros(batch_size, dtype=torch.long, device=device)
            eps_p = self.protein_denoiser(protein, protein_mask, timesteps)
            eps_l = self.ligand_denoiser(ligand, ligand_mask, timesteps)
            protein_hat = self._predict_x0(protein, timesteps, eps_p)
            ligand_hat = self._predict_x0(ligand, timesteps, eps_l)
            diffusion_loss = None

        for block in self.cross_attn_blocks:
            protein_hat, ligand_hat = block(
                protein_hat, ligand_hat, protein_mask, ligand_mask
            )

        protein_pooled = self.protein_pool(protein_hat, protein_mask)
        ligand_pooled = self.ligand_pool(ligand_hat, ligand_mask)
        combined = torch.cat([protein_pooled, ligand_pooled], dim=-1)

        cls_logits, reg_output = self.task_head(combined)
        result = {
            "classification": cls_logits,
            "regression": reg_output,
        }

        if diffusion_loss is not None:
            result["aux_loss"] = diffusion_loss * self.diffusion_loss_weight

        if return_attention:
            result["protein_repr"] = protein_pooled
            result["ligand_repr"] = ligand_pooled

        return result

    def forward_classification(
        self,
        protein_matrix: torch.Tensor,
        ligand_matrix: torch.Tensor,
        protein_mask: Optional[torch.Tensor] = None,
        ligand_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output = self.forward(protein_matrix, ligand_matrix, protein_mask, ligand_mask)
        return output["classification"]

    def get_architecture_info(self) -> Dict[str, Any]:
        return {
            "model_type": "DiffusionAffinityModel",
            "hidden_dim": self.hidden_dim,
            "diffusion_steps": self.diffusion_steps,
            "diffusion_loss_weight": self.diffusion_loss_weight,
            "diffusion_cross_attn_layers": len(self.cross_attn_blocks),
            "separate_denoisers": True,
            "positional_encoding": "sinusoidal",
            "classification_only": self.classification_only,
            "snr_sampling_gamma": self.snr_sampling_gamma,
            "snr_sampling_mix": self.snr_sampling_mix,
        }
