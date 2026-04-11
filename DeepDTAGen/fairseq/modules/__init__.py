"""fairseq.modules shim — TransformerEncoderLayer & TransformerDecoderLayer.

Reimplements fairseq's transformer layers with separate Q/K/V projections
to match the pretrained checkpoint parameter names exactly:
    self_attn.{k_proj,v_proj,q_proj,out_proj}.{weight,bias}
    encoder_attn.{k_proj,v_proj,q_proj,out_proj}.{weight,bias}
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class MultiheadAttentionSeparateProj(nn.Module):
    """Multi-head attention with separate Q/K/V projections (fairseq-compatible).
    
    Uses separate nn.Linear for k_proj, v_proj, q_proj, out_proj instead of
    PyTorch's monolithic in_proj_weight, matching fairseq checkpoint keys.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim

        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        """
        Args:
            query: (T, B, C)
            key: (S, B, C)
            value: (S, B, C)
            key_padding_mask: (B, S) bool, True = ignore
            attn_mask: (T, S) float mask
        Returns:
            output: (T, B, C)
            attn_weights: None (not needed)
        """
        tgt_len, bsz, _ = query.size()
        src_len = key.size(0)

        q = self.q_proj(query)  # (T, B, C)
        k = self.k_proj(key)    # (S, B, C)
        v = self.v_proj(value)  # (S, B, C)

        # Reshape: (T, B, C) -> (B * num_heads, T, head_dim)
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        # Scaled dot-product attention
        attn_weights = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)

        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask.unsqueeze(0)

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float('-inf')
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.bmm(attn_weights, v)  # (B*H, T, head_dim)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)

        output = self.out_proj(attn_output)
        return output, None


class TransformerEncoderLayer(nn.Module):
    """Pre-norm Transformer encoder layer matching fairseq's parameter names."""

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        num_heads = args.encoder_attention_heads
        ff_dim = args.encoder_ffn_embed_dim
        dropout = getattr(args, 'dropout', 0.1)
        attn_dropout = getattr(args, 'attention_dropout', 0.0)
        self.normalize_before = getattr(args, 'encoder_normalize_before', False)

        self.self_attn = MultiheadAttentionSeparateProj(
            self.embed_dim, num_heads, dropout=attn_dropout
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout_module = nn.Dropout(dropout)

    def forward(self, x, encoder_padding_mask=None):
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(x, x, x, key_padding_mask=encoder_padding_mask)
        x = self.dropout_module(x)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = F.relu(self.fc1(x))
        x = self.dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)

        return x


class TransformerDecoderLayer(nn.Module):
    """Pre-norm Transformer decoder layer matching fairseq's parameter names."""

    def __init__(self, args, no_encoder_attn=False):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        num_heads = args.decoder_attention_heads
        ff_dim = args.decoder_ffn_embed_dim
        dropout = getattr(args, 'dropout', 0.1)
        attn_dropout = getattr(args, 'attention_dropout', 0.0)
        self.normalize_before = getattr(args, 'decoder_normalize_before', False)
        self.no_encoder_attn = no_encoder_attn

        self.self_attn = MultiheadAttentionSeparateProj(
            self.embed_dim, num_heads, dropout=attn_dropout
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        if not no_encoder_attn:
            self.encoder_attn = MultiheadAttentionSeparateProj(
                self.embed_dim, num_heads, dropout=attn_dropout
            )
            self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.fc1 = nn.Linear(self.embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout_module = nn.Dropout(dropout)

    def forward(
        self,
        x,
        encoder_out=None,
        encoder_padding_mask=None,
        incremental_state=None,
        self_attn_mask=None,
        self_attn_padding_mask=None,
    ):
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            x, x, x,
            attn_mask=self_attn_mask,
            key_padding_mask=self_attn_padding_mask,
        )
        x = self.dropout_module(x)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if not self.no_encoder_attn and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            x, _ = self.encoder_attn(
                x, encoder_out, encoder_out,
                key_padding_mask=encoder_padding_mask,
            )
            x = self.dropout_module(x)
            x = residual + x
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = F.relu(self.fc1(x))
        x = self.dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)

        return x, None
