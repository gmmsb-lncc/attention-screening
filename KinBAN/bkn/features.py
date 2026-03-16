"""Protein (ESM-2) and drug (MoLFormer) feature extraction with disk caching."""
from __future__ import annotations

import pickle
import sys
import types
from pathlib import Path

# ---------------------------------------------------------------------------
# Compatibility shims for MoLFormer's cached HuggingFace code vs modern
# transformers (>= 4.40).  MoLFormer's dynamic modules import symbols that
# were renamed or removed.  We patch at module‐level before any transformers
# import so the cached configuration/modeling files don't crash.
# ---------------------------------------------------------------------------
import transformers as _transformers_pkg

# 1) transformers.onnx.OnnxConfig was removed in >= 4.40
if "transformers.onnx" not in sys.modules:
    _onnx_stub = types.ModuleType("transformers.onnx")

    class _OnnxConfig:  # minimal stub — only used as a base class
        pass

    _onnx_stub.OnnxConfig = _OnnxConfig  # type: ignore[attr-defined]
    sys.modules["transformers.onnx"] = _onnx_stub
    _transformers_pkg.onnx = _onnx_stub  # type: ignore[attr-defined]

# 2) find_pruneable_heads_and_indices was renamed / removed in transformers >= 4.45
from transformers import pytorch_utils as _pu

if not hasattr(_pu, "find_pruneable_heads_and_indices"):
    if hasattr(_pu, "find_prunable_heads_and_indices"):
        _pu.find_pruneable_heads_and_indices = _pu.find_prunable_heads_and_indices
    else:
        # Neither name exists — provide the canonical implementation directly
        import torch as _torch

        def _find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):  # type: ignore[no-untyped-def]
            mask = _torch.ones(n_heads, head_size)
            heads = set(heads) - already_pruned_heads
            for head in heads:
                head -= sum(1 if h < head else 0 for h in already_pruned_heads)
                mask[head] = 0
            index = _torch.arange(mask.numel())[mask.view(-1).bool()]
            return heads, index

        _pu.find_pruneable_heads_and_indices = _find_pruneable_heads_and_indices  # type: ignore[attr-defined]

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from . import constants as _constants
from .constants import (
    ESM2_MAX_SEQ_LEN,
    ESM2_MODEL_NAME,
    MOLFORMER_MAX_LEN,
)
from .pooling import cls_guided_attention_pool


def extract_esm2_features(df: pd.DataFrame, device: torch.device) -> pd.DataFrame:
    """Extract ESM-2 650M protein embeddings (1280-d, CLS-guided attention pooling).

    Replaces GraphBAN's ESM-1b mean-pool with:
      - Model: esm2_t33_650M_UR50D (1280-d hidden dim, same as ESM-1b)
      - Pooling: CLS-guided attention pooling over per-residue representations
        (CLS token from layer 33 used as query; residues 1..L as keys/values)

    The result is stored in a new column ``"esm"`` on the returned DataFrame.
    """
    import esm

    print(f"\n  Loading ESM-2 650M ({ESM2_MODEL_NAME})...")
    model, alphabet = getattr(esm.pretrained, ESM2_MODEL_NAME)()
    batch_converter = alphabet.get_batch_converter()
    model = model.eval().to(device)

    df = df.copy()
    df["Protein"] = df["Protein"].apply(
        lambda x: x[:ESM2_MAX_SEQ_LEN] if len(x) > ESM2_MAX_SEQ_LEN else x
    )

    pro_list = df["Protein"].unique()
    print(
        f"  Extracting ESM-2 features for {len(pro_list)} unique proteins "
        f"(CLS-guided attention pooling)..."
    )

    dictionary: dict[str, np.ndarray] = {}
    data_tmp = [(f"protein{i}", p) for i, p in enumerate(pro_list)]

    batch_size = 5
    for i in tqdm(range(0, len(data_tmp), batch_size), desc="  ESM-2"):
        batch = data_tmp[i : i + batch_size]
        if not batch:
            continue
        _, _, batch_tokens = batch_converter(batch)
        batch_tokens = batch_tokens.to(device)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
        token_reps = results["representations"][33]  # [B, L+2, 1280]

        for j, (_, seq) in enumerate(batch):
            # CLS token is at position 0; residues at positions 1..len(seq)
            residues = token_reps[j, 1 : len(seq) + 1]  # [L, 1280]
            cls = token_reps[j, 0]                       # [1280]
            emb = cls_guided_attention_pool(
                residues, cls, dim=1280,
            ).cpu().numpy()
            dictionary[seq] = emb

    esm_df = pd.DataFrame(list(dictionary.items()), columns=["Protein", "esm"])
    df = pd.merge(df, esm_df, on="Protein", how="left")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("  ESM-2 feature extraction complete.")
    return df


def extract_molformer_features(df: pd.DataFrame, device: torch.device) -> pd.DataFrame:
    """Extract MoLFormer-XL drug embeddings (768-d, CLS-guided attention pooling).

    Replaces GraphBAN's ChemBERTa CLS-token with:
      - Model: ibm/MoLFormer-XL-both-10pct (768-d)
      - Pooling: CLS-guided attention pooling over all (non-padding) token reps

    The result is stored in column ``"fcfp"`` (GraphBAN's expected column name).
    """
    from transformers import AutoModel, AutoTokenizer

    # Read at call-time so --molformer-path override in run_bkn.py takes effect
    molformer_name = _constants.MOLFORMER_MODEL_NAME
    print(f"\n  Loading MoLFormer-XL ({molformer_name})...")
    tokenizer = AutoTokenizer.from_pretrained(
        molformer_name, trust_remote_code=True
    )
    model = AutoModel.from_pretrained(molformer_name, trust_remote_code=True)

    # Force float32 — bfloat16/float16 weights can overflow to NaN on some GPUs
    model = model.float().to(device).eval()

    # ── Compatibility shims for MoLFormer dynamic code + transformers >= 4.40 ──

    if not hasattr(type(model), "get_head_mask"):
        def _get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
            if head_mask is not None:
                head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
                if is_attention_chunked is True:
                    head_mask = head_mask.unsqueeze(-1)
            else:
                head_mask = [None] * num_hidden_layers
            return head_mask
        type(model).get_head_mask = _get_head_mask

    if not hasattr(type(model), "_convert_head_mask_to_5d"):
        def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            return head_mask.to(dtype=next(self.parameters()).dtype)
        type(model)._convert_head_mask_to_5d = _convert_head_mask_to_5d

    # ── Validation: quick forward pass WITHOUT attention_mask ──
    # MoLFormer uses linear (Performer) attention — there is no softmax,
    # so passing an additive -inf mask (the standard BERT approach used by
    # get_extended_attention_mask in transformers >= 4.40) propagates NaN
    # through the kernel computations.  The safe strategy is to call
    # model(input_ids=...) only and apply the padding mask manually at
    # pooling time (using the stored attention_mask tensor).
    _test_enc = tokenizer("CCO", return_tensors="pt", max_length=10,
                          padding="max_length", truncation=True)
    _test_ids = _test_enc["input_ids"].to(device)

    # Check weights first
    _n_nan_w = sum(torch.isnan(p).sum().item() for p in model.parameters())
    if _n_nan_w > 0:
        print(f"  ERROR: model weights contain {_n_nan_w} NaN values (corrupted download?)")

    with torch.no_grad():
        _test_out = model(input_ids=_test_ids)          # ← no attention_mask
    _test_ok = not torch.isnan(_test_out.last_hidden_state).any().item()
    if _test_ok:
        print("  MoLFormer validation: forward pass OK (no NaN)")
    else:
        print("  WARNING: MoLFormer forward() still produces NaN!")
        print(f"  Model type: {type(model).__name__}")
        for name, child in model.named_children():
            print(f"    .{name} = {type(child).__name__}")

    df_unique = df.drop_duplicates(subset="SMILES").copy()
    print(
        f"  Extracting MoLFormer features for {len(df_unique)} unique SMILES "
        f"(CLS-guided attention pooling)..."
    )

    emblist: list[np.ndarray] = []
    n_nan_fallback = 0
    for idx, (_, row) in enumerate(tqdm(df_unique.iterrows(), total=len(df_unique), desc="  MoLFormer")):
        encodings = tokenizer(
            row["SMILES"],
            return_tensors="pt",
            padding="max_length",
            max_length=MOLFORMER_MAX_LEN,
            truncation=True,
        )
        input_ids = encodings["input_ids"].to(device)
        mask = encodings["attention_mask"][0]          # [L] — kept for manual pooling
        with torch.no_grad():
            output = model(input_ids=input_ids)        # ← no attention_mask (linear attn)
        token_reps = output.last_hidden_state[0]  # [L, 768]
        cls = token_reps[0]                        # [768] — CLS token

        # Sanity check on first molecule
        if idx == 0:
            has_nan = torch.isnan(token_reps).any().item()
            has_inf = torch.isinf(token_reps).any().item()
            rng_lo = token_reps[~torch.isnan(token_reps)].min().item() if not has_nan else float("nan")
            rng_hi = token_reps[~torch.isnan(token_reps)].max().item() if not has_nan else float("nan")
            print(f"  MoLFormer sanity check (first SMILES): NaN={has_nan}, Inf={has_inf}, "
                  f"shape={list(token_reps.shape)}, range=[{rng_lo:.3f}, {rng_hi:.3f}]")

        emb = cls_guided_attention_pool(
            token_reps, cls, dim=768, mask=mask,
        ).cpu().numpy().astype(np.float64)

        # Fallback: if CLS-guided pool produced NaN, use simple mean pooling
        if np.any(np.isnan(emb)):
            valid = mask.bool().cpu()
            emb = token_reps[valid].mean(dim=0).cpu().numpy().astype(np.float64)
            # If still NaN (model itself outputs NaN), zero-fill
            if np.any(np.isnan(emb)):
                emb = np.zeros(768, dtype=np.float64)
            n_nan_fallback += 1
        emblist.append(emb)

    if n_nan_fallback > 0:
        print(f"  WARNING: {n_nan_fallback}/{len(df_unique)} SMILES produced NaN -> mean-pool fallback used")

    df_unique["fcfp"] = emblist
    df = pd.merge(df, df_unique[["SMILES", "fcfp"]], on="SMILES", how="left")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("  MoLFormer feature extraction complete.")
    return df


def extract_features_cached(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    device: torch.device,
    cache_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Extract ESM-2 + MoLFormer features with disk caching (once for all seeds).

    Concatenates all three splits, extracts features jointly (avoids duplicate
    work for shared proteins/SMILES), then re-splits back to original sizes.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "features_extracted_bkn.pkl"

    if cache_file.exists():
        print("\n  Loading cached BKN features...")
        with open(cache_file, "rb") as f:
            cached_train, cached_val, cached_test = pickle.load(f)
        # Validate cache: reject if 'fcfp'/'esm' columns missing or all NaN
        stale = False
        for col in ("fcfp", "esm"):
            if col not in cached_train.columns:
                print(f"  WARNING: cached features missing '{col}' column — deleting stale cache")
                stale = True
                break
        if not stale:
            sample_emb = cached_train["fcfp"].iloc[0]
            if isinstance(sample_emb, np.ndarray) and np.all(np.isnan(sample_emb)):
                print("  WARNING: cached MoLFormer embeddings are all NaN — deleting stale cache")
                stale = True
        if stale:
            cache_file.unlink()
            # Fall through to re-extraction below using the ORIGINAL
            # train_df/val_df/test_df (from CSV, without stale columns).
        else:
            print("  Cache loaded (validated).")
            return cached_train, cached_val, cached_test

    print("\n  Extracting features (ESM-2 + MoLFormer + AttentionPool)...")
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    all_df = extract_esm2_features(all_df, device)
    all_df = extract_molformer_features(all_df, device)

    n_train = len(train_df)
    n_val = len(val_df)
    train_df = all_df.iloc[:n_train].reset_index(drop=True)
    val_df = all_df.iloc[n_train : n_train + n_val].reset_index(drop=True)
    test_df = all_df.iloc[n_train + n_val :].reset_index(drop=True)

    with open(cache_file, "wb") as f:
        pickle.dump((train_df, val_df, test_df), f, protocol=4)
    print(f"  Features cached to {cache_file}")

    return train_df, val_df, test_df
