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

# 2) find_pruneable_heads_and_indices was renamed to find_prunable_heads_and_indices
from transformers import pytorch_utils as _pu

if not hasattr(_pu, "find_pruneable_heads_and_indices"):
    if hasattr(_pu, "find_prunable_heads_and_indices"):
        _pu.find_pruneable_heads_and_indices = _pu.find_prunable_heads_and_indices

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .constants import (
    ESM2_MAX_SEQ_LEN,
    ESM2_MODEL_NAME,
    MOLFORMER_MAX_LEN,
    MOLFORMER_MODEL_NAME,
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

    print(f"\n  Loading MoLFormer-XL ({MOLFORMER_MODEL_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(
        MOLFORMER_MODEL_NAME, trust_remote_code=True
    )
    model = AutoModel.from_pretrained(MOLFORMER_MODEL_NAME, trust_remote_code=True)
    model = model.to(device).eval()

    df_unique = df.drop_duplicates(subset="SMILES").copy()
    print(
        f"  Extracting MoLFormer features for {len(df_unique)} unique SMILES "
        f"(CLS-guided attention pooling)..."
    )

    emblist: list[np.ndarray] = []
    for _, row in tqdm(df_unique.iterrows(), total=len(df_unique), desc="  MoLFormer"):
        encodings = tokenizer(
            row["SMILES"],
            return_tensors="pt",
            padding="max_length",
            max_length=MOLFORMER_MAX_LEN,
            truncation=True,
        )
        encodings = {k: v.to(device) for k, v in encodings.items()}
        with torch.no_grad():
            output = model(**encodings)
        token_reps = output.last_hidden_state[0]  # [L, 768]
        mask = encodings["attention_mask"][0]      # [L]
        cls = token_reps[0]                        # [768] — CLS token

        emb = cls_guided_attention_pool(
            token_reps, cls, dim=768, mask=mask,
        ).cpu().numpy().astype(np.float64)
        emblist.append(emb)

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
            train_df, val_df, test_df = pickle.load(f)
        print("  Cache loaded.")
        return train_df, val_df, test_df

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
