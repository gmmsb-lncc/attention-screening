"""Shared on-the-fly encoders for inference: ESM-2 8M + MoLFormer.

Wraps:
    - facebookresearch/esm pretrained.load_model_and_alphabet("esm2_t6_8M_UR50D")
    - HuggingFace ibm/MoLFormer-XL-both-10pct via AutoModel + AutoTokenizer

Both produce per-token matrices (no pooling) consistent with the cached
embeddings the DT-Kinase v7+F training pipeline expects: protein matrix
[seq_len, 320] (BOS/EOS stripped); ligand matrix [tok_len, 768] (CLS/SEP
preserved as in training cache).

Optional disk cache (data/reference/cache/{esm2_8m,molformer}/<id>.npy)
keyed by uniprot for proteins and by canonical SMILES SHA1 hash for
ligands. Cache hits skip re-encoding.

Usage:
    from scripts.inference.encoders import (
        load_esm2_8m, encode_proteins,
        load_molformer, encode_ligands,
    )
    model_p, alpha_p = load_esm2_8m(device)
    prot_mats = encode_proteins(model_p, alpha_p, sequences, device, cache_dir=...)
"""
from __future__ import annotations

import gc
import hashlib
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "reference" / "cache"

ESM2_MODEL_NAME = "esm2_t6_8M_UR50D"
ESM2_DIM = 320
ESM2_MAX_LEN = 1024

MOLFORMER_HF_NAME = "ibm/MoLFormer-XL-both-10pct"
MOLFORMER_DIM = 768
MOLFORMER_MAX_LEN = 512


# ======================================================================
# Cache helpers
# ======================================================================

def _ligand_cache_key(smiles: str) -> str:
    """Stable cache filename for a SMILES (sha1 hex of canonical string)."""
    return hashlib.sha1(smiles.encode("utf-8")).hexdigest()[:16]


def _load_cached(cache_dir: Path, key: str) -> np.ndarray | None:
    if cache_dir is None:
        return None
    p = cache_dir / f"{key}.npy"
    if p.exists():
        return np.load(p)
    return None


def _save_cached(cache_dir: Path, key: str, mat: np.ndarray) -> None:
    if cache_dir is None:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / f"{key}.npy", mat.astype(np.float32))


# ======================================================================
# ESM-2 8M (proteins)
# ======================================================================

def load_esm2_8m(device: torch.device):
    """Return (model, alphabet) for ESM-2 8M from local llm/ESM clone."""
    sys.path.insert(0, str(REPO_ROOT / "llm" / "ESM"))
    import esm  # type: ignore
    model, alphabet = esm.pretrained.load_model_and_alphabet(ESM2_MODEL_NAME)
    model = model.to(device).eval()
    return model, alphabet


@torch.inference_mode()
def _encode_protein_one(
    model, alphabet, sequence: str, device: torch.device,
) -> np.ndarray:
    """Encode a single sequence; returns [seq_len, 320] matrix (no BOS/EOS)."""
    valid = set("ACDEFGHIKLMNPQRSTVWY")
    seq = "".join(c for c in sequence.upper() if c in valid)
    if len(seq) == 0:
        raise ValueError("empty/invalid sequence")
    if len(seq) > ESM2_MAX_LEN:
        seq = seq[:ESM2_MAX_LEN]

    converter = alphabet.get_batch_converter()
    _, _, tokens = converter([("seq", seq)])
    tokens = tokens.to(device)
    out = model(tokens, repr_layers=[model.num_layers], return_contacts=False)
    rep = out["representations"][model.num_layers][0, 1:-1]  # strip BOS/EOS
    return rep.float().cpu().numpy()


def encode_proteins(
    model, alphabet,
    items: Iterable[tuple[str, str]],   # (uniprot_or_id, sequence)
    device: torch.device,
    cache_dir: Path | None = None,
) -> dict[str, np.ndarray]:
    """Encode many proteins; cache by uniprot/id; returns {id: matrix}."""
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR / "esm2_8m"
    out: dict[str, np.ndarray] = {}
    for pid, seq in items:
        cached = _load_cached(cache_dir, pid)
        if cached is not None:
            out[pid] = cached
            continue
        mat = _encode_protein_one(model, alphabet, seq, device)
        _save_cached(cache_dir, pid, mat)
        out[pid] = mat
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return out


# ======================================================================
# MoLFormer (ligands)
# ======================================================================

def load_molformer(device: torch.device):
    """Return (model, tokenizer) for MoLFormer-XL-both-10pct."""
    from transformers import AutoModel, AutoTokenizer  # lazy
    tokenizer = AutoTokenizer.from_pretrained(
        MOLFORMER_HF_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MOLFORMER_HF_NAME, trust_remote_code=True)
    model = model.to(device).eval()
    return model, tokenizer


@torch.inference_mode()
def _encode_ligand_batch(
    model, tokenizer, smiles_batch: list[str], device: torch.device,
    max_length: int = MOLFORMER_MAX_LEN,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode a batch; returns (hidden [B, L, 768], attn_mask [B, L])."""
    enc = tokenizer(
        smiles_batch, max_length=max_length, truncation=True,
        padding=True, return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    out = model(input_ids=input_ids, attention_mask=attn)
    hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    return hidden.float().cpu().numpy(), attn.cpu().numpy()


def encode_ligands(
    model, tokenizer,
    items: Iterable[tuple[str, str]],   # (chembl_id, canonical_smiles)
    device: torch.device,
    cache_dir: Path | None = None,
    batch_size: int = 64,
) -> dict[str, np.ndarray]:
    """Encode many SMILES; cache by sha1 hash of canonical SMILES.

    Returns {chembl_id: matrix [tok_len, 768]} with padding tokens stripped
    via the attention mask (matrix length = number of valid tokens).
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR / "molformer"
    items = list(items)

    # Cache hits first
    out: dict[str, np.ndarray] = {}
    pending: list[tuple[str, str]] = []
    for cid, smi in items:
        cached = _load_cached(cache_dir, _ligand_cache_key(smi))
        if cached is not None:
            out[cid] = cached
        else:
            pending.append((cid, smi))

    # Encode in batches
    for i in range(0, len(pending), batch_size):
        batch = pending[i : i + batch_size]
        cids, smis = zip(*batch)
        hidden, attn = _encode_ligand_batch(model, tokenizer, list(smis), device)
        for j, (cid, smi) in enumerate(zip(cids, smis)):
            n_valid = int(attn[j].sum())
            mat = hidden[j, :n_valid].astype(np.float32)  # strip padding
            _save_cached(cache_dir, _ligand_cache_key(smi), mat)
            out[cid] = mat

    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return out
