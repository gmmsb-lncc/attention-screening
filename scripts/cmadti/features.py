"""Ragged memmap cache for CMA-DTI's frozen ESM-2/ChemBERTa features."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer


class RaggedStore:
    def __init__(self, root: Path):
        metadata = json.loads((root / "metadata.json").read_text())
        self.keys = metadata["keys"]
        self.index = {key: i for i, key in enumerate(self.keys)}
        self.offsets = np.asarray(metadata["offsets"], dtype=np.int64)
        self.lengths = np.asarray(metadata["lengths"], dtype=np.int64)
        self.dim = int(metadata["dim"])
        self.values = np.load(root / "values.npy", mmap_mode="r")

    def get(self, key: str) -> np.ndarray:
        idx = self.index[str(key)]
        start, length = int(self.offsets[idx]), int(self.lengths[idx])
        return np.asarray(self.values[start:start + length], dtype=np.float32)


def _build_store(
    values: list[str], root: Path, model_name: str, revision: str,
    max_length: int, device: torch.device, batch_size: int, masked_lm: bool,
    keys: list[str] | None = None,
) -> None:
    keys = values if keys is None else keys
    content_sha256 = hashlib.sha256("\0".join(values).encode("utf-8")).hexdigest()
    if (root / "metadata.json").exists() and (root / "values.npy").exists():
        metadata = json.loads((root / "metadata.json").read_text())
        if (metadata.get("keys") != keys or metadata.get("revision") != revision
                or metadata.get("content_sha256") != content_sha256):
            raise ValueError(f"feature cache provenance mismatch: {root}")
        return
    root.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    lengths = []
    for start in range(0, len(values), 2048):
        encoded = tokenizer(values[start:start + 2048], truncation=True,
                            max_length=max_length, add_special_tokens=True)
        lengths.extend(map(len, encoded["input_ids"]))
    offsets = np.cumsum([0] + lengths[:-1], dtype=np.int64)

    model_cls = AutoModelForMaskedLM if masked_lm else AutoModel
    model = model_cls.from_pretrained(model_name, revision=revision).to(device).eval()
    dim = int(model.config.hidden_size)
    mmap = np.lib.format.open_memmap(
        root / "values.npy", mode="w+", dtype=np.float16,
        shape=(int(sum(lengths)), dim),
    )
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            batch_values = values[start:start + batch_size]
            encoded = tokenizer(
                batch_values, padding=True, truncation=True, max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(**encoded, output_hidden_states=masked_lm)
            hidden = (outputs.hidden_states[-1] if masked_lm else outputs.last_hidden_state)
            hidden = hidden.detach().cpu().to(torch.float16).numpy()
            for local in range(len(batch_values)):
                index = start + local
                length = lengths[index]
                offset = int(offsets[index])
                mmap[offset:offset + length] = hidden[local, :length]
            mmap.flush()
    metadata = {
        "model": model_name, "revision": revision, "max_length": max_length,
        "content_sha256": content_sha256,
        "dim": dim, "dtype": "float16", "keys": keys,
        "offsets": offsets.tolist(), "lengths": lengths,
    }
    (root / "metadata.json").write_text(json.dumps(metadata) + "\n")
    del model, mmap
    if device.type == "cuda":
        torch.cuda.empty_cache()


def ensure_feature_cache(frame, cache_root: Path, config: dict, device: torch.device,
                         batch_size: int = 16) -> tuple[RaggedStore, RaggedStore]:
    enc = config["encoders"]
    proteins = frame[["target_id", "Protein"]].drop_duplicates("target_id")
    conflicts = frame.groupby("target_id")["Protein"].nunique(dropna=False)
    if (conflicts > 1).any():
        raise ValueError("same target_id maps to multiple protein sequences")
    protein_keys = proteins["target_id"].astype(str).tolist()
    protein_values = proteins["Protein"].astype(str).tolist()
    smiles_values = frame["SMILES"].astype(str).drop_duplicates().tolist()
    _build_store(
        protein_values, cache_root / "protein", enc["esm_model"], enc["esm_revision"],
        int(enc["esm_max_length"]), device, batch_size, masked_lm=True,
        keys=protein_keys,
    )
    _build_store(
        smiles_values, cache_root / "smiles", enc["chemberta_model"],
        enc["chemberta_revision"], int(enc["chemberta_max_length"]),
        device, batch_size, masked_lm=False,
    )
    return RaggedStore(cache_root / "protein"), RaggedStore(cache_root / "smiles")
