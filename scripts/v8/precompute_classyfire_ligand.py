#!/usr/bin/env python3
"""Pre-compute ClassyFire taxonomy (Superclass + Class) as one-hot per compound.

ClassyFire (Feunang et al., 2016, https://cfb.fiehnlab.ucdavis.edu/) is
a curated hierarchical chemical taxonomy. Lookups use the free REST API
with aggressive async concurrency + a resumable SQLite cache.

Two-phase workflow:
  1. Fetch JSON responses for each InChIKey, cache in SQLite.
  2. Build Superclass + Class vocabulary, write one-hot .npy per compound.

Output:
    data/embeddings/v8/classyfire_{corpus}/{chembl_id}.npy  (shape ~(D,), float32)
    data/embeddings/v8/classyfire_{corpus}.vocab.json       (vocab for reload)

Usage:
    python3 scripts/v8/precompute_classyfire_ligand.py --corpus non_human
    python3 scripts/v8/precompute_classyfire_ligand.py --corpus non_human --phase build

Set CLASSYFIRE_URL env var to override the default endpoint (e.g. local mirror).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CLASSYFIRE_URL = os.environ.get(
    "CLASSYFIRE_URL", "https://cfb.fiehnlab.ucdavis.edu/entities/"
)
MAX_CONCURRENT = int(os.environ.get("CLASSYFIRE_CONCURRENCY", "32"))

_CORPUS_TSV = {
    "human":     REPO / "scaffolds_splits/output/human",
    "non_human": REPO / "scaffolds_splits/output/non_human",
    "all":       REPO / "scaffolds_splits/output/universal",
}


def _load_unique(corpus: str) -> pd.DataFrame:
    stem = _CORPUS_TSV[corpus]
    frames = []
    for split in ("train", "val", "test"):
        path = Path(f"{stem}_{split}.tsv")
        if not path.exists():
            continue
        frames.append(pd.read_csv(path, sep="\t",
                                   usecols=["chembl_id", "canonical_smiles"]))
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["chembl_id"]).reset_index(drop=True)


def _smiles_to_inchikey(smiles: str) -> str | None:
    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import inchi as _inchi  # type: ignore
    except ImportError:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return _inchi.MolToInchiKey(mol)


# ---------------------- Phase 1: async fetch ---------------------------


async def _fetch_one(session, inchikey: str, semaphore) -> tuple[str, str | None]:
    url = f"{CLASSYFIRE_URL}{inchikey}.json"
    async with semaphore:
        try:
            async with session.get(url, timeout=30) as resp:
                if resp.status != 200:
                    return inchikey, None
                return inchikey, await resp.text()
        except Exception:
            return inchikey, None


async def _fetch_all(inchikeys: list[str], db_path: Path) -> None:
    try:
        import aiohttp  # type: ignore
    except ImportError:
        print("[fatal] aiohttp not installed (pip install aiohttp)", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE IF NOT EXISTS cf (inchikey TEXT PRIMARY KEY, payload TEXT)")
    existing = {r[0] for r in conn.execute("SELECT inchikey FROM cf")}
    todo = [k for k in inchikeys if k not in existing]
    print(f"  {len(todo)} inchikeys faltam (skip {len(existing)} cached)")
    if not todo:
        conn.close(); return

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    connector = __import__("aiohttp").TCPConnector(limit=MAX_CONCURRENT)
    async with __import__("aiohttp").ClientSession(connector=connector) as session:
        batch = 256
        t0 = time.time()
        done = 0
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            results = await asyncio.gather(*(_fetch_one(session, k, semaphore) for k in chunk))
            conn.executemany("INSERT OR REPLACE INTO cf VALUES (?, ?)", results)
            conn.commit()
            done += len(chunk)
            rate = done / max(time.time() - t0, 1e-6)
            print(f"  {done}/{len(todo)} fetched ({rate:.1f} req/s)", flush=True)
    conn.close()


# ---------------------- Phase 2: build OHE ----------------------------


def _extract_classes(payload: str | None) -> dict[str, str | None]:
    if payload is None:
        return {"superclass": None, "class": None}
    try:
        d = json.loads(payload)
    except Exception:
        return {"superclass": None, "class": None}
    sc = (d.get("superclass") or {}).get("name")
    cl = (d.get("class") or {}).get("name")
    return {"superclass": sc, "class": cl}


def _build_vocab(classes: list[dict]) -> list[str]:
    vocab: set[str] = set()
    for c in classes:
        for key in ("superclass", "class"):
            v = c.get(key)
            if v:
                vocab.add(f"{key}:{v}")
    return sorted(vocab)


def _encode(c: dict, vocab: list[str]) -> np.ndarray:
    idx = {v: i for i, v in enumerate(vocab)}
    vec = np.zeros(len(vocab), dtype=np.float32)
    for key in ("superclass", "class"):
        v = c.get(key)
        if v and f"{key}:{v}" in idx:
            vec[idx[f"{key}:{v}"]] = 1.0
    return vec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=list(_CORPUS_TSV))
    ap.add_argument("--output-root", default="data/embeddings/v8")
    ap.add_argument("--phase", default="both", choices=["fetch", "build", "both"])
    args = ap.parse_args()

    out_dir = REPO / args.output_root / f"classyfire_{args.corpus}"
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir.parent / f"classyfire_{args.corpus}.sqlite"
    vocab_path = out_dir.parent / f"classyfire_{args.corpus}.vocab.json"

    df = _load_unique(args.corpus)
    print(f"[classyfire/{args.corpus}] {len(df)} compostos únicos")

    # Map chembl_id → inchikey (idempotent; rdkit call is deterministic)
    chid_to_ik: dict[str, str] = {}
    for _, row in df.iterrows():
        ik = _smiles_to_inchikey(row.canonical_smiles)
        if ik is not None:
            chid_to_ik[row.chembl_id] = ik

    if args.phase in ("fetch", "both"):
        asyncio.run(_fetch_all(sorted(set(chid_to_ik.values())), db_path))

    if args.phase in ("build", "both"):
        conn = sqlite3.connect(db_path)
        rows = {r[0]: r[1] for r in conn.execute("SELECT inchikey, payload FROM cf")}
        conn.close()
        # Parse classes for every (chembl_id)
        parsed = {cid: _extract_classes(rows.get(ik)) for cid, ik in chid_to_ik.items()}
        vocab = _build_vocab(list(parsed.values()))
        vocab_path.write_text(json.dumps(vocab))
        print(f"  vocab size: {len(vocab)}  → {vocab_path}")

        for cid, classes in parsed.items():
            vec = _encode(classes, vocab)
            np.save(out_dir / f"{cid}.npy", vec)
        # Fill compounds without InChIKey with zero vectors
        for cid in df["chembl_id"]:
            path = out_dir / f"{cid}.npy"
            if not path.exists():
                np.save(path, np.zeros(len(vocab), dtype=np.float32))
        print(f"[done] cache → {out_dir}")


if __name__ == "__main__":
    main()
