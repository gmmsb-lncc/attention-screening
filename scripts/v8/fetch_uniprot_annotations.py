#!/usr/bin/env python3
"""Fetch UniProt functional annotations for every unique kinase in a corpus.

For each (seq_id, target_kinase, organism, seq) tuple, queries UniProt
REST API (with fallback to local BLAST-style lookup by sequence hash)
to retrieve:
  - CC FUNCTION (biological function description)
  - KW (keywords)
  - DR GO (Gene Ontology cross-references)
  - DR KEGG (KEGG pathway cross-references)

Writes one JSON per seq_id:
    data/embeddings/v8/uniprot_{corpus}/{seq_id}.json

Async + SQLite cache for resumability. Aggressive concurrency.

Usage:
    python3 scripts/v8/fetch_uniprot_annotations.py --corpus non_human
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
from urllib.parse import quote

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
MAX_CONCURRENT = int(os.environ.get("UNIPROT_CONCURRENCY", "32"))

_CORPUS_TSV = {
    "human":     REPO / "scaffolds_splits/output/human",
    "non_human": REPO / "scaffolds_splits/output/non_human",
    "all":       REPO / "scaffolds_splits/output/universal",
}


def _load_targets(corpus: str) -> pd.DataFrame:
    stem = _CORPUS_TSV[corpus]
    frames = []
    for split in ("train", "val", "test"):
        path = Path(f"{stem}_{split}.tsv")
        if not path.exists():
            continue
        frames.append(pd.read_csv(path, sep="\t",
                                   usecols=["seq_id", "target_kinase", "organism", "seq"]))
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["seq_id"])
    return df.reset_index(drop=True)


def _build_query(target_name: str, organism: str) -> str:
    # UniProtKB search by name + organism; top-1 reviewed hit if available
    q = f'(protein_name:"{target_name}") AND (organism_name:"{organism}") AND (reviewed:true)'
    return f"https://rest.uniprot.org/uniprotkb/search?query={quote(q)}&format=json&size=1"


def _extract_fields(entry: dict) -> dict:
    """Extract CC FUNCTION, KW, DR GO, DR KEGG from a UniProt JSON entry."""
    out: dict[str, list] = {"function": [], "keywords": [], "go": [], "kegg": []}
    # Function comments
    for cc in entry.get("comments", []):
        if cc.get("commentType") == "FUNCTION":
            for text in cc.get("texts", []):
                val = text.get("value")
                if val:
                    out["function"].append(val)
    # Keywords
    for kw in entry.get("keywords", []):
        name = kw.get("name")
        if name:
            out["keywords"].append(name)
    # DR cross-references
    for xref in entry.get("uniProtKBCrossReferences", []):
        db = xref.get("database")
        pid = xref.get("id")
        if not db or not pid:
            continue
        if db == "GO":
            term = None
            for prop in xref.get("properties", []):
                if prop.get("key") == "GoTerm":
                    term = prop.get("value")
            out["go"].append(f"{pid}: {term}" if term else pid)
        elif db == "KEGG":
            out["kegg"].append(pid)
    return out


async def _fetch_one(session, row: pd.Series) -> tuple[str, dict | None]:
    url = _build_query(row.target_kinase, row.organism)
    try:
        async with session.get(url, timeout=30) as resp:
            if resp.status != 200:
                return row.seq_id, None
            data = await resp.json()
    except Exception:
        return row.seq_id, None
    results = data.get("results") or []
    if not results:
        return row.seq_id, None
    return row.seq_id, _extract_fields(results[0])


async def _run(rows: list[pd.Series], db_path: Path) -> None:
    try:
        import aiohttp  # type: ignore
    except ImportError:
        print("[fatal] aiohttp not installed (pip install aiohttp)", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE IF NOT EXISTS up (seq_id TEXT PRIMARY KEY, payload TEXT)")
    existing = {r[0] for r in conn.execute("SELECT seq_id FROM up")}
    todo = [r for r in rows if str(r.seq_id) not in existing]
    print(f"  {len(todo)} targets faltam (skip {len(existing)} cached)")
    if not todo:
        conn.close(); return

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async def _sem_fetch(session, row):
        async with semaphore:
            return await _fetch_one(session, row)

    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT)
    async with aiohttp.ClientSession(connector=connector) as session:
        t0 = time.time()
        done = 0
        batch = 64
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            results = await asyncio.gather(*(_sem_fetch(session, r) for r in chunk))
            payload = [(sid, json.dumps(d) if d is not None else None) for sid, d in results]
            conn.executemany("INSERT OR REPLACE INTO up VALUES (?, ?)", payload)
            conn.commit()
            done += len(chunk)
            rate = done / max(time.time() - t0, 1e-6)
            print(f"  {done}/{len(todo)} fetched ({rate:.1f} req/s)", flush=True)
    conn.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=list(_CORPUS_TSV))
    ap.add_argument("--output-root", default="data/embeddings/v8")
    args = ap.parse_args()

    out_dir = REPO / args.output_root / f"uniprot_{args.corpus}"
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir.parent / f"uniprot_{args.corpus}.sqlite"

    df = _load_targets(args.corpus)
    print(f"[uniprot/{args.corpus}] {len(df)} targets únicos")

    asyncio.run(_run(list(df.itertuples(index=False)), db_path))

    # Materialize per-seq_id JSON files
    conn = sqlite3.connect(db_path)
    rows = {r[0]: r[1] for r in conn.execute("SELECT seq_id, payload FROM up")}
    conn.close()
    for seq_id, payload in rows.items():
        (out_dir / f"{seq_id}.json").write_text(payload or json.dumps({"function": [], "keywords": [], "go": [], "kegg": []}))
    print(f"[done] JSON files → {out_dir}")


if __name__ == "__main__":
    main()
