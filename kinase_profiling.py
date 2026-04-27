#!/usr/bin/env python3
"""kinase_profiling: single-command kinase-ligand interaction profiling.

Auto-detects input type and runs the appropriate inference pipeline:

  python kinase_profiling.py "CC(=O)Oc1ccccc1C(=O)O"      → SMILES vs human kinome
  python kinase_profiling.py protein.fa                   → FASTA vs ligand library
  python kinase_profiling.py batch.csv                    → batch (auto-detects which
                                                             column is SMILES vs sequence)
  python kinase_profiling.py pairs.tsv                    → idem (TSV)

Detection rules:
  1. If input is a path to an existing file:
       - File starts with '>' OR extension in {.fa, .fasta, .faa}     → FASTA
       - File extension in {.csv, .tsv, .txt}                          → batch
     Batch column order does not matter: the script tries to parse each
     column with RDKit; the column that parses as valid SMILES is taken
     as ligand, the other column is taken as protein sequence.
  2. Otherwise input is treated as an inline SMILES string. The string
     must parse via RDKit (Chem.MolFromSmiles).

Output directory:
  results/inference/<run_id>/
where <run_id> = kinase_profiling_<timestamp>_<input_hash>

Usage examples:
  python kinase_profiling.py "CC(=O)Oc1ccccc1C(=O)O"
  python kinase_profiling.py my_protein.fa --organism human --top-k 20
  python kinase_profiling.py pairs.csv --out custom/path
  python kinase_profiling.py CHEMBL941_inputs.txt --models dtkinase,drugban
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
INFER_DIR = REPO / "scripts" / "inference"


# ======================================================================
# Input detection
# ======================================================================

class InputType:
    SMILES_STRING = "smiles_string"
    FASTA_FILE    = "fasta_file"
    BATCH_FILE    = "batch_file"


def detect_input(arg: str) -> tuple[str, dict]:
    """Return (input_type, parsed_payload).

    Payload dict keys per type:
      SMILES_STRING:  {"smiles": str}
      FASTA_FILE:     {"path": Path, "n_records": int}
      BATCH_FILE:     {"path": Path, "smiles_col": str, "seq_col": str,
                       "delimiter": str, "n_rows": int}
    """
    # Lazy-import RDKit (heavy, only needed for SMILES validation)
    from rdkit import Chem

    p = Path(arg)
    if p.exists() and p.is_file():
        return _detect_from_file(p, Chem)

    mol = Chem.MolFromSmiles(arg)
    if mol is not None:
        return InputType.SMILES_STRING, {"smiles": Chem.MolToSmiles(mol, canonical=True)}

    raise SystemExit(
        f"Cannot interpret input: {arg!r}\n"
        f"  - Not a valid path to an existing file\n"
        f"  - Not a valid SMILES string parseable by RDKit\n"
        f"Try one of:\n"
        f"  python kinase_profiling.py 'CC(=O)Oc1ccccc1C(=O)O'   # SMILES\n"
        f"  python kinase_profiling.py protein.fa                # FASTA file\n"
        f"  python kinase_profiling.py pairs.csv                 # batch file\n"
    )


def _detect_from_file(path: Path, Chem) -> tuple[str, dict]:
    """Inspect file head + extension to classify."""
    with open(path) as fh:
        head = fh.read(4096).lstrip()

    # FASTA: starts with '>' or extension hints
    if head.startswith(">") or path.suffix.lower() in {".fa", ".fasta", ".faa"}:
        n = sum(1 for line in open(path) if line.startswith(">"))
        if n == 0:
            raise SystemExit(f"file {path} looks like FASTA but has no '>' headers")
        return InputType.FASTA_FILE, {"path": path, "n_records": n}

    # Batch file: CSV/TSV/TXT — sniff delimiter, find SMILES column
    if path.suffix.lower() in {".csv", ".tsv", ".txt"} or _is_tabular(head):
        return _detect_batch_columns(path, Chem)

    raise SystemExit(
        f"cannot classify file {path}: extension {path.suffix!r} unknown.\n"
        f"  Supported: .fa/.fasta/.faa (FASTA), .csv/.tsv/.txt (batch)"
    )


def _is_tabular(head: str) -> bool:
    """Heuristic: at least 2 newline-separated lines with consistent delimiter."""
    lines = [l for l in head.split("\n") if l.strip()][:5]
    if len(lines) < 2:
        return False
    for delim in ("\t", ","):
        n = lines[0].count(delim)
        if n >= 1 and all(line.count(delim) == n for line in lines[1:]):
            return True
    return False


def _sniff_delimiter(path: Path) -> str:
    """Return '\\t' for TSV, ',' for CSV; sniffs from first non-empty line."""
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if "\t" in line:
                return "\t"
            return ","
    return ","


def _detect_batch_columns(path: Path, Chem) -> tuple[str, dict]:
    """Auto-detect which column holds SMILES vs sequence.

    Strategy: read first ~50 data rows; for each candidate column try
    RDKit parse on every value. The column with the highest fraction of
    valid SMILES is the ligand column; the other column with text-like
    content is the sequence column.

    Header detection is robust to unknown column names: if the first
    row's column scoring matches the rest of the file (high fraction
    parse as SMILES), treat as data; if the first row has zero parseable
    SMILES while subsequent rows do, treat as header.
    """
    delim = _sniff_delimiter(path)
    with open(path, newline="") as fh:
        reader = csv.reader(fh, delimiter=delim)
        rows = list(reader)
    if not rows:
        raise SystemExit(f"{path} is empty")

    # Two-stage header detection:
    #   1. Known column names hint
    #   2. RDKit-parse heuristic: if no cell in row 0 parses as SMILES but
    #      cells in later rows do, row 0 is likely a header.
    has_header = _looks_like_header(rows[0])
    if not has_header and len(rows) >= 2:
        row0_parses = any(Chem.MolFromSmiles(c.strip()) is not None for c in rows[0])
        sample_below = rows[1:11]
        below_parses = any(
            Chem.MolFromSmiles(c.strip()) is not None
            for r in sample_below for c in r
        )
        if (not row0_parses) and below_parses:
            has_header = True

    header = rows[0] if has_header else [f"col{i}" for i in range(len(rows[0]))]
    data = rows[1:] if has_header else rows
    if not data:
        raise SystemExit(f"{path}: no data rows after header")

    n_cols = len(header)
    if n_cols < 2:
        raise SystemExit(
            f"{path}: only {n_cols} column(s). Need at least 2 (one SMILES, "
            f"one sequence). Got delimiter={delim!r}, header={header}"
        )

    # Score each column: fraction of cells that parse as valid SMILES.
    sample = data[:50]
    smiles_score = [0.0] * n_cols
    for j in range(n_cols):
        valid = sum(1 for row in sample if j < len(row)
                    and Chem.MolFromSmiles(row[j].strip()) is not None)
        smiles_score[j] = valid / max(len(sample), 1)

    smi_col_idx = max(range(n_cols), key=lambda j: smiles_score[j])
    if smiles_score[smi_col_idx] < 0.5:
        raise SystemExit(
            f"{path}: no column parses as valid SMILES (best fraction "
            f"{smiles_score[smi_col_idx]:.2f}). Confirm input format."
        )

    # Sequence column: pick the other column with longest mean text length
    # (proteins are 100s of AA; chembl_id and other ids are short).
    other = [j for j in range(n_cols) if j != smi_col_idx]
    mean_len = {
        j: sum(len(row[j].strip()) for row in sample if j < len(row))
           / max(len(sample), 1)
        for j in other
    }
    seq_col_idx = max(mean_len, key=mean_len.get)

    return InputType.BATCH_FILE, {
        "path":       path,
        "smiles_col": header[smi_col_idx],
        "seq_col":    header[seq_col_idx],
        "delimiter":  delim,
        "has_header": has_header,
        "n_rows":     len(data),
        "smiles_idx": smi_col_idx,
        "seq_idx":    seq_col_idx,
    }


def _looks_like_header(row: list[str]) -> bool:
    """Heuristic: first row has any cell that's a known column name."""
    if not row:
        return False
    known = {
        "smiles", "canonical_smiles", "ligand", "drug",
        "sequence", "seq", "protein", "target", "fasta",
        "uniprot", "seq_id", "chembl_id", "compound", "molecule",
    }
    return any(cell.strip().lower() in known for cell in row)


# ======================================================================
# Pipeline runners — dispatch on detected type
# ======================================================================

def run_smiles_string(smiles: str, args: argparse.Namespace) -> Path:
    """Mode 1: SMILES string vs kinome reference (delegates to committee.py)."""
    cmd = [
        sys.executable, str(INFER_DIR / "committee.py"),
        "--smiles",   smiles,
        "--organism", args.organism,
        "--out",      str(args.out),
        "--ckpt-corpus", args.ckpt_corpus,
        "--top-k",    str(args.top_k),
        "--models",   args.models,
    ]
    if args.parallel: cmd.append("--parallel")
    if args.dry_run:  cmd.append("--dry-run")
    print(f"[run] mode = smiles → kinome × ligand")
    print(f"[run] cmd  = {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return args.out / "consensus.csv"


def run_fasta_file(path: Path, args: argparse.Namespace) -> Path:
    """Mode 2: FASTA → ligand library (single sequence) OR direct pair."""
    cmd = [
        sys.executable, str(INFER_DIR / "committee.py"),
        "--fasta", str(path),
        "--out",   str(args.out),
        "--ckpt-corpus", args.ckpt_corpus,
        "--top-k", str(args.top_k),
        "--models", args.models,
    ]
    if args.parallel: cmd.append("--parallel")
    if args.dry_run:  cmd.append("--dry-run")
    print(f"[run] mode = fasta → ligand library")
    print(f"[run] cmd  = {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return args.out / "consensus.csv"


def run_batch_file(path: Path, payload: dict, args: argparse.Namespace) -> Path:
    """Mode 3: batch CSV/TSV — order-agnostic column detection.

    Normalizes the user's batch to the canonical pairs.tsv schema
    (uniprot, sequence, chembl_id, smiles, source) and feeds that to
    committee.py via --pairs.
    """
    args.out.mkdir(parents=True, exist_ok=True)
    canon_path = args.out / "pairs.tsv"

    # Translate user batch → canonical pairs.tsv
    delim = payload["delimiter"]
    smi_col, seq_col = payload["smiles_col"], payload["seq_col"]
    has_header = payload["has_header"]

    rows_out = []
    with open(path, newline="") as fh:
        reader = csv.reader(fh, delimiter=delim)
        rows = list(reader)
    data = rows[1:] if has_header else rows
    smi_idx, seq_idx = payload["smiles_idx"], payload["seq_idx"]

    # Optional: detect uniprot/chembl_id columns if present in header.
    header = rows[0] if has_header else [f"col{i}" for i in range(len(rows[0]))]
    header_lc = [c.strip().lower() for c in header]
    uniprot_idx = next((i for i, c in enumerate(header_lc)
                        if c in {"uniprot", "seq_id", "target", "protein_id"}), None)
    chembl_idx  = next((i for i, c in enumerate(header_lc)
                        if c in {"chembl_id", "compound", "ligand_id", "drug_id"}), None)

    for i, row in enumerate(data):
        if smi_idx >= len(row) or seq_idx >= len(row):
            continue
        smi = row[smi_idx].strip()
        seq = row[seq_idx].strip()
        uid = row[uniprot_idx].strip() if uniprot_idx is not None and uniprot_idx < len(row) else f"USER_PROT_{i}"
        cid = row[chembl_idx].strip()  if chembl_idx  is not None and chembl_idx  < len(row) else f"USER_LIG_{i}"
        rows_out.append({
            "uniprot":   uid,
            "sequence":  seq,
            "chembl_id": cid,
            "smiles":    smi,
            "source":    "batch_user",
        })

    if not rows_out:
        raise SystemExit(f"no usable rows in {path}")

    cols = ["uniprot", "sequence", "chembl_id", "smiles", "source"]
    with open(canon_path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
        wr.writeheader()
        for r in rows_out:
            wr.writerow(r)

    print(f"[run] mode = batch ({len(rows_out)} pairs from {path.name})")
    print(f"[run] auto-detected: smiles_col={smi_col!r}, seq_col={seq_col!r}, delim={delim!r}")
    print(f"[run] normalized → {canon_path}")

    cmd = [
        sys.executable, str(INFER_DIR / "committee.py"),
        "--pairs", str(canon_path),
        "--out",   str(args.out),
        "--ckpt-corpus", args.ckpt_corpus,
        "--top-k", str(args.top_k),
        "--models", args.models,
    ]
    if args.parallel: cmd.append("--parallel")
    if args.dry_run:  cmd.append("--dry-run")
    subprocess.run(cmd, check=True)
    return args.out / "consensus.csv"


# ======================================================================
# Output annotation (organism + kinase name when input was SMILES vs kinome)
# ======================================================================

def annotate_consensus(consensus: Path, kind: str) -> None:
    """Best-effort metadata join from scaffolds_splits/output/universal_test.tsv."""
    if not consensus.exists():
        return
    try:
        import pandas as pd
        c  = pd.read_csv(consensus)
        df = pd.read_csv(REPO/"scaffolds_splits/output/universal_test.tsv", sep="\t")
        meta = (df.drop_duplicates(["seq_id"])
                  [["seq_id", "target_kinase", "organism"]]
                  .rename(columns={"seq_id": "uniprot"}))
        meta["uniprot"] = meta["uniprot"].astype(str)
        c["uniprot"]    = c["uniprot"].astype(str)
        c.merge(meta, on="uniprot", how="left").to_csv(
            consensus.with_name(consensus.stem + ".annotated.csv"), index=False)
    except Exception as e:
        print(f"[warn] annotation step skipped: {e}", file=sys.stderr)


# ======================================================================
# Main CLI
# ======================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="kinase_profiling: auto-detect input + run inference pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("input",
                    help="SMILES string OR path to FASTA/CSV/TSV/TXT")
    ap.add_argument("--organism", choices=["human", "non_human", "all"], default="human",
                    help="kinome subset for SMILES-only mode (default: human)")
    ap.add_argument("--ckpt-corpus", choices=["human", "non_human", "all"], default="all",
                    help="training corpus for the model checkpoints (default: all)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output dir (default: results/inference/kinase_profiling_<TS>)")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--models", default="dtkinase,drugban,graphban,conplex",
                    help="comma-separated subset of models to run")
    ap.add_argument("--parallel", action="store_true",
                    help="run model scoring subprocesses in parallel")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the resolved pipeline command without executing")
    args = ap.parse_args()

    # Auto-resolve output dir if user did not specify
    if args.out is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        h  = hashlib.sha1(args.input.encode("utf-8")).hexdigest()[:8]
        args.out = REPO / "results" / "inference" / f"kinase_profiling_{ts}_{h}"
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    # Step 1: detect input type
    # ----------------------------------------------------------
    print("=" * 70)
    print(f" kinase_profiling")
    print(f"   input : {args.input!r}")
    print(f"   out   : {args.out}")
    print("=" * 70)

    kind, payload = detect_input(args.input)

    print(f"\n[detect] input type = {kind}")
    if kind == InputType.SMILES_STRING:
        print(f"[detect] canonical SMILES = {payload['smiles']}")
        print(f"[detect] will expand against {args.organism} kinome reference")
    elif kind == InputType.FASTA_FILE:
        print(f"[detect] FASTA records = {payload['n_records']}")
        if payload['n_records'] == 1:
            print(f"[detect] will expand against ligand library "
                  f"({REPO/'data/reference/ligand_library.tsv'})")
        else:
            print(f"[detect] WARN: multi-record FASTA — only first record "
                  f"will be used by committee.py (single-protein mode)")
    elif kind == InputType.BATCH_FILE:
        print(f"[detect] batch rows = {payload['n_rows']}")
        print(f"[detect] SMILES column = {payload['smiles_col']!r} (idx {payload['smiles_idx']})")
        print(f"[detect] sequence column = {payload['seq_col']!r} (idx {payload['seq_idx']})")
        print(f"[detect] delimiter = {payload['delimiter']!r}")

    # ----------------------------------------------------------
    # Step 2: dispatch
    # ----------------------------------------------------------
    print()
    if kind == InputType.SMILES_STRING:
        consensus = run_smiles_string(payload["smiles"], args)
    elif kind == InputType.FASTA_FILE:
        consensus = run_fasta_file(payload["path"], args)
    elif kind == InputType.BATCH_FILE:
        consensus = run_batch_file(payload["path"], payload, args)
    else:
        raise SystemExit(f"unhandled input type: {kind}")

    if args.dry_run:
        return

    # ----------------------------------------------------------
    # Step 3: annotate + report
    # ----------------------------------------------------------
    annotate_consensus(consensus, kind)

    print()
    print("=" * 70)
    print(" DONE")
    print("=" * 70)
    print(f"output dir: {args.out}")
    if consensus.exists():
        print(f"consensus : {consensus.relative_to(REPO) if consensus.is_relative_to(REPO) else consensus}")
        annotated = consensus.with_name(consensus.stem + ".annotated.csv")
        if annotated.exists():
            print(f"annotated : {annotated.relative_to(REPO) if annotated.is_relative_to(REPO) else annotated}")


if __name__ == "__main__":
    main()
