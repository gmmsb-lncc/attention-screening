"""Expand user input into a list of (protein, ligand) pairs to score.

Modes (auto-dispatched from CLI flags):
  --smiles SMI                          → (each kinase in library) × SMI
  --fasta FILE                          → P × (each ligand in library)
  --smiles SMI --fasta FILE             → 1 explicit pair
  --pairs FILE                          → batch TSV (uniprot, sequence, chembl_id, smiles)

Library filters:
  --organism {human, non_human, all}    → kinase library subset
  --family   {TK, CMGC, AGC, ...}       → optional family filter

Emits: pairs.tsv (uniprot, sequence, chembl_id, smiles, source).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import csv
import sys
from typing import Iterator

from rdkit import Chem


REFERENCE_DIR = Path(__file__).resolve().parents[2] / "data" / "reference"
KINOME_FILES = {
    "human":     REFERENCE_DIR / "kinome_human.fasta",
    "non_human": REFERENCE_DIR / "kinome_non_human.fasta",
    "all":       REFERENCE_DIR / "kinome_full.fasta",
}
LIGAND_LIBRARY = REFERENCE_DIR / "ligand_library.tsv"


def parse_fasta(path: Path) -> Iterator[tuple[str, str]]:
    """Yield (header_id, sequence) tuples from a FASTA file."""
    header, buf = None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(buf)
                header = line[1:].split()[0]
                buf = []
            else:
                buf.append(line)
        if header is not None:
            yield header, "".join(buf)


def validate_smiles(smiles: str) -> str:
    """Return canonical SMILES; raise ValueError if invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    return Chem.MolToSmiles(mol, canonical=True)


def validate_sequence(seq: str, max_len: int = 1024) -> str:
    """Strip non-AA, validate alphabet, truncate to max_len (C-terminal)."""
    valid = set("ACDEFGHIKLMNPQRSTVWY")
    seq = "".join(c for c in seq.upper() if c in valid)
    if len(seq) == 0:
        raise ValueError("empty/invalid amino-acid sequence")
    if len(seq) > max_len:
        seq = seq[:max_len]
    return seq


def load_kinome(organism: str, family: str | None = None) -> list[tuple[str, str]]:
    """Return [(uniprot, sequence), ...] from organism FASTA, optionally filtered."""
    path = KINOME_FILES.get(organism)
    if path is None or not path.exists():
        raise FileNotFoundError(
            f"kinome FASTA missing for organism={organism}: {path}\n"
            f"Run scripts/inference/references/build_kinome_cache.py first."
        )
    pairs = [(h, validate_sequence(s)) for h, s in parse_fasta(path)]
    if family is not None:
        # placeholder: requires kinase family annotation TSV
        raise NotImplementedError("family filter pending family annotation TSV")
    return pairs


def load_ligand_library() -> list[tuple[str, str]]:
    """Return [(chembl_id, canonical_smiles), ...] from curated ligand TSV."""
    if not LIGAND_LIBRARY.exists():
        raise FileNotFoundError(
            f"ligand library missing: {LIGAND_LIBRARY}\n"
            f"Run scripts/inference/references/build_ligand_cache.py first."
        )
    out = []
    with open(LIGAND_LIBRARY) as fh:
        rdr = csv.DictReader(fh, delimiter="\t")
        for row in rdr:
            try:
                smi = validate_smiles(row["smiles"])
            except ValueError:
                continue
            out.append((row["chembl_id"], smi))
    return out


def expand(args: argparse.Namespace) -> list[dict]:
    """Dispatch on (smiles, fasta, pairs) combination → list of pair dicts."""
    if args.pairs:
        out = []
        with open(args.pairs) as fh:
            rdr = csv.DictReader(fh, delimiter="\t")
            for row in rdr:
                out.append({
                    "uniprot":   row["uniprot"],
                    "sequence":  validate_sequence(row["sequence"]),
                    "chembl_id": row["chembl_id"],
                    "smiles":    validate_smiles(row["smiles"]),
                    "source":    "user_batch",
                })
        return out

    if args.smiles and args.fasta:
        kinases = list(parse_fasta(Path(args.fasta)))
        if len(kinases) != 1:
            raise ValueError(
                f"--fasta with --smiles must contain exactly 1 sequence, got {len(kinases)}"
            )
        uid, seq = kinases[0]
        return [{
            "uniprot":   uid,
            "sequence":  validate_sequence(seq),
            "chembl_id": "USER_LIGAND",
            "smiles":    validate_smiles(args.smiles),
            "source":    "user_pair",
        }]

    if args.smiles:
        kinome = load_kinome(args.organism, args.family)
        smi = validate_smiles(args.smiles)
        return [{
            "uniprot": uid, "sequence": seq,
            "chembl_id": "USER_LIGAND", "smiles": smi,
            "source": f"smiles_vs_{args.organism}",
        } for uid, seq in kinome]

    if args.fasta:
        kinases = list(parse_fasta(Path(args.fasta)))
        if len(kinases) != 1:
            raise ValueError(
                f"--fasta-only mode expects 1 sequence, got {len(kinases)}"
            )
        uid, seq = kinases[0]
        seq = validate_sequence(seq)
        ligands = load_ligand_library()
        return [{
            "uniprot": uid, "sequence": seq,
            "chembl_id": cid, "smiles": smi,
            "source": "fasta_vs_ligand_library",
        } for cid, smi in ligands]

    raise ValueError("provide one of: --smiles, --fasta, both, or --pairs")


def write_pairs(pairs: list[dict], out_path: Path) -> None:
    cols = ["uniprot", "sequence", "chembl_id", "smiles", "source"]
    with open(out_path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
        wr.writeheader()
        for p in pairs:
            wr.writerow(p)


def main() -> None:
    ap = argparse.ArgumentParser(description="Expand input into protein-ligand pairs.")
    ap.add_argument("--smiles", type=str)
    ap.add_argument("--fasta",  type=str)
    ap.add_argument("--pairs",  type=str)
    ap.add_argument("--organism", choices=["human", "non_human", "all"], default="human")
    ap.add_argument("--family", type=str, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    pairs = expand(args)
    write_pairs(pairs, args.out)
    print(f"wrote {len(pairs)} pairs → {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
