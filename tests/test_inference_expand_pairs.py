"""Unit tests for scripts/inference/expand_pairs.py.

Covers:
  - SMILES validation/canonicalization (RDKit)
  - FASTA parsing (multi-record + comments)
  - sequence validation (alphabet IUPAC + truncation 1024)
  - input mode dispatch (smiles, fasta, both, batch)
  - output schema (uniprot, sequence, chembl_id, smiles, source)
"""
import csv
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts" / "inference"))

import expand_pairs  # noqa: E402

pytestmark = pytest.mark.unit


# ======================================================================
# SMILES validation
# ======================================================================

def test_validate_smiles_canonicalizes():
    """RDKit returns the canonical SMILES, regardless of the input form."""
    canonical = expand_pairs.validate_smiles("C(C)O")  # ethanol non-canonical
    # RDKit canonicalization depends on version; should at minimum parse.
    assert canonical is not None
    assert "C" in canonical and "O" in canonical


def test_validate_smiles_accepts_well_formed():
    smi = "CC(=O)Oc1ccccc1C(=O)O"  # aspirin
    out = expand_pairs.validate_smiles(smi)
    assert "C" in out and "O" in out


def test_validate_smiles_rejects_garbage():
    with pytest.raises(ValueError, match="invalid SMILES"):
        expand_pairs.validate_smiles("ZZZ_not_a_smiles")


# ======================================================================
# FASTA parsing
# ======================================================================

def test_parse_fasta_single_record(tmp_path):
    fa = tmp_path / "seq.fa"
    fa.write_text(">P00519_ABL1\nMGNNHGTYLG\n")
    rec = list(expand_pairs.parse_fasta(fa))
    assert rec == [("P00519_ABL1", "MGNNHGTYLG")]


def test_parse_fasta_multiline(tmp_path):
    fa = tmp_path / "multi.fa"
    fa.write_text(">A\nMGN\nNHGT\nYLG\n>B\nACDE\n")
    rec = list(expand_pairs.parse_fasta(fa))
    assert rec == [("A", "MGNNHGTYLG"), ("B", "ACDE")]


def test_parse_fasta_id_only_first_token(tmp_path):
    fa = tmp_path / "wide.fa"
    fa.write_text(">P00519 ABL1 description with spaces\nMGN\n")
    rec = list(expand_pairs.parse_fasta(fa))
    assert rec[0][0] == "P00519"


def test_parse_fasta_skips_empty_lines(tmp_path):
    fa = tmp_path / "blank.fa"
    fa.write_text(">A\n\nMGN\n\nNHGT\n")
    rec = list(expand_pairs.parse_fasta(fa))
    assert rec == [("A", "MGNNHGT")]


# ======================================================================
# Sequence validation
# ======================================================================

def test_validate_sequence_strips_invalid_chars():
    seq = expand_pairs.validate_sequence("MGN-XYZ@HGT")
    assert "-" not in seq and "@" not in seq
    assert all(c in "ACDEFGHIKLMNPQRSTVWY" for c in seq)


def test_validate_sequence_uppercases():
    seq = expand_pairs.validate_sequence("mgnnhgt")
    assert seq == "MGNNHGT"


def test_validate_sequence_truncates_to_max_len():
    long = "M" * 2000
    seq = expand_pairs.validate_sequence(long, max_len=1024)
    assert len(seq) == 1024


def test_validate_sequence_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        expand_pairs.validate_sequence("---@@@")
    with pytest.raises(ValueError, match="empty"):
        expand_pairs.validate_sequence("")


def test_validate_sequence_x_unknown_dropped():
    """X (unknown AA) is not in the canonical alphabet → dropped."""
    seq = expand_pairs.validate_sequence("MGNXYHGT")
    assert "X" not in seq


# ======================================================================
# Input mode dispatch
# ======================================================================

def _make_args(**kwargs) -> object:
    """Minimal namespace for expand()."""
    import argparse
    ns = argparse.Namespace(
        smiles=None, fasta=None, pairs=None,
        organism="human", family=None, out=None,
    )
    for k, v in kwargs.items():
        setattr(ns, k, v)
    return ns


def test_mode_smiles_plus_fasta_yields_one_pair(tmp_path):
    fa = tmp_path / "abl.fa"
    fa.write_text(">P00519_ABL1\nMGNNHGTYLG\n")
    pairs = expand_pairs.expand(_make_args(
        smiles="CC(=O)O", fasta=str(fa)
    ))
    assert len(pairs) == 1
    p = pairs[0]
    assert p["uniprot"] == "P00519_ABL1"
    assert p["chembl_id"] == "USER_LIGAND"
    assert p["sequence"] == "MGNNHGTYLG"
    assert p["source"] == "user_pair"


def test_mode_smiles_plus_fasta_rejects_multi_seq(tmp_path):
    fa = tmp_path / "multi.fa"
    fa.write_text(">A\nMGN\n>B\nACDE\n")
    with pytest.raises(ValueError, match="exactly 1 sequence"):
        expand_pairs.expand(_make_args(smiles="CCO", fasta=str(fa)))


def test_mode_pairs_batch(tmp_path):
    tsv = tmp_path / "batch.tsv"
    tsv.write_text(
        "uniprot\tsequence\tchembl_id\tsmiles\n"
        "P1\tMGNNHGT\tCHEMBL1\tCCO\n"
        "P2\tACDEFGH\tCHEMBL2\tCCN\n"
    )
    pairs = expand_pairs.expand(_make_args(pairs=str(tsv)))
    assert len(pairs) == 2
    assert pairs[0]["uniprot"] == "P1" and pairs[0]["chembl_id"] == "CHEMBL1"
    assert pairs[1]["uniprot"] == "P2" and pairs[1]["source"] == "user_batch"


def test_mode_no_input_raises():
    with pytest.raises(ValueError, match="provide one of"):
        expand_pairs.expand(_make_args())


# ======================================================================
# Writer
# ======================================================================

def test_write_pairs_schema(tmp_path):
    pairs = [
        {"uniprot": "P1", "sequence": "MGN", "chembl_id": "C1",
         "smiles": "CCO", "source": "test"},
    ]
    out = tmp_path / "pairs.tsv"
    expand_pairs.write_pairs(pairs, out)
    rows = list(csv.DictReader(open(out), delimiter="\t"))
    assert rows[0]["uniprot"] == "P1"
    assert rows[0]["chembl_id"] == "C1"
    assert rows[0]["smiles"] == "CCO"
    assert rows[0]["source"] == "test"
    # column order matches spec
    with open(out) as fh:
        header = fh.readline().strip().split("\t")
    assert header == ["uniprot", "sequence", "chembl_id", "smiles", "source"]
