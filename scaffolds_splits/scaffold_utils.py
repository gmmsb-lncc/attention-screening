"""Scaffold computation and scaffold-level aggregations."""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm

UNKNOWN_SCAFFOLD = "UNKNOWN"


def murcko_scaffold(smiles: str) -> str:
    """Return Murcko scaffold SMILES, or UNKNOWN on parsing failure."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return UNKNOWN_SCAFFOLD
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None:
            return UNKNOWN_SCAFFOLD
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return UNKNOWN_SCAFFOLD


def build_compound_scaffold_table(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Build unique compound table with one scaffold per `chembl_id`."""
    smiles_counts = df.groupby("chembl_id")["canonical_smiles"].nunique(dropna=True)
    conflicts = int((smiles_counts > 1).sum())
    if conflicts > 0:
        print(f"[{dataset_name}] warning: {conflicts} compounds map to >1 SMILES; using first occurrence")

    compounds = (
        df[["chembl_id", "canonical_smiles"]]
        .drop_duplicates(subset=["chembl_id"], keep="first")
        .copy()
        .reset_index(drop=True)
    )

    scaffolds = []
    for smi in tqdm(
        compounds["canonical_smiles"].tolist(),
        desc=f"[{dataset_name}] computing Murcko scaffolds",
        disable=len(compounds) < 1000,
    ):
        scaffolds.append(murcko_scaffold(smi))

    compounds["scaffold"] = scaffolds
    unknown = int((compounds["scaffold"] == UNKNOWN_SCAFFOLD).sum())
    unique_scaffolds = int(compounds["scaffold"].nunique())
    print(
        f"[{dataset_name}] scaffold table: compounds={len(compounds):,} "
        f"scaffolds={unique_scaffolds:,} unknown={unknown:,}"
    )
    return compounds


def attach_scaffolds(df: pd.DataFrame, compound_scaffolds: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(
        compound_scaffolds[["chembl_id", "scaffold"]],
        on="chembl_id",
        how="left",
    )
    out["scaffold"] = out["scaffold"].fillna(UNKNOWN_SCAFFOLD)
    return out


def scaffold_stats(df_with_scaffold: pd.DataFrame) -> pd.DataFrame:
    """Aggregate scaffold statistics used by optimization routines."""
    grouped = df_with_scaffold.groupby("scaffold", dropna=False)

    stats = grouped.agg(
        unique_compounds=("chembl_id", "nunique"),
        rows_total=("chembl_id", "size"),
        rows_pos=("label", lambda x: int((x == 1).sum())),
        rows_neg=("label", lambda x: int((x == 0).sum())),
    ).reset_index()

    return stats


def scaffold_to_compound_count(stats_df: pd.DataFrame) -> Dict[str, int]:
    return dict(zip(stats_df["scaffold"], stats_df["unique_compounds"]))
