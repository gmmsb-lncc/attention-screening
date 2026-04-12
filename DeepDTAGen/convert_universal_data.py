#!/usr/bin/env python3
"""
convert_universal_data.py

Converts the universal kinase scaffold-split datasets to DeepDTAGen's
expected format (CSVs + PyTorch .pt files).

Source data: scaffolds_splits/output/ (TSV files)
    Columns: chembl_id, molregno, target_kinase, canonical_smiles,
             standard_value, standard_type, pchembl_value, compound_name,
             organism, seq, seq_id, label, scaffold, dataset_source

Destination: data/ (DeepDTAGen format)
    CSVs with: compound_iso_smiles, target_smiles, target_sequence, affinity
    PyTorch .pt files for training

Dataset source filtering:
    non_human: dataset_source != 'human'
    human:     dataset_source == 'human'
    all:       no filter

Usage:
    python convert_universal_data.py [--scaffold-dir PATH]
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
import pickle
from collections import OrderedDict
from rdkit import Chem
import networkx as nx
import torch
from tqdm import tqdm

# --- Import from existing DeepDTAGen code ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from create_data import smile_to_graph, seq_cat, seq_dict, max_seq_len
from utils import Tokenizer, TestbedDataset


DEFAULT_SCAFFOLD_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'scaffolds_splits', 'output',
)

DATASETS = {
    'non_human': lambda df: df[df['dataset_source'] != 'human'],
    'human':     lambda df: df[df['dataset_source'] == 'human'],
    'all':       lambda df: df,
}

SPLITS = {
    'train': 'scenarios/Sc/universal_train.tsv',
    'val':   'scenarios/Sc/universal_val.tsv',
    'test':  'universal_test.tsv',
}


def load_universal_split(scaffold_dir: str, split_file: str) -> pd.DataFrame:
    """Load a universal TSV split file."""
    path = os.path.join(scaffold_dir, split_file)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Split file not found: {path}")
    df = pd.read_csv(path, sep='\t')
    print(f"  Loaded {path}: {len(df)} rows")
    return df


def universal_to_deepdtagen(df: pd.DataFrame) -> pd.DataFrame:
    """Convert universal TSV format to DeepDTAGen CSV format.

    Maps:
        canonical_smiles -> compound_iso_smiles, target_smiles
        seq              -> target_sequence
        label            -> affinity (binary: 0/1)
    """
    out = pd.DataFrame({
        'compound_iso_smiles': df['canonical_smiles'],
        'target_smiles': df['canonical_smiles'],  # MTS = same SMILES
        'target_sequence': df['seq'],
        'affinity': df['label'].astype(float),
    })

    # Filter invalid SMILES
    valid_mask = out['compound_iso_smiles'].apply(
        lambda s: Chem.MolFromSmiles(s) is not None
    )
    invalid_count = (~valid_mask).sum()
    if invalid_count > 0:
        print(f"    [WARN] Removing {invalid_count} invalid SMILES")
        out = out[valid_mask].reset_index(drop=True)

    return out


def main():
    parser = argparse.ArgumentParser(description="Convert universal kinase data for DeepDTAGen")
    parser.add_argument('--scaffold-dir', default=DEFAULT_SCAFFOLD_DIR,
                        help="Path to scaffolds_splits/output/")
    args = parser.parse_args()

    scaffold_dir = os.path.abspath(args.scaffold_dir)
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    os.makedirs(os.path.join(data_dir, 'processed'), exist_ok=True)

    print(f"Scaffold splits: {scaffold_dir}")
    print(f"Output dir:      {data_dir}")

    # --- Load all universal splits ---
    raw_splits = {}
    for split_name, split_file in SPLITS.items():
        raw_splits[split_name] = load_universal_split(scaffold_dir, split_file)

    # --- Process each dataset ---
    for dataset_name, filter_fn in DATASETS.items():
        print(f"\n{'='*60}")
        print(f" Processing: {dataset_name}")
        print(f"{'='*60}")

        all_smiles_for_tokenizer = set()
        dfs = {}

        for split_name in ['train', 'val', 'test']:
            raw_df = raw_splits[split_name]
            filtered = filter_fn(raw_df).reset_index(drop=True)
            print(f"  [{dataset_name}/{split_name}] {len(filtered)} rows after filter")

            if len(filtered) == 0:
                print(f"    [SKIP] No data for {dataset_name}/{split_name}")
                continue

            df = universal_to_deepdtagen(filtered)
            dfs[split_name] = df
            all_smiles_for_tokenizer.update(df['compound_iso_smiles'].tolist())

            # Save CSV
            csv_path = os.path.join(data_dir, f'kinase_{dataset_name}_{split_name}.csv')
            df.to_csv(csv_path, index=False)
            print(f"    Saved: {csv_path} ({len(df)} samples)")

        if not dfs:
            print(f"  [SKIP] No data for {dataset_name}")
            continue

        # Build graph representations
        print(f"  Building molecular graphs...")
        smile_graph = {}
        failed = 0
        for smile in tqdm(all_smiles_for_tokenizer, desc=f"  Graphs ({dataset_name})"):
            try:
                g = smile_to_graph(smile)
                c_size, features, edge_index, edge_feats = g
                if len(edge_index) == 0 or len(edge_index[0]) == 0:
                    failed += 1
                    continue
                smile_graph[smile] = g
            except Exception:
                failed += 1
        if failed:
            print(f"  [WARN] {failed} molecules failed graph conversion")

        # Build tokenizer
        print(f"  Building tokenizer...")
        tokenizer = Tokenizer(Tokenizer.gen_vocabs(all_smiles_for_tokenizer))
        tokenizer_path = os.path.join(data_dir, f'kinase_{dataset_name}_tokenizer.pkl')
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(tokenizer, f)
        print(f"  Tokenizer saved: {tokenizer_path} (vocab={len(tokenizer)})")

        # Convert to PyTorch format
        for split_name, df in dfs.items():
            pt_path = os.path.join(data_dir, 'processed', f'kinase_{dataset_name}_{split_name}.pt')
            if os.path.exists(pt_path):
                os.remove(pt_path)

            drugs = list(df['compound_iso_smiles'])
            MTS = list(df['target_smiles'])
            prots = list(df['target_sequence'])
            Y = list(df['affinity'])

            # Encode proteins
            XT = [seq_cat(t) for t in prots]
            XT = np.asarray(XT)

            # Tokenize target SMILES
            XD = [torch.LongTensor(tokenizer.parse(s)) for s in MTS]

            # Filter to molecules with valid graphs
            valid_indices = [i for i, s in enumerate(drugs) if s in smile_graph]
            if len(valid_indices) < len(drugs):
                print(f"    [WARN] Filtered {len(drugs) - len(valid_indices)} samples without graphs")
                drugs = [drugs[i] for i in valid_indices]
                XD = [XD[i] for i in valid_indices]
                XT = XT[valid_indices]
                Y = [Y[i] for i in valid_indices]

            drugs = np.asarray(drugs)
            Y = np.asarray(Y)

            print(f"  Building PyTorch dataset: kinase_{dataset_name}_{split_name} ({len(drugs)} samples)")
            TestbedDataset(
                root='data',
                dataset=f'kinase_{dataset_name}_{split_name}',
                xd=drugs, xdt=XD, xt=XT, y=Y,
                smile_graph=smile_graph,
            )
            print(f"    Saved: {pt_path}")

    print(f"\n{'='*60}")
    print(" All datasets converted successfully!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
