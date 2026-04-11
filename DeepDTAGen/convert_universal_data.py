#!/usr/bin/env python3
"""
convert_universal_data.py

Converts the universal kinase scaffold-split datasets (from DrugBAN/GraphBAN)
to DeepDTAGen's expected CSV format.

DeepDTAGen expects:
    compound_iso_smiles, target_smiles, target_sequence, affinity

Universal datasets have:
    SMILES, Protein, Y (binary: 0/1)

Key adaptations:
    1. Y (binary) -> affinity: DeepDTAGen originally uses continuous affinity
       (pKd/pKi). For binary labels, we use Y directly (0.0 or 1.0).
       The model's MSE loss will treat this as a regression target.
    2. target_smiles: Required for the generative decoder (MTS = Modified Target
       SMILES). We set it equal to compound_iso_smiles since we don't have a
       separate target SMILES representation.
    3. target_sequence: Maps directly from Protein column.

Usage:
    python convert_universal_data.py
"""
import os
import sys
import pandas as pd
import numpy as np
import pickle
from collections import OrderedDict
from rdkit import Chem
import networkx as nx
import torch
from torch.nn.utils.rnn import pad_sequence

# --- Import from existing DeepDTAGen code ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from create_data import smile_to_graph, seq_cat, seq_dict, max_seq_len
from utils import Tokenizer, TestbedDataset


DRUGBAN_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'DrugBAN', 'datasets', 'kinase')

DATASETS = ['non_human', 'human', 'all']
SPLITS = ['train', 'test']


def convert_csv(dataset_name, split):
    """Read universal CSV and convert to DeepDTAGen format."""
    src_path = os.path.join(DRUGBAN_BASE, dataset_name, 'scaffold', f'{split}.csv')
    if not os.path.exists(src_path):
        print(f"[WARN] {src_path} not found, skipping.")
        return None

    df = pd.read_csv(src_path)
    print(f"  [{dataset_name}/{split}] Read {len(df)} rows")

    # Filter invalid SMILES
    valid_mask = df['SMILES'].apply(lambda s: Chem.MolFromSmiles(s) is not None)
    invalid_count = (~valid_mask).sum()
    if invalid_count > 0:
        print(f"  [WARN] Removing {invalid_count} invalid SMILES")
        df = df[valid_mask].reset_index(drop=True)

    # Convert to DeepDTAGen format
    out = pd.DataFrame({
        'compound_iso_smiles': df['SMILES'],
        'target_smiles': df['SMILES'],  # MTS = same SMILES (no separate target SMILES)
        'target_sequence': df['Protein'],
        'affinity': df['Y'].astype(float),
    })
    return out


def main():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    os.makedirs(os.path.join(data_dir, 'processed'), exist_ok=True)

    for dataset in DATASETS:
        print(f"\n{'='*60}")
        print(f" Processing: {dataset}")
        print(f"{'='*60}")

        all_smiles_for_tokenizer = set()
        dfs = {}

        for split in SPLITS:
            df = convert_csv(dataset, split)
            if df is not None:
                dfs[split] = df
                all_smiles_for_tokenizer.update(df['compound_iso_smiles'].tolist())

                # Save CSV
                csv_path = os.path.join(data_dir, f'kinase_{dataset}_{split}.csv')
                df.to_csv(csv_path, index=False)
                print(f"  Saved: {csv_path}")

        if not dfs:
            print(f"  [SKIP] No data for {dataset}")
            continue

        # Build graph representations
        print(f"  Building molecular graphs...")
        smile_graph = {}
        failed = 0
        for smile in all_smiles_for_tokenizer:
            try:
                g = smile_to_graph(smile)
                c_size, features, edge_index, edge_feats = g
                # Filter out single-atom molecules (empty edge_index)
                # These cause IndexError in utils.py transpose(1,0)
                if len(edge_index) == 0 or len(edge_index[0]) == 0:
                    failed += 1
                    continue
                smile_graph[smile] = g
            except Exception as e:
                failed += 1
        if failed:
            print(f"  [WARN] {failed} molecules failed graph conversion or have no edges")

        # Build tokenizer
        print(f"  Building tokenizer...")
        tokenizer = Tokenizer(Tokenizer.gen_vocabs(all_smiles_for_tokenizer))
        tokenizer_path = os.path.join(data_dir, f'kinase_{dataset}_tokenizer.pkl')
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(tokenizer, f)
        print(f"  Tokenizer saved: {tokenizer_path} (vocab={len(tokenizer)})")

        # Convert to PyTorch format
        for split, df in dfs.items():
            pt_path = os.path.join(data_dir, 'processed', f'kinase_{dataset}_{split}.pt')
            if os.path.exists(pt_path):
                os.remove(pt_path)
                print(f"  [INFO] Removed old {pt_path}, regenerating...")

            drugs = list(df['compound_iso_smiles'])
            MTS = list(df['target_smiles'])
            prots = list(df['target_sequence'])
            Y = list(df['affinity'])

            # Encode proteins
            XT = [seq_cat(t) for t in prots]
            XT = np.asarray(XT)

            # Tokenize target SMILES
            XD = [torch.LongTensor(tokenizer.parse(s)) for s in MTS]

            # Filter to only molecules with valid graphs
            valid_indices = [i for i, s in enumerate(drugs) if s in smile_graph]
            if len(valid_indices) < len(drugs):
                print(f"  [WARN] Filtered {len(drugs) - len(valid_indices)} samples without graphs")
                drugs = [drugs[i] for i in valid_indices]
                XD = [XD[i] for i in valid_indices]
                XT = XT[valid_indices]
                Y = [Y[i] for i in valid_indices]

            drugs = np.asarray(drugs)
            Y = np.asarray(Y)

            print(f"  Building PyTorch dataset: kinase_{dataset}_{split} ({len(drugs)} samples)")
            data = TestbedDataset(
                root='data',
                dataset=f'kinase_{dataset}_{split}',
                xd=drugs, xdt=XD, xt=XT, y=Y,
                smile_graph=smile_graph
            )
            print(f"  Saved: {pt_path}")

    print(f"\n{'='*60}")
    print(" All datasets converted successfully!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
