#!/usr/bin/env python3
"""
prepare_kinase_datasets.py — Generate ConPLex-format kinase CSVs from benchmark splits.

Scans for kinase dataset splits in standard locations (DrugBAN, GraphBAN, or
universal benchmark) and converts them to ConPLex format.

ConPLex format:  ,SMILES,Target Sequence,Label  (with integer index)
DrugBAN format:  SMILES,Protein,Y               (no index)

Usage:
    python prepare_kinase_datasets.py
    python prepare_kinase_datasets.py --source-root /path/to/repo
"""

import os
import sys
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser


# Column mapping from various formats to ConPLex
COLUMN_MAPS = {
    # DrugBAN / GraphBAN format
    ('SMILES', 'Protein', 'Y'): {
        'SMILES': 'SMILES',
        'Protein': 'Target Sequence',
        'Y': 'Label',
    },
    # DT-Kinase / universal benchmark format
    ('smiles', 'sequence', 'label'): {
        'smiles': 'SMILES',
        'sequence': 'Target Sequence',
        'label': 'Label',
    },
    # Already ConPLex format
    ('SMILES', 'Target Sequence', 'Label'): None,
}


def detect_format(csv_path: Path) -> dict:
    """Detect CSV format and return column mapping."""
    df = pd.read_csv(csv_path, nrows=2)
    cols = tuple(df.columns[:3])
    
    # Handle indexed CSVs (first col is unnamed index)
    if cols[0] == 'Unnamed: 0' or cols[0] == '':
        cols = tuple(df.columns[1:4])
    
    for key_cols, mapping in COLUMN_MAPS.items():
        if all(c in df.columns for c in key_cols):
            return mapping
    
    # Try case-insensitive match
    col_lower = {c.lower(): c for c in df.columns}
    if 'smiles' in col_lower and ('protein' in col_lower or 'sequence' in col_lower or 'target sequence' in col_lower):
        target_col = col_lower.get('protein') or col_lower.get('sequence') or col_lower.get('target sequence')
        label_col = col_lower.get('y') or col_lower.get('label')
        return {
            col_lower['smiles']: 'SMILES',
            target_col: 'Target Sequence',
            label_col: 'Label',
        }
    
    raise ValueError(f"Unknown format in {csv_path}: columns = {list(df.columns)}")


def convert_to_conplex(src_dir: Path, dst_dir: Path) -> bool:
    """Convert train/val/test CSVs from src_dir to ConPLex format in dst_dir."""
    splits = ['train', 'val', 'test']
    
    for split in splits:
        src_file = src_dir / f'{split}.csv'
        if not src_file.exists():
            print(f"    ⚠ Missing: {src_file}")
            return False
    
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    for split in splits:
        src_file = src_dir / f'{split}.csv'
        dst_file = dst_dir / f'{split}.csv'
        
        mapping = detect_format(src_file)
        df = pd.read_csv(src_file)
        
        if mapping is not None:
            df = df.rename(columns=mapping)
        
        # Ensure correct columns exist
        required = ['SMILES', 'Target Sequence', 'Label']
        for col in required:
            if col not in df.columns:
                print(f"    ✗ Column '{col}' missing after mapping in {src_file}")
                return False
        
        # Write ConPLex format (with integer index)
        df[required].to_csv(dst_file, index=True)
        print(f"    ✓ {split}: {len(df)} samples → {dst_file.name}")
    
    return True


def find_source_splits(source_root: Path, dataset: str) -> Path:
    """Find kinase splits in standard locations."""
    # Search order (most specific first)
    candidates = [
        # Already in ConPLex format
        source_root / 'ConPLex' / 'dataset' / f'kinase_{dataset}',
        # DrugBAN standard location
        source_root / 'DrugBAN' / 'datasets' / 'kinase' / dataset / 'scaffold',
        # GraphBAN standard location
        source_root / 'GraphBAN' / 'datasets' / 'kinase' / dataset / 'scaffold',
        # Universal benchmark
        source_root / 'benchmark' / 'datasets' / 'kinase' / dataset,
        source_root / 'benchmark' / 'datasets' / f'kinase_{dataset}',
        # Flat structure
        source_root / 'datasets' / 'kinase' / dataset / 'scaffold',
        source_root / 'dataset' / f'kinase_{dataset}',
    ]
    
    for candidate in candidates:
        if candidate.exists() and (candidate / 'train.csv').exists():
            return candidate
    
    return None


def main():
    parser = ArgumentParser(description="Prepare kinase datasets for ConPLex")
    parser.add_argument(
        '--source-root', default=None,
        help='Root directory to search for benchmark splits. '
             'Auto-detects: parent of ConPLex dir, /data/docktkinase, etc.'
    )
    parser.add_argument(
        '--output-dir', default='./dataset',
        help='Output directory for ConPLex datasets (default: ./dataset)'
    )
    args = parser.parse_args()
    
    script_dir = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir)
    
    # Auto-detect source root
    if args.source_root:
        search_roots = [Path(args.source_root)]
    else:
        search_roots = [
            script_dir.parent,                    # ../  (attention-screening/)
            script_dir,                           # .    (ConPLex/)
            Path('/data/docktkinase'),             # GPU machine standard
            Path('/storage/leon/attention-screening'),  # old GPU machine
        ]
    
    datasets = ['non_human', 'human', 'all']
    
    print("═══════════════════════════════════════════════════════════════")
    print(" Prepare Kinase Datasets for ConPLex")
    print("═══════════════════════════════════════════════════════════════")
    
    all_ok = True
    for dataset in datasets:
        dst = output_dir / f'kinase_{dataset}'
        
        # Skip if already exists and complete
        if dst.exists() and all((dst / f'{s}.csv').exists() for s in ['train', 'val', 'test']):
            print(f"\n  ✓ kinase_{dataset}: already exists ({dst})")
            continue
        
        print(f"\n  → kinase_{dataset}: searching for source splits...")
        
        found = None
        for root in search_roots:
            if not root.exists():
                continue
            found = find_source_splits(root, dataset)
            if found:
                print(f"    Found: {found}")
                break
        
        if found is None:
            print(f"    ✗ No source splits found for kinase_{dataset}")
            print(f"      Searched: {[str(r) for r in search_roots if r.exists()]}")
            all_ok = False
            continue
        
        # Check if source is already ConPLex format
        if found == dst:
            print(f"    ✓ Already in place")
            continue
        
        ok = convert_to_conplex(found, dst)
        if not ok:
            all_ok = False
    
    print()
    if all_ok:
        print("  All datasets ready.")
    else:
        print("  ⚠ Some datasets could not be prepared.")
        print("    Ensure benchmark splits exist in one of the standard locations.")
        sys.exit(1)


if __name__ == "__main__":
    main()
