#!/usr/bin/env python3
"""
Script to separate kinase data into 3 files:
1. kinase_all_compounds.tsv (all - already exists)
2. kinase_human_compounds.tsv (Homo sapiens only)
3. kinase_non_human_compounds.tsv (except Homo sapiens)
"""

import pandas as pd
import os
from pathlib import Path

def split_kinase_data(input_file, output_dir):
    """
    Separate kinase data into human and non-human.
    
    Args:
        input_file: Path to kinase_all_compounds.tsv file
        output_dir: Directory where files will be saved
    """
    
    print("=" * 80)
    print("🧬 KINASE DATA SEPARATION")
    print("=" * 80)
    
    # Check if file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"❌ File not found: {input_file}")
    
    print(f"\n📂 Reading file: {input_file}")
    print(f"📊 File size: {os.path.getsize(input_file) / (1024**3):.2f} GB")
    
    # Read the TSV file
    print("\n⏳ Loading data (may take a few minutes)...")
    df = pd.read_csv(input_file, sep='\t', low_memory=False)
    
    print(f"✅ Data loaded: {len(df):,} records")
    print(f"\n📋 Available columns:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")
    
    # Check unique values in organism column
    print(f"\n🌍 Unique organisms found: {df['organism'].nunique():,}")
    print("\n📊 Top 10 most frequent organisms:")
    organism_counts = df['organism'].value_counts()
    for org, count in organism_counts.head(10).items():
        percentage = (count / len(df)) * 100
        print(f"  • {org}: {count:,} ({percentage:.2f}%)")
    
    # Separate data
    print("\n" + "=" * 80)
    print("🔬 SEPARATING DATA...")
    print("=" * 80)
    
    # Filter humans
    df_humans = df[df['organism'] == 'Homo sapiens'].copy()
    print(f"\n✅ Human data: {len(df_humans):,} records ({len(df_humans)/len(df)*100:.2f}%)")
    
    # Filter non-humans
    df_non_humans = df[df['organism'] != 'Homo sapiens'].copy()
    print(f"✅ Non-human data: {len(df_non_humans):,} records ({len(df_non_humans)/len(df)*100:.2f}%)")
    
    # Verify sum
    assert len(df_humans) + len(df_non_humans) == len(df), "❌ Error: sum mismatch!"
    print(f"✅ Verification: {len(df_humans):,} + {len(df_non_humans):,} = {len(df):,} ✓")
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save files
    print("\n" + "=" * 80)
    print("💾 SAVING FILES...")
    print("=" * 80)
    
    # Human file
    output_humans = os.path.join(output_dir, 'kinase_human_compounds.tsv')
    print(f"\n📝 Saving: {output_humans}")
    df_humans.to_csv(output_humans, sep='\t', index=False)
    size_humans = os.path.getsize(output_humans) / (1024**2)
    print(f"   ✅ Saved: {size_humans:.2f} MB")
    
    # Non-human file
    output_non_humans = os.path.join(output_dir, 'kinase_non_human_compounds.tsv')
    print(f"\n📝 Saving: {output_non_humans}")
    df_non_humans.to_csv(output_non_humans, sep='\t', index=False)
    size_non_humans = os.path.getsize(output_non_humans) / (1024**2)
    print(f"   ✅ Saved: {size_non_humans:.2f} MB")
    
    # Copy original file to output directory (if needed)
    output_all = os.path.join(output_dir, 'kinase_all_compounds.tsv')
    if os.path.abspath(input_file) != os.path.abspath(output_all):
        print(f"\n📝 Copying original file to: {output_all}")
        df.to_csv(output_all, sep='\t', index=False)
        size_all = os.path.getsize(output_all) / (1024**2)
        print(f"   ✅ Saved: {size_all:.2f} MB")
    
    # Final summary
    print("\n" + "=" * 80)
    print("🎉 PROCESSING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    
    print("\n📊 SUMMARY OF GENERATED FILES:")
    print(f"\n1️⃣  kinase_all_compounds.tsv")
    print(f"   • Records: {len(df):,}")
    print(f"   • Size: {os.path.getsize(input_file) / (1024**2):.2f} MB")
    
    print(f"\n2️⃣  kinase_human_compounds.tsv")
    print(f"   • Records: {len(df_humans):,}")
    print(f"   • Size: {size_humans:.2f} MB")
    print(f"   • Organism: Homo sapiens")
    
    print(f"\n3️⃣  kinase_non_human_compounds.tsv")
    print(f"   • Records: {len(df_non_humans):,}")
    print(f"   • Size: {size_non_humans:.2f} MB")
    print(f"   • Organisms: {df_non_humans['organism'].nunique():,} species")
    
    # Sequence statistics
    print("\n" + "=" * 80)
    print("🧬 SEQUENCE STATISTICS:")
    print("=" * 80)
    
    print(f"\n📊 All compounds:")
    print(f"   • With sequence: {df['seq'].notna().sum():,} ({df['seq'].notna().sum()/len(df)*100:.2f}%)")
    print(f"   • Without sequence: {df['seq'].isna().sum():,} ({df['seq'].isna().sum()/len(df)*100:.2f}%)")
    print(f"   • Unique sequences: {df['seq'].nunique():,}")
    
    print(f"\n📊 Human compounds:")
    print(f"   • With sequence: {df_humans['seq'].notna().sum():,} ({df_humans['seq'].notna().sum()/len(df_humans)*100:.2f}%)")
    print(f"   • Without sequence: {df_humans['seq'].isna().sum():,} ({df_humans['seq'].isna().sum()/len(df_humans)*100:.2f}%)")
    print(f"   • Unique sequences: {df_humans['seq'].nunique():,}")
    
    print(f"\n📊 Non-human compounds:")
    print(f"   • With sequence: {df_non_humans['seq'].notna().sum():,} ({df_non_humans['seq'].notna().sum()/len(df_non_humans)*100:.2f}%)")
    print(f"   • Without sequence: {df_non_humans['seq'].isna().sum():,} ({df_non_humans['seq'].isna().sum()/len(df_non_humans)*100:.2f}%)")
    print(f"   • Unique sequences: {df_non_humans['seq'].nunique():,}")
    
    print("\n" + "=" * 80)
    print("✅ All files have been generated successfully!")
    print("=" * 80 + "\n")
    
    return df, df_humans, df_non_humans


if __name__ == "__main__":
    # Path configuration
    input_file = os.path.expanduser("~/Desktop/2024_desktop/chembl_35/kinase_all_compounds.tsv")
    
    # Output directories in docktkinase project (relative path to script)
    base_dir = Path(__file__).parent  # docktkinase/src/database -> parent = src
    output_dirs = {
        'all': base_dir / 'kinase_all',
        'humans': base_dir / 'kinase_humans',
        'non_humans': base_dir / 'kinase_non_humans'
    }
    
    # Create directories if they don't exist
    for dir_path in output_dirs.values():
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"📁 Directory verified/created: {dir_path}")
    
    # Execute the separation
    try:
        print("\n" + "=" * 80)
        print("🚀 STARTING PROCESSING WITH NEW DIRECTORIES")
        print("=" * 80)
        
        # Read data
        print(f"\n📂 Reading file: {input_file}")
        df = pd.read_csv(input_file, sep='\t', low_memory=False)
        print(f"✅ Data loaded: {len(df):,} records")
        
        # Separate data
        df_humans = df[df['organism'] == 'Homo sapiens'].copy()
        df_non_humans = df[df['organism'] != 'Homo sapiens'].copy()
        
        print(f"\n✅ Human data: {len(df_humans):,} records")
        print(f"✅ Non-human data: {len(df_non_humans):,} records")
        
        # Save to correct directories
        print("\n" + "=" * 80)
        print("💾 SAVING FILES TO PROJECT DIRECTORIES")
        print("=" * 80)
        
        # 1. ALL file
        output_all = os.path.join(output_dirs['all'], 'kinase_all_compounds.tsv')
        print(f"\n1️⃣  Saving: {output_all}")
        df.to_csv(output_all, sep='\t', index=False)
        size_all = os.path.getsize(output_all) / (1024**2)
        print(f"   ✅ Saved: {size_all:.2f} MB ({len(df):,} records)")
        
        # 2. HUMANS file
        output_humans = os.path.join(output_dirs['humans'], 'kinase_human_compounds.tsv')
        print(f"\n2️⃣  Saving: {output_humans}")
        df_humans.to_csv(output_humans, sep='\t', index=False)
        size_humans = os.path.getsize(output_humans) / (1024**2)
        print(f"   ✅ Saved: {size_humans:.2f} MB ({len(df_humans):,} records)")
        
        # 3. NON-HUMANS file
        output_non_humans = os.path.join(output_dirs['non_humans'], 'kinase_non_human_compounds.tsv')
        print(f"\n3️⃣  Saving: {output_non_humans}")
        df_non_humans.to_csv(output_non_humans, sep='\t', index=False)
        size_non_humans = os.path.getsize(output_non_humans) / (1024**2)
        print(f"   ✅ Saved: {size_non_humans:.2f} MB ({len(df_non_humans):,} records)")
        
        # Final summary
        print("\n" + "=" * 80)
        print("🎉 ALL FILES HAVE BEEN SAVED SUCCESSFULLY!")
        print("=" * 80)
        
        print("\n📊 FILE LOCATIONS:")
        print(f"\n1️⃣  {output_all}")
        print(f"   • {len(df):,} records | {size_all:.2f} MB")
        
        print(f"\n2️⃣  {output_humans}")
        print(f"   • {len(df_humans):,} records | {size_humans:.2f} MB")
        
        print(f"\n3️⃣  {output_non_humans}")
        print(f"   • {len(df_non_humans):,} records | {size_non_humans:.2f} MB")
        
        print("\n✅ Script executed successfully!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
