#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for comparative analysis between human and non-human kinases.

This script compares compound interaction data with human and non-human kinases,
generating statistics and visualizations to understand the differences
between the two datasets.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_data(human_file, non_human_file):
    """
    Load TSV files for human and non-human kinases.
    
    Args:
        human_file (str): Path to the human kinases file
        non_human_file (str): Path to the non-human kinases file
        
    Returns:
        tuple: DataFrames with human and non-human data
    """
    print("Loading data...")
    
    # Check if files exist
    if not os.path.exists(human_file):
        raise FileNotFoundError(f"File not found: {human_file}")
    
    if not os.path.exists(non_human_file):
        raise FileNotFoundError(f"File not found: {non_human_file}")
    
    # Load the data
    df_human = pd.read_csv(human_file, sep='\t')
    df_non_human = pd.read_csv(non_human_file, sep='\t')
    
    print(f"Human kinases: {len(df_human)} records")
    print(f"Non-human kinases: {len(df_non_human)} records")
    
    return df_human, df_non_human

def basic_statistics(df_human, df_non_human):
    """
    Calculate basic statistics of the datasets.
    
    Args:
        df_human (pd.DataFrame): DataFrame with human kinase data
        df_non_human (pd.DataFrame): DataFrame with non-human kinase data
    """
    print("\n=== BASIC STATISTICS ===")
    
    # Number of unique compounds
    human_compounds = df_human['molregno'].nunique()
    non_human_compounds = df_non_human['molregno'].nunique()
    
    print(f"Unique compounds - Human: {human_compounds}")
    print(f"Unique compounds - Non-Human: {non_human_compounds}")
    
    # Number of unique kinases
    human_kinases = df_human['target_kinase'].nunique()
    non_human_kinases = df_non_human['target_kinase'].nunique()
    
    print(f"Unique kinases - Human: {human_kinases}")
    print(f"Unique kinases - Non-Human: {non_human_kinases}")
    
    # Most common organisms (for non-human)
    print("\nMost common organisms (non-human):")
    organism_counts = df_non_human['organism'].value_counts().head(10)
    for organism, count in organism_counts.items():
        print(f"  {organism}: {count}")
    
    # Standard value types
    print("\nStandard value types:")
    print("Human:", df_human['standard_type'].value_counts().to_dict())
    print("Non-Human:", df_non_human['standard_type'].value_counts().to_dict())

def activity_distribution(df_human, df_non_human):
    """
    Analyze the distribution of activity values.
    
    Args:
        df_human (pd.DataFrame): DataFrame with human kinase data
        df_non_human (pd.DataFrame): DataFrame with non-human kinase data
    """
    print("\n=== ACTIVITY DISTRIBUTION ===")
    
    # Convert values to log scale (pIC50)
    df_human['pIC50'] = -np.log10(df_human['standard_value'] * 1e-9)
    df_non_human['pIC50'] = -np.log10(df_non_human['standard_value'] * 1e-9)
    
    # pIC50 statistics
    print("pIC50 Statistics:")
    print("Human - Mean: {:.2f}, Median: {:.2f}, Std: {:.2f}".format(
        df_human['pIC50'].mean(), df_human['pIC50'].median(), df_human['pIC50'].std()))
    print("Non-Human - Mean: {:.2f}, Median: {:.2f}, Std: {:.2f}".format(
        df_non_human['pIC50'].mean(), df_non_human['pIC50'].median(), df_non_human['pIC50'].std()))

def compound_overlap_analysis(df_human, df_non_human):
    """
    Analyze the overlap of compounds between human and non-human kinases.
    
    Args:
        df_human (pd.DataFrame): DataFrame with human kinase data
        df_non_human (pd.DataFrame): DataFrame with non-human kinase data
    """
    print("\n=== COMPOUND OVERLAP ANALYSIS ===")
    
    human_compounds = set(df_human['molregno'].unique())
    non_human_compounds = set(df_non_human['molregno'].unique())
    
    overlap = human_compounds.intersection(non_human_compounds)
    
    print(f"Compounds interacting with human kinases: {len(human_compounds)}")
    print(f"Compounds interacting with non-human kinases: {len(non_human_compounds)}")
    print(f"Compounds interacting with both: {len(overlap)}")
    
    if len(overlap) > 0:
        overlap_percentage_human = len(overlap) / len(human_compounds) * 100
        overlap_percentage_non_human = len(overlap) / len(non_human_compounds) * 100
        
        print(f"Percentage of human compounds also in non-human: {overlap_percentage_human:.2f}%")
        print(f"Percentage of non-human compounds also in human: {overlap_percentage_non_human:.2f}%")

def kinase_family_analysis(df_human, df_non_human):
    """
    Analyze kinase families present in each dataset.
    
    Args:
        df_human (pd.DataFrame): DataFrame with human kinase data
        df_non_human (pd.DataFrame): DataFrame with non-human kinase data
    """
    print("\n=== KINASE FAMILY ANALYSIS ===")
    
    # Extract kinase families from name (words before "kinase")
    def extract_kinase_family(name):
        if pd.isna(name):
            return "Unknown"
        parts = name.split()
        if len(parts) > 1 and parts[-1].lower() == 'kinase':
            return ' '.join(parts[:-1])
        return name
    
    df_human['kinase_family'] = df_human['target_kinase'].apply(extract_kinase_family)
    df_non_human['kinase_family'] = df_non_human['target_kinase'].apply(extract_kinase_family)
    
    # Top families
    print("Top 10 kinase families - Human:")
    human_families = df_human['kinase_family'].value_counts().head(10)
    for family, count in human_families.items():
        print(f"  {family}: {count}")
    
    print("\nTop 10 kinase families - Non-Human:")
    non_human_families = df_non_human['kinase_family'].value_counts().head(10)
    for family, count in non_human_families.items():
        print(f"  {family}: {count}")

def generate_visualizations(df_human, df_non_human, output_dir="analysis_output"):
    """
    Generate comparative visualizations.
    
    Args:
        df_human (pd.DataFrame): DataFrame with human kinase data
        df_non_human (pd.DataFrame): DataFrame with non-human kinase data
        output_dir (str): Directory to save visualizations
    """
    print("\n=== GENERATING VISUALIZATIONS ===")
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Convert values to pIC50
    df_human['pIC50'] = -np.log10(df_human['standard_value'] * 1e-9)
    df_non_human['pIC50'] = -np.log10(df_non_human['standard_value'] * 1e-9)
    
    # 1. pIC50 distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df_human['pIC50'].dropna(), bins=50, alpha=0.7, label='Human', density=True)
    plt.hist(df_non_human['pIC50'].dropna(), bins=50, alpha=0.7, label='Non-Human', density=True)
    plt.xlabel('pIC50')
    plt.ylabel('Density')
    plt.title('pIC50 Distribution - Human vs Non-Human')
    plt.legend()
    plt.savefig(f"{output_dir}/pic50_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Boxplot of pIC50 by standard value type
    combined_df = pd.concat([
        df_human[['standard_type', 'pIC50']].assign(source='Human'),
        df_non_human[['standard_type', 'pIC50']].assign(source='Non-Human')
    ])
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=combined_df, x='standard_type', y='pIC50', hue='source')
    plt.xlabel('Standard Value Type')
    plt.ylabel('pIC50')
    plt.title('pIC50 Distribution by Standard Value Type')
    plt.xticks(rotation=45)
    plt.savefig(f"{output_dir}/pic50_by_standard_type.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to: {output_dir}")

def main():
    """Main function for running the analysis."""
    # File paths (adjust as needed)
    human_file = "src/database/kinase_human_compounds.tsv"
    non_human_file = "src/database/kinase_non_human_compounds.tsv"
    
    try:
        # Load data
        df_human, df_non_human = load_data(human_file, non_human_file)
        
        # Analyses
        basic_statistics(df_human, df_non_human)
        activity_distribution(df_human, df_non_human)
        compound_overlap_analysis(df_human, df_non_human)
        kinase_family_analysis(df_human, df_non_human)
        generate_visualizations(df_human, df_non_human)
        
        print("\n✅ Analysis completed successfully!")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Make sure the TSV files were generated by the SQL scripts.")
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
