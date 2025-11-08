"""
Data Loader for Embeddings Generation

Manages loading and preprocessing of input data (protein sequences, SMILES, etc.)
for embedding generation.
"""

from typing import List, Optional, Union, Tuple
from pathlib import Path
import pandas as pd


class DataManager:
    """
    Manages data loading and preprocessing for embedding generation.
    
    Handles:
    - Loading protein sequences from various formats
    - Loading SMILES strings
    - Validation of input data
    - Batch preparation
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize DataManager.
        
        Args:
            verbose: Whether to print progress information
        """
        self.verbose = verbose
        
    def load_sequences(
        self,
        source: Union[str, Path, List[str], pd.DataFrame],
        sequence_column: str = 'sequence',
        id_column: Optional[str] = None
    ) -> Tuple[List[str], List[str]]:
        """
        Load protein sequences from various sources.
        
        Args:
            source: Path to file, list of sequences, or DataFrame
            sequence_column: Column name containing sequences (for DataFrame/CSV)
            id_column: Column name for IDs (optional)
            
        Returns:
            Tuple of (sequences, ids)
        """
        if isinstance(source, (str, Path)):
            return self._load_from_file(source, sequence_column, id_column)
        elif isinstance(source, list):
            return self._load_from_list(source)
        elif isinstance(source, pd.DataFrame):
            return self._load_from_dataframe(source, sequence_column, id_column)
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")
    
    def _load_from_file(
        self, 
        filepath: Union[str, Path],
        sequence_column: str,
        id_column: Optional[str]
    ) -> Tuple[List[str], List[str]]:
        """Load sequences from file (CSV, TSV, FASTA)."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Check file extension
        if filepath.suffix in ['.csv', '.tsv', '.txt']:
            sep = '\t' if filepath.suffix == '.tsv' else ','
            df = pd.read_csv(filepath, sep=sep)
            return self._load_from_dataframe(df, sequence_column, id_column)
        
        elif filepath.suffix in ['.fasta', '.fa']:
            return self._load_from_fasta(filepath)
        
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    def _load_from_fasta(self, filepath: Path) -> Tuple[List[str], List[str]]:
        """Load sequences from FASTA file."""
        sequences = []
        ids = []
        current_id = None
        current_seq = []
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    # Save previous sequence
                    if current_id is not None:
                        ids.append(current_id)
                        sequences.append(''.join(current_seq))
                    # Start new sequence
                    current_id = line[1:].split()[0]  # Get first word after >
                    current_seq = []
                elif line:
                    current_seq.append(line)
            
            # Save last sequence
            if current_id is not None:
                ids.append(current_id)
                sequences.append(''.join(current_seq))
        
        if self.verbose:
            print(f"   ✅ Loaded {len(sequences)} sequences from FASTA")
        
        return sequences, ids
    
    def _load_from_list(self, sequences: List[str]) -> Tuple[List[str], List[str]]:
        """Load sequences from list."""
        ids = [f"seq_{i}" for i in range(len(sequences))]
        
        if self.verbose:
            print(f"   ✅ Loaded {len(sequences)} sequences from list")
        
        return sequences, ids
    
    def _load_from_dataframe(
        self,
        df: pd.DataFrame,
        sequence_column: str,
        id_column: Optional[str]
    ) -> Tuple[List[str], List[str]]:
        """Load sequences from DataFrame."""
        if sequence_column not in df.columns:
            raise ValueError(f"Column '{sequence_column}' not found in DataFrame")
        
        sequences = df[sequence_column].tolist()
        
        if id_column and id_column in df.columns:
            ids = df[id_column].tolist()
        else:
            ids = [f"seq_{i}" for i in range(len(sequences))]
        
        if self.verbose:
            print(f"   ✅ Loaded {len(sequences)} sequences from DataFrame")
        
        return sequences, ids
    
    def load_smiles(
        self,
        source: Union[str, Path, List[str], pd.DataFrame],
        smiles_column: str = 'smiles',
        id_column: Optional[str] = None
    ) -> Tuple[List[str], List[str]]:
        """
        Load SMILES strings.
        
        Args:
            source: Path to file, list of SMILES, or DataFrame
            smiles_column: Column name containing SMILES
            id_column: Column name for IDs (optional)
            
        Returns:
            Tuple of (smiles_list, ids)
        """
        # Reuse sequence loading logic (works the same way)
        return self.load_sequences(source, smiles_column, id_column)
