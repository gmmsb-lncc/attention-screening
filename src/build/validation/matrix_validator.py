"""
Matrix validation for embedding matrices and concatenated arrays.

Validates matrix dimensions, alignment, and data integrity
for the protein-ligand interaction prediction pipeline.
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from pathlib import Path

from ..core import BuildConfig
from .base_validator import BaseValidator


class MatrixValidator(BaseValidator):
    """Validates embedding matrices and data alignment."""
    
    def __init__(self, config: BuildConfig):
        """Initialize matrix validator."""
        super().__init__(config)
        self.matrices: Dict[str, np.ndarray] = {}
        self.matrix_paths: Dict[str, Path] = {}
    
    def _validate_config(self) -> None:
        """Validate matrix validator configuration."""
        # Base validation - pode ser expandida conforme necessário
        pass
    
    def build(self) -> Dict[str, Any]:
        """
        Build validation results.
        
        Returns:
            Dictionary with validation results
        """
        return {
            'status': 'success',
            'matrices': len(self.matrices),
            'message': 'MatrixValidator initialized successfully'
        }
    
    def validate(self, *args, **kwargs) -> bool:
        """
        Perform matrix validation.
        
        Returns:
            True if validation passes, False otherwise
        """
        # Implementação básica - pode ser expandida
        return True
    
    def load_matrix(self, matrix_path: Union[str, Path], name: str) -> Optional[np.ndarray]:
        """
        Load matrix from file.
        
        Args:
            matrix_path: Path to matrix file
            name: Name identifier for the matrix
            
        Returns:
            Loaded matrix or None if failed
        """
        try:
            matrix_path = Path(matrix_path)
            
            if not self.check_file_exists(matrix_path, f"Matrix {name}"):
                return None
            
            matrix = np.load(str(matrix_path), allow_pickle=True)
            self.matrices[name] = matrix
            self.matrix_paths[name] = matrix_path
            
            self.logger.info(f"Loaded matrix {name}: {matrix.shape}")
            return matrix
            
        except Exception as e:
            self.add_error(f"Failed to load matrix {name} from {matrix_path}: {e}")
            return None
    
    def validate_matrix_dimensions(self, 
                                 matrix_name: str,
                                 expected_rows: Optional[int] = None,
                                 expected_cols: Optional[int] = None,
                                 min_rows: Optional[int] = None,
                                 min_cols: Optional[int] = None) -> bool:
        """
        Validate matrix dimensions.
        
        Args:
            matrix_name: Name of the matrix to validate
            expected_rows: Expected number of rows (exact)
            expected_cols: Expected number of columns (exact)
            min_rows: Minimum number of rows
            min_cols: Minimum number of columns
            
        Returns:
            True if dimensions are valid, False otherwise
        """
        if matrix_name not in self.matrices:
            self.add_error(f"Matrix {matrix_name} not loaded")
            return False
        
        matrix = self.matrices[matrix_name]
        rows, cols = matrix.shape
        valid = True
        
        # Check exact dimensions
        if expected_rows is not None and rows != expected_rows:
            self.add_error(f"Matrix {matrix_name} has {rows} rows, expected {expected_rows}")
            valid = False
        
        if expected_cols is not None and cols != expected_cols:
            self.add_error(f"Matrix {matrix_name} has {cols} columns, expected {expected_cols}")
            valid = False
        
        # Check minimum dimensions
        if min_rows is not None and rows < min_rows:
            self.add_error(f"Matrix {matrix_name} has {rows} rows, minimum required {min_rows}")
            valid = False
        
        if min_cols is not None and cols < min_cols:
            self.add_error(f"Matrix {matrix_name} has {cols} columns, minimum required {min_cols}")
            valid = False
        
        if valid:
            self.validation_results[f"{matrix_name}_dimensions"] = {
                'rows': rows,
                'cols': cols,
                'shape': f"({rows}, {cols})",
                'valid': True
            }
        
        return valid
    
    def validate_matrix_alignment(self, *matrix_names: str) -> bool:
        """
        Validate that matrices have aligned dimensions (same number of rows).
        
        Args:
            matrix_names: Names of matrices to check alignment
            
        Returns:
            True if matrices are aligned, False otherwise
        """
        if len(matrix_names) < 2:
            self.add_error("Need at least 2 matrices for alignment check")
            return False
        
        loaded_matrices = []
        shapes = []
        
        for name in matrix_names:
            if name not in self.matrices:
                self.add_error(f"Matrix {name} not loaded for alignment check")
                return False
            loaded_matrices.append(self.matrices[name])
            shapes.append(self.matrices[name].shape)
        
        # Check if all matrices have the same number of rows
        first_rows = shapes[0][0]
        aligned = True
        
        for i, (name, shape) in enumerate(zip(matrix_names, shapes)):
            if shape[0] != first_rows:
                self.add_error(f"Matrix {name} has {shape[0]} rows, "
                             f"but {matrix_names[0]} has {first_rows} rows")
                aligned = False
        
        if aligned:
            self.validation_results['matrix_alignment'] = {
                'matrices': list(matrix_names),
                'aligned_rows': first_rows,
                'shapes': {name: shape for name, shape in zip(matrix_names, shapes)},
                'valid': True
            }
        
        return aligned
    
    def validate_matrix_data_integrity(self, matrix_name: str) -> bool:
        """
        Validate matrix data integrity (no NaN, inf, etc.).
        
        Args:
            matrix_name: Name of the matrix to validate
            
        Returns:
            True if data integrity is good, False otherwise
        """
        if matrix_name not in self.matrices:
            self.add_error(f"Matrix {matrix_name} not loaded")
            return False
        
        matrix = self.matrices[matrix_name]
        valid = True
        
        # Check for NaN values
        if not self.check_no_nan_values(matrix, f"Matrix {matrix_name}"):
            valid = False
        
        # Check for infinite values
        if not self.check_no_inf_values(matrix, f"Matrix {matrix_name}"):
            valid = False
        
        # Check for all-zero rows (warning only)
        self.check_all_zeros(matrix, f"Matrix {matrix_name}", axis=1)
        
        # Additional statistics
        if valid:
            self.validation_results[f"{matrix_name}_data_integrity"] = {
                'nan_count': int(np.isnan(matrix).sum()),
                'inf_count': int(np.isinf(matrix).sum()),
                'zero_rows': int(np.all(matrix == 0, axis=1).sum()),
                'min_value': float(np.min(matrix)),
                'max_value': float(np.max(matrix)),
                'mean_value': float(np.mean(matrix)),
                'std_value': float(np.std(matrix)),
                'valid': True
            }
        
        return valid
    
    def validate_concatenated_embeddings(self, 
                                       concatenated_path: Union[str, Path],
                                       labels_path: Union[str, Path],
                                       original_tsv_path: Optional[Union[str, Path]] = None) -> bool:
        """
        Validate concatenated embeddings matrix (main validation from checkConcatenate.py).
        
        Args:
            concatenated_path: Path to concatenated embeddings matrix
            labels_path: Path to labels matrix
            original_tsv_path: Path to original TSV for row count verification
            
        Returns:
            True if validation passes, False otherwise
        """
        self.clear_results()
        
        # Load matrices
        concatenated_matrix = self.load_matrix(concatenated_path, "concatenated_embeddings")
        labels_matrix = self.load_matrix(labels_path, "interaction_labels")
        
        if concatenated_matrix is None or labels_matrix is None:
            return False
        
        valid = True
        
        # Get expected row count from original TSV if provided
        expected_rows = None
        if original_tsv_path is not None:
            try:
                original_tsv_path = Path(original_tsv_path)
                if self.check_file_exists(original_tsv_path, "Original TSV"):
                    df = pd.read_csv(original_tsv_path, sep='\t')
                    expected_rows = len(df)
                    self.logger.info(f"Expected rows from TSV: {expected_rows}")
            except Exception as e:
                self.add_warning(f"Could not read original TSV for row count: {e}")
        
        # Validate concatenated embeddings dimensions
        if not self.validate_matrix_dimensions(
            "concatenated_embeddings",
            expected_rows=expected_rows,
            min_rows=1,
            min_cols=1
        ):
            valid = False
        
        # Validate labels dimensions
        if not self.validate_matrix_dimensions(
            "interaction_labels",
            expected_rows=expected_rows,
            min_rows=1
        ):
            valid = False
        
        # Validate matrix alignment
        if not self.validate_matrix_alignment("concatenated_embeddings", "interaction_labels"):
            valid = False
        
        # Validate data integrity
        if not self.validate_matrix_data_integrity("concatenated_embeddings"):
            valid = False
        
        if not self.validate_matrix_data_integrity("interaction_labels"):
            valid = False
        
        return valid
    
    def validate_embedding_matrix_construction(self,
                                             protein_embeddings_path: Union[str, Path],
                                             ligand_embeddings_path: Union[str, Path],
                                             concatenated_path: Union[str, Path]) -> bool:
        """
        Validate embedding matrix construction process.
        
        Args:
            protein_embeddings_path: Path to protein embeddings
            ligand_embeddings_path: Path to ligand embeddings  
            concatenated_path: Path to concatenated matrix
            
        Returns:
            True if construction is valid, False otherwise
        """
        self.clear_results()
        
        # Load all matrices
        protein_matrix = self.load_matrix(protein_embeddings_path, "protein_embeddings")
        ligand_matrix = self.load_matrix(ligand_embeddings_path, "ligand_embeddings")
        concatenated_matrix = self.load_matrix(concatenated_path, "concatenated_embeddings")
        
        if None in [protein_matrix, ligand_matrix, concatenated_matrix]:
            return False
        
        valid = True
        
        # Check that protein and ligand have same number of rows
        if not self.validate_matrix_alignment("protein_embeddings", "ligand_embeddings"):
            valid = False
        
        # Check concatenated dimensions
        expected_rows = protein_matrix.shape[0]
        expected_cols = protein_matrix.shape[1] + ligand_matrix.shape[1]
        
        if not self.validate_matrix_dimensions(
            "concatenated_embeddings",
            expected_rows=expected_rows,
            expected_cols=expected_cols
        ):
            valid = False
        
        # Validate data integrity for all matrices
        for matrix_name in ["protein_embeddings", "ligand_embeddings", "concatenated_embeddings"]:
            if not self.validate_matrix_data_integrity(matrix_name):
                valid = False
        
        # Check if concatenation was done correctly
        if valid:
            try:
                reconstructed = np.concatenate([protein_matrix, ligand_matrix], axis=1)
                if not np.allclose(concatenated_matrix, reconstructed, rtol=1e-10):
                    self.add_error("Concatenated matrix does not match reconstruction from components")
                    valid = False
                else:
                    self.validation_results['concatenation_check'] = {
                        'reconstruction_matches': True,
                        'protein_dims': protein_matrix.shape,
                        'ligand_dims': ligand_matrix.shape,
                        'concatenated_dims': concatenated_matrix.shape
                    }
            except Exception as e:
                self.add_error(f"Failed to verify concatenation: {e}")
                valid = False
        
        return valid
    
    def validate(self, 
                concatenated_path: Union[str, Path],
                labels_path: Union[str, Path],
                original_tsv_path: Optional[Union[str, Path]] = None,
                protein_embeddings_path: Optional[Union[str, Path]] = None,
                ligand_embeddings_path: Optional[Union[str, Path]] = None) -> bool:
        """
        Main validation method.
        
        Args:
            concatenated_path: Path to concatenated embeddings
            labels_path: Path to labels
            original_tsv_path: Path to original TSV (optional)
            protein_embeddings_path: Path to protein embeddings (optional)
            ligand_embeddings_path: Path to ligand embeddings (optional)
            
        Returns:
            True if all validations pass, False otherwise
        """
        valid = True
        
        # Main concatenated embeddings validation
        if not self.validate_concatenated_embeddings(
            concatenated_path, labels_path, original_tsv_path
        ):
            valid = False
        
        # Optional: validate construction process if component matrices are provided
        if protein_embeddings_path and ligand_embeddings_path:
            if not self.validate_embedding_matrix_construction(
                protein_embeddings_path, ligand_embeddings_path, concatenated_path
            ):
                valid = False
        
        return valid
