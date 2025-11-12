"""
Base interface for validation components.

Provides abstract base class for different types of validators
in the protein-ligand interaction prediction pipeline.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from pathlib import Path

from src.build.core import BaseBuilder, BuildConfig


class BaseValidator(BaseBuilder):
    """Abstract base class for validation components."""
    
    def __init__(self, config: BuildConfig):
        """Initialize validator."""
        super().__init__(config)
        self.validation_results: Dict[str, Any] = {}
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def _validate_config(self) -> None:
        """Valida configuração base para validação."""
        super()._validate_config()
        # Validações específicas para validação podem ser adicionadas aqui
    
    def build(self, *args, **kwargs) -> Dict[str, Any]:
        """Build method that delegates to validate."""
        validation_success = self.validate(*args, **kwargs)
        return {
            'validation_passed': validation_success,
            'errors': self.errors,
            'warnings': self.warnings,
            'results': self.validation_results
        }
    
    @abstractmethod
    def validate(self, *args, **kwargs) -> bool:
        """
        Perform validation.
        
        Returns:
            True if validation passes, False otherwise
        """
        pass
    
    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.errors.append(message)
        self.logger.error(message)
    
    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)
        self.logger.warning(message)
    
    def clear_results(self) -> None:
        """Clear previous validation results."""
        self.validation_results.clear()
        self.errors.clear()
        self.warnings.clear()
    
    def has_errors(self) -> bool:
        """Check if there are validation errors."""
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        """Check if there are validation warnings."""
        return len(self.warnings) > 0
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get validation summary.
        
        Returns:
            Dictionary with validation summary
        """
        return {
            'passed': not self.has_errors(),
            'error_count': len(self.errors),
            'warning_count': len(self.warnings),
            'errors': self.errors.copy(),
            'warnings': self.warnings.copy(),
            'results': self.validation_results.copy()
        }
    
    def print_summary(self) -> None:
        """Print validation summary to console."""
        summary = self.get_summary()
        
        if summary['passed']:
            print("\n✅ Validation PASSED")
        else:
            print("\n❌ Validation FAILED")
        
        if summary['error_count'] > 0:
            print(f"\nErrors ({summary['error_count']}):")
            for error in summary['errors']:
                print(f"  ❌ {error}")
        
        if summary['warning_count'] > 0:
            print(f"\nWarnings ({summary['warning_count']}):")
            for warning in summary['warnings']:
                print(f"  ⚠️ {warning}")
        
        if summary['results']:
            print("\nValidation Results:")
            self._print_results(summary['results'])
    
    def _print_results(self, results: Dict[str, Any], indent: int = 0) -> None:
        """Recursively print validation results."""
        spaces = "  " * indent
        for key, value in results.items():
            if isinstance(value, dict):
                print(f"{spaces}{key}:")
                self._print_results(value, indent + 1)
            else:
                print(f"{spaces}{key}: {value}")
    
    def save_report(self, output_path: Union[str, Path]) -> bool:
        """
        Save validation report to file.
        
        Args:
            output_path: Path to save report
            
        Returns:
            True if successful, False otherwise
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            summary = self.get_summary()
            self.save_json(summary, output_path)
            
            self.logger.info(f"Validation report saved to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving validation report: {e}")
            return False
    
    def check_file_exists(self, file_path: Union[str, Path], 
                         description: str = "File") -> bool:
        """
        Check if file exists and add error if not.
        
        Args:
            file_path: Path to check
            description: Description for error message
            
        Returns:
            True if file exists, False otherwise
        """
        file_path = Path(file_path)
        if not file_path.exists():
            self.add_error(f"{description} not found: {file_path}")
            return False
        return True
    
    def check_array_shape(self, array: np.ndarray, 
                         expected_shape: Tuple[int, ...],
                         name: str) -> bool:
        """
        Check if array has expected shape.
        
        Args:
            array: Array to check
            expected_shape: Expected shape tuple
            name: Name for error message
            
        Returns:
            True if shape matches, False otherwise
        """
        if array.shape != expected_shape:
            self.add_error(f"Incorrect shape for {name}. "
                         f"Expected {expected_shape}, got {array.shape}")
            return False
        return True
    
    def check_no_nan_values(self, array: np.ndarray, name: str) -> bool:
        """
        Check if array contains NaN values.
        
        Args:
            array: Array to check
            name: Name for error message
            
        Returns:
            True if no NaN values, False otherwise
        """
        try:
            # Try to check for NaN - works for numeric types
            if np.isnan(array).any():
                self.add_error(f"{name} contains NaN values")
                return False
        except (TypeError, ValueError):
            # For non-numeric types (e.g., object dtype), try converting
            try:
                array_numeric = array.astype(float)
                if np.isnan(array_numeric).any():
                    self.add_error(f"{name} contains NaN values")
                    return False
            except (TypeError, ValueError):
                # If conversion fails, skip NaN check for this array
                self.add_warning(f"{name} has non-numeric dtype, skipping NaN check")
        return True
    
    def check_no_inf_values(self, array: np.ndarray, name: str) -> bool:
        """
        Check if array contains infinite values.
        
        Args:
            array: Array to check
            name: Name for error message
            
        Returns:
            True if no infinite values, False otherwise
        """
        try:
            # Try to check for inf - works for numeric types
            if np.isinf(array).any():
                self.add_error(f"{name} contains infinite values")
                return False
        except (TypeError, ValueError):
            # For non-numeric types, try converting
            try:
                array_numeric = array.astype(float)
                if np.isinf(array_numeric).any():
                    self.add_error(f"{name} contains infinite values")
                    return False
            except (TypeError, ValueError):
                # If conversion fails, skip inf check for this array
                self.add_warning(f"{name} has non-numeric dtype, skipping inf check")
        return True
    
    def check_all_zeros(self, array: np.ndarray, name: str, 
                       axis: Optional[int] = None) -> bool:
        """
        Check for all-zero rows/columns/values.
        
        Args:
            array: Array to check
            name: Name for warning message
            axis: Axis to check along (None for entire array)
            
        Returns:
            True if no all-zero sections found, False otherwise
        """
        if axis is None:
            if np.all(array == 0):
                self.add_warning(f"{name} contains all zero values")
                return False
        else:
            zero_mask = np.all(array == 0, axis=axis)
            if np.any(zero_mask):
                zero_count = np.sum(zero_mask)
                axis_name = "rows" if axis == 1 else "columns"
                self.add_warning(f"{name} contains {zero_count} all-zero {axis_name}")
                return False
        return True
    
    def check_value_range(self, array: np.ndarray, name: str,
                         min_val: Optional[float] = None,
                         max_val: Optional[float] = None) -> bool:
        """
        Check if array values are within expected range.
        
        Args:
            array: Array to check
            name: Name for error message
            min_val: Minimum expected value
            max_val: Maximum expected value
            
        Returns:
            True if values are in range, False otherwise
        """
        valid = True
        
        if min_val is not None:
            if np.any(array < min_val):
                self.add_error(f"{name} contains values below {min_val}")
                valid = False
        
        if max_val is not None:
            if np.any(array > max_val):
                self.add_error(f"{name} contains values above {max_val}")
                valid = False
        
        return valid
