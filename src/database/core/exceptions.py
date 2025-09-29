"""
Custom exceptions for database operations.
"""


class DatabaseError(Exception):
    """Base exception for database operations."""
    pass


class AnalysisError(DatabaseError):
    """Exception raised during analysis operations."""
    pass


class ProcessingError(DatabaseError):
    """Exception raised during data processing operations."""
    pass


class ConfigurationError(DatabaseError):
    """Exception raised for configuration-related issues."""
    pass


class DataValidationError(DatabaseError):
    """Exception raised when data validation fails."""
    pass
