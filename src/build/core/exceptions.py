"""
Custom exceptions for the build module.
"""

class BuildException(Exception):
    """Base exception for build module errors."""
    pass

class ConfigurationError(BuildException):
    """Configuration error."""
    pass

class EmbeddingError(BuildException):
    """Embedding generation error."""
    pass

class MatrixError(BuildException):
    """Matrix construction error."""
    pass

class ValidationError(BuildException):
    """Validation error."""
    pass

class PipelineError(BuildException):
    """Pipeline execution error."""
    pass

class DependencyError(BuildException):
    """Dependency not found error."""
    pass

class BuildFileNotFoundError(BuildException):
    """File not found error (custom to avoid builtin conflict)."""
    pass

class BuildMemoryError(BuildException):
    """Memory insufficient error (custom to avoid builtin conflict)."""
    pass

class ModelLoadError(EmbeddingError):
    """Model loading error."""
    pass
