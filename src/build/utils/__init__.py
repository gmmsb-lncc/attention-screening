"""
Utilities module for the build system.

Provides basic utilities with fallback for missing functions.
"""

# Import what we can from memory_utils
try:
    from build.utils.memory_utils import *
except ImportError as e:
    print(f"Warning: Could not import memory_utils: {e}")
    MemoryContext = None
    memory_monitor = lambda *args, **kwargs: lambda f: f  # Dummy decorator

# Import what we can from file_utils
try:
    from build.utils.file_utils import *
except ImportError as e:
    print(f"Warning: Could not import file_utils: {e}")
    ensure_directory = lambda path: None
    load_numpy = lambda path: None
    save_numpy = lambda data, path: None
    load_tsv = lambda path: None

# Import what we can from spark_utils
try:
    from build.utils.spark_utils import SparkManager
except ImportError as e:
    print(f"Warning: Could not import spark_utils: {e}")
    SparkManager = None

# Import what we can from logging_utils
try:
    from build.utils.logging_utils import ProgressLogger
except ImportError as e:
    print(f"Warning: Could not import logging_utils: {e}")
    ProgressLogger = None

# Dummy functions for missing utilities
def optimize_batch_size(*args, **kwargs):
    return 32  # Default batch size

__all__ = [
    'MemoryContext',
    'SparkManager', 
    'ProgressLogger',
    'memory_monitor',
    'ensure_directory',
    'optimize_batch_size',
    'load_numpy',
    'save_numpy',
    'load_tsv'
]
