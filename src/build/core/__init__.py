"""
Core module for the build system.

Provides base classes, configuration management, constants, and exceptions.
"""

from build.core.config import BuildConfig
from build.core.base_builder import BaseBuilder
from build.core.exceptions import BuildException
from build.core.constants import BuildConstants

__all__ = [
    'BuildConfig',
    'BaseBuilder',
    'BuildException',
    'BuildConstants'
]
