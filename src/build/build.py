#!/usr/bin/env python3
"""
DockTKinase Build System - Main Entry Point

This script serves as the main entry point for the modular build system.
For demonstrations and examples, use build_demo.py instead.
"""

import sys
from pathlib import Path

def main():
    """Main entry point for the build system."""
    print("🚀 DockTKinase Build System")
    print("=" * 40)
    print("Main entry point for the modular build system.")
    print("\nFor demonstrations and examples:")
    print("  python build_demo.py")
    print("\nFor production use, import modules directly:")
    print("  from src.build.core import BuildConfig")
    print("  from src.build.pipeline import BuildPipeline")
    print("  from src.build.embeddings import ProteinEmbedding, LigandEmbedding")
    print("  from src.build.matrix import EmbeddingMatrix")
    print("\nSystem is ready for production use!")

if __name__ == "__main__":
    main()
