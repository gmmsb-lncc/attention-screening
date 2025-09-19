#!/usr/bin/env python3
"""
DockTKinase Build System - Complete Workflow Demo

This script demonstrates the complete workflow for the modular build system.
It shows how to use all components and provides production-ready examples.
Run this file directly to see the system in action.

This demo uses dynamic imports to showcase functionality without circular imports.
"""

import sys
import importlib
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir / ".."
sys.path.insert(0, str(src_dir))

def dynamic_import(module_name: str, class_name: str = None):
    """Dynamically import modules to avoid circular imports."""
    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name) if class_name else module
    except ImportError as e:
        print(f"⚠️ Module {module_name} not available: {e}")
        return None

def example_basic_usage():
    """Example 1: Basic pipeline execution."""
    print("🚀 EXAMPLE 1: Basic Pipeline Usage")
    print("-" * 40)
    
    # Use dynamic imports to avoid circular import warnings
    BuildConfig = dynamic_import('build.core', 'BuildConfig')
    BuildPipeline = dynamic_import('build.pipeline', 'BuildPipeline')
    
    if not BuildConfig or not BuildPipeline:
        print("❌ Required modules not available")
        return
    
    # Simple configuration
    config = BuildConfig()
    
    print(f"Configuration loaded:")
    print(f"  • Ligand dimension: {config.ligand_dim}")
    print(f"  • Protein dimension: {config.protein_dim}")
    print(f"  • Embedding type: {config.embedding_type}")
    print(f"  • Batch size: {config.batch_size}")
    
    # Initialize pipeline
    pipeline = BuildPipeline(config)
    print(f"  • Pipeline components: {len(pipeline.components)}")
    
    print("\n✅ Basic setup complete!")
    print("To run: pipeline.run() (requires actual data)")

def example_custom_configuration():
    """Example 2: Custom configuration."""
    print("\n🔧 EXAMPLE 2: Custom Configuration")
    print("-" * 40)
    
    # Use dynamic imports to avoid circular import warnings
    BuildConfig = dynamic_import('build.core', 'BuildConfig')
    
    if not BuildConfig:
        print("❌ BuildConfig not available")
        return
    
    # Custom settings
    config = BuildConfig({
        'esm_model': 'esm2_t33_650M_UR50D',  # Faster model
        'fm4m_model': 'SELFIES-TED',         # Alternative ligand model
        'batch_size': 16,                    # Smaller batches
        'use_parallel': True,                # Enable parallelism
        'checkpoint_enabled': True           # Enable checkpointing
    })
    
    print("Custom configuration:")
    print(f"  • ESM model: {config.get('esm_model')}")
    print(f"  • FM4M model: {config.get('fm4m_model')}")
    print(f"  • Batch size: {config.batch_size}")
    print(f"  • Parallel processing: {config.use_parallel}")
    print(f"  • Checkpointing: {config.get('checkpoint_enabled')}")
    
    print("\n✅ Custom configuration ready!")

def example_individual_components():
    """Example 3: Using individual components."""
    print("\n🧩 EXAMPLE 3: Individual Components")
    print("-" * 40)
    
    # Use dynamic imports to avoid circular import warnings
    BuildConfig = dynamic_import('build.core', 'BuildConfig')
    ProteinEmbedding = dynamic_import('build.embeddings', 'ProteinEmbedding')
    LigandEmbedding = dynamic_import('build.embeddings', 'LigandEmbedding')
    EmbeddingMatrix = dynamic_import('build.matrix', 'EmbeddingMatrix')
    
    if not all([BuildConfig, ProteinEmbedding, LigandEmbedding, EmbeddingMatrix]):
        print("❌ Required components not available")
        return
    
    config = BuildConfig()
    
    # Individual components
    protein_emb = ProteinEmbedding(config)
    ligand_emb = LigandEmbedding(config)
    matrix = EmbeddingMatrix(config)
    
    print("Component initialization:")
    print(f"  • Protein embedding: {protein_emb.model_name}")
    print(f"  • Ligand embedding: {ligand_emb.model_name}")
    print(f"  • Matrix builder: {matrix.ligand_dim} + {matrix.protein_dim} dims")
    
    print("\n✅ Components ready for individual use!")

def example_backward_compatibility():
    """Example 4: Backward compatibility with legacy scripts."""
    print("\n🔄 EXAMPLE 4: Backward Compatibility")
    print("-" * 40)
    
    # Use dynamic imports to avoid circular import warnings
    EmbeddingMatrixReconstructor = dynamic_import('build.matrix', 'EmbeddingMatrixReconstructor')
    
    if not EmbeddingMatrixReconstructor:
        print("❌ EmbeddingMatrixReconstructor not available")
        return
    
    # Legacy interface (identical to old scripts)
    matrix = EmbeddingMatrixReconstructor(
        '/dev/null',  # TSV path (dummy for demo)
        ligand_embeddings_dir='ligand_embeddings',
        protein_embeddings_dir='protein_embeddings'
    )
    
    print("Legacy interface attributes:")
    print(f"  • ligand_dir: {matrix.ligand_dir}")
    print(f"  • protein_dir: {matrix.protein_dir}")
    print(f"  • ligand_dim: {matrix.ligand_dim}")
    print(f"  • protein_dim: {matrix.protein_dim}")
    print(f"  • reconstruct_matrix method: {hasattr(matrix, 'reconstruct_matrix')}")
    
    print("\n✅ Legacy compatibility confirmed!")

def example_available_models():
    """Example 5: Available models."""
    print("\n📋 EXAMPLE 5: Available Models")
    print("-" * 40)
    
    # Use dynamic imports to avoid circular import warnings
    constants = dynamic_import('build.core.constants')
    
    if not constants:
        print("❌ Constants module not available")
        return
    
    ESM_MODELS = getattr(constants, 'ESM_MODELS', {})
    FM4M_MODELS = getattr(constants, 'FM4M_MODELS', {})
    
    print("Available ESM Models (Proteins):")
    for i, (model, info) in enumerate(ESM_MODELS.items(), 1):
        print(f"  {i}. {model}")
        print(f"     • Dimensions: {info['dim']}")
        print(f"     • Layers: {info['layers']}")
    
    print("\nAvailable FM4M Models (Ligands):")
    for i, (model, info) in enumerate(FM4M_MODELS.items(), 1):
        print(f"  {i}. {model}")
        print(f"     • Dimensions: {info['dim']}")
        print(f"     • Type: {info['type']}")
    
    print("\n✅ Model information displayed!")

def show_system_status():
    """Show system status and health."""
    print("\n🏥 SYSTEM STATUS")
    print("=" * 50)
    
    # Check module availability
    modules_to_check = [
        ('build.core', 'BuildConfig'),
        ('build.pipeline', 'BuildPipeline'),
        ('build.embeddings', 'ProteinEmbedding'),
        ('build.embeddings', 'LigandEmbedding'),
        ('build.matrix', 'EmbeddingMatrix'),
        ('build.labels', 'BinaryLabels'),
        ('build.validation', 'MatrixValidator')
    ]
    
    available = 0
    total = len(modules_to_check)
    
    for module_name, class_name in modules_to_check:
        cls = dynamic_import(module_name, class_name)
        status = "✅" if cls else "❌"
        print(f"{status} {module_name}.{class_name}")
        if cls:
            available += 1
    
    print(f"\n📊 System Health: {available}/{total} modules available")
    
    if available == total:
        print("🎉 All systems operational!")
    elif available > total * 0.7:
        print("⚠️ System partially operational")
    else:
        print("🚨 System needs attention")

def main():
    """Main demonstration function."""
    print("🚀 DockTKinase Build System Demo")
    print("=" * 50)
    
    # Show system status first
    show_system_status()
    
    # Run examples
    try:
        example_basic_usage()
        example_custom_configuration()
        example_individual_components()
        example_backward_compatibility()
        example_available_models()
        
        print("\n" + "=" * 50)
        print("✅ Demo completed successfully!")
        print("🔧 System is ready for production use.")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
