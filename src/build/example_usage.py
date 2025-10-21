"""
Example configuration and usage script for the modular build system.

This script demonstrates how to use the new modular architecture
to replace the original build.py functionality.
"""

import os
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from build.core import BuildConfig
from build.pipeline import BuildPipeline


def create_example_config(config_path: str = "build_config.json") -> BuildConfig:
    """Create an example configuration file."""
    
    config_data = {
        "project_name": "docktkinase_embeddings",
        "log_level": "INFO",
        "output_directory": "concatenated_embeddings",
        "spark_config": {
            "app_name": "DocktKinaseEmbeddings", 
            "master": "local[*]",
            "memory": "8g",
            "max_result_size": "2g"
        },
        "esm_config": {
            "model_name": "esm2_t33_650M_UR50D",
            "batch_size": 16,
            "device": "auto",
            "max_sequence_length": 1024
        },
        "fm4m_config": {
            "model_path": "/home/leon/Desktop/latent_extractor/ibm/FM4M",
            "batch_size": 32,
            "device": "auto"
        },
        "matrix_config": {
            "normalize": True,
            "cache_embeddings": True,
            "chunk_size": 1000
        },
        "binary_threshold": 1000.0,
        "validation": {
            "check_nan": True,
            "check_inf": True,
            "check_alignment": True,
            "save_reports": True
        }
    }
    
    config = BuildConfig(config_data)
    config.save(config_path)
    print(f"Example configuration saved to: {config_path}")
    
    return config


def run_modular_pipeline():
    """Run the complete modular pipeline."""
    
    # Get paths (adjust these to match your setup)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    # Input and output paths
    input_tsv = project_root / "src" / "database" / "kinase_non_human_compounds.tsv"
    output_dir = Path.cwd() / "concatenated_embeddings_modular"
    config_path = "build_config.json"
    
    print("🚀 Starting Modular Build Pipeline")
    print(f"Input TSV: {input_tsv}")
    print(f"Output Directory: {output_dir}")
    
    try:
        # Create or load configuration
        if not Path(config_path).exists():
            print(f"Creating example configuration...")
            config = create_example_config(config_path)
        else:
            print(f"Loading configuration from {config_path}")
            config = BuildConfig.from_json(config_path)
        
        # Initialize pipeline
        pipeline = BuildPipeline(config)
        
        # Run complete pipeline
        success = pipeline.run_complete_pipeline(
            input_tsv_path=input_tsv,
            output_dir=output_dir,
            matrix_type='embedding',  # or 'kinase' for kinase-specific processing
            binary_threshold=1000.0,  # 1000 nM threshold
            run_validation=True
        )
        
        if success:
            print("\n🎉 Pipeline completed successfully!")
            
            # Print summary
            summary = pipeline.get_pipeline_summary()
            print(f"\nExecuted steps: {summary['executed_steps']}")
            print(f"Total steps: {summary['total_steps']}")
            print(f"Overall success: {summary['success']}")
            
            # Results are saved automatically to output_dir/pipeline_results.json
            print(f"Detailed results saved to: {output_dir}/pipeline_results.json")
        else:
            print("\n❌ Pipeline failed!")
            sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Error running pipeline: {e}")
        sys.exit(1)


def run_individual_components():
    """Example of running individual components."""
    
    print("🔧 Running Individual Components Example")
    
    # Create config
    config = create_example_config("component_config.json")
    
    # Example: Generate only protein embeddings
    from src.build.embeddings import ProteinEmbedding
    
    protein_emb = ProteinEmbedding(config)
    # protein_emb.generate_embeddings(...)
    
    # Example: Validate existing matrices
    from src.build.validation import MatrixValidator
    
    validator = MatrixValidator(config)
    # validator.validate(...)
    
    print("Component examples initialized (see code for usage)")


def run_stratification_example():
    """Example of using the new stratification functionality."""
    
    print("🎯 Running Stratification Example")
    
    # Create config with stratification enabled
    config = create_example_config("stratification_config.json")
    
    # Enable stratification in config
    config.update({
        'stratification_enabled': True,
        'stratification_params': {
            'clustering_algorithm': 'dbscan',
            'similarity_threshold': 0.75,
            'cluster_min_size': 3,
            'stratify_by': 'both'
        }
    })
    
    # Example: Initialize and use stratifier directly
    from build.stratification import Stratifier, SplitValidator
    
    stratifier = Stratifier(config)
    print(f"Stratifier initialized with algorithm: {stratifier.clustering_algorithm}")
    print(f"Similarity threshold: {stratifier.similarity_threshold}")
    
    # Example: Validate existing pipeline integration
    pipeline = BuildPipeline(config)
    print(f"Pipeline includes stratifier: {'stratifier' in pipeline.components}")
    print(f"Pipeline includes split validator: {'split_validator' in pipeline.components}")
    
    print("Stratification example completed (see code for full usage)")


def main():
    """Main entry point."""
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "components":
            run_individual_components()
        elif sys.argv[1] == "stratification":
            run_stratification_example()
        else:
            print(f"Unknown command: {sys.argv[1]}")
            print("Available commands: components, stratification")
    else:
        run_modular_pipeline()


if __name__ == "__main__":
    main()
