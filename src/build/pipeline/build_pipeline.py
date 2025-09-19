"""
Main build pipeline orchestrator.

Coordinates all modules to execute the complete embedding matrix
construction pipeline for protein-ligand interaction prediction.
"""

import os
import sys
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from ..core import BaseBuilder, BuildConfig
from ..embeddings import ProteinEmbedding, LigandEmbedding
from ..matrix import EmbeddingMatrix, KinaseMatrix  
from ..labels import InteractionLabels, BinaryLabels
from ..validation import MatrixValidator


class BuildPipeline(BaseBuilder):
    """Main pipeline coordinator for the build process."""
    
    def __init__(self, config: BuildConfig):
        """Initialize build pipeline."""
        super().__init__(config)
        self.results: Dict[str, Any] = {}
        self.components: Dict[str, BaseBuilder] = {}
        
        # Initialize components
        self._initialize_components()
    
    def _validate_config(self) -> None:
        """Validate pipeline configuration."""
        # Base validation - pode ser expandida conforme necessário
        pass
    
    def build(self) -> Dict[str, Any]:
        """
        Execute the complete build pipeline.
        
        Returns:
            Dictionary with pipeline results
        """
        try:
            self.logger.info("Starting complete build pipeline")
            
            # Implementação básica - pode ser expandida
            results = {
                'status': 'success',
                'components': len(self.components),
                'message': 'Pipeline initialized successfully'
            }
            
            return results
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            return {
                'status': 'error', 
                'error': str(e),
                'message': 'Pipeline execution failed'
            }
    
    def _initialize_components(self) -> None:
        """Initialize all pipeline components."""
        try:
            self.components = {
                'protein_embedding': ProteinEmbedding(self.config),
                'ligand_embedding': LigandEmbedding(self.config),
                'embedding_matrix': EmbeddingMatrix(self.config),
                'kinase_matrix': KinaseMatrix(self.config),
                'matrix_validator': MatrixValidator(self.config)
            }
            self.logger.info("Initialized all pipeline components")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    def run_embedding_generation(self, 
                               input_tsv_path: Union[str, Path],
                               output_dir: Optional[Union[str, Path]] = None) -> bool:
        """
        Run embedding generation for proteins and ligands.
        
        Args:
            input_tsv_path: Path to input TSV file
            output_dir: Output directory (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info("🟡 Starting embedding generation phase")
            
            input_tsv_path = Path(input_tsv_path)
            
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate protein embeddings
            self.logger.info("Generating protein embeddings...")
            protein_embedding = self.components['protein_embedding']
            protein_success = protein_embedding.generate_embeddings(
                tsv_path=input_tsv_path,
                output_dir=output_dir
            )
            
            if not protein_success:
                self.logger.error("Failed to generate protein embeddings")
                return False
            
            # Generate ligand embeddings  
            self.logger.info("Generating ligand embeddings...")
            ligand_embedding = self.components['ligand_embedding']
            ligand_success = ligand_embedding.generate_embeddings(
                tsv_path=input_tsv_path,
                output_dir=output_dir
            )
            
            if not ligand_success:
                self.logger.error("Failed to generate ligand embeddings")
                return False
            
            # Store results
            self.results['embedding_generation'] = {
                'protein_embeddings': protein_embedding.get_embeddings_info(),
                'ligand_embeddings': ligand_embedding.get_embeddings_info(),
                'success': True
            }
            
            self.logger.info("✅ Embedding generation completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in embedding generation: {e}")
            return False
    
    def run_matrix_construction(self,
                              output_dir: Optional[Union[str, Path]] = None,
                              matrix_type: str = 'embedding') -> bool:
        """
        Run matrix construction phase.
        
        Args:
            output_dir: Output directory (optional)
            matrix_type: Type of matrix ('embedding' or 'kinase')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info("🟡 Starting matrix construction phase")
            
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
            
            # Choose matrix builder
            if matrix_type == 'kinase':
                matrix_builder = self.components['kinase_matrix']
            else:
                matrix_builder = self.components['embedding_matrix']
            
            # Get embedding paths from previous step
            protein_embedding = self.components['protein_embedding']
            ligand_embedding = self.components['ligand_embedding']
            
            protein_path = protein_embedding.get_output_path()
            ligand_path = ligand_embedding.get_output_path()
            
            if not protein_path or not ligand_path:
                self.logger.error("Embedding paths not available. Run embedding generation first.")
                return False
            
            # Build matrix
            self.logger.info(f"Building {matrix_type} matrix...")
            matrix_success = matrix_builder.build_matrix(
                protein_embeddings_path=protein_path,
                ligand_embeddings_path=ligand_path,
                output_dir=output_dir
            )
            
            if not matrix_success:
                self.logger.error(f"Failed to build {matrix_type} matrix")
                return False
            
            # Store results
            self.results['matrix_construction'] = {
                'matrix_type': matrix_type,
                'matrix_info': matrix_builder.get_matrix_info(),
                'output_path': str(matrix_builder.get_output_path()),
                'success': True
            }
            
            self.logger.info("✅ Matrix construction completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in matrix construction: {e}")
            return False
    
    def run_label_generation(self,
                           input_tsv_path: Union[str, Path],
                           output_dir: Optional[Union[str, Path]] = None,
                           binary_threshold: float = 1000.0) -> bool:
        """
        Run label generation phase.
        
        Args:
            input_tsv_path: Path to input TSV file
            output_dir: Output directory (optional)
            binary_threshold: Threshold for binary labels (nM)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info("🟡 Starting label generation phase")
            
            input_tsv_path = Path(input_tsv_path)
            
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate interaction labels
            self.logger.info("Generating interaction labels...")
            interaction_labels = InteractionLabels(self.config, input_tsv_path)
            interaction_success = interaction_labels.generate_labels()
            
            if interaction_success is None:
                self.logger.error("Failed to generate interaction labels")
                return False
            
            # Save interaction labels
            labels_path = output_dir / "interaction_labels" if output_dir else Path("interaction_labels")
            if not interaction_labels.save_labels(labels_path):
                self.logger.error("Failed to save interaction labels")
                return False
            
            # Generate binary labels
            self.logger.info(f"Generating binary labels (threshold: {binary_threshold} nM)...")
            binary_labels = BinaryLabels(self.config)
            binary_success = binary_labels.generate_labels(
                interaction_data=interaction_labels.labels,
                threshold=binary_threshold
            )
            
            if binary_success is None:
                self.logger.error("Failed to generate binary labels")
                return False
            
            # Save binary labels
            binary_path = output_dir / "binary_labels" if output_dir else Path("binary_labels")
            if not binary_labels.save_labels(binary_path):
                self.logger.error("Failed to save binary labels")
                return False
            
            # Store results
            self.results['label_generation'] = {
                'interaction_labels': {
                    'count': len(interaction_labels.labels),
                    'statistics': interaction_labels.get_interaction_statistics(),
                    'path': str(labels_path)
                },
                'binary_labels': {
                    'count': len(binary_labels.labels),
                    'statistics': binary_labels.get_class_balance_info(),
                    'threshold': binary_threshold,
                    'path': str(binary_path)
                },
                'success': True
            }
            
            self.logger.info("✅ Label generation completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in label generation: {e}")
            return False
    
    def run_validation(self,
                      concatenated_path: Optional[Union[str, Path]] = None,
                      labels_path: Optional[Union[str, Path]] = None,
                      original_tsv_path: Optional[Union[str, Path]] = None) -> bool:
        """
        Run validation phase.
        
        Args:
            concatenated_path: Path to concatenated embeddings (optional)
            labels_path: Path to labels (optional)
            original_tsv_path: Path to original TSV (optional)
            
        Returns:
            True if validation passes, False otherwise
        """
        try:
            self.logger.info("🟡 Starting validation phase")
            
            # Use paths from previous results if not provided
            if concatenated_path is None and 'matrix_construction' in self.results:
                concatenated_path = self.results['matrix_construction']['output_path']
            
            if labels_path is None and 'label_generation' in self.results:
                labels_path = self.results['label_generation']['interaction_labels']['path'] + '.npy'
            
            if not concatenated_path or not labels_path:
                self.logger.error("Required paths for validation not available")
                return False
            
            # Run validation
            validator = self.components['matrix_validator']
            validation_success = validator.validate(
                concatenated_path=concatenated_path,
                labels_path=labels_path,
                original_tsv_path=original_tsv_path
            )
            
            # Store results
            self.results['validation'] = {
                'passed': validation_success,
                'summary': validator.get_summary(),
                'success': True
            }
            
            # Print validation summary
            validator.print_summary()
            
            if validation_success:
                self.logger.info("✅ Validation completed successfully")
            else:
                self.logger.error("❌ Validation failed")
            
            return validation_success
            
        except Exception as e:
            self.logger.error(f"Error in validation: {e}")
            return False
    
    def run_complete_pipeline(self,
                            input_tsv_path: Union[str, Path],
                            output_dir: Union[str, Path],
                            matrix_type: str = 'embedding',
                            binary_threshold: float = 1000.0,
                            run_validation: bool = True) -> bool:
        """
        Run the complete build pipeline.
        
        Args:
            input_tsv_path: Path to input TSV file
            output_dir: Output directory
            matrix_type: Type of matrix to build ('embedding' or 'kinase')
            binary_threshold: Threshold for binary labels (nM)
            run_validation: Whether to run validation step
            
        Returns:
            True if entire pipeline succeeds, False otherwise
        """
        try:
            self.logger.info("🚀 Starting complete build pipeline")
            
            # Ensure output directory exists
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Step 1: Generate embeddings
            if not self.run_embedding_generation(input_tsv_path, output_dir):
                return False
            
            # Step 2: Build matrix
            if not self.run_matrix_construction(output_dir, matrix_type):
                return False
            
            # Step 3: Generate labels
            if not self.run_label_generation(input_tsv_path, output_dir, binary_threshold):
                return False
            
            # Step 4: Optional validation
            if run_validation:
                if not self.run_validation(original_tsv_path=input_tsv_path):
                    self.logger.warning("Validation failed, but pipeline completed")
            
            # Save pipeline results
            results_path = output_dir / "pipeline_results.json"
            self.save_json(self.results, results_path)
            
            self.logger.info("🎉 Complete pipeline executed successfully!")
            self.logger.info(f"Results saved to: {results_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in complete pipeline: {e}")
            return False
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """
        Get summary of pipeline execution.
        
        Returns:
            Dictionary with pipeline summary
        """
        summary = {
            'executed_steps': list(self.results.keys()),
            'total_steps': len(self.results),
            'success': all(result.get('success', False) for result in self.results.values()),
            'results': self.results.copy()
        }
        
        return summary
