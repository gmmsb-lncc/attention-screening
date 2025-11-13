"""
Main build pipeline orchestrator.

Coordinates all modules to execute the complete embedding matrix
construction pipeline for protein-ligand interaction prediction.
"""

import os
import sys
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from src.build.core import BaseBuilder, BuildConfig
from src.build.embeddings import ProteinEmbedding, LigandEmbedding
from src.build.matrix import EmbeddingMatrix, KinaseMatrix  
from src.build.labels import InteractionLabels, BinaryLabels
from src.build.validation import MatrixValidator
from src.build.stratification import Stratifier, SplitValidator
from src.build.pipeline.stratification_manager import StratificationManager
from src.build.pipeline.split_indices import SplitIndices


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
            # Obter configuração de modelos (priorizar esm_model sobre protein_model)
            protein_model = self.config.get('esm_model') or self.config.get('protein_model', 'esm2_t6_8M_UR50D')
            ligand_model = self.config.get('ligand_model', 'fm4m')
            
            self.components = {
                'protein_embedding': ProteinEmbedding(
                    self.config, 
                    model_name=protein_model,
                    use_gpu=self.config.use_gpu
                ),
                'ligand_embedding': LigandEmbedding(
                    self.config,
                    model_name=ligand_model
                ),
                'embedding_matrix': EmbeddingMatrix(self.config),
                'kinase_matrix': KinaseMatrix(self.config),
                'matrix_validator': MatrixValidator(self.config),
                'stratifier': Stratifier(self.config),
                'split_validator': SplitValidator(self.config),
                'stratification_manager': StratificationManager(
                    self.config,
                    clustering_algorithm=self.config.get('stratification_params', {}).get('clustering_algorithm', 'kmeans'),
                    protein_weight=self.config.get('stratification_params', {}).get('protein_weight', 0.6),
                    ligand_weight=self.config.get('stratification_params', {}).get('ligand_weight', 0.4),
                    random_state=self.config.get('random_state', 42)
                )
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
                              matrix_type: str = 'embedding',
                              data_path: Optional[Union[str, Path]] = None) -> bool:
        """
        Run matrix construction phase.
        
        Args:
            output_dir: Output directory (optional)
            matrix_type: Type of matrix ('embedding' or 'kinase')
            data_path: Path to original TSV file (required for matrix construction)
            
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
                output_dir=output_dir,
                data_path=data_path
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
                labels_path = str(Path(self.results['label_generation']['interaction_labels']['path']).with_suffix('.npy'))
            
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
            if not self.run_matrix_construction(output_dir, matrix_type, data_path=input_tsv_path):
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
    
    def run_stratification(self,
                          protein_embeddings_path: Optional[Union[str, Path]] = None,
                          ligand_embeddings_path: Optional[Union[str, Path]] = None,
                          concatenated_matrix_path: Optional[Union[str, Path]] = None,
                          labels_path: Optional[Union[str, Path]] = None,
                          test_size: float = 0.2,
                          val_size: float = 0.1,
                          clustering_algorithm: Optional[str] = None,
                          similarity_threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Run stratification to create train/test/validation splits.
        
        Args:
            protein_embeddings_path: Path to protein embeddings (optional if using concatenated)
            ligand_embeddings_path: Path to ligand embeddings (optional if using concatenated)
            concatenated_matrix_path: Path to concatenated embeddings matrix
            labels_path: Path to labels file
            test_size: Proportion of test set
            val_size: Proportion of validation set
            clustering_algorithm: Algorithm for clustering ('dbscan', 'hierarchical', 'kmeans', 'random')
            similarity_threshold: Similarity threshold for clustering
            
        Returns:
            Dictionary with split information and validation results
        """
        try:
            self.logger.info("🟡 Starting stratification process")
            
            # Load concatenated embeddings matrix
            if concatenated_matrix_path:
                import numpy as np
                concatenated_embeddings = np.load(concatenated_matrix_path)
                protein_embeddings = None
                ligand_embeddings = None
            elif protein_embeddings_path and ligand_embeddings_path:
                # Load separate embeddings
                import numpy as np
                protein_embeddings = np.load(protein_embeddings_path)
                ligand_embeddings = np.load(ligand_embeddings_path)
                # Also create concatenated for fallback
                concatenated_embeddings = np.concatenate([protein_embeddings, ligand_embeddings], axis=1)
            else:
                raise ValueError("Either concatenated_matrix_path or both protein_embeddings_path and ligand_embeddings_path must be provided")
            
            # Load labels
            if labels_path:
                import numpy as np
                labels = np.load(labels_path)
            else:
                # Generate dummy labels if not available (for testing)
                labels = np.random.randint(0, 2, size=(concatenated_embeddings.shape[0],))
            
            # Initialize and configure stratifier
            stratifier = self.components['stratifier']
            
            # Use provided parameters or fall back to config values
            if clustering_algorithm is not None:
                stratifier.clustering_algorithm = clustering_algorithm
            if similarity_threshold is not None:
                stratifier.similarity_threshold = similarity_threshold
            
            # Perform stratified split using multi-view if separate embeddings available
            if protein_embeddings is not None and ligand_embeddings is not None:
                self.logger.info("Using multi-view stratification (protein + ligand)")
                self.logger.info(f"  Protein weight: {stratifier.protein_weight}")
                self.logger.info(f"  Ligand weight: {stratifier.ligand_weight}")
                train_idx, val_idx, test_idx = stratifier.multi_view_stratified_split(
                    protein_embeddings=protein_embeddings,
                    ligand_embeddings=ligand_embeddings,
                    labels=labels,
                    test_size=test_size,
                    val_size=val_size,
                    protein_weight=stratifier.protein_weight,
                    ligand_weight=stratifier.ligand_weight
                )
            else:
                self.logger.info("Using combined embeddings stratification")
                train_idx, val_idx, test_idx = stratifier.stratified_split(
                    embeddings=concatenated_embeddings,
                    labels=labels,
                    test_size=test_size,
                    val_size=val_size
                )
            
            # Validate splits
            split_validator = self.components['split_validator']
            validation_report = split_validator.validate_splits_comprehensively(
                embeddings=concatenated_embeddings,
                labels=labels,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx
            )
            
            # Store results
            stratification_results = {
                'train_indices': train_idx.tolist(),
                'val_indices': val_idx.tolist(),
                'test_indices': test_idx.tolist(),
                'split_sizes': {
                    'train': len(train_idx),
                    'validation': len(val_idx),
                    'test': len(test_idx)
                },
                'validation_report': validation_report
            }
            
            self.results['stratification'] = stratification_results
            self.logger.info(f"✅ Stratification completed: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test samples")
            
            return stratification_results
            
        except Exception as e:
            self.logger.error(f"Error in stratification: {e}")
            raise
    
    def get_split_indices(self) -> Optional[SplitIndices]:
        """
        Get the split indices from the last stratification.
        
        Returns:
            SplitIndices object if stratification was performed, None otherwise
        """
        if 'stratification' not in self.results:
            return None
        
        # Check if we have the new format with splits_file
        if 'splits_file' in self.results['stratification']:
            splits_file = self.results['stratification']['splits_file']
            return SplitIndices.load(splits_file)
        
        return None
    
    def run_complete_pipeline(self,
                            input_tsv_path: Union[str, Path],
                            output_dir: Union[str, Path],
                            matrix_type: str = 'embedding',
                            binary_threshold: float = 1000.0,
                            run_validation: bool = True,
                            stratify_splits: bool = False,
                            test_size: float = 0.2,
                            val_size: float = 0.1) -> bool:
        """
        Run the complete build pipeline.
        
        Args:
            input_tsv_path: Path to input TSV file
            output_dir: Output directory
            matrix_type: Type of matrix to build ('embedding' or 'kinase')
            binary_threshold: Threshold for binary labels (nM)
            run_validation: Whether to run validation step
            stratify_splits: Whether to perform stratified splits
            test_size: Proportion of test set (if stratifying)
            val_size: Proportion of validation set (if stratifying)
            
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
            if not self.run_matrix_construction(output_dir, matrix_type, data_path=input_tsv_path):
                return False
            
            # Step 3: Generate labels
            if not self.run_label_generation(input_tsv_path, output_dir, binary_threshold):
                return False
            
            # Step 4: Optional stratification using StratificationManager
            # Use parameter value if provided, otherwise check config
            perform_stratification = stratify_splits or self.config.get('stratification_enabled', False)
            split_indices: Optional[SplitIndices] = None
            
            if perform_stratification:
                self.logger.info("🟡 Starting stratification phase with StratificationManager")
                
                # Load embeddings from results stored by previous steps
                import numpy as np
                
                # Get protein embeddings info
                protein_info = self.results['embedding_generation']['protein_embeddings']
                protein_emb_dir = output_dir / "protein_embeddings"
                protein_emb_path = protein_emb_dir / "protein_embeddings.npy"
                
                # Get ligand embeddings info
                ligand_info = self.results['embedding_generation']['ligand_embeddings']
                ligand_emb_dir = output_dir / "ligand_embeddings"
                ligand_emb_path = ligand_emb_dir / "ligand_embeddings.npy"
                
                # Load embeddings
                self.logger.info(f"Loading protein embeddings from: {protein_emb_path}")
                protein_embeddings = np.load(str(protein_emb_path))
                
                self.logger.info(f"Loading ligand embeddings from: {ligand_emb_path}")
                ligand_embeddings = np.load(str(ligand_emb_path))
                
                # Load interaction labels (use interaction labels for stratification)
                labels_path = Path(self.results['label_generation']['interaction_labels']['path']).with_suffix('.npy')
                self.logger.info(f"Loading labels from: {labels_path}")
                labels = np.load(str(labels_path))
                
                # Perform stratification using StratificationManager
                stratification_manager = self.components['stratification_manager']
                split_indices = stratification_manager.stratify(
                    protein_embeddings=protein_embeddings,
                    ligand_embeddings=ligand_embeddings,
                    labels=labels,
                    test_size=test_size,
                    val_size=val_size
                )
                
                # Save splits using SplitIndices
                splits_output_dir = output_dir / "splits"
                splits_output_dir.mkdir(exist_ok=True)
                split_indices.save(str(splits_output_dir / "stratified_splits.npz"))
                
                # Also save as individual .npy files for backward compatibility
                np.save(splits_output_dir / "train_indices.npy", split_indices.train_idx)
                np.save(splits_output_dir / "val_indices.npy", split_indices.val_idx)
                np.save(splits_output_dir / "test_indices.npy", split_indices.test_idx)
                
                # Store results
                self.results['stratification'] = {
                    'train_size': len(split_indices.train_idx),
                    'val_size': len(split_indices.val_idx),
                    'test_size': len(split_indices.test_idx),
                    'splits_file': str(splits_output_dir / "stratified_splits.npz"),
                    'metadata': split_indices.metadata
                }
                
                self.logger.info(f"✅ Stratification complete: {split_indices}")
                self.logger.info(f"✅ Splits saved to: {splits_output_dir / 'stratified_splits.npz'}")
            
            # Step 5: Optional validation
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
