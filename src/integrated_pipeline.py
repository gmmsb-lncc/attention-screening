#!/usr/bin/env python3
"""
DockTKinase Integrated Pipeline (Refactored)
=============================================

End-to-end integration system that orchestrates all modules:
- build: Embedding and matrix generation
- classifier: Binary classification (active/inactive)
- regression: Quantitative prediction (pKi/IC50)

Refactored following SOLID, KISS, and Clean Code principles:
- Single Responsibility: Each class has one job
- Open/Closed: Easy to extend with new phases
- DRY: Centralized serialization and checkpointing
- Clean: Clear naming, small methods, good documentation

Usage:
    # CLI
    python -m integrated_pipeline --input data.tsv --output results/

    # Python API
    from integrated_pipeline import IntegratedPipeline
    
    pipeline = IntegratedPipeline(
        input_tsv="data.tsv",
        output_dir="results/"
    )
    results = pipeline.run()
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from dataclasses import dataclass
import logging

import numpy as np

# Add paths
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / 'src'))

# Import utilities
from utils.json_serializer import make_json_serializable, save_json
from utils.checkpoint_manager import CheckpointManager

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class IntegratedConfig:
    """Configuration for all pipeline modules."""
    
    # Input/Output
    input_tsv: str
    output_dir: str = "results/integrated"
    use_checkpoints: bool = True
    
    # Build module
    esm_model: str = "esm2_t6_8M_UR50D"
    esm_dim: Optional[int] = None
    ligand_model: str = "SMI-TED"
    batch_size: int = 8
    device: str = "cpu"
    save_matrices: bool = False
    
    # Embedding directories (reuse pre-computed)
    protein_embeddings_dir: Optional[str] = None
    ligand_embeddings_dir: Optional[str] = None
    
    # Data split
    test_size: float = 0.1
    val_size: float = 0.1
    random_state: int = 42
    
    # Stratification
    stratifier_auto_threshold: bool = True
    stratifier_threshold: Optional[float] = None
    stratifier_method: str = 'target'
    
    # Classification
    run_classification: bool = True
    use_multi_model_classification: bool = False
    classification_models: Optional[List[str]] = None
    classifier_epochs: int = 50
    classifier_cv_folds: int = 5
    
    # Regression
    run_regression: bool = True
    regression_models: Optional[List[str]] = None
    regression_cv_folds: int = 5
    
    # Binary threshold
    binary_threshold: float = 1000.0  # nM
    
    # Options
    verbose: bool = True
    save_models: bool = True
    create_visualizations: bool = True


# =============================================================================
# PIPELINE PHASES (Single Responsibility)
# =============================================================================

class BuildPhase:
    """Handles embedding generation and matrix construction."""
    
    def __init__(self, config: IntegratedConfig, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.build_dir = output_dir / "build"
        self.build_dir.mkdir(exist_ok=True)
    
    def run(self) -> Dict[str, Any]:
        """Execute build phase."""
        from build.pipeline import BuildPipeline
        from build.core import BuildConfig
        
        if self.config.verbose:
            print("\n📦 Running Build Phase...")
        
        # Configure
        build_config = BuildConfig(
            input_tsv=self.config.input_tsv,
            output_dir=str(self.build_dir),
            esm_model=self.config.esm_model,
            esm_dim=self.config.esm_dim,
            ligand_model=self.config.ligand_model,
            batch_size=self.config.batch_size,
            device=self.config.device,
            binary_threshold=self.config.binary_threshold,
            test_size=self.config.test_size,
            val_size=self.config.val_size,
            random_state=self.config.random_state,
            protein_embeddings_dir=self.config.protein_embeddings_dir,
            ligand_embeddings_dir=self.config.ligand_embeddings_dir,
            stratification_enabled=True,
            stratification_params={
                'clustering_algorithm': 'adaptive',
                'similarity_threshold': self.config.stratifier_threshold,
                'adaptive_method': self.config.stratifier_method,
                'cluster_min_size': 3,
                'stratify_by': 'both',
                'protein_weight': 0.6,
                'ligand_weight': 0.4
            }
        )
        
        # Run
        pipeline = BuildPipeline(build_config)
        success = pipeline.run_complete_pipeline(
            input_tsv_path=self.config.input_tsv,
            output_dir=self.build_dir,
            matrix_type='embedding',
            binary_threshold=self.config.binary_threshold,
            run_validation=True
        )
        
        if not success:
            raise RuntimeError("Build phase failed")
        
        # Load split indices
        split_indices = self._load_split_indices()
        
        # Collect results
        results = self._collect_results(split_indices)
        
        if self.config.verbose:
            print("✅ Build phase completed")
            print(f"   Samples: {results.get('n_samples', 0)}")
            print(f"   Features: {results.get('embedding_dim', 0)}")
        
        return results
    
    def _load_split_indices(self):
        """Load stratified splits if available."""
        from build.pipeline.split_indices import SplitIndices
        
        splits_file = self.build_dir / "splits" / "stratified_splits.npz"
        
        if splits_file.exists():
            try:
                split_indices = SplitIndices.load(str(splits_file))
                if self.config.verbose:
                    print(f"   Loaded splits: train={len(split_indices.train_idx)}, "
                          f"val={len(split_indices.val_idx)}, test={len(split_indices.test_idx)}")
                return split_indices
            except Exception as e:
                logger.warning(f"Could not load splits: {e}")
        
        return None
    
    def _collect_results(self, split_indices) -> Dict[str, Any]:
        """Collect build phase results."""
        results = {
            'success': True,
            'embeddings': {
                'protein_dir': str(self.build_dir / "proteins"),
                'ligand_dir': str(self.build_dir / "ligands"),
                'concatenated': str(self.build_dir / "embedding_matrix.npy")
            },
            'labels': {
                'binary': str(self.build_dir / "binary_labels.npy"),
                'regression': str(self.build_dir / "interaction_labels.npy")
            },
            'splits': {
                'train_indices': str(self.build_dir / "splits" / "train_indices.npy"),
                'val_indices': str(self.build_dir / "splits" / "val_indices.npy"),
                'test_indices': str(self.build_dir / "splits" / "test_indices.npy")
            },
            'split_indices': split_indices
        }
        
        # Load matrix statistics
        matrix_path = self.build_dir / "embedding_matrix.npy"
        if matrix_path.exists():
            matrix = np.load(matrix_path)
            results['n_samples'] = matrix.shape[0]
            results['embedding_dim'] = matrix.shape[1]
        
        return results


class ClassificationPhase:
    """Handles binary classification model training."""
    
    def __init__(self, config: IntegratedConfig, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.classifier_dir = output_dir / "classifier"
        self.classifier_dir.mkdir(exist_ok=True)
    
    def run(self, build_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute classification phase."""
        if self.config.verbose:
            print("\n🎯 Running Classification Phase...")
        
        embeddings_path = build_results['embeddings']['concatenated']
        labels_path = build_results['labels']['binary']
        split_indices = build_results.get('split_indices')
        
        if self.config.use_multi_model_classification:
            results = self._run_multi_model(embeddings_path, labels_path, split_indices)
        else:
            results = self._run_mlp(embeddings_path, labels_path, split_indices)
        
        if self.config.verbose:
            self._print_results(results)
        
        return results
    
    def _run_mlp(self, embeddings_path: str, labels_path: str, split_indices) -> Dict[str, Any]:
        """Run MLP classification."""
        from classifier.modular_pipeline import MLPEmbeddingPipeline
        
        classifier = MLPEmbeddingPipeline(
            embeddings_path=embeddings_path,
            labels_path=labels_path,
            batch_size=32,
            lr=0.001,
            epochs=self.config.classifier_epochs,
            test_split=self.config.test_size,
            val_split=self.config.val_size,
            early_stopping_patience=10,
            model_output=str(self.classifier_dir / "mlp_model.pth"),
            metrics_output=str(self.classifier_dir / "metrics.json"),
            split_indices=split_indices
        )
        
        classifier.load_data()
        val_loss = classifier.train()
        cv_results = classifier.cross_validate(k=self.config.classifier_cv_folds)
        test_metrics = classifier.evaluate(classifier.model, classifier.test_loader)
        
        return {
            'success': True,
            'mode': 'MLP',
            'val_loss': float(val_loss),
            'test_metrics': {k: float(v) for k, v in test_metrics.items()},
            'cv_results': {
                'mean_roc_auc': float(cv_results.get('mean_roc_auc', 0)),
                'std_roc_auc': float(cv_results.get('std_roc_auc', 0)),
                'n_folds': self.config.classifier_cv_folds
            },
            'model_path': str(self.classifier_dir / "mlp_model.pth")
        }
    
    def _run_multi_model(self, embeddings_path: str, labels_path: str, split_indices) -> Dict[str, Any]:
        """Run multi-model classification."""
        from classifier.multi_model_pipeline import MultiModelClassificationPipeline
        
        pipeline = MultiModelClassificationPipeline(
            embeddings_path=embeddings_path,
            labels_path=labels_path,
            output_dir=str(self.classifier_dir),
            models_to_train=self.config.classification_models,
            test_size=self.config.test_size,
            val_size=self.config.val_size,
            random_state=self.config.random_state,
            verbose=self.config.verbose
        )
        
        test_metrics = pipeline.run()
        
        # Find best model
        best_model = max(test_metrics.items(), key=lambda x: x[1].get('ROC_AUC', 0))
        
        return {
            'success': True,
            'mode': 'MultiModel',
            'n_models_trained': len(test_metrics),
            'best_model': best_model[0],
            'best_metrics': {k: float(v) for k, v in best_model[1].items()},
            'individual_results': {
                name: {k: float(v) for k, v in metrics.items()}
                for name, metrics in test_metrics.items()
            }
        }
    
    def _print_results(self, results: Dict[str, Any]) -> None:
        """Print classification results."""
        print("✅ Classification completed")
        if results.get('mode') == 'MultiModel':
            print(f"   Best: {results['best_model']} (ROC-AUC: {results['best_metrics'].get('ROC_AUC', 0):.4f})")
        else:
            print(f"   ROC-AUC: {results['test_metrics'].get('roc_auc', 0):.4f}")


class RegressionPhase:
    """Handles regression model training."""
    
    def __init__(self, config: IntegratedConfig, output_dir: Path, checkpoint_manager: CheckpointManager):
        self.config = config
        self.output_dir = output_dir
        self.regression_dir = output_dir / "regression"
        self.regression_dir.mkdir(exist_ok=True)
        self.checkpoint_manager = checkpoint_manager
    
    def run(self, build_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute regression phase."""
        if self.config.verbose:
            print("\n📈 Running Regression Phase...")
        
        from regression.modular_pipeline import RegressionPipeline
        
        regression = RegressionPipeline(
            embeddings_path=build_results['embeddings']['concatenated'],
            targets_path=build_results['labels']['regression'],
            output_dir=str(self.regression_dir),
            models_to_train=self.config.regression_models,
            test_size=self.config.test_size,
            val_size=self.config.val_size,
            random_state=self.config.random_state,
            verbose=self.config.verbose,
            split_indices=build_results.get('split_indices')
        )
        
        # Load data
        regression.load_data()
        
        # Train with checkpoint support
        train_results = self._train_with_checkpoint(regression)
        
        # Evaluate
        test_results = self._evaluate_with_checkpoint(regression)
        
        # Save
        regression.val_metrics = train_results
        regression.test_metrics = test_results
        regression.save_results()
        
        # Compile results
        results = self._compile_results(train_results, test_results)
        
        if self.config.verbose:
            self._print_results(results)
        
        return results
    
    def _train_with_checkpoint(self, regression) -> Dict[str, Any]:
        """Train with checkpoint support."""
        checkpoint = self.checkpoint_manager.load('regression_train')
        if checkpoint:
            regression.val_metrics = checkpoint
            return checkpoint
        
        results = regression.train_models()
        self.checkpoint_manager.save('regression_train', results)
        return results
    
    def _evaluate_with_checkpoint(self, regression) -> Dict[str, Any]:
        """Evaluate with checkpoint support."""
        checkpoint = self.checkpoint_manager.load('regression_test')
        if checkpoint:
            regression.test_metrics = checkpoint
            return checkpoint
        
        results = regression.evaluate_on_test()
        self.checkpoint_manager.save('regression_test', results)
        return results
    
    def _compile_results(self, train_results: Dict, test_results: Dict) -> Dict[str, Any]:
        """Compile regression results."""
        # Filter out None results (failed models)
        valid_train = {k: v for k, v in train_results.items() if v is not None}
        valid_test = {k: v for k, v in test_results.items() if v is not None}
        
        if not valid_train:
            return {
                'success': False,
                'error': 'All regression models failed to train',
                'models_trained': 0
            }
        
        # Find best model by validation MAE
        best_model = min(valid_train.items(), key=lambda x: x[1].get('MAE', float('inf')))
        best_name = best_model[0]
        
        best_test = valid_test.get(best_name, {})
        
        return {
            'success': True,
            'best_model': best_name,
            'best_val_mae': float(best_model[1].get('MAE', 0)),
            'best_val_r2': float(best_model[1].get('R2', 0)),
            'best_test_mae': float(best_test.get('MAE', 0)),
            'best_test_r2': float(best_test.get('R2', 0)),
            'best_mae': float(best_test.get('MAE', 0)),
            'best_r2': float(best_test.get('R2', 0)),
            'models_trained': len(valid_train),
            'individual_results': {
                name: {k: v if isinstance(v, (int, float)) else str(v) for k, v in metrics.items()}
                for name, metrics in valid_train.items()
            },
            'test_results': {
                name: {k: v if isinstance(v, (int, float)) else str(v) for k, v in metrics.items()}
                for name, metrics in valid_test.items()
            }
        }
    
    def _print_results(self, results: Dict[str, Any]) -> None:
        """Print regression results."""
        print("✅ Regression completed")
        print(f"   Best: {results['best_model']}")
        print(f"   Test MAE: {results['best_test_mae']:.3f}, R²: {results['best_test_r2']:.4f}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

class IntegratedPipeline:
    """
    DockTKinase end-to-end integrated pipeline.
    
    Orchestrates all phases:
    1. Build: Generate embeddings and matrices
    2. Classifier: Train binary classification model  
    3. Regression: Train quantitative regression models
    """
    
    def __init__(self, config: Union[IntegratedConfig, Dict[str, Any]]):
        """Initialize pipeline."""
        if isinstance(config, dict):
            config = IntegratedConfig(**config)
        
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        for subdir in ['build', 'classifier', 'regression']:
            (self.output_dir / subdir).mkdir(exist_ok=True)
        
        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.output_dir / 'checkpoints',
            enabled=config.use_checkpoints,
            verbose=config.verbose
        )
        
        # Results storage
        self.results = {
            'config': make_json_serializable(config),
            'build': {},
            'classifier': {},
            'regression': {},
            'status': 'initialized',
            'timestamp_start': None,
            'timestamp_end': None,
            'total_time_seconds': None
        }
    
    def run(self) -> Dict[str, Any]:
        """Run complete pipeline."""
        start_time = time.time()
        self.results['timestamp_start'] = datetime.now().isoformat()
        
        if self.config.verbose:
            self._print_header()
        
        try:
            # Phase 1: Build
            build_results = self._run_build_phase()
            self.results['build'] = make_json_serializable(build_results)
            
            # Phase 2: Classification (optional)
            if self.config.run_classification:
                classifier_results = self._run_classification_phase(build_results)
                self.results['classifier'] = classifier_results
            
            # Phase 3: Regression (optional)
            if self.config.run_regression:
                regression_results = self._run_regression_phase(build_results)
                self.results['regression'] = regression_results
            
            self.results['status'] = 'completed'
            
        except Exception as e:
            self.results['status'] = 'failed'
            self.results['error'] = str(e)
            logger.exception("Pipeline failed")
            raise
        
        finally:
            self.results['timestamp_end'] = datetime.now().isoformat()
            self.results['total_time_seconds'] = time.time() - start_time
            self._save_results()
            
            if self.config.verbose:
                self._print_summary()
        
        return self.results
    
    def _run_build_phase(self) -> Dict[str, Any]:
        """Run build phase with checkpoint support."""
        if self.config.verbose:
            print("\n" + "="*80)
            print("PHASE 1: BUILD - Embedding Generation & Matrix Construction")
            print("="*80)
        
        # Check checkpoint
        checkpoint = self.checkpoint_manager.load('build')
        if checkpoint:
            # Update old checkpoints if needed
            return self._update_build_checkpoint(checkpoint)
        
        # Run build
        build_phase = BuildPhase(self.config, self.output_dir)
        results = build_phase.run()
        
        # Save checkpoint (without split_indices object - it will be reloaded)
        checkpoint_data = {k: v for k, v in results.items() if k != 'split_indices'}
        self.checkpoint_manager.save('build', checkpoint_data)
        
        return results
    
    def _update_build_checkpoint(self, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        """Update old checkpoint with missing fields."""
        if 'n_samples' not in checkpoint:
            matrix_path = self.output_dir / "build" / "embedding_matrix.npy"
            if matrix_path.exists():
                matrix = np.load(matrix_path)
                checkpoint['n_samples'] = matrix.shape[0]
                checkpoint['embedding_dim'] = matrix.shape[1]
        
        # Reload split indices
        from build.pipeline.split_indices import SplitIndices
        splits_file = self.output_dir / "build" / "splits" / "stratified_splits.npz"
        if splits_file.exists():
            try:
                checkpoint['split_indices'] = SplitIndices.load(str(splits_file))
            except Exception:
                checkpoint['split_indices'] = None
        
        return checkpoint
    
    def _run_classification_phase(self, build_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run classification phase with checkpoint support."""
        if self.config.verbose:
            print("\n" + "="*80)
            print("PHASE 2: CLASSIFICATION - Binary Activity Prediction")
            print("="*80)
        
        checkpoint = self.checkpoint_manager.load('classifier')
        if checkpoint:
            return self._normalize_classifier_checkpoint(checkpoint)
        
        phase = ClassificationPhase(self.config, self.output_dir)
        results = phase.run(build_results)
        self.checkpoint_manager.save('classifier', results)
        return results
    
    def _normalize_classifier_checkpoint(self, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize old classifier checkpoint format."""
        if 'best_model' in checkpoint:
            return checkpoint
        
        # Old format: model_name -> metrics
        best_model = max(checkpoint.items(), key=lambda x: x[1].get('ROC_AUC', 0) if isinstance(x[1], dict) else 0)
        
        return {
            'success': True,
            'mode': 'MultiModel',
            'n_models_trained': len([k for k, v in checkpoint.items() if isinstance(v, dict)]),
            'best_model': best_model[0],
            'best_metrics': best_model[1] if isinstance(best_model[1], dict) else {},
            'individual_results': {k: v for k, v in checkpoint.items() if isinstance(v, dict)}
        }
    
    def _run_regression_phase(self, build_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run regression phase."""
        if self.config.verbose:
            print("\n" + "="*80)
            print("PHASE 3: REGRESSION - Quantitative Activity Prediction")
            print("="*80)
        
        checkpoint = self.checkpoint_manager.load('regression')
        if checkpoint:
            return checkpoint
        
        phase = RegressionPhase(self.config, self.output_dir, self.checkpoint_manager)
        results = phase.run(build_results)
        self.checkpoint_manager.save('regression', results)
        return results
    
    def _save_results(self) -> None:
        """Save final results."""
        results_file = self.output_dir / "integrated_results.json"
        save_json(self.results, results_file)
        
        if self.config.verbose:
            print(f"\n📁 Results saved to: {results_file}")
    
    def _print_header(self) -> None:
        """Print pipeline header."""
        print("\n" + "="*80)
        print(" " * 20 + "🧬 DOCKTKINASE INTEGRATED PIPELINE 🧬")
        print("="*80)
        print(f"Input TSV: {self.config.input_tsv}")
        print(f"Output Dir: {self.config.output_dir}")
        print(f"ESM Model: {self.config.esm_model}")
        print(f"Device: {self.config.device}")
        print(f"Random Seed: {self.config.random_state}")
        print("\nModules to run:")
        print(f"  • Build: ✅")
        mode = "Multi-Model" if self.config.use_multi_model_classification else "MLP"
        print(f"  • Classification: {'✅ (' + mode + ')' if self.config.run_classification else '❌'}")
        print(f"  • Regression: {'✅' if self.config.run_regression else '❌'}")
        print("="*80)
    
    def _print_summary(self) -> None:
        """Print final summary."""
        print("\n" + "="*80)
        print(" " * 25 + "🎉 PIPELINE SUMMARY 🎉")
        print("="*80)
        
        print(f"\n📊 Status: {self.results['status'].upper()}")
        print(f"⏱️  Total time: {self.results['total_time_seconds']:.2f} seconds")
        
        if self.results.get('build', {}).get('success'):
            print("\n✅ Build Phase: SUCCESS")
        
        if self.config.run_classification and self.results.get('classifier', {}).get('success'):
            clf = self.results['classifier']
            print("\n✅ Classification Phase: SUCCESS")
            if clf.get('mode') == 'MultiModel':
                print(f"   Best model: {clf.get('best_model')}")
                print(f"   ROC-AUC: {clf.get('best_metrics', {}).get('ROC_AUC', 0):.4f}")
            else:
                print(f"   ROC-AUC: {clf.get('test_metrics', {}).get('roc_auc', 0):.4f}")
        
        if self.config.run_regression and self.results.get('regression', {}).get('success'):
            reg = self.results['regression']
            print("\n✅ Regression Phase: SUCCESS")
            print(f"   Best model: {reg['best_model']}")
            print(f"   Test MAE: {reg['best_mae']:.3f}, R²: {reg['best_r2']:.4f}")
        
        print("\n" + "="*80)
        print(f"📁 All results saved to: {self.output_dir}")
        print("="*80 + "\n")


# =============================================================================
# CLI
# =============================================================================

def main():
    """Command line entry point."""
    parser = argparse.ArgumentParser(
        description="DockTKinase Integrated Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Complete workflow
  python -m integrated_pipeline --input data.tsv --output results/

  # Classification only
  python -m integrated_pipeline --input data.tsv --no-regression

  # Regression only  
  python -m integrated_pipeline --input data.tsv --no-classification
        """
    )
    
    parser.add_argument('--input', required=True, help='Input TSV file')
    parser.add_argument('--output', default='results/integrated', help='Output directory')
    parser.add_argument('--esm-model', default='esm2_t6_8M_UR50D', help='ESM model')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'mps'])
    parser.add_argument('--no-classification', action='store_true')
    parser.add_argument('--no-regression', action='store_true')
    parser.add_argument('--regression-models', nargs='+', 
                        default=['Ridge', 'Lasso', 'ElasticNet', 'RandomForest', 'XGBoost'])
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--quiet', action='store_true')
    
    args = parser.parse_args()
    
    config = IntegratedConfig(
        input_tsv=args.input,
        output_dir=args.output,
        esm_model=args.esm_model,
        device=args.device,
        run_classification=not args.no_classification,
        run_regression=not args.no_regression,
        regression_models=args.regression_models,
        random_state=args.random_state,
        verbose=not args.quiet
    )
    
    pipeline = IntegratedPipeline(config)
    results = pipeline.run()
    
    return 0 if results['status'] == 'completed' else 1


if __name__ == '__main__':
    sys.exit(main())
