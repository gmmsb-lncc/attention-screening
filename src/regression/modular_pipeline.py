#!/usr/bin/env python3
"""
Modular Regression Pipeline - DockTKinase
==========================================

Complete modularized regression pipeline following the pattern
of the modular classifier.

This implementation maintains EXACTLY the same functionality as the
original pipeline, but in a modularized and organized way.
"""

import time
import json
import warnings
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

# Suppress harmless sklearn/LGBM warnings
warnings.filterwarnings('ignore', message='X does not have valid feature names')
warnings.filterwarnings('ignore', message='.*was fitted with feature names.*')
warnings.filterwarnings('ignore', message='.*Liblinear failed to converge.*')
warnings.filterwarnings('ignore', message='An input array is constant')

# Import SplitIndices for external stratification
try:
    from src.build.pipeline.split_indices import SplitIndices
except ImportError:
    SplitIndices = None  # Fallback if not available

# Imports from modularized modules
try:
    from .core import RegressionEvaluator, DataManager, RegressionTrainer
    from .utils import MetricsCalculator
    from .models.models import RegressionModels  # UPDATED: usar models/models.py
except ImportError:
    # Fallback para execução direta
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    
    from core import RegressionEvaluator, DataManager, RegressionTrainer
    from utils import MetricsCalculator
    from models.models import RegressionModels  # UPDATED: usar models/models.py


class RegressionPipeline:
    """
    Modular regression pipeline for activity prediction.
    
    Implements complete pipeline following the classifier pattern:
    1. Load data (embeddings + targets)
    2. Split into train/validation/test
    3. Train multiple models
    4. Evaluate and compare results
    5. Save metrics and predictions
    
    Maintains 100% compatibility with original pipeline.
    """
    
    def __init__(
        self,
        embeddings_path: str,
        targets_path: str,
        output_dir: str = 'results/regression',
        models_to_train: Optional[List[str]] = None,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        verbose: bool = True,
        split_indices: Optional['SplitIndices'] = None,
        use_pchembl: bool = True
    ):
        """
        Initialize regression pipeline.
        
        Args:
            embeddings_path: Path to embeddings (.npy or .npz)
            targets_path: Path to regression targets (.npy)
            output_dir: Directory to save results
            models_to_train: List of models to train (None = all)
            test_size: Test set proportion (0.2 = 20%)
            val_size: Validation set proportion (0.1 = 10%)
            random_state: Seed for reproducibility
            verbose: Show progress
            split_indices: Optional SplitIndices object with pre-defined train/val/test splits.
                          If provided, these indices will be used instead of random splitting.
                          This ensures consistency with other pipelines (e.g., classification).
            use_pchembl: If True, converts nM values to pChEMBL (-log10(M)).
                        This is RECOMMENDED for regression since Ki/IC50 values
                        vary across several orders of magnitude.
        """
        self.embeddings_path = embeddings_path
        self.targets_path = targets_path
        self.output_dir = Path(output_dir)
        self.models_to_train = models_to_train
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.verbose = verbose
        self.split_indices = split_indices  # Store external splits if provided
        self.use_pchembl = use_pchembl
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'models').mkdir(exist_ok=True)
        (self.output_dir / 'predictions').mkdir(exist_ok=True)
        (self.output_dir / 'metrics').mkdir(exist_ok=True)
        
        # Modularized components
        self.data_manager = DataManager(embeddings_path, targets_path, use_pchembl=use_pchembl)
        self.metrics_calculator = MetricsCalculator()
        self.evaluator = RegressionEvaluator()
        
        # Data (filled in load_data)
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
        # Results
        self.trained_models = {}
        self.train_metrics = {}
        self.val_metrics = {}
        self.test_metrics = {}
        
        # Pipeline stats
        self.stats = {
            'pipeline': 'regression_modular',
            'timestamp': datetime.now().isoformat(),
            'random_state': random_state,
            'embeddings_path': str(embeddings_path),
            'targets_path': str(targets_path)
        }
        
    def load_data(self) -> None:
        """
        Load embeddings and targets, split into train/val/test.
        
        Uses stratified split based on quantile bins to
        maintain similar distribution across all sets.
        
        **NEW**: If split_indices was provided during initialization, those indices
        will be used instead of automatic splitting. This ensures consistent splits
        across classification and regression pipelines.
        """
        if self.verbose:
            print('📊 STEP 1: Data Loading and Splitting')
            print('=' * 70)
        
        start_time = time.time()
        
        # Check if external split indices are provided
        if self.split_indices is not None:
            # Use external splits - load data and apply indices
            if self.verbose:
                print("   📌 Using external split indices (from stratification)")
            
            # Load full data first
            X, y = self.data_manager.load_data()
            
            # Apply external indices
            self.X_train = X[self.split_indices.train_idx]
            self.X_val = X[self.split_indices.val_idx]
            self.X_test = X[self.split_indices.test_idx]
            self.y_train = y[self.split_indices.train_idx]
            self.y_val = y[self.split_indices.val_idx]
            self.y_test = y[self.split_indices.test_idx]
        else:
            # Use automatic splitting (default behavior)
            self.X_train, self.X_val, self.X_test, \
            self.y_train, self.y_val, self.y_test = self.data_manager.split_data(
                test_size=self.test_size,
                val_size=self.val_size,
                random_state=self.random_state
            )
        
        # Estatísticas
        stats = self.data_manager.get_stats()
        target_unit = stats.get('target_unit', 'nM')
        
        if self.verbose:
            print(f"✅ Data loaded successfully!")
            print(f"   Total samples: {stats['n_samples']:,}")
            print(f"   Embeddings dimension: {stats['embedding_dim']:,}")
            print(f"   Train: {len(self.X_train):,} samples")
            print(f"   Validation: {len(self.X_val):,} samples")
            print(f"   Test: {len(self.X_test):,} samples")
            print(f"\n   Target ({target_unit}):")
            print(f"      Média: {stats['target_mean']:.2f}")
            print(f"      Std: {stats['target_std']:.2f}")
            print(f"      Min: {stats['target_min']:.2f}")
            print(f"      Max: {stats['target_max']:.2f}")
            
            # Show original values in nM if using pChEMBL
            if self.use_pchembl and 'target_original_nm' in stats:
                orig = stats['target_original_nm']
                print(f"\n   Target original (nM):")
                print(f"      Mean: {orig['mean']:.2f}")
                print(f"      Min: {orig['min']:.2f}")
                print(f"      Max: {orig['max']:.2f}")
                print(f"\n   ℹ️  Using pChEMBL scale (logarithmic) - recommended for regression")
            
            print(f"\n   Time: {time.time() - start_time:.2f}s")
            print('=' * 70)
            print()
        
        # Update stats
        self.stats.update({
            'n_samples_total': stats['n_samples'],
            'n_samples_train': len(self.X_train),
            'n_samples_val': len(self.X_val),
            'n_samples_test': len(self.X_test),
            'embedding_dim': stats['embedding_dim'],
            'use_pchembl': self.use_pchembl,
            'target_unit': target_unit,
            'target_stats': {
                'mean': stats['target_mean'],
                'std': stats['target_std'],
                'min': stats['target_min'],
                'max': stats['target_max']
            }
        })
    
    def train_models(self) -> Dict[str, Any]:
        """
        Train all regression models.
        
        Returns:
            Dict with validation metrics for all models
        """
        if self.verbose:
            print('🤖 STEP 2: Model Training')
            print('=' * 70)
        
        # Get models
        all_models = RegressionModels.get_all_models(
            random_state=self.random_state,
            verbose=self.verbose
        )
        
        # Filter models if specified
        if self.models_to_train:
            models = {k: v for k, v in all_models.items() if k in self.models_to_train}
        else:
            models = all_models
        
        if self.verbose:
            print(f"   Models to train: {len(models)}")
            print(f"   Models: {', '.join(models.keys())}")
            print()
        
        # Create trainer
        self.trainer = RegressionTrainer(
            models_dict=models,
            verbose=self.verbose,
            random_state=self.random_state
        )
        
        # Train all
        start_time = time.time()
        self.trainer.train_all(self.X_train, self.y_train, self.X_val, self.y_val)
        training_time = time.time() - start_time
        
        # Store results
        self.trained_models = self.trainer.trained_models
        self.train_metrics = self.trainer.train_results
        self.val_metrics = self.trainer.val_results
        
        if self.verbose:
            print(f"\n✅ Training complete!")
            print(f"   Total time: {training_time:.2f}s")
            print(f"   Average time per model: {training_time/len(models):.2f}s")
            print('=' * 70)
            print()
        
        self.stats['training_time'] = training_time
        self.stats['n_models_trained'] = len(models)
        
        return self.val_metrics
    
    def evaluate_on_test(self) -> Dict[str, Any]:
        """
        Evaluate all models on the test set.
        
        Returns:
            Dict with test metrics for all models
        """
        # Use the trainer's evaluate_on_test method which has the correct formatted print
        if hasattr(self, 'trainer') and self.trainer:
            self.test_metrics = self.trainer.evaluate_on_test(self.X_test, self.y_test)
        else:
            # Fallback if trainer is not available
            if self.verbose:
                print('📈 STEP 3: Evaluation on Test Set')
                print('=' * 70)
            
            for model_name, model in self.trained_models.items():
                if self.verbose:
                    print(f"   Evaluating {model_name}...")
                
                # Predictions
                y_pred = model.predict(self.X_test)
                
                # Calcular métricas
                metrics = self.metrics_calculator.calculate_all_metrics(
                    self.y_test,
                    y_pred,
                    model_name
                )
                
                self.test_metrics[model_name] = metrics
            
            if self.verbose:
                print("\n✅ Test set evaluation complete!")
                print('=' * 70)
                print()
        
        return self.test_metrics
    
    def print_results_summary(self) -> None:
        """Print results summary."""
        if not self.test_metrics:
            print("⚠️  No test results available")
            return
        
        print('📊 RESULTS SUMMARY (Test Set)')
        print('=' * 80)
        
        # Sort by MAE
        sorted_results = sorted(
            self.test_metrics.items(),
            key=lambda x: x[1]['MAE']
        )
        
        # Header
        header = f"{'Model':<20} {'MAE':>10} {'RMSE':>10} {'R²':>10} {'MedianAE':>10}"
        print(header)
        print('-' * 80)
        
        # Results
        for model_name, metrics in sorted_results:
            row = (
                f"{model_name:<20} "
                f"{metrics['MAE']:>10.4f} "
                f"{metrics['RMSE']:>10.4f} "
                f"{metrics['R2']:>10.4f} "
                f"{metrics['MedianAE']:>10.4f}"
            )
            print(row)
        
        print('=' * 80)
        
        # Best model
        best_model_name = sorted_results[0][0]
        best_metrics = sorted_results[0][1]
        
        print(f"\n🏆 BEST MODEL: {best_model_name}")
        print(f"   MAE: {best_metrics['MAE']:.4f} nM")
        print(f"   RMSE: {best_metrics['RMSE']:.4f} nM")
        print(f"   R²: {best_metrics['R2']:.4f}")
        print()
    
    def save_results(self) -> None:
        """Save metrics, models and statistics to files."""
        if self.verbose:
            print('💾 STEP 4: Saving Results')
            print('=' * 70)
        
        # Save test metrics
        metrics_file = self.output_dir / 'metrics' / 'test_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(self.test_metrics, f, indent=2)
        
        if self.verbose:
            print(f"   ✅ Test metrics saved: {metrics_file}")
        
        # Save validation metrics
        val_metrics_file = self.output_dir / 'metrics' / 'validation_metrics.json'
        with open(val_metrics_file, 'w') as f:
            json.dump(self.val_metrics, f, indent=2)
        
        if self.verbose:
            print(f"   ✅ Validation metrics saved: {val_metrics_file}")
        
        # Save trained models (if trainer is available)
        # Select best model based on TEST performance (consistent with displayed ranking)
        if hasattr(self, 'trainer') and self.trainer and hasattr(self.trainer, 'save_models'):
            models_dir = self.output_dir / 'models'
            self.trainer.save_models(str(models_dir), save_all=True, select_by='test')
            if self.verbose:
                print(f"   ✅ Models saved: {models_dir}")
        
        # Save pipeline stats
        stats_file = self.output_dir / 'pipeline_stats.json'
        self.stats['test_metrics_summary'] = {
            'best_model': min(self.test_metrics.items(), key=lambda x: x[1]['MAE'])[0] if self.test_metrics else None,
            'best_mae': min(m['MAE'] for m in self.test_metrics.values()) if self.test_metrics else None,
            'best_r2': max(m['R2'] for m in self.test_metrics.values()) if self.test_metrics else None
        }
        
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        if self.verbose:
            print(f"   ✅ Pipeline stats saved: {stats_file}")
            print('=' * 70)
            print()
    
    def run(self) -> Dict[str, Any]:
        """
        Execute complete pipeline.
        
        Returns:
            Dict with test metrics
        """
        if self.verbose:
            print('🚀 PIPELINE MODULAR DE REGRESSÃO - DockTKinase')
            print('=' * 70)
            print()
        
        start_time = time.time()
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Train models
        self.train_models()
        
        # Step 3: Evaluate on test
        self.evaluate_on_test()
        
        # Step 4: Save results
        self.save_results()
        
        # Summary
        self.print_results_summary()
        
        total_time = time.time() - start_time
        
        if self.verbose:
            print(f'✅ PIPELINE COMPLETE!')
            print(f'   Total time: {total_time:.2f}s ({total_time/60:.2f} min)')
            print(f'   Results saved to: {self.output_dir}')
            print('=' * 70)
        
        return self.test_metrics


# Convenience function
def run_regression_pipeline(
    embeddings_path: str,
    targets_path: str,
    output_dir: str = 'results/regression',
    models: Optional[List[str]] = None,
    random_state: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to execute complete pipeline.
    
    Args:
        embeddings_path: Path to embeddings
        targets_path: Path to targets
        output_dir: Output directory
        models: List of models (None = all)
        random_state: Seed
        verbose: Show progress
        
    Returns:
        Dict with test metrics
    """
    pipeline = RegressionPipeline(
        embeddings_path=embeddings_path,
        targets_path=targets_path,
        output_dir=output_dir,
        models_to_train=models,
        random_state=random_state,
        verbose=verbose
    )
    
    return pipeline.run()


if __name__ == '__main__':
    print("Modular Regression Pipeline - DockTKinase")
    print("=" * 70)
    print("\nTo use this module, import it:")
    print("\n  from regression.modular_pipeline import RegressionPipeline")
    print("\n  pipeline = RegressionPipeline(")
    print("      embeddings_path='embeddings.npy',")
    print("      targets_path='targets.npy'")
    print("  )")
    print("  results = pipeline.run()")
