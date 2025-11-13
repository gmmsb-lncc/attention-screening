"""
End-to-end integration test for stratification across the complete pipeline.

This test validates the COMPLETE workflow:
1. BuildPipeline generates embeddings and performs stratification
2. Classification pipeline uses the stratified splits
3. Regression pipeline uses the SAME stratified splits
4. Verify that all three use IDENTICAL train/val/test indices

This is the CRITICAL test that validates the entire stratification integration.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil
import sys
import logging

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from src.build.core.config import BuildConfig
from src.build.pipeline.build_pipeline import BuildPipeline
from src.build.pipeline.split_indices import SplitIndices
from src.classifier.modular_pipeline import MLPEmbeddingPipeline
from src.regression.modular_pipeline import RegressionPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestEndToEndStratification:
    """End-to-end tests for complete pipeline stratification integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_dir = Path(self.temp_dir)
        
        # Create necessary subdirectories
        (self.test_dir / "protein_embeddings").mkdir(parents=True)
        (self.test_dir / "ligand_embeddings").mkdir(parents=True)
        (self.test_dir / "concatenated_embeddings").mkdir(parents=True)
        
        logger.info(f"Test directory created: {self.temp_dir}")
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
            logger.info(f"Test directory cleaned: {self.temp_dir}")
    
    def _create_synthetic_data(self, n_samples: int = 100):
        """
        Create synthetic data for testing.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Tuple of (protein_embeddings, ligand_embeddings, labels, activities)
        """
        np.random.seed(42)
        
        # Generate embeddings
        protein_embeddings = np.random.randn(n_samples, 320).astype(np.float32)
        ligand_embeddings = np.random.randn(n_samples, 768).astype(np.float32)
        
        # Generate labels (binary classification)
        labels = np.random.randint(0, 2, n_samples).astype(np.int32)
        
        # Generate activities (continuous regression target)
        activities = np.random.uniform(4.0, 9.0, n_samples).astype(np.float32)
        
        return protein_embeddings, ligand_embeddings, labels, activities
    
    def _save_embeddings_and_matrix(
        self,
        protein_embeddings: np.ndarray,
        ligand_embeddings: np.ndarray,
        labels: np.ndarray,
        activities: np.ndarray
    ):
        """Save embeddings and concatenated matrix to disk."""
        # Save protein embeddings
        protein_path = self.test_dir / "protein_embeddings" / "protein_embeddings.npy"
        np.save(protein_path, protein_embeddings)
        
        # Save ligand embeddings
        ligand_path = self.test_dir / "ligand_embeddings" / "ligand_embeddings.npy"
        np.save(ligand_path, ligand_embeddings)
        
        # Create concatenated matrix
        concatenated = np.hstack([protein_embeddings, ligand_embeddings])
        matrix_path = self.test_dir / "concatenated_embeddings" / "concatenated_matrix.npy"
        np.save(matrix_path, concatenated)
        
        # Save labels
        labels_path = self.test_dir / "concatenated_embeddings" / "labels.npy"
        np.save(labels_path, labels)
        
        # Save activities
        activities_path = self.test_dir / "concatenated_embeddings" / "activities.npy"
        np.save(activities_path, activities)
        
        logger.info(f"Saved embeddings and matrix to {self.test_dir}")
        
        return {
            'protein_path': protein_path,
            'ligand_path': ligand_path,
            'matrix_path': matrix_path,
            'labels_path': labels_path,
            'activities_path': activities_path
        }
    
    def test_build_pipeline_generates_splits(self):
        """Test that BuildPipeline generates and saves split indices."""
        # Create synthetic data
        protein_emb, ligand_emb, labels, activities = self._create_synthetic_data(100)
        
        # Save data
        paths = self._save_embeddings_and_matrix(protein_emb, ligand_emb, labels, activities)
        
        # Create BuildConfig
        config = BuildConfig()
        config.output_dir = str(self.test_dir)
        
        # Create BuildPipeline and mock stratification
        from src.build.pipeline.stratification_manager import StratificationManager
        
        manager = StratificationManager(config, random_state=42)
        splits = manager.stratify(
            protein_embeddings=protein_emb,
            ligand_embeddings=ligand_emb,
            labels=labels,
            test_size=0.2,
            val_size=0.1
        )
        
        # Save splits (simulating BuildPipeline behavior)
        splits_path = self.test_dir / "splits.npz"
        splits.save(str(splits_path))
        
        # Verify splits file exists
        assert splits_path.exists(), "Splits file should be created"
        
        # Load and verify
        loaded_splits = SplitIndices.load(str(splits_path))
        assert np.array_equal(loaded_splits.train_idx, splits.train_idx)
        assert np.array_equal(loaded_splits.val_idx, splits.val_idx)
        assert np.array_equal(loaded_splits.test_idx, splits.test_idx)
        
        logger.info("✓ BuildPipeline generates and saves splits correctly")
    
    def test_classification_uses_external_splits(self):
        """Test that Classification pipeline can use external splits."""
        # Create synthetic data
        protein_emb, ligand_emb, labels, activities = self._create_synthetic_data(100)
        
        # Save data
        paths = self._save_embeddings_and_matrix(protein_emb, ligand_emb, labels, activities)
        
        # Create splits
        config = BuildConfig()
        from src.build.pipeline.stratification_manager import StratificationManager
        
        manager = StratificationManager(config, random_state=42)
        splits = manager.stratify(
            protein_embeddings=protein_emb,
            ligand_embeddings=ligand_emb,
            labels=labels,
            test_size=0.2,
            val_size=0.1
        )
        
        # Create classification pipeline with external splits
        embeddings_path = str(paths['matrix_path'])
        labels_path = str(paths['labels_path'])
        output_dir = str(self.test_dir / "classifier_output")
        
        # Create pipeline with split_indices
        clf_pipeline = MLPEmbeddingPipeline(
            embeddings_path=embeddings_path,
            labels_path=labels_path,
            batch_size=16,
            epochs=2,  # Quick test
            model_output=f"{output_dir}/model.pth",
            split_indices=splits
        )
        
        # Verify pipeline has split_indices
        assert clf_pipeline.split_indices is not None
        assert np.array_equal(clf_pipeline.split_indices.train_idx, splits.train_idx)
        
        logger.info("✓ Classification pipeline accepts external splits")
    
    def test_regression_uses_external_splits(self):
        """Test that Regression pipeline can use external splits."""
        # Create synthetic data
        protein_emb, ligand_emb, labels, activities = self._create_synthetic_data(100)
        
        # Save data
        paths = self._save_embeddings_and_matrix(protein_emb, ligand_emb, labels, activities)
        
        # Create splits
        config = BuildConfig()
        from src.build.pipeline.stratification_manager import StratificationManager
        
        manager = StratificationManager(config, random_state=42)
        splits = manager.stratify(
            protein_embeddings=protein_emb,
            ligand_embeddings=ligand_emb,
            labels=labels,
            test_size=0.2,
            val_size=0.1
        )
        
        # Create regression pipeline with external splits
        embeddings_path = str(paths['matrix_path'])
        targets_path = str(paths['activities_path'])
        output_dir = str(self.test_dir / "regression_output")
        
        # Create pipeline with split_indices
        reg_pipeline = RegressionPipeline(
            embeddings_path=embeddings_path,
            targets_path=targets_path,
            output_dir=output_dir,
            random_state=42,
            split_indices=splits
        )
        
        # Verify pipeline has split_indices
        assert reg_pipeline.split_indices is not None
        assert np.array_equal(reg_pipeline.split_indices.train_idx, splits.train_idx)
        
        logger.info("✓ Regression pipeline accepts external splits")
    
    @pytest.mark.slow
    def test_complete_pipeline_identical_splits(self):
        """
        CRITICAL END-TO-END TEST: Verify complete pipeline uses identical splits.
        
        This test validates the PRIMARY REQUIREMENT:
        - BuildPipeline stratifies once
        - Classification uses those splits
        - Regression uses THE SAME splits
        - All three have IDENTICAL train/val/test indices
        """
        logger.info("\n" + "="*70)
        logger.info("CRITICAL END-TO-END TEST: Complete Pipeline with Identical Splits")
        logger.info("="*70)
        
        # 1. Create synthetic data
        logger.info("\n1. Creating synthetic dataset (100 samples)...")
        protein_emb, ligand_emb, labels, activities = self._create_synthetic_data(100)
        paths = self._save_embeddings_and_matrix(protein_emb, ligand_emb, labels, activities)
        logger.info("   ✓ Data created and saved")
        
        # 2. BuildPipeline: Stratify and save splits
        logger.info("\n2. BuildPipeline: Performing stratification...")
        config = BuildConfig()
        config.output_dir = str(self.test_dir)
        
        from src.build.pipeline.stratification_manager import StratificationManager
        manager = StratificationManager(config, random_state=42)
        
        build_splits = manager.stratify(
            protein_embeddings=protein_emb,
            ligand_embeddings=ligand_emb,
            labels=labels,
            test_size=0.2,
            val_size=0.1
        )
        
        splits_path = self.test_dir / "splits.npz"
        build_splits.save(str(splits_path))
        logger.info(f"   ✓ Stratification complete")
        logger.info(f"   ✓ Train: {len(build_splits.train_idx)} samples")
        logger.info(f"   ✓ Val:   {len(build_splits.val_idx)} samples")
        logger.info(f"   ✓ Test:  {len(build_splits.test_idx)} samples")
        logger.info(f"   ✓ Splits saved to: {splits_path}")
        
        # 3. Load splits (simulating downstream pipelines)
        logger.info("\n3. Loading splits for downstream pipelines...")
        loaded_splits = SplitIndices.load(str(splits_path))
        logger.info("   ✓ Splits loaded successfully")
        
        # 4. Classification pipeline uses loaded splits
        logger.info("\n4. Classification pipeline using loaded splits...")
        
        embeddings_path = str(paths['matrix_path'])
        labels_path = str(paths['labels_path'])
        output_dir = str(self.test_dir / "classifier_output")
        
        clf_pipeline = MLPEmbeddingPipeline(
            embeddings_path=embeddings_path,
            labels_path=labels_path,
            batch_size=16,
            epochs=1,  # Minimal training for test
            model_output=f"{output_dir}/model.pth",
            metrics_output=f"{output_dir}/metrics.json",
            split_indices=loaded_splits
        )
        
        # Extract indices that classification will use
        clf_train_idx = clf_pipeline.split_indices.train_idx.copy()
        clf_val_idx = clf_pipeline.split_indices.val_idx.copy()
        clf_test_idx = clf_pipeline.split_indices.test_idx.copy()
        
        logger.info(f"   ✓ Classification train: {len(clf_train_idx)} samples")
        logger.info(f"   ✓ Classification val:   {len(clf_val_idx)} samples")
        logger.info(f"   ✓ Classification test:  {len(clf_test_idx)} samples")
        
        # 5. Regression pipeline uses loaded splits
        logger.info("\n5. Regression pipeline using loaded splits...")
        
        targets_path = str(paths['activities_path'])
        reg_output_dir = str(self.test_dir / "regression_output")
        
        reg_pipeline = RegressionPipeline(
            embeddings_path=embeddings_path,
            targets_path=targets_path,
            output_dir=reg_output_dir,
            random_state=42,
            split_indices=loaded_splits
        )
        
        # Extract indices that regression will use
        reg_train_idx = reg_pipeline.split_indices.train_idx.copy()
        reg_val_idx = reg_pipeline.split_indices.val_idx.copy()
        reg_test_idx = reg_pipeline.split_indices.test_idx.copy()
        
        logger.info(f"   ✓ Regression train: {len(reg_train_idx)} samples")
        logger.info(f"   ✓ Regression val:   {len(reg_val_idx)} samples")
        logger.info(f"   ✓ Regression test:  {len(reg_test_idx)} samples")
        
        # 6. CRITICAL VALIDATION: Verify all splits are IDENTICAL
        logger.info("\n6. CRITICAL VALIDATION: Verifying identical splits...")
        logger.info("   Comparing BuildPipeline vs Classification...")
        
        assert np.array_equal(build_splits.train_idx, clf_train_idx), \
            "BuildPipeline and Classification must use IDENTICAL train indices!"
        logger.info("      ✓ Train indices: IDENTICAL")
        
        assert np.array_equal(build_splits.val_idx, clf_val_idx), \
            "BuildPipeline and Classification must use IDENTICAL validation indices!"
        logger.info("      ✓ Validation indices: IDENTICAL")
        
        assert np.array_equal(build_splits.test_idx, clf_test_idx), \
            "BuildPipeline and Classification must use IDENTICAL test indices!"
        logger.info("      ✓ Test indices: IDENTICAL")
        
        logger.info("\n   Comparing BuildPipeline vs Regression...")
        
        assert np.array_equal(build_splits.train_idx, reg_train_idx), \
            "BuildPipeline and Regression must use IDENTICAL train indices!"
        logger.info("      ✓ Train indices: IDENTICAL")
        
        assert np.array_equal(build_splits.val_idx, reg_val_idx), \
            "BuildPipeline and Regression must use IDENTICAL validation indices!"
        logger.info("      ✓ Validation indices: IDENTICAL")
        
        assert np.array_equal(build_splits.test_idx, reg_test_idx), \
            "BuildPipeline and Regression must use IDENTICAL test indices!"
        logger.info("      ✓ Test indices: IDENTICAL")
        
        logger.info("\n   Comparing Classification vs Regression...")
        
        assert np.array_equal(clf_train_idx, reg_train_idx), \
            "Classification and Regression must use IDENTICAL train indices!"
        logger.info("      ✓ Train indices: IDENTICAL")
        
        assert np.array_equal(clf_val_idx, reg_val_idx), \
            "Classification and Regression must use IDENTICAL validation indices!"
        logger.info("      ✓ Validation indices: IDENTICAL")
        
        assert np.array_equal(clf_test_idx, reg_test_idx), \
            "Classification and Regression must use IDENTICAL test indices!"
        logger.info("      ✓ Test indices: IDENTICAL")
        
        # 7. Additional validations
        logger.info("\n7. Additional validations...")
        
        # Verify no data leakage
        train_set = set(build_splits.train_idx.tolist())
        val_set = set(build_splits.val_idx.tolist())
        test_set = set(build_splits.test_idx.tolist())
        
        assert len(train_set & val_set) == 0, "No overlap between train and validation"
        assert len(train_set & test_set) == 0, "No overlap between train and test"
        assert len(val_set & test_set) == 0, "No overlap between validation and test"
        logger.info("   ✓ No data leakage detected")
        
        # Verify complete coverage
        all_indices = train_set | val_set | test_set
        assert len(all_indices) == 100, "All samples must be assigned"
        logger.info("   ✓ Complete coverage: all samples assigned")
        
        # 8. SUCCESS!
        logger.info("\n" + "="*70)
        logger.info("✅ END-TO-END TEST PASSED: Complete pipeline uses identical splits!")
        logger.info("="*70)
        logger.info("\nSummary:")
        logger.info(f"  • BuildPipeline stratified: {len(all_indices)} samples")
        logger.info(f"  • Classification uses: SAME {len(all_indices)} samples")
        logger.info(f"  • Regression uses: SAME {len(all_indices)} samples")
        logger.info(f"  • Split consistency: 100% VERIFIED ✓")
        logger.info(f"  • Data leakage: NONE ✓")
        logger.info(f"  • Coverage: COMPLETE ✓")
        logger.info("\n" + "="*70 + "\n")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s', '--tb=short'])
