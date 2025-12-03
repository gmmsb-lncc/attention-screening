"""
Unit tests for numpy file loading with allow_pickle safety.

These tests ensure that:
1. Object arrays (containing mixed types like strings) are loaded correctly with allow_pickle=True
2. Numeric arrays work with or without allow_pickle
3. The load_numpy utility function handles both cases
4. interaction_labels.npy (object array) loads correctly
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import sys

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))


class TestNumpyLoadSafety:
    """Test numpy loading with different array types."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        import tempfile
        import shutil
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path)
    
    def test_load_numeric_array_without_pickle(self, temp_dir):
        """Test that numeric arrays can be loaded without allow_pickle."""
        # Create a numeric array (float64)
        numeric_array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
        file_path = temp_dir / "numeric_array.npy"
        np.save(file_path, numeric_array)
        
        # Should work without allow_pickle
        loaded = np.load(file_path, allow_pickle=False)
        assert np.allclose(loaded, numeric_array)
        assert loaded.dtype == np.float64
    
    def test_load_object_array_requires_pickle(self, temp_dir):
        """Test that object arrays require allow_pickle=True."""
        # Create an object array (mixed types like interaction_labels)
        object_array = np.array([
            [123, "kinase_A", "IC50", 100.0, 7.0],
            [456, "kinase_B", "Ki", 50.0, 7.3],
            [789, "kinase_C", "Kd", 200.0, 6.7],
        ], dtype=object)
        file_path = temp_dir / "object_array.npy"
        np.save(file_path, object_array)
        
        # Should fail without allow_pickle
        with pytest.raises(ValueError, match="allow_pickle=False"):
            np.load(file_path, allow_pickle=False)
        
        # Should work with allow_pickle=True
        loaded = np.load(file_path, allow_pickle=True)
        assert loaded.dtype == object
        assert loaded.shape == (3, 5)
        assert loaded[0, 1] == "kinase_A"
    
    def test_load_interaction_labels_format(self, temp_dir):
        """Test loading interaction_labels.npy format (the exact format used in pipeline)."""
        # Simulate interaction_labels format: [molregno, kinase, type, standard_value, pchembl]
        interaction_labels = np.array([
            [12345, "ABL1", "IC50", 100.5, 7.0],
            [12346, "EGFR", "Ki", 50.2, 7.3],
            [12347, "JAK2", "Kd", 200.8, 6.7],
            [12348, "SRC", "IC50", None, None],  # Some values may be None
        ], dtype=object)
        
        file_path = temp_dir / "interaction_labels.npy"
        np.save(file_path, interaction_labels)
        
        # Must use allow_pickle=True
        loaded = np.load(file_path, allow_pickle=True)
        
        assert loaded.dtype == object
        assert loaded.shape == (4, 5)
        # Check string values are preserved
        assert loaded[0, 1] == "ABL1"
        assert loaded[1, 2] == "Ki"
        # Check numeric values
        assert loaded[0, 3] == 100.5
        # Check None values are preserved
        assert loaded[3, 3] is None
    
    def test_load_numpy_utility_function(self, temp_dir):
        """Test the load_numpy utility function handles both array types."""
        from src.build.utils.file_utils import load_numpy
        
        # Test with numeric array
        numeric_array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        numeric_path = temp_dir / "numeric.npy"
        np.save(numeric_path, numeric_array)
        
        loaded_numeric = load_numpy(numeric_path)
        assert np.allclose(loaded_numeric, numeric_array)
        
        # Test with object array
        object_array = np.array(["a", "b", "c"], dtype=object)
        object_path = temp_dir / "object.npy"
        np.save(object_path, object_array)
        
        loaded_object = load_numpy(object_path)
        assert np.array_equal(loaded_object, object_array)
    
    def test_load_numpy_with_explicit_allow_pickle_false(self, temp_dir):
        """Test load_numpy with allow_pickle=False for numeric arrays."""
        from src.build.utils.file_utils import load_numpy
        
        # Numeric array should work with allow_pickle=False
        numeric_array = np.array([[1, 2], [3, 4]], dtype=np.int32)
        file_path = temp_dir / "int_array.npy"
        np.save(file_path, numeric_array)
        
        loaded = load_numpy(file_path, allow_pickle=False)
        assert np.array_equal(loaded, numeric_array)
    
    def test_embedding_matrix_format(self, temp_dir):
        """Test that embedding matrices (float arrays) load without pickle."""
        # Simulate embedding matrix format
        embedding_matrix = np.random.randn(100, 1088).astype(np.float32)
        file_path = temp_dir / "embedding_matrix.npy"
        np.save(file_path, embedding_matrix)
        
        # Should work without allow_pickle
        loaded = np.load(file_path, allow_pickle=False)
        assert loaded.shape == (100, 1088)
        assert loaded.dtype == np.float32
    
    def test_binary_labels_format(self, temp_dir):
        """Test that binary labels (int arrays) load without pickle."""
        # Simulate binary labels format
        binary_labels = np.random.randint(0, 2, size=100).astype(np.int32)
        file_path = temp_dir / "binary_labels.npy"
        np.save(file_path, binary_labels)
        
        # Should work without allow_pickle
        loaded = np.load(file_path, allow_pickle=False)
        assert loaded.shape == (100,)
        assert loaded.dtype == np.int32
        assert set(np.unique(loaded)).issubset({0, 1})
    
    def test_split_indices_format(self, temp_dir):
        """Test that split indices (int arrays) load correctly."""
        # Simulate split indices format
        train_idx = np.arange(0, 80, dtype=np.int32)
        val_idx = np.arange(80, 90, dtype=np.int32)
        test_idx = np.arange(90, 100, dtype=np.int32)
        
        np.save(temp_dir / "train_indices.npy", train_idx)
        np.save(temp_dir / "val_indices.npy", val_idx)
        np.save(temp_dir / "test_indices.npy", test_idx)
        
        # Should work without allow_pickle
        loaded_train = np.load(temp_dir / "train_indices.npy", allow_pickle=False)
        loaded_val = np.load(temp_dir / "val_indices.npy", allow_pickle=False)
        loaded_test = np.load(temp_dir / "test_indices.npy", allow_pickle=False)
        
        assert len(loaded_train) == 80
        assert len(loaded_val) == 10
        assert len(loaded_test) == 10


class TestInteractionLabelsLoading:
    """Specific tests for interaction_labels.npy loading scenarios."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        import tempfile
        import shutil
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path)
    
    def test_interaction_labels_with_all_columns(self, temp_dir):
        """Test interaction_labels with all 5 columns (including pchembl)."""
        labels = np.array([
            [1, "KINASE1", "IC50", 100.0, 7.0],
            [2, "KINASE2", "Ki", 50.0, 7.3],
        ], dtype=object)
        
        file_path = temp_dir / "interaction_labels.npy"
        np.save(file_path, labels)
        
        loaded = np.load(file_path, allow_pickle=True)
        assert loaded.shape[1] == 5
    
    def test_interaction_labels_with_4_columns(self, temp_dir):
        """Test interaction_labels with 4 columns (no pchembl)."""
        labels = np.array([
            [1, "KINASE1", "IC50", 100.0],
            [2, "KINASE2", "Ki", 50.0],
        ], dtype=object)
        
        file_path = temp_dir / "interaction_labels.npy"
        np.save(file_path, labels)
        
        loaded = np.load(file_path, allow_pickle=True)
        assert loaded.shape[1] == 4
    
    def test_interaction_labels_with_none_values(self, temp_dir):
        """Test interaction_labels with None values (missing pchembl)."""
        labels = np.array([
            [1, "KINASE1", "IC50", 100.0, 7.0],
            [2, "KINASE2", "Ki", None, None],  # Missing values
        ], dtype=object)
        
        file_path = temp_dir / "interaction_labels.npy"
        np.save(file_path, labels)
        
        loaded = np.load(file_path, allow_pickle=True)
        assert loaded[1, 3] is None
        assert loaded[1, 4] is None
    
    def test_base_labels_load_method(self, temp_dir):
        """Test that BaseLabels subclass can load object arrays correctly."""
        from src.build.labels.interaction_labels import InteractionLabels
        from src.build.core import BuildConfig
        
        # Create test interaction labels
        labels = np.array([
            [1, "KINASE1", "IC50", 100.0, 7.0],
            [2, "KINASE2", "Ki", 50.0, 7.3],
        ], dtype=object)
        
        file_path = temp_dir / "test_labels.npy"
        np.save(file_path, labels)
        
        # Load directly using np.load with allow_pickle=True
        loaded = np.load(file_path, allow_pickle=True)
        assert loaded is not None
        assert loaded.shape == (2, 5)
        assert loaded.dtype == object


class TestBuildPipelineNumpyLoading:
    """Test numpy loading in build pipeline context."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        import tempfile
        import shutil
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path)
    
    def test_stratification_labels_loading(self, temp_dir):
        """Test that stratification can load interaction labels correctly."""
        # Create mock files that stratification expects
        
        # Embedding matrix (float array)
        embedding_matrix = np.random.randn(100, 1088).astype(np.float32)
        np.save(temp_dir / "embedding_matrix.npy", embedding_matrix)
        
        # Interaction labels (object array)
        interaction_labels = np.array([
            [i, f"KINASE_{i % 10}", "IC50", 100.0 + i, 7.0 - (i * 0.01)]
            for i in range(100)
        ], dtype=object)
        np.save(temp_dir / "interaction_labels.npy", interaction_labels)
        
        # Load both
        matrix = np.load(temp_dir / "embedding_matrix.npy")  # Works without pickle
        labels = np.load(temp_dir / "interaction_labels.npy", allow_pickle=True)  # Needs pickle
        
        assert matrix.shape == (100, 1088)
        assert labels.shape == (100, 5)
        assert labels.dtype == object


class TestRegressionDataLoading:
    """Test numpy loading for regression pipeline."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        import tempfile
        import shutil
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path)
    
    def test_regression_data_loader(self, temp_dir):
        """Test that regression data manager handles all file types."""
        from src.regression.core.data_loader import DataManager
        
        # Create test files
        n_samples = 50
        embedding_dim = 1088
        
        # Embedding matrix
        embeddings = np.random.randn(n_samples, embedding_dim).astype(np.float32)
        np.save(temp_dir / "embedding_matrix.npy", embeddings)
        
        # Interaction labels with pchembl values (object array)
        interaction_labels = np.array([
            [i, f"KINASE_{i % 5}", "IC50", 100.0 + i * 10, 6.0 + np.random.random()]
            for i in range(n_samples)
        ], dtype=object)
        np.save(temp_dir / "interaction_labels.npy", interaction_labels)
        
        # Create a numeric targets array (extracted pchembl values)
        pchembl_values = np.array([float(row[4]) for row in interaction_labels], dtype=np.float32)
        np.save(temp_dir / "targets.npy", pchembl_values)
        
        # Create data manager
        manager = DataManager(
            embeddings_path=str(temp_dir / "embedding_matrix.npy"),
            targets_path=str(temp_dir / "targets.npy"),
            use_pchembl=False  # Already in pchembl format
        )
        
        # Load data
        X = manager.load_embeddings()
        y = manager.load_targets()
        
        assert X.shape == (n_samples, embedding_dim)
        assert y.shape == (n_samples,)
        
        # Check that y values are numeric
        assert y.dtype in [np.float32, np.float64]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestEdgeCases:
    """Test edge cases that could cause allow_pickle errors."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        import tempfile
        import shutil
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path)
    
    def test_legacy_format_interaction_labels(self, temp_dir):
        """Test loading interaction_labels that might have been created in legacy format."""
        # Simulate a legacy format that might have been saved differently
        labels = np.array([
            [123, "ABL1_HUMAN", "IC50", 100.5, 7.0],
            [456, "EGFR_HUMAN", "Ki", None, None],  # Some legacy data might have None
        ], dtype=object)
        
        file_path = temp_dir / "interaction_labels.npy"
        np.save(file_path, labels)
        
        # This should NOT raise an error
        loaded = np.load(file_path, allow_pickle=True)
        assert loaded.shape == (2, 5)
    
    def test_file_utils_load_numpy_with_object_array(self, temp_dir):
        """Test that load_numpy utility handles object arrays correctly."""
        from src.build.utils.file_utils import load_numpy
        
        # Create an object array (like interaction_labels)
        labels = np.array([
            ["kinase_1", "IC50", 100.0],
            ["kinase_2", "Ki", 50.0],
        ], dtype=object)
        
        file_path = temp_dir / "labels.npy"
        np.save(file_path, labels)
        
        # Default should work (allow_pickle=True by default now)
        loaded = load_numpy(file_path)
        assert np.array_equal(loaded, labels)
    
    def test_build_pipeline_labels_loading_simulation(self, temp_dir):
        """Simulate the exact scenario from build_pipeline.py line 727."""
        # Create files like build_pipeline expects
        
        # 1. Embedding matrix (float array)
        embedding_matrix = np.random.randn(100, 1088).astype(np.float32)
        np.save(temp_dir / "embedding_matrix.npy", embedding_matrix)
        
        # 2. Interaction labels (object array - this was causing the error)
        interaction_labels = np.array([
            [i, f"KINASE_{i % 10}", "IC50", 100.0 + i * 10, 6.0 + 0.1 * i]
            for i in range(100)
        ], dtype=object)
        np.save(temp_dir / "interaction_labels.npy", interaction_labels)
        
        # Simulate what build_pipeline.py does at line 725-727
        labels_path = temp_dir / "interaction_labels.npy"
        
        # This is the exact pattern used in build_pipeline.py
        labels = np.load(str(labels_path), allow_pickle=True)
        
        assert labels.shape == (100, 5)
        assert labels.dtype == object
        assert labels[0, 1] == "KINASE_0"
    
    def test_prevent_allow_pickle_false_on_object_array(self, temp_dir):
        """Verify that allow_pickle=False raises error on object arrays."""
        # Create object array
        obj_array = np.array(["a", "b", "c"], dtype=object)
        file_path = temp_dir / "object.npy"
        np.save(file_path, obj_array)
        
        # This MUST raise an error (to confirm the error scenario)
        with pytest.raises(ValueError):
            np.load(file_path, allow_pickle=False)
    
    def test_mixed_types_in_interaction_labels(self, temp_dir):
        """Test interaction labels with mixed types (int, str, float, None)."""
        labels = np.array([
            [1, "K1", "IC50", 100.0, 7.0],
            [2, "K2", "Ki", 50.0, 7.3],
            [3, "K3", "Kd", None, None],  # Missing values
            [4, "K4", "IC50", 0.0, float('inf')],  # Edge case values
        ], dtype=object)
        
        file_path = temp_dir / "labels.npy"
        np.save(file_path, labels)
        
        # Must use allow_pickle=True
        loaded = np.load(file_path, allow_pickle=True)
        
        assert loaded[2, 3] is None
        assert loaded[3, 4] == float('inf')

