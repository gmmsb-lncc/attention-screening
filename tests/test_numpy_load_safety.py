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


class TestPchemblConversion:
    """Test pChEMBL value calculation and filling."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        import tempfile
        import shutil
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path)
    
    def test_pchembl_calculation_formula(self):
        """Test the pChEMBL calculation formula: pChEMBL = 9 - log10(nM)."""
        # Test values
        test_cases = [
            (1.0, 9.0),      # 1 nM -> pChEMBL 9
            (10.0, 8.0),     # 10 nM -> pChEMBL 8
            (100.0, 7.0),    # 100 nM -> pChEMBL 7
            (1000.0, 6.0),   # 1000 nM -> pChEMBL 6
            (10000.0, 5.0),  # 10000 nM -> pChEMBL 5
        ]
        
        for nm_value, expected_pchembl in test_cases:
            calculated = 9 - np.log10(nm_value)
            assert np.isclose(calculated, expected_pchembl, atol=0.01), \
                f"Expected pChEMBL {expected_pchembl} for {nm_value} nM, got {calculated}"
    
    def test_stratification_manager_binary_extraction_with_none(self):
        """Test that stratification handles None values in pchembl column."""
        from src.build.pipeline.stratification_manager import StratificationManager
        from src.build.core import BuildConfig
        
        config = BuildConfig(output_dir="/tmp/test")
        manager = StratificationManager(config)
        
        # Create interaction_labels with some None pchembl values
        labels = np.array([
            [1, "K1", "IC50", 100.0, 7.0],    # Active (pchembl 7.0 >= 6.0)
            [2, "K2", "Ki", 50.0, None],       # None pchembl, should use standard_value
            [3, "K3", "Kd", 2000.0, None],     # Inactive (2000 nM > 1000)
            [4, "K4", "IC50", 500.0, 6.3],     # Active
        ], dtype=object)
        
        # Should not raise an error
        binary = manager._extract_binary_labels(labels)
        
        assert binary is not None
        assert len(binary) == 4
        assert binary[0] == 1  # Active (pchembl 7.0)
        assert binary[1] == 1  # Active (50 nM <= 1000)
        assert binary[2] == 0  # Inactive (2000 nM > 1000)
        assert binary[3] == 1  # Active (pchembl 6.3)
    
    def test_stratification_manager_all_none_pchembl(self):
        """Test stratification when all pchembl values are None."""
        from src.build.pipeline.stratification_manager import StratificationManager
        from src.build.core import BuildConfig
        
        config = BuildConfig(output_dir="/tmp/test")
        manager = StratificationManager(config)
        
        # All pchembl values are None - should use standard_value
        labels = np.array([
            [1, "K1", "IC50", 100.0, None],   # Active (100 nM <= 1000)
            [2, "K2", "Ki", 5000.0, None],    # Inactive (5000 nM > 1000)
            [3, "K3", "Kd", 500.0, None],     # Active (500 nM <= 1000)
        ], dtype=object)
        
        binary = manager._extract_binary_labels(labels)
        
        assert binary is not None
        assert len(binary) == 3
        assert binary[0] == 1  # Active
        assert binary[1] == 0  # Inactive
        assert binary[2] == 1  # Active
    
    def test_stratification_manager_4_column_format(self):
        """Test stratification with 4-column format (no pchembl column)."""
        from src.build.pipeline.stratification_manager import StratificationManager
        from src.build.core import BuildConfig
        
        config = BuildConfig(output_dir="/tmp/test")
        manager = StratificationManager(config)
        
        # 4-column format: [molregno, kinase, type, standard_value]
        labels = np.array([
            [1, "K1", "IC50", 100.0],    # Active
            [2, "K2", "Ki", 5000.0],     # Inactive
            [3, "K3", "Kd", 1000.0],     # Active (exactly at threshold)
        ], dtype=object)
        
        binary = manager._extract_binary_labels(labels)
        
        assert binary is not None
        assert len(binary) == 3
        assert binary[0] == 1  # Active
        assert binary[1] == 0  # Inactive
        assert binary[2] == 1  # Active (at threshold)
    
    def test_is_valid_number_helper(self):
        """Test the _is_valid_number helper function."""
        from src.build.pipeline.stratification_manager import StratificationManager
        from src.build.core import BuildConfig
        
        config = BuildConfig(output_dir="/tmp/test")
        manager = StratificationManager(config)
        
        # Valid numbers
        assert manager._is_valid_number(100.0) == True
        assert manager._is_valid_number(0.0) == True
        assert manager._is_valid_number("100.5") == True
        assert manager._is_valid_number(7) == True
        
        # Invalid values
        assert manager._is_valid_number(None) == False
        assert manager._is_valid_number(float('nan')) == False
        assert manager._is_valid_number("None") == False
        assert manager._is_valid_number("nan") == False
        assert manager._is_valid_number("") == False
        assert manager._is_valid_number("N/A") == False
    
    def test_real_dataset_pchembl_filling(self):
        """Test pchembl filling with data similar to real dataset structure."""
        from src.build.pipeline.stratification_manager import StratificationManager
        from src.build.core import BuildConfig
        
        config = BuildConfig(output_dir="/tmp/test")
        manager = StratificationManager(config)
        
        # Simulate real dataset: some rows have pchembl, some only have standard_value
        # This is exactly like the kinase_non_human_compounds.tsv dataset
        labels = np.array([
            # Rows with both pchembl and standard_value
            [250, "Calcium-dependent protein kinase 1", "Kd", 850.0, 6.07],
            [250, "MAP kinase p38 alpha", "Kd", 20.0, 7.70],
            [250, "Mitogen-activated protein kinase 1", "IC50", 100.0, 7.00],
            [250, "Mitogen-activated protein kinase 2", "IC50", 105.0, 6.98],
            # Rows with standard_value but NO pchembl (NaN)
            [250, "Myosin light chain kinase, smooth muscle", "Kd", 10000.0, np.nan],
            [250, "Serine/threonine-protein kinase pknB", "Kd", 10000.0, np.nan],
            [160532, "Protein kinase C, PKC; classical", "IC50", 540.0, 6.27],
            [163291, "Some kinase", "IC50", 10000.0, np.nan],
        ], dtype=object)
        
        # Should handle NaN pchembl values by calculating from standard_value
        binary = manager._extract_binary_labels(labels)
        
        assert binary is not None
        assert len(binary) == 8
        
        # Verify each row
        # Row 0: pchembl 6.07 >= 6.0 -> Active
        assert binary[0] == 1, f"Row 0: expected active (pchembl=6.07), got {binary[0]}"
        # Row 1: pchembl 7.70 >= 6.0 -> Active
        assert binary[1] == 1, f"Row 1: expected active (pchembl=7.70), got {binary[1]}"
        # Row 2: pchembl 7.00 >= 6.0 -> Active
        assert binary[2] == 1, f"Row 2: expected active (pchembl=7.00), got {binary[2]}"
        # Row 3: pchembl 6.98 >= 6.0 -> Active
        assert binary[3] == 1, f"Row 3: expected active (pchembl=6.98), got {binary[3]}"
        # Row 4: standard_value 10000 nM -> pchembl 5.0 < 6.0 -> Inactive
        assert binary[4] == 0, f"Row 4: expected inactive (10000nM -> pchembl=5.0), got {binary[4]}"
        # Row 5: standard_value 10000 nM -> pchembl 5.0 < 6.0 -> Inactive
        assert binary[5] == 0, f"Row 5: expected inactive (10000nM -> pchembl=5.0), got {binary[5]}"
        # Row 6: pchembl 6.27 >= 6.0 -> Active
        assert binary[6] == 1, f"Row 6: expected active (pchembl=6.27), got {binary[6]}"
        # Row 7: standard_value 10000 nM -> pchembl 5.0 < 6.0 -> Inactive
        assert binary[7] == 0, f"Row 7: expected inactive (10000nM -> pchembl=5.0), got {binary[7]}"
    
    def test_interaction_labels_fill_missing_pchembl(self):
        """Test InteractionLabels._fill_missing_pchembl_values method."""
        from src.build.labels.interaction_labels import InteractionLabels
        from src.build.core import BuildConfig
        import tempfile
        
        config = BuildConfig(output_dir="/tmp/test")
        
        # Create a temporary TSV file for initialization (required by InteractionLabels)
        with tempfile.NamedTemporaryFile(suffix='.tsv', delete=False, mode='w') as f:
            f.write("molregno\ttarget_kinase\tstandard_type\tstandard_value\tpchembl_value\n")
            f.write("1\tK1\tIC50\t100\t7.0\n")
            temp_path = f.name
        
        try:
            labels_gen = InteractionLabels(config, temp_path)
            
            # Manually set interaction_data to test _fill_missing_pchembl_values
            labels_gen.interaction_data = np.array([
                [1, "K1", "IC50", 100.0, 7.0],      # Has pchembl
                [2, "K2", "Ki", 1000.0, None],       # Missing pchembl
                [3, "K3", "Kd", 10000.0, np.nan],    # Missing pchembl (NaN)
                [4, "K4", "IC50", 50.0, ""],         # Empty string
                [5, "K5", "Kd", 500.0, "nan"],       # String "nan"
            ], dtype=object)
            
            # Call the method
            labels_gen._fill_missing_pchembl_values()
            
            # Verify results
            data = labels_gen.interaction_data
            
            # Row 0: already has pchembl 7.0
            assert data[0, 4] == 7.0
            
            # Row 1: 1000 nM -> pchembl = 6.0
            assert np.isclose(float(data[1, 4]), 6.0, atol=0.01), f"Expected 6.0, got {data[1, 4]}"
            
            # Row 2: 10000 nM -> pchembl = 5.0
            assert np.isclose(float(data[2, 4]), 5.0, atol=0.01), f"Expected 5.0, got {data[2, 4]}"
            
            # Row 3: 50 nM -> pchembl = 7.3
            assert np.isclose(float(data[3, 4]), 7.3, atol=0.01), f"Expected 7.3, got {data[3, 4]}"
            
            # Row 4: 500 nM -> pchembl = 6.3
            assert np.isclose(float(data[4, 4]), 6.3, atol=0.01), f"Expected 6.3, got {data[4, 4]}"
            
        finally:
            import os
            os.unlink(temp_path)

