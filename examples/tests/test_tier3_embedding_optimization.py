"""
Test Suite for Tier 3.1 Embedding Optimization Components

Tests profiling, quantization, and integration functionality.
"""

import unittest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import json

from src.classifier.core.embedding_profiler import EmbeddingProfiler, ProfileStats
from src.classifier.core.embedding_quantizer import EmbeddingQuantizer, QuantizationConfig
from src.classifier.core.embedding_integration import (
    OptimizedEmbeddingExtractor,
    EmbeddingOptimizationContext,
    create_optimized_extractor,
    ExtractionMetric,
    ExtractionMetrics
)


class TestEmbeddingProfiler(unittest.TestCase):
    """Tests for EmbeddingProfiler component."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.profiler = EmbeddingProfiler()
    
    def test_profiler_initialization(self):
        """Test profiler initializes correctly."""
        self.assertIsNotNone(self.profiler)
        self.assertEqual(len(self.profiler.stats), 0)
    
    def test_component_timing(self):
        """Test component timing measurement."""
        import time
        
        # Measure a simple operation using context manager
        with self.profiler.context("test_op"):
            time.sleep(0.01)  # Sleep 10ms
        
        # Check that timing was recorded
        self.assertIn("test_op", self.profiler.stats)
        stats = self.profiler.stats["test_op"]
        self.assertGreater(stats.total_time_ms, 0)
        self.assertEqual(stats.num_calls, 1)
    
    def test_multiple_component_calls(self):
        """Test multiple calls to same component."""
        import time
        
        for _ in range(3):
            with self.profiler.context("op"):
                time.sleep(0.001)
        
        stats = self.profiler.stats["op"]
        self.assertEqual(stats.num_calls, 3)
        self.assertGreater(stats.total_time_ms, 0)
    
    def test_get_report(self):
        """Test getting profiling report."""
        import time
        
        with self.profiler.context("test"):
            time.sleep(0.001)
        
        report = self.profiler.get_report()
        self.assertIsNotNone(report)
        self.assertIn("PROFILING REPORT", report)
        self.assertIn("test", report)
    
    def test_reset(self):
        """Test resetting profiler."""
        import time
        
        with self.profiler.context("test"):
            time.sleep(0.001)
        
        self.assertIn("test", self.profiler.stats)
        
        self.profiler.reset()
        self.assertEqual(len(self.profiler.stats), 0)


class TestEmbeddingQuantizer(unittest.TestCase):
    """Tests for EmbeddingQuantizer component."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_embedding = np.random.randn(1024).astype(np.float32)
        self.config = QuantizationConfig(method="fp16")
        self.quantizer = EmbeddingQuantizer(self.config)
    
    def test_quantizer_initialization(self):
        """Test quantizer initializes correctly."""
        self.assertIsNotNone(self.quantizer)
        self.assertEqual(self.quantizer.config.method, "fp16")
    
    def test_fp16_quantization(self):
        """Test FP16 quantization."""
        quantized = self.quantizer.quantize_fp16(self.test_embedding)
        
        # Check output type and shape
        self.assertEqual(quantized.dtype, np.float16)
        self.assertEqual(quantized.shape, self.test_embedding.shape)
        
        # Check memory reduction
        reduction_ratio = self.test_embedding.nbytes / quantized.nbytes
        self.assertAlmostEqual(reduction_ratio, 2.0, places=0)
    
    def test_fp16_dequantization(self):
        """Test FP16 dequantization."""
        quantized = self.quantizer.quantize_fp16(self.test_embedding)
        dequantized = quantized.astype(np.float32)
        
        # Check recovery
        self.assertEqual(dequantized.dtype, np.float32)
        self.assertEqual(dequantized.shape, self.test_embedding.shape)
    
    def test_int8_quantization_config(self):
        """Test INT8 quantization configuration."""
        config = QuantizationConfig(
            method="int8",
            preserve_accuracy=True,
            dynamic=True
        )
        quantizer = EmbeddingQuantizer(config)
        self.assertEqual(quantizer.config.method, "int8")
    
    def test_quantizer_with_batch(self):
        """Test quantizer with batch of embeddings."""
        batch = np.random.randn(10, 1024).astype(np.float32)
        quantized = self.quantizer.quantize_fp16(batch)
        
        self.assertEqual(quantized.shape, batch.shape)
        self.assertEqual(quantized.dtype, np.float16)
    
    def test_invalid_method(self):
        """Test handling of invalid quantization method."""
        config = QuantizationConfig(method="invalid")
        
        # Should raise ValueError for invalid method
        with self.assertRaises(ValueError):
            quantizer = EmbeddingQuantizer(config)


class TestExtractionMetrics(unittest.TestCase):
    """Tests for ExtractionMetrics dataclass."""
    
    def test_metrics_creation(self):
        """Test creating ExtractionMetrics."""
        metric = ExtractionMetrics(
            total_time=0.5,
            components={"forward": 0.3, "extract": 0.2},
            memory_before=1024,
            memory_after=1024,
            memory_peak=2048,
            sequence_length=100,
            embedding_size=1024,
            quantization_method="fp16",
            speedup=1.5,
            accuracy_preserved=True
        )
        
        self.assertEqual(metric.total_time, 0.5)
        self.assertEqual(metric.sequence_length, 100)
        self.assertEqual(metric.quantization_method, "fp16")
    
    def test_metrics_serialization(self):
        """Test metrics can be serialized."""
        from dataclasses import asdict
        
        metric = ExtractionMetrics(
            total_time=0.5,
            components={"forward": 0.3},
            memory_before=1024,
            memory_after=1024,
            memory_peak=2048,
            sequence_length=100,
            embedding_size=1024,
            quantization_method="fp16",
            speedup=1.5,
            accuracy_preserved=True
        )
        
        metric_dict = asdict(metric)
        self.assertIsInstance(metric_dict, dict)
        self.assertEqual(metric_dict["total_time"], 0.5)


class TestOptimizedEmbeddingExtractor(unittest.TestCase):
    """Tests for OptimizedEmbeddingExtractor integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = OptimizedEmbeddingExtractor(
            enable_profiling=True,
            enable_quantization=True,
            quantization_method="fp16"
        )
    
    def test_extractor_initialization(self):
        """Test extractor initializes correctly."""
        self.assertIsNotNone(self.extractor)
        self.assertTrue(self.extractor.enable_profiling)
        self.assertTrue(self.extractor.enable_quantization)
        self.assertEqual(self.extractor.quantization_method, "fp16")
    
    def test_profiling_disabled_initialization(self):
        """Test extractor with profiling disabled."""
        extractor = OptimizedEmbeddingExtractor(
            enable_profiling=False,
            enable_quantization=True
        )
        
        self.assertFalse(extractor.enable_profiling)
        self.assertIsNone(extractor.profiler)
    
    def test_quantization_disabled_initialization(self):
        """Test extractor with quantization disabled."""
        extractor = OptimizedEmbeddingExtractor(
            enable_profiling=True,
            enable_quantization=False
        )
        
        self.assertFalse(extractor.enable_quantization)
        self.assertIsNone(extractor.quantizer)
    
    def test_metrics_tracking(self):
        """Test that metrics are tracked after extraction."""
        # Mock the extraction
        with patch.object(self.extractor, '_tokenize', return_value=None):
            with patch.object(self.extractor, '_forward', return_value=np.zeros(1024)):
                self.extractor.extract(
                    "SEQUENCE",
                    Mock(),
                    Mock(),
                    Mock()
                )
        
        self.assertEqual(len(self.extractor.extraction_history), 1)
        metric = self.extractor.extraction_history[0]
        self.assertIsInstance(metric, ExtractionMetrics)
    
    def test_get_report_empty(self):
        """Test report generation with no extractions."""
        report = self.extractor.get_report()
        self.assertEqual(report["status"], "No extractions yet")
    
    def test_get_report_with_data(self):
        """Test report generation with extraction data."""
        # Add test metrics
        with patch.object(self.extractor, '_tokenize', return_value=None):
            with patch.object(self.extractor, '_forward', return_value=np.zeros(1024)):
                self.extractor.extract("SEQ1", Mock(), Mock(), Mock())
                self.extractor.extract("SEQ2", Mock(), Mock(), Mock())
        
        report = self.extractor.get_report()
        self.assertEqual(report["extraction_count"], 2)
        self.assertIn("total_time", report)
        self.assertIn("average_time", report)
        self.assertIn("average_speedup", report)
    
    def test_bottleneck_identification(self):
        """Test bottleneck identification."""
        # Add test metrics
        with patch.object(self.extractor, '_tokenize', return_value=None):
            with patch.object(self.extractor, '_forward', return_value=np.zeros(1024)):
                self.extractor.extract("SEQ1", Mock(), Mock(), Mock())
        
        bottleneck, time_val = self.extractor.get_bottleneck()
        self.assertIsNotNone(bottleneck)
        self.assertGreaterEqual(time_val, 0)
    
    def test_reset_metrics(self):
        """Test resetting metrics."""
        # Add test metrics
        with patch.object(self.extractor, '_tokenize', return_value=None):
            with patch.object(self.extractor, '_forward', return_value=np.zeros(1024)):
                self.extractor.extract("SEQ1", Mock(), Mock(), Mock())
        
        self.assertEqual(len(self.extractor.extraction_history), 1)
        
        # Reset
        self.extractor.reset_metrics()
        self.assertEqual(len(self.extractor.extraction_history), 0)
    
    def test_save_report(self):
        """Test saving report to file."""
        # Add test metrics
        with patch.object(self.extractor, '_tokenize', return_value=None):
            with patch.object(self.extractor, '_forward', return_value=np.zeros(1024)):
                self.extractor.extract("SEQ1", Mock(), Mock(), Mock())
        
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "report.json"
            self.extractor.save_report(report_path)
            
            self.assertTrue(report_path.exists())
            
            # Verify JSON content
            with open(report_path) as f:
                data = json.load(f)
            
            self.assertIn("extraction_count", data)


class TestEmbeddingOptimizationContext(unittest.TestCase):
    """Tests for EmbeddingOptimizationContext."""
    
    def test_context_manager_basic(self):
        """Test context manager basic functionality."""
        with EmbeddingOptimizationContext() as optimizer:
            self.assertIsInstance(optimizer, OptimizedEmbeddingExtractor)
    
    def test_context_manager_with_method(self):
        """Test context manager with specific quantization method."""
        with EmbeddingOptimizationContext(quantization_method="int8") as optimizer:
            self.assertEqual(optimizer.quantization_method, "int8")


class TestUtilityFunctions(unittest.TestCase):
    """Tests for utility functions."""
    
    def test_create_optimized_extractor(self):
        """Test create_optimized_extractor function."""
        extractor = create_optimized_extractor("fp16")
        
        self.assertIsInstance(extractor, OptimizedEmbeddingExtractor)
        self.assertTrue(extractor.enable_profiling)
        self.assertTrue(extractor.enable_quantization)
        self.assertEqual(extractor.quantization_method, "fp16")
    
    def test_create_optimized_extractor_int8(self):
        """Test create_optimized_extractor with INT8."""
        extractor = create_optimized_extractor("int8")
        self.assertEqual(extractor.quantization_method, "int8")


class TestIntegrationWorkflow(unittest.TestCase):
    """Integration tests for complete workflows."""
    
    def test_full_extraction_workflow(self):
        """Test complete extraction workflow."""
        extractor = OptimizedEmbeddingExtractor(
            enable_profiling=True,
            enable_quantization=True
        )
        
        # Mock model and alphabet
        model = Mock()
        alphabet = Mock()
        device = Mock()
        
        # Mock the extraction pipeline
        with patch.object(extractor, '_tokenize', return_value=None):
            with patch.object(extractor, '_forward', return_value=np.zeros(1024)):
                embeddings = extractor.extract("TEST", model, alphabet, device)
        
        # Verify results
        self.assertIsInstance(embeddings, np.ndarray)
        self.assertEqual(len(extractor.extraction_history), 1)
    
    def test_batch_extraction_workflow(self):
        """Test batch extraction workflow."""
        extractor = OptimizedEmbeddingExtractor()
        
        sequences = ["SEQ1", "SEQ2", "SEQ3"]
        
        with patch.object(extractor, '_tokenize', return_value=None):
            with patch.object(extractor, '_forward', return_value=np.zeros(1024)):
                for seq in sequences:
                    extractor.extract(seq, Mock(), Mock(), Mock())
        
        report = extractor.get_report()
        self.assertEqual(report["extraction_count"], len(sequences))
    
    def test_performance_tracking_workflow(self):
        """Test performance tracking through multiple extractions."""
        extractor = OptimizedEmbeddingExtractor()
        
        with patch.object(extractor, '_tokenize', return_value=None):
            with patch.object(extractor, '_forward', return_value=np.zeros(1024)):
                # Multiple extractions
                for i in range(5):
                    extractor.extract(f"SEQ{i}", Mock(), Mock(), Mock())
        
        # Verify metrics aggregation
        report = extractor.get_report()
        self.assertEqual(report["extraction_count"], 5)
        self.assertGreater(report["average_time"], 0)


# ==============================================================================
# Test Runners
# ==============================================================================

def run_profiler_tests():
    """Run all profiler tests."""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEmbeddingProfiler)
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


def run_quantizer_tests():
    """Run all quantizer tests."""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEmbeddingQuantizer)
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


def run_integration_tests():
    """Run all integration tests."""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestOptimizedEmbeddingExtractor)
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


def run_all_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestEmbeddingProfiler))
    suite.addTests(loader.loadTestsFromTestCase(TestEmbeddingQuantizer))
    suite.addTests(loader.loadTestsFromTestCase(TestExtractionMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestOptimizedEmbeddingExtractor))
    suite.addTests(loader.loadTestsFromTestCase(TestEmbeddingOptimizationContext))
    suite.addTests(loader.loadTestsFromTestCase(TestUtilityFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegrationWorkflow))
    
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("TIER 3.1 EMBEDDING OPTIMIZATION - TEST SUITE")
    print("="*70 + "\n")
    
    result = run_all_tests()
    
    print("\n" + "="*70)
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*70)
