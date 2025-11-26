"""
Tier 3.1 Integration Validation Test

This test validates that the profiler, quantizer, and integration
work correctly together.
"""

import unittest
import tempfile
from pathlib import Path
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from src.classifier.core.embedding_profiler import EmbeddingProfiler, ProfileStats
from src.classifier.core.embedding_quantizer import EmbeddingQuantizer, QuantizationConfig
from src.classifier.core.embedding_integration import (
    OptimizedEmbeddingExtractor,
    EmbeddingOptimizationContext,
    create_optimized_extractor,
    ExtractionMetrics
)


class TestEmbeddingProfiler(unittest.TestCase):
    """Test profiler functionality."""
    
    def setUp(self):
        self.profiler = EmbeddingProfiler()
    
    def test_profiler_context_manager(self):
        """Test context manager for timing."""
        with self.profiler.context("test_component"):
            pass
        
        # Should have recorded timing
        self.assertIn("test_component", self.profiler.components)
    
    def test_profiler_start_end(self):
        """Test manual start/end."""
        self.profiler.start_component("component1")
        self.profiler.end_component("component1")
        
        self.assertIn("component1", self.profiler.components)
    
    def test_profiler_report(self):
        """Test report generation."""
        for _ in range(3):
            with self.profiler.context("test"):
                pass
        
        report = self.profiler.get_report()
        self.assertIsInstance(report, ProfileStats)
        self.assertEqual(report.count, 3)
        self.assertGreaterEqual(report.total_time, 0)


class TestEmbeddingQuantizer(unittest.TestCase):
    """Test quantizer functionality."""
    
    def setUp(self):
        self.embeddings = np.random.randn(100).astype(np.float32)
    
    def test_quantizer_creation_fp16(self):
        """Test FP16 quantizer creation."""
        config = QuantizationConfig(method="fp16")
        quantizer = EmbeddingQuantizer(config)
        
        self.assertIsNotNone(quantizer)
        self.assertEqual(quantizer.config.method, "fp16")
    
    def test_quantizer_creation_int8(self):
        """Test INT8 quantizer creation."""
        config = QuantizationConfig(method="int8")
        quantizer = EmbeddingQuantizer(config)
        
        self.assertIsNotNone(quantizer)
        self.assertEqual(quantizer.config.method, "int8")
    
    def test_fp16_quantization(self):
        """Test FP16 quantization."""
        config = QuantizationConfig(method="fp16")
        quantizer = EmbeddingQuantizer(config)
        
        quantized = quantizer.quantize_fp16(self.embeddings)
        self.assertIsInstance(quantized, np.ndarray)
        self.assertEqual(quantized.dtype, np.float16)
    
    def test_int8_quantization(self):
        """Test INT8 quantization and dequantization."""
        config = QuantizationConfig(method="int8", preserve_accuracy=False)
        quantizer = EmbeddingQuantizer(config)
        
        # First calibrate
        calibration_data = [
            np.random.randn(100).astype(np.float32) for _ in range(10)
        ]
        quantizer.calibrate_int8(calibration_data)
        
        # Then quantize
        quantized = quantizer.quantize_int8(self.embeddings)
        self.assertIsInstance(quantized, np.ndarray)
        self.assertEqual(quantized.dtype, np.int8)
        
        # Dequantize
        dequantized = quantizer.dequantize_int8(quantized)
        self.assertIsInstance(dequantized, np.ndarray)


class TestOptimizedEmbeddingExtractor(unittest.TestCase):
    """Test integration extractor."""
    
    def setUp(self):
        self.extractor = OptimizedEmbeddingExtractor(
            enable_profiling=True,
            enable_quantization=True,
            quantization_method="fp16"
        )
    
    def test_extractor_creation(self):
        """Test extractor initialization."""
        self.assertIsNotNone(self.extractor.profiler)
        self.assertIsNotNone(self.extractor.quantizer)
        self.assertTrue(self.extractor.enable_profiling)
        self.assertTrue(self.extractor.enable_quantization)
    
    def test_extractor_metrics(self):
        """Test metrics tracking."""
        metrics = ExtractionMetrics(
            total_time=0.05,
            components={"tokenization": 0.001, "forward": 0.048},
            memory_before=100,
            memory_after=150,
            memory_peak=200,
            sequence_length=50,
            embedding_size=1024,
            quantization_method="fp16",
            speedup=2.0,
            accuracy_preserved=True
        )
        
        self.assertEqual(metrics.total_time, 0.05)
        self.assertEqual(metrics.speedup, 2.0)
    
    def test_report_generation(self):
        """Test report generation."""
        # Add some dummy metrics
        for _ in range(3):
            metrics = ExtractionMetrics(
                total_time=0.05,
                components={"test": 0.05},
                memory_before=100,
                memory_after=100,
                memory_peak=100,
                sequence_length=50,
                embedding_size=1024,
                quantization_method="fp16",
                speedup=2.0,
                accuracy_preserved=True
            )
            self.extractor.extraction_history.append(metrics)
        
        report = self.extractor.get_report()
        self.assertEqual(report['extraction_count'], 3)
        self.assertIn('average_time', report)
        self.assertIn('components', report)
    
    def test_bottleneck_detection(self):
        """Test bottleneck identification."""
        metrics = ExtractionMetrics(
            total_time=0.05,
            components={
                "tokenization": 0.001,
                "forward": 0.048,  # Largest
                "quantization": 0.001
            },
            memory_before=100,
            memory_after=100,
            memory_peak=100,
            sequence_length=50,
            embedding_size=1024,
            quantization_method="fp16",
            speedup=2.0,
            accuracy_preserved=True
        )
        self.extractor.extraction_history.append(metrics)
        
        bottleneck, time_val = self.extractor.get_bottleneck()
        self.assertEqual(bottleneck, "forward")
        self.assertAlmostEqual(time_val, 0.048, places=3)
    
    def test_metrics_reset(self):
        """Test metrics reset."""
        metrics = ExtractionMetrics(
            total_time=0.05,
            components={"test": 0.05},
            memory_before=100,
            memory_after=100,
            memory_peak=100,
            sequence_length=50,
            embedding_size=1024,
            quantization_method="fp16",
            speedup=2.0,
            accuracy_preserved=True
        )
        self.extractor.extraction_history.append(metrics)
        
        self.assertEqual(len(self.extractor.extraction_history), 1)
        self.extractor.reset_metrics()
        self.assertEqual(len(self.extractor.extraction_history), 0)
    
    def test_save_report(self):
        """Test report saving."""
        metrics = ExtractionMetrics(
            total_time=0.05,
            components={"test": 0.05},
            memory_before=100,
            memory_after=100,
            memory_peak=100,
            sequence_length=50,
            embedding_size=1024,
            quantization_method="fp16",
            speedup=2.0,
            accuracy_preserved=True
        )
        self.extractor.extraction_history.append(metrics)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "report.json"
            self.extractor.save_report(output_file)
            
            self.assertTrue(output_file.exists())
            
            import json
            with open(output_file) as f:
                data = json.load(f)
            
            self.assertEqual(data['extraction_count'], 1)


class TestContextManager(unittest.TestCase):
    """Test context manager functionality."""
    
    def test_context_manager_creation(self):
        """Test context manager initialization."""
        with EmbeddingOptimizationContext() as extractor:
            self.assertIsNotNone(extractor)
            self.assertTrue(extractor.enable_profiling)
    
    def test_context_manager_with_quantization(self):
        """Test context manager with different quantization."""
        with EmbeddingOptimizationContext(quantization_method="int8") as extractor:
            self.assertEqual(extractor.quantization_method, "int8")


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions."""
    
    def test_create_optimized_extractor_fp16(self):
        """Test creating extractor with FP16."""
        extractor = create_optimized_extractor("fp16")
        
        self.assertIsNotNone(extractor)
        self.assertEqual(extractor.quantization_method, "fp16")
    
    def test_create_optimized_extractor_int8(self):
        """Test creating extractor with INT8."""
        extractor = create_optimized_extractor("int8")
        
        self.assertIsNotNone(extractor)
        self.assertEqual(extractor.quantization_method, "int8")


class TestIntegrationWorkflow(unittest.TestCase):
    """Test complete integration workflow."""
    
    def test_complete_extraction_workflow(self):
        """Test full extraction workflow."""
        extractor = create_optimized_extractor("fp16")
        
        # Simulate multiple extractions
        for i in range(3):
            metrics = ExtractionMetrics(
                total_time=0.05 - (i * 0.005),  # Improving
                components={
                    "tokenization": 0.001,
                    "forward": 0.048 - (i * 0.005),
                    "quantization": 0.001
                },
                memory_before=100,
                memory_after=100,
                memory_peak=100 + (i * 10),
                sequence_length=50,
                embedding_size=1024,
                quantization_method="fp16",
                speedup=2.0 + (i * 0.1),
                accuracy_preserved=True
            )
            extractor.extraction_history.append(metrics)
        
        # Get report
        report = extractor.get_report()
        self.assertEqual(report['extraction_count'], 3)
        self.assertGreater(report['average_speedup'], 2.0)
        
        # Get bottleneck
        bottleneck, time_val = extractor.get_bottleneck()
        self.assertEqual(bottleneck, "forward")


def run_validation_tests():
    """Run all validation tests."""
    print("\n" + "="*70)
    print("🧪 TIER 3.1 INTEGRATION VALIDATION TESTS")
    print("="*70 + "\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestEmbeddingProfiler))
    suite.addTests(loader.loadTestsFromTestCase(TestEmbeddingQuantizer))
    suite.addTests(loader.loadTestsFromTestCase(TestOptimizedEmbeddingExtractor))
    suite.addTests(loader.loadTestsFromTestCase(TestContextManager))
    suite.addTests(loader.loadTestsFromTestCase(TestConvenienceFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegrationWorkflow))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*70)
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_validation_tests()
    exit(0 if success else 1)
