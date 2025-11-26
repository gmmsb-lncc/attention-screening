"""
Tier 3.1 Integration Validation Test (Updated)

This test validates that the profiler, quantizer, and integration
work correctly together using the current API.
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


class TestEmbeddingProfilerAPI(unittest.TestCase):
    """Test profiler functionality with current API."""
    
    def setUp(self):
        self.profiler = EmbeddingProfiler()
    
    def test_profiler_context_manager(self):
        """Test context manager for timing."""
        with self.profiler.context("test_component"):
            pass
        
        # Current API stores in 'stats' dict
        self.assertIn("test_component", self.profiler.stats)
    
    def test_profiler_report_generation(self):
        """Test report generation returns string."""
        for _ in range(3):
            with self.profiler.context("test"):
                pass
        
        report = self.profiler.get_report()
        # get_report() returns a formatted string
        self.assertIsInstance(report, str)
        self.assertIn("test", report)


class TestEmbeddingQuantizerAPI(unittest.TestCase):
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
        """Test INT8 quantization returns tuple."""
        config = QuantizationConfig(method="int8", preserve_accuracy=False)
        quantizer = EmbeddingQuantizer(config)
        
        # Calibrate first
        calibration_data = [
            np.random.randn(100).astype(np.float32) for _ in range(10)
        ]
        quantizer.calibrate_int8(calibration_data)
        
        # quantize_int8 returns (quantized_array, scale)
        result = quantizer.quantize_int8(self.embeddings)
        self.assertIsInstance(result, tuple)
        quantized, scale = result
        self.assertIsInstance(quantized, np.ndarray)
        self.assertEqual(quantized.dtype, np.int8)
        self.assertIsInstance(scale, (float, np.floating))


class TestOptimizedExtractorAPI(unittest.TestCase):
    """Test the OptimizedEmbeddingExtractor API."""
    
    def setUp(self):
        self.config = QuantizationConfig(method="fp16")
        self.extractor = OptimizedEmbeddingExtractor(quantization_config=self.config)
    
    def test_extractor_creation(self):
        """Test extractor initialization."""
        self.assertIsNotNone(self.extractor)
    
    def test_metrics_tracking(self):
        """Test metrics tracking."""
        # Metrics should be tracked after operations
        self.assertIsNotNone(self.extractor.metrics)
    
    def test_metrics_reset(self):
        """Test metrics reset."""
        self.extractor.reset_metrics()
        # Should not raise exception
        self.assertIsNotNone(self.extractor.metrics)
    
    def test_report_generation(self):
        """Test report generation returns correct structure."""
        report = self.extractor.get_report()
        
        # Report should have expected keys
        self.assertIsInstance(report, dict)
        expected_keys = ['extraction_count', 'total_time', 'average_time']
        for key in expected_keys:
            self.assertIn(key, report)
    
    def test_save_report(self):
        """Test report saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "report.json"
            self.extractor.save_report(str(report_path))
            self.assertTrue(report_path.exists())


class TestContextManager(unittest.TestCase):
    """Test context manager functionality."""
    
    def test_context_manager_creation(self):
        """Test context manager initialization."""
        with EmbeddingOptimizationContext(quantization_method="fp16") as ctx:
            self.assertIsNotNone(ctx.extractor)
    
    def test_context_manager_with_quantization(self):
        """Test context manager with different quantization."""
        with EmbeddingOptimizationContext(quantization_method="int8") as ctx:
            self.assertEqual(
                ctx.extractor.quantizer.config.method,
                "int8"
            )


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience factory functions."""
    
    def test_create_optimized_extractor_fp16(self):
        """Test creating extractor with FP16."""
        extractor = create_optimized_extractor(quantization_method="fp16")
        self.assertIsNotNone(extractor)
        self.assertEqual(extractor.quantizer.config.method, "fp16")
    
    def test_create_optimized_extractor_int8(self):
        """Test creating extractor with INT8."""
        extractor = create_optimized_extractor(quantization_method="int8")
        self.assertIsNotNone(extractor)
        self.assertEqual(extractor.quantizer.config.method, "int8")


class TestIntegrationWorkflow(unittest.TestCase):
    """Test complete integration workflow."""
    
    def test_complete_extraction_workflow(self):
        """Test full extraction workflow."""
        with EmbeddingOptimizationContext(quantization_method="fp16") as ctx:
            # Should have extractor
            self.assertIsNotNone(ctx.extractor)
            
            # Should be able to get bottleneck
            bottleneck = ctx.extractor.get_bottleneck()
            # Bottleneck returns component name (could be various values)
            self.assertIsInstance(bottleneck, (str, type(None)))
            
            # Should be able to get report
            report = ctx.extractor.get_report()
            self.assertIsInstance(report, dict)


if __name__ == "__main__":
    unittest.main()
