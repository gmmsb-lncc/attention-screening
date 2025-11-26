"""
TIER 3.1 QUICK REFERENCE - Developer Cheat Sheet

Print this or bookmark it for fast access to integration info.
"""

QUICK_START = """
╔═══════════════════════════════════════════════════════════════════════════╗
║                     TIER 3.1 QUICK REFERENCE CARD                         ║
╚═══════════════════════════════════════════════════════════════════════════╝

1️⃣  BASIC IMPORT & USAGE
─────────────────────────────────────────────────────────────────────────────

from src.classifier.core.embedding_integration import create_optimized_extractor

extractor = create_optimized_extractor("fp16")  # or "int8"
embeddings = extractor.extract(sequence, model, alphabet, device, batch_converter)

report = extractor.get_report()
print(f"Speedup: {report['average_speedup']:.2f}x")


2️⃣  CONTEXT MANAGER (AUTO-REPORTING)
─────────────────────────────────────────────────────────────────────────────

from src.classifier.core.embedding_integration import EmbeddingOptimizationContext

with EmbeddingOptimizationContext(quantization_method="fp16") as opt:
    embeddings = opt.extract(sequence, model, alphabet, device)
    # Auto-report on exit


3️⃣  IDENTIFY BOTTLENECK
─────────────────────────────────────────────────────────────────────────────

component, time = extractor.get_bottleneck()
print(f"Bottleneck: {component} ({time:.3f}s)")

# If still high after Tier 3.1:
# - model_forward → Use Tier 3.3 (GPU optimization)
# - tokenization → Use Tier 3.2 (batch processing)
# - I/O → Use Tier 3.4 (advanced caching)


4️⃣  SAVE PROFILING REPORT
─────────────────────────────────────────────────────────────────────────────

extractor.save_report(Path("logs/embedding_profile.json"))

# Report includes:
# - extraction_count, total_time, average_time
# - Component breakdown (tokenization, forward, quantization)
# - Performance metrics (speedup, accuracy)


5️⃣  QUANTIZATION COMPARISON
─────────────────────────────────────────────────────────────────────────────

        Speedup         Memory      Accuracy    When to Use
FP16    2.0-2.5x        50%         99.9%       Balanced (DEFAULT)
INT8    4.0-5.0x        25%         99.0%       Max throughput
NONE    1.0x            100%        100%        Baseline/Testing


6️⃣  PERFORMANCE TARGETS
─────────────────────────────────────────────────────────────────────────────

Per-sequence:
  Baseline: 50-100ms
  FP16:     25-50ms (2x speedup)
  INT8:     12-25ms (4x speedup)

For 100 proteins:
  Baseline: 5-10s
  FP16:     2.5-5s
  INT8:     1.2-2.5s


7️⃣  INTEGRATION INTO protein_embedding.py
─────────────────────────────────────────────────────────────────────────────

# 1. Add import
from src.classifier.core.embedding_integration import OptimizedEmbeddingExtractor

# 2. Initialize in __init__
self.optimizer = OptimizedEmbeddingExtractor(
    enable_profiling=True,
    enable_quantization=True,
    quantization_method="fp16"
)

# 3. Wrap extraction in _generate_single_embedding
return self.optimizer.extract(
    sequence=sequence,
    model=self.model,
    alphabet=self.alphabet,
    device=self.device,
    batch_converter=self.batch_converter
)

# 4. Log report after batch
report = self.optimizer.get_report()
self.logger.info(f"Speedup: {report['average_speedup']:.2f}x")


8️⃣  TROUBLESHOOTING
─────────────────────────────────────────────────────────────────────────────

Q: Quantization not activating?
A: Check if quantizer is None
   if extractor.quantizer is None:
       print("Quantizer initialization failed")

Q: Slower than expected?
A: Get component breakdown
   report = extractor.get_report()
   for comp, stats in report['components'].items():
       print(f"{comp}: {stats['avg']*1000:.1f}ms")

Q: Memory issues with INT8?
A: Use FP16 instead or increase calibration samples
   extractor = OptimizedEmbeddingExtractor(
       quantization_method="int8",
       calibration_samples=200
   )


9️⃣  SPEEDUP ROADMAP
─────────────────────────────────────────────────────────────────────────────

Current (Tier 1+2):    8-10x
After 3.1 (FP16):      12-18x (3-5x additional)
After 3.2 (Batching):  60-150x (10-50x additional)
After 3.3 (GPU):       90-250x (1.5-2x additional)
After 3.4 (I/O):       100-300x (20-30% additional)


🔟  FILE LOCATIONS
─────────────────────────────────────────────────────────────────────────────

Main module:    src/classifier/core/embedding_integration.py
Profiler:       src/classifier/core/embedding_profiler.py
Quantizer:      src/classifier/core/embedding_quantizer.py
Examples:       examples/tier_3_1_integration_example.py
Tests:          tests/test_tier_3_1_integration.py
Guide:          docs/03-architecture/TIER_3_1_INTEGRATION_GUIDE.md


1️⃣1️⃣  KEY CLASSES
─────────────────────────────────────────────────────────────────────────────

OptimizedEmbeddingExtractor
  Methods:
    extract() - Perform optimized extraction with profiling
    get_report() - Get profiling report
    get_bottleneck() - Identify main bottleneck
    save_report(path) - Export report to JSON
    reset_metrics() - Clear collected data

EmbeddingOptimizationContext
  Usage:
    with EmbeddingOptimizationContext() as opt:
        opt.extract(...)
    # Auto-report on exit

ExtractionMetrics (dataclass)
  Tracks:
    total_time, components, memory_before/after/peak,
    sequence_length, embedding_size, quantization_method,
    speedup, accuracy_preserved


1️⃣2️⃣  CONFIGURATION
─────────────────────────────────────────────────────────────────────────────

opt = OptimizedEmbeddingExtractor(
    enable_profiling=True,           # Enable timing
    enable_quantization=True,        # Enable FP16/INT8
    quantization_method="fp16",      # "fp16", "int8", "auto"
    calibration_samples=100,         # For INT8 calibration
    logger=logging.getLogger()       # Optional logger
)


🎯  NEXT STEPS AFTER INTEGRATION
─────────────────────────────────────────────────────────────────────────────

1. Integrate into protein_embedding.py
2. Run profiling on real data
3. Validate 3-5x speedup
4. Measure memory savings
5. Check accuracy (target: >99%)
6. Save profiling reports
7. If speedup <5x, proceed to Tier 3.2 (batching)


💡  BEST PRACTICES
─────────────────────────────────────────────────────────────────────────────

✓ Start with FP16 (best balance)
✓ Use INT8 for max throughput
✓ Calibrate INT8 with diverse data (100+ samples)
✓ Save reports for trending
✓ Monitor bottleneck changes
✓ Test with real data before production
✓ Reset metrics between runs
✓ Use context manager for automatic reporting


📚  USEFUL COMMANDS
─────────────────────────────────────────────────────────────────────────────

# Run integration tests
python -m pytest tests/test_tier_3_1_integration.py -v

# Run examples
python examples/tier_3_1_integration_example.py

# View integration guide
cat docs/03-architecture/TIER_3_1_INTEGRATION_GUIDE.md

# Check compilation
python -m py_compile src/classifier/core/embedding_integration.py


═══════════════════════════════════════════════════════════════════════════════
For detailed information, see: docs/03-architecture/TIER_3_1_INTEGRATION_GUIDE.md
For examples, see: examples/tier_3_1_integration_example.py
═══════════════════════════════════════════════════════════════════════════════
"""

if __name__ == "__main__":
    print(QUICK_START)
