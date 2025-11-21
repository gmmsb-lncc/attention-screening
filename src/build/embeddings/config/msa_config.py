"""
MSA Configuration for Protein Structure/Embedding Models

This module provides MSA (Multiple Sequence Alignment) configuration settings
optimized for large-scale protein embedding extraction using the ColabFold MSA server.

ARCHITECTURE OVERVIEW:
=====================
This configuration system is designed to be model-agnostic and extensible.
Currently supports OpenFold3, with planned support for:
- Boltz-2 (biomolecular structure prediction)
- ESM-3 (evolutionary scale modeling)
- AlphaFold3 (structure prediction)

MSA STRATEGY FOR EMBEDDING EXTRACTION:
======================================
For DockTKinase with 700+ protein sequences, Main MSA mode is recommended:

1. Main MSA (RECOMMENDED):
   - Processes all unique sequences in a single batch
   - Uses ColabFold server (UniRef90 + BFD + MGnify + MetaEuk)
   - Time: ~3-5 minutes for 700 sequences
   - Best for: Embedding extraction, independent proteins

2. Paired MSA (NOT RECOMMENDED for 700+ seqs):
   - Processes each complex separately (1 query per complex)
   - Used for co-evolution analysis
   - Time: ~150s × number of complexes
   - Best for: Protein-protein interaction studies

FUTURE EXTENSIBILITY (Boltz-2):
===============================
Boltz-2 integration notes:
- Boltz-2 uses similar MSA input format to OpenFold3
- Both support ColabFold MSA server
- Key difference: Boltz-2 may require different template handling
- This config class can be reused with minor adaptations:
  * Add Boltz-2 specific modes if needed
  * Extend MsaMode enum for Boltz-2 requirements
  * Override colabfold_settings property in subclass if needed

DESIGN PATTERNS:
===============
- Dependency Injection: MsaConfig injected into strategy classes
- Factory Pattern: Class methods (for_production, for_development, etc.)
- Strategy Pattern: Different MSA modes for different use cases
- Template Method: colabfold_settings provides base implementation

Author: DockTKinase Team
Date: 2025-11-20
License: Apache 2.0
"""

from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional
import tempfile


# =============================================================================
# ENUMS
# =============================================================================

class MsaMode(str, Enum):
    """MSA computation modes for protein structure/embedding models.
    
    These modes are compatible with:
    - OpenFold3: Full support for all modes
    - Boltz-2: Compatible (planned) - same ColabFold infrastructure
    - ESM-3: May use NONE mode (sequence-only language model)
    
    Mode Selection Guide:
    ====================
    MAIN_STANDARD: Production embeddings (recommended for 700+ sequences)
    MAIN_FAST: Development/testing (2x faster, slightly lower quality)
    MAIN_HIGH_QUALITY: Research (maximum evolutionary information)
    PAIRED: Protein complexes (NOT for large-scale embedding extraction)
    NONE: Sequence-only (fastest, no MSA overhead)
    
    ColabFold Server Details:
    ========================
    - Main modes: Single query for all unique sequences
    - Paired mode: Separate query per complex (slow for many sequences)
    - Deduplication: Automatically handled by ColabFold
    - Rate limiting: Server may throttle excessive requests
    """
    
    # Main MSA modes (recommended for embedding extraction)
    # -------------------------------------------------------
    # These modes query the ColabFold server once for all sequences,
    # making them ideal for large-scale batch processing (700+ sequences).
    
    MAIN_STANDARD = "main_standard"        # Standard: UniRef90 + env DBs + filter
    MAIN_FAST = "main_fast"                # Fast: UniRef90 only (50% time)
    MAIN_HIGH_QUALITY = "main_high_quality"  # Research: All DBs, no filter (larger MSAs)
    
    # Paired MSA mode (for protein-protein complexes)
    # ------------------------------------------------
    # WARNING: Creates 1 query per complex - NOT scalable for 700+ sequences
    # Use only when studying co-evolution or protein-protein interactions
    
    PAIRED = "paired"                      # Paired: Co-evolution analysis (SLOW)
    
    # No MSA mode (sequence-only)
    # ----------------------------
    # Skips MSA computation entirely. Useful for:
    # - Quick prototyping
    # - Language models (ESM-3) that don't need MSAs
    # - When MSA enrichment is not beneficial
    
    NONE = "none"                          # No MSA: Instant, sequence-only
    
    def __str__(self) -> str:
        return self.value


class MsaFileFormat(str, Enum):
    """Output format for MSA files."""
    
    NPZ = "npz"  # Pre-parsed numpy format (faster loading)
    A3M = "a3m"  # Raw a3m format (human-readable)
    
    def __str__(self) -> str:
        return self.value


# =============================================================================
# MSA CONFIGURATION DATACLASS
# =============================================================================

@dataclass
class MsaConfig:
    """
    Configuration for MSA computation using ColabFold server.
    
    DESIGN PHILOSOPHY:
    ==================
    This configuration class is designed to be:
    1. Model-agnostic: Works with OpenFold3, Boltz-2, and other structure models
    2. Extensible: Easy to add new modes or parameters
    3. Production-ready: Sensible defaults for large-scale processing
    4. Cache-friendly: Supports MSA reuse across multiple runs
    
    CURRENT SUPPORT:
    ===============
    - OpenFold3: Full integration (openfold_strategy.py)
    - Boltz-2: Planned integration (will use same ColabFold infrastructure)
    - ESM-3: Partial (may use NONE mode for sequence-only)
    
    OPTIMIZATION FOR DOCKTKINASE (700+ sequences):
    ==============================================
    This configuration is specifically tuned for large-scale embedding extraction:
    - Main MSA mode processes all sequences in one batch
    - Automatic deduplication reduces redundant computations
    - Caching prevents re-computing identical MSAs
    - NPZ format optimized for fast loading in Python/NumPy
    
    INTEGRATION POINTS FOR NEW MODELS (e.g., Boltz-2):
    ==================================================
    To integrate a new model:
    
    1. Reuse this config class as-is (most cases)
    2. OR extend with model-specific subclass:
       class Boltz2MsaConfig(MsaConfig):
           # Add Boltz-2 specific parameters
           boltz2_specific_param: bool = True
    
    3. Override colabfold_settings if needed:
       @property
       def colabfold_settings(self):
           settings = super().colabfold_settings
           settings['boltz2_option'] = self.boltz2_specific_param
           return settings
    
    4. Use in strategy class:
       class Boltz2Strategy(BaseProteinStrategy):
           def __init__(self, msa_config: MsaConfig = None):
               self.msa_config = msa_config or MsaConfig.for_production()
    
    Attributes:
        # CORE MSA SETTINGS
        mode: MSA computation mode (see MsaMode enum)
        file_format: Output format - NPZ (fast) or A3M (human-readable)
        
        # COLABFOLD SERVER SETTINGS
        use_env: Whether to use environmental databases (BFD, MGnify, MetaEuk, SMAG)
                 True = slower but higher quality MSAs
                 False = faster, UniRef90 only
        use_filter: Whether to apply diversity filter to MSAs
                    True = smaller, cleaner MSAs
                    False = larger, more diverse MSAs
        use_templates: Whether to fetch template structures from PDB70
                       False = recommended for embedding extraction (faster)
                       True = needed for structure prediction (slower)
        
        # SERVER CONNECTION
        server_url: ColabFold MSA server URL (default: https://api.colabfold.com)
        user_agent: User agent for API requests (REQUIRED for API compliance)
                    Format: 'toolname/version contact@email'
        
        # STORAGE AND CACHING
        output_directory: Directory to save MSA files
                         Organized as: output_dir/main/<seq_hash>/colabfold_main.a3m
        cleanup_after_use: Whether to cleanup raw MSA files after processing
                          True = save disk space
                          False = keep for inspection/debugging
        enable_caching: Whether to cache MSA results for reuse
                       True = reuse MSAs across runs (RECOMMENDED for 700+ seqs)
                       False = always recompute
        
        # PERFORMANCE TUNING
        chunk_size: Number of sequences to process per batch
                   None = process all at once (recommended for < 1000 sequences)
                   int = split into chunks (for very large datasets)
        timeout_seconds: Request timeout in seconds (increase for large batches)
        max_retries: Maximum number of retry attempts (handles transient failures)
        
    Examples:
        >>> # Standard production mode
        >>> config = MsaConfig.for_production()
        >>> 
        >>> # Fast development mode
        >>> config = MsaConfig.for_development()
        >>> 
        >>> # High quality research mode
        >>> config = MsaConfig.for_research()
    """
    
    # MSA computation settings
    mode: MsaMode = MsaMode.MAIN_STANDARD
    file_format: MsaFileFormat = MsaFileFormat.NPZ
    
    # ColabFold server settings
    use_env: bool = True
    use_filter: bool = True
    use_templates: bool = False
    
    # Server configuration
    server_url: str = "https://api.colabfold.com"
    user_agent: str = "docktkinase/1.0"
    
    # Output settings
    output_directory: Path = field(default_factory=lambda: Path(tempfile.gettempdir()) / "docktkinase_msa")
    cleanup_after_use: bool = True
    enable_caching: bool = True
    
    # Performance settings
    chunk_size: Optional[int] = None  # None = process all at once
    timeout_seconds: int = 300
    max_retries: int = 5
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Convert string paths to Path objects
        if isinstance(self.output_directory, str):
            self.output_directory = Path(self.output_directory)
        
        # Create output directory if it doesn't exist
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
        # Validate mode-specific settings
        if self.mode == MsaMode.PAIRED:
            # Paired mode doesn't use templates
            self.use_templates = False
        
        # Ensure user agent is set for API compliance
        if not self.user_agent:
            raise ValueError(
                "user_agent must be set for ColabFold API compliance. "
                "Format: 'toolname/version contact@email'"
            )
    
    @property
    def colabfold_settings(self):
        """
        Convert to ColabFold-compatible settings dictionary.
        
        ARCHITECTURE NOTE:
        ==================
        This property acts as an adapter pattern, translating our internal
        configuration to ColabFold's expected format. This abstraction allows:
        
        1. Model Independence: Different models (OpenFold3, Boltz-2) can use
           the same MsaConfig but get model-specific ColabFold settings
        
        2. Version Compatibility: Easy to update for new ColabFold API versions
           without changing the rest of the codebase
        
        3. Extensibility: Subclasses can override this method to add
           model-specific parameters:
           
           class Boltz2MsaConfig(MsaConfig):
               @property
               def colabfold_settings(self):
                   settings = super().colabfold_settings
                   settings['boltz2_specific'] = True
                   return settings
        
        COLABFOLD API PARAMETERS:
        =========================
        - use_pairing: bool - Enable paired MSA mode (slow)
        - use_templates: bool - Fetch template structures
        - use_env: bool - Use environmental databases
        - use_filter: bool - Apply diversity filter
        - pairing_strategy: str - 'greedy' or 'complete' (for paired mode)
        - user_agent: str - API user identification (required)
        - host_url: str - Server endpoint
        
        MAPPING STRATEGY:
        ================
        - MAIN_STANDARD → use_env=True, use_filter=True
        - MAIN_FAST → use_env=False (UniRef90 only)
        - MAIN_HIGH_QUALITY → use_env=True, use_filter=False (larger MSAs)
        - PAIRED → use_pairing=True, pairing_strategy='greedy'
        - NONE → N/A (no ColabFold call)
        
        Returns:
            Dictionary with settings for ColabFold MSA server
        
        Example:
            >>> config = MsaConfig.for_production()
            >>> settings = config.colabfold_settings
            >>> # Use in OpenFold3 or Boltz-2 MSA pipeline
            >>> msas = query_colabfold_msa_server(sequences, **settings)
        """
        # Base settings
        settings = {
            "use_pairing": self.mode == MsaMode.PAIRED,
            "use_templates": self.use_templates,
            "user_agent": self.user_agent,
            "host_url": self.server_url,
        }
        
        # Mode-specific settings
        if self.mode == MsaMode.MAIN_FAST:
            settings.update({
                "use_env": False,  # Only UniRef90
                "use_filter": True,
            })
        elif self.mode == MsaMode.MAIN_HIGH_QUALITY:
            settings.update({
                "use_env": True,
                "use_filter": False,  # No diversity filter
            })
        elif self.mode == MsaMode.MAIN_STANDARD:
            settings.update({
                "use_env": self.use_env,
                "use_filter": self.use_filter,
            })
        elif self.mode == MsaMode.PAIRED:
            settings.update({
                "use_env": self.use_env,
                "pairing_strategy": "greedy",  # More efficient than "complete"
            })
        
        return settings
    
    @classmethod
    def for_production(cls, output_dir: Optional[Path] = None) -> "MsaConfig":
        """
        Create configuration optimized for production embedding extraction.
        
        WHEN TO USE:
        ============
        - DockTKinase with 700+ protein sequences (RECOMMENDED)
        - Production pipeline runs
        - When quality and speed balance is important
        - Any model requiring MSAs (OpenFold3, Boltz-2)
        
        CHARACTERISTICS:
        ===============
        - Main MSA: Single batch for all sequences (scalable)
        - Environmental databases: UniRef90 + BFD + MGnify + MetaEuk (high quality)
        - Diversity filter: Enabled (cleaner MSAs, better for ML)
        - File format: NPZ (fast NumPy loading, 2-3x faster than A3M)
        - Caching: Enabled (reuse MSAs across runs, saves 90% time on reruns)
        - Cleanup: Enabled (saves disk space, ~500MB for 700 sequences)
        
        PERFORMANCE:
        ===========
        - Time (700 unique sequences): 3-5 minutes first run, < 1 min cached
        - Memory: ~1-2 GB during processing
        - Disk (with cleanup): ~200 MB (NPZ files only)
        - Disk (without cleanup): ~700 MB (includes raw A3M files)
        
        ARCHITECTURE:
        ============
        This factory method implements the Factory Pattern, providing a
        pre-configured instance with production-optimal settings. This ensures:
        
        1. Consistency: All production runs use same settings
        2. Maintainability: Update defaults in one place
        3. Simplicity: Users don't need to know optimal parameters
        
        EXTENSIBILITY (Boltz-2):
        =======================
        This method can be reused for Boltz-2 without modification:
        
        >>> # OpenFold3
        >>> openfold_strategy = OpenFoldStrategy(
        ...     msa_config=MsaConfig.for_production()
        ... )
        >>> 
        >>> # Boltz-2 (future)
        >>> boltz2_strategy = Boltz2Strategy(
        ...     msa_config=MsaConfig.for_production()
        ... )
        
        Args:
            output_dir: Optional custom output directory.
                       Default: {tempdir}/docktkinase_msa
                       Tip: Use persistent directory for caching across sessions
        
        Returns:
            MsaConfig instance optimized for production
        
        Example:
            >>> # Default (temp directory)
            >>> config = MsaConfig.for_production()
            >>> 
            >>> # Custom persistent cache
            >>> config = MsaConfig.for_production(
            ...     output_dir=Path("./persistent_msa_cache")
            ... )
            >>> # First run: 3-5 min, subsequent: < 1 min
        """
        return cls(
            mode=MsaMode.MAIN_STANDARD,
            file_format=MsaFileFormat.NPZ,
            use_env=True,
            use_filter=True,
            use_templates=False,
            cleanup_after_use=True,
            enable_caching=True,
            output_directory=output_dir or Path(tempfile.gettempdir()) / "docktkinase_msa",
            user_agent="docktkinase/1.0 production",
        )
    
    @classmethod
    def for_development(cls, output_dir: Optional[Path] = None) -> "MsaConfig":
        """
        Create configuration optimized for fast development/testing.
        
        Useful for prototyping and quick iterations.
        
        Characteristics:
        - Fast mode (UniRef90 only)
        - Diversity filter enabled
        - NPZ format
        - Caching enabled
        - ~50% faster than production mode
        
        Args:
            output_dir: Optional custom output directory
        
        Returns:
            MsaConfig instance optimized for development
        
        Example:
            >>> config = MsaConfig.for_development()
            >>> # Process 700 sequences in ~1-2 minutes
        """
        return cls(
            mode=MsaMode.MAIN_FAST,
            file_format=MsaFileFormat.NPZ,
            use_env=False,  # Fast mode
            use_filter=True,
            use_templates=False,
            cleanup_after_use=True,
            enable_caching=True,
            output_directory=output_dir or Path(tempfile.gettempdir()) / "docktkinase_msa_dev",
            user_agent="docktkinase/1.0 development",
        )
    
    @classmethod
    def for_research(cls, output_dir: Optional[Path] = None) -> "MsaConfig":
        """
        Create configuration optimized for research (maximum quality).
        
        Best for detailed evolutionary analysis or when quality > speed.
        
        Characteristics:
        - High quality mode (no diversity filter)
        - All environmental databases
        - A3M format (human-readable)
        - Caching enabled
        - Larger MSAs, more processing time
        
        Args:
            output_dir: Optional custom output directory
        
        Returns:
            MsaConfig instance optimized for research
        
        Example:
            >>> config = MsaConfig.for_research()
            >>> # Process 700 sequences in ~5-10 minutes
        """
        return cls(
            mode=MsaMode.MAIN_HIGH_QUALITY,
            file_format=MsaFileFormat.A3M,  # Human-readable
            use_env=True,
            use_filter=False,  # Maximum diversity
            use_templates=False,
            cleanup_after_use=False,  # Keep raw files for inspection
            enable_caching=True,
            output_directory=output_dir or Path(tempfile.gettempdir()) / "docktkinase_msa_research",
            user_agent="docktkinase/1.0 research",
        )
    
    @classmethod
    def no_msa(cls) -> "MsaConfig":
        """
        Create configuration for no MSA computation.
        
        Use only raw sequence without MSA enrichment.
        Fastest option but lower quality embeddings.
        
        Returns:
            MsaConfig instance with MSA disabled
        
        Example:
            >>> config = MsaConfig.no_msa()
            >>> # Instant processing, sequence-only embeddings
        """
        return cls(
            mode=MsaMode.NONE,
            file_format=MsaFileFormat.NPZ,
            use_env=False,
            use_filter=False,
            use_templates=False,
            cleanup_after_use=True,
            enable_caching=False,
            user_agent="docktkinase/1.0 no-msa",
        )
    
    def get_cache_key(self, sequence: str) -> str:
        """
        Generate cache key for a sequence based on current settings.
        
        Args:
            sequence: Protein sequence
        
        Returns:
            Cache key string
        """
        import hashlib
        
        # Create unique key based on sequence and settings
        settings_str = f"{self.mode}_{self.use_env}_{self.use_filter}_{sequence}"
        return hashlib.sha256(settings_str.encode()).hexdigest()[:16]
    
    def summary(self) -> str:
        """
        Get human-readable summary of configuration.
        
        Returns:
            Formatted configuration summary
        """
        return f"""
MSA Configuration Summary:
  Mode: {self.mode}
  File Format: {self.file_format}
  
  ColabFold Settings:
    - Use ENV databases: {self.use_env}
    - Use diversity filter: {self.use_filter}
    - Use templates: {self.use_templates}
    - Server: {self.server_url}
    - User agent: {self.user_agent}
  
  Performance:
    - Chunk size: {self.chunk_size or 'Auto'}
    - Timeout: {self.timeout_seconds}s
    - Max retries: {self.max_retries}
  
  Storage:
    - Output directory: {self.output_directory}
    - Caching enabled: {self.enable_caching}
    - Cleanup after use: {self.cleanup_after_use}
  
  Estimated time for 700 sequences:
    - {self._estimate_time()}
        """.strip()
    
    def _estimate_time(self) -> str:
        """Estimate processing time based on mode."""
        time_estimates = {
            MsaMode.MAIN_STANDARD: "3-5 minutes",
            MsaMode.MAIN_FAST: "1-2 minutes",
            MsaMode.MAIN_HIGH_QUALITY: "5-10 minutes",
            MsaMode.PAIRED: "Not recommended for 700+ sequences",
            MsaMode.NONE: "Instant (no MSA)",
        }
        return time_estimates.get(self.mode, "Unknown")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_default_config() -> MsaConfig:
    """
    Get default MSA configuration for DockTKinase.
    
    Returns production-optimized configuration.
    
    Returns:
        Default MsaConfig instance
    """
    return MsaConfig.for_production()


def validate_msa_config(config: MsaConfig) -> bool:
    """
    Validate MSA configuration.
    
    Args:
        config: MsaConfig instance to validate
    
    Returns:
        True if valid
    
    Raises:
        ValueError: If configuration is invalid
    """
    # Check output directory is writable
    if not config.output_directory.exists():
        raise ValueError(f"Output directory does not exist: {config.output_directory}")
    
    # Check user agent is set
    if not config.user_agent:
        raise ValueError("user_agent must be set for ColabFold API")
    
    # Warn about paired mode for large datasets
    if config.mode == MsaMode.PAIRED:
        import warnings
        warnings.warn(
            "Paired MSA mode is not recommended for 700+ sequences. "
            "Consider using MAIN_STANDARD mode instead.",
            UserWarning,
            stacklevel=2
        )
    
    return True


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    # Example 1: Production mode (recommended)
    print("="*70)
    print("PRODUCTION MODE (Recommended for DockTKinase)")
    print("="*70)
    config_prod = MsaConfig.for_production()
    print(config_prod.summary())
    
    # Example 2: Development mode
    print("\n" + "="*70)
    print("DEVELOPMENT MODE (Fast testing)")
    print("="*70)
    config_dev = MsaConfig.for_development()
    print(config_dev.summary())
    
    # Example 3: Research mode
    print("\n" + "="*70)
    print("RESEARCH MODE (Maximum quality)")
    print("="*70)
    config_res = MsaConfig.for_research()
    print(config_res.summary())
    
    # Example 4: No MSA mode
    print("\n" + "="*70)
    print("NO MSA MODE (Fastest)")
    print("="*70)
    config_none = MsaConfig.no_msa()
    print(config_none.summary())
    
    # Example 5: Custom configuration
    print("\n" + "="*70)
    print("CUSTOM CONFIGURATION")
    print("="*70)
    config_custom = MsaConfig(
        mode=MsaMode.MAIN_STANDARD,
        use_env=True,
        use_filter=False,  # Custom: no filter
        output_directory=Path("./my_msa_cache"),
        user_agent="myproject/2.0 contact@myemail.com"
    )
    print(config_custom.summary())
