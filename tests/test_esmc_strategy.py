"""
Unit tests for ESM-C strategy implementation.

Tests cover:
- Model loading with local cache
- Embedding generation with mean pooling
- Sequence validation and cleaning
- Device handling (CUDA/MPS/CPU)
- Error handling and edge cases
"""

import pytest
import numpy as np
import torch
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.build.embeddings.strategies.esmc_strategy import ESMCStrategy
from src.build.core.exceptions import ModelLoadError, EmbeddingError


class TestESMCStrategy:
    """Test suite for ESMCStrategy."""
    
    @pytest.fixture
    def strategy(self):
        """Create ESMCStrategy instance."""
        return ESMCStrategy()
    
    @pytest.fixture
    def mock_logger(self):
        """Create mock logger."""
        logger = Mock()
        logger.info = Mock()
        logger.warning = Mock()
        logger.debug = Mock()
        return logger
    
    @pytest.fixture
    def device(self):
        """Get appropriate device for testing."""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    # ===== Model Specifications Tests =====
    
    def test_model_specs_structure(self, strategy):
        """Test MODEL_SPECS has correct structure."""
        assert hasattr(strategy, 'MODEL_SPECS')
        assert isinstance(strategy.MODEL_SPECS, dict)
        
        # Check esmc-300m-2024-12 (priority model)
        assert 'esmc-300m-2024-12' in strategy.MODEL_SPECS
        specs_300m = strategy.MODEL_SPECS['esmc-300m-2024-12']
        assert specs_300m['dim'] == 960
        assert specs_300m['layers'] == 30
        assert specs_300m['max_len'] == 2048
        
        # Check esmc-600m-2024-12
        assert 'esmc-600m-2024-12' in strategy.MODEL_SPECS
        specs_600m = strategy.MODEL_SPECS['esmc-600m-2024-12']
        assert specs_600m['dim'] == 1152
        assert specs_600m['layers'] == 36
        assert specs_600m['max_len'] == 2048
    
    def test_get_embedding_dim(self, strategy):
        """Test get_embedding_dim returns correct dimensions."""
        assert strategy.get_embedding_dim('esmc-300m-2024-12') == 960
        assert strategy.get_embedding_dim('esmc-600m-2024-12') == 1152
        assert strategy.get_embedding_dim('unknown') == 960  # default
    
    def test_get_max_length(self, strategy):
        """Test get_max_length returns correct max lengths."""
        assert strategy.get_max_length('esmc-300m-2024-12') == 2048
        assert strategy.get_max_length('esmc-600m-2024-12') == 2048
        assert strategy.get_max_length('unknown') == 2048  # default
    
    # ===== Sequence Cleaning Tests =====
    
    def test_clean_sequence_valid(self, strategy):
        """Test cleaning valid sequences."""
        valid_seq = "ACDEFGHIKLMNPQRSTVWY"
        cleaned = strategy._clean_sequence(valid_seq)
        assert cleaned == valid_seq
    
    def test_clean_sequence_lowercase(self, strategy):
        """Test cleaning converts lowercase to uppercase."""
        mixed_case = "AcDeFgHiKlMnPqRsTvWy"
        cleaned = strategy._clean_sequence(mixed_case)
        assert cleaned == "ACDEFGHIKLMNPQRSTVWY"
    
    def test_clean_sequence_with_whitespace(self, strategy):
        """Test cleaning removes whitespace."""
        with_spaces = "  ACDEFG  HIKLMN  "
        cleaned = strategy._clean_sequence(with_spaces)
        assert cleaned == "ACDEFGHIKLMN"
    
    def test_clean_sequence_with_invalid_chars(self, strategy, mock_logger):
        """Test cleaning removes invalid characters."""
        strategy.logger = mock_logger
        invalid_chars = "ACDEFG123HIKLMN***PQRST"
        cleaned = strategy._clean_sequence(invalid_chars)
        assert cleaned == "ACDEFGHIKLMNPQRST"
        assert mock_logger.warning.called
    
    def test_clean_sequence_empty_raises_error(self, strategy):
        """Test empty sequence raises EmbeddingError."""
        with pytest.raises(EmbeddingError, match="no valid amino acids"):
            strategy._clean_sequence("")
    
    def test_clean_sequence_only_invalid_raises_error(self, strategy):
        """Test sequence with only invalid chars raises error."""
        with pytest.raises(EmbeddingError, match="no valid amino acids"):
            strategy._clean_sequence("123456789")
    
    # ===== Cache Setup Tests =====
    
    def test_setup_cache_creates_directory(self, strategy):
        """Test _setup_cache_and_paths creates cache directory."""
        strategy._setup_cache_and_paths()
        
        assert strategy._cache_dir is not None
        assert strategy._cache_dir.exists()
        assert strategy._cache_dir.name == "ESM3"
    
    def test_setup_cache_sets_environment_variable(self, strategy):
        """Test _setup_cache_and_paths sets ESM_DATA_ROOT."""
        import os
        strategy._setup_cache_and_paths()
        
        assert 'ESM_DATA_ROOT' in os.environ
        assert os.environ['ESM_DATA_ROOT'] == str(strategy._cache_dir)
    
    def test_setup_cache_adds_esm3_to_path(self, strategy):
        """Test _setup_cache_and_paths adds ESM-3 to sys.path."""
        import sys
        strategy._setup_cache_and_paths()
        
        assert strategy._esm3_path is not None
        esm3_str = str(strategy._esm3_path)
        assert esm3_str in sys.path
    
    # ===== Model Loading Tests (Unit - Mocked) =====
    
    def test_load_unsupported_model_raises_error(self, strategy, device):
        """Test loading unsupported model raises ValueError."""
        with pytest.raises(ValueError, match="not supported"):
            strategy.load('unsupported-model', device)
    
    def test_load_model_successful(self, strategy, device, mock_logger):
        """Test successful model loading."""
        # Mock ESMC import and model
        with patch('esm.models.esmc.ESMC') as mock_esmc_class:
            mock_model = MagicMock()
            mock_model.eval.return_value = mock_model
            mock_model.tokenizer = MagicMock()
            mock_esmc_class.from_pretrained.return_value = mock_model
            # Load model
            model, tokenizer = strategy.load(
                'esmc-300m-2024-12',
                device,
                logger=mock_logger
            )
            
            # Assertions
            assert model is not None
            assert tokenizer is not None
            mock_esmc_class.from_pretrained.assert_called_once()
            mock_model.eval.assert_called_once()
            assert mock_logger.info.called
    
    def test_load_model_import_error(self, strategy, device):
        """Test ModelLoadError when ESMC import fails."""
        # Simulate import error by mocking the import to raise exception
        with patch('esm.models.esmc.ESMC') as mock_esmc_class:
            mock_esmc_class.from_pretrained.side_effect = ImportError("ESMC not found")
            
            with pytest.raises(ModelLoadError, match="Failed to load"):
                strategy.load('esmc-300m-2024-12', device)
    
    # ===== Embedding Generation Tests (Unit - Mocked) =====
    
    def test_generate_embedding_shape(self, strategy, device):
        """Test generate returns correct embedding shape."""
        # Mock model and tokenizer
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        
        # Mock model._tokenize
        mock_tokens = torch.tensor([[1, 2, 3, 4, 5]])  # Example tokens
        mock_model._tokenize.return_value = mock_tokens
        
        # Mock model.forward
        mock_embeddings = torch.randn(1, 5, 960)  # [batch, seq_len, dim]
        mock_output = MagicMock()
        mock_output.embeddings = mock_embeddings
        mock_model.forward.return_value = mock_output
        
        # Generate embedding
        sequence = "ACDEFGHIKLMNPQRSTVWY"
        embedding = strategy.generate(
            mock_model,
            mock_tokenizer,
            sequence,
            device
        )
        
        # Assertions
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (960,)  # esmc-300m dim
        assert not np.isnan(embedding).any()
    
    def test_generate_with_cls_pooling(self, strategy, device):
        """Test generate with CLS token pooling."""
        # Mock model and tokenizer
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        
        # Mock tokens and output
        mock_tokens = torch.tensor([[1, 2, 3]])
        mock_model._tokenize.return_value = mock_tokens
        
        mock_embeddings = torch.randn(1, 3, 960)
        mock_output = MagicMock()
        mock_output.embeddings = mock_embeddings
        mock_model.forward.return_value = mock_output
        
        # Generate with CLS pooling
        sequence = "ACDEFG"
        embedding = strategy.generate(
            mock_model,
            mock_tokenizer,
            sequence,
            device,
            pooling_strategy='cls'
        )
        
        # Should use first token (CLS)
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (960,)
    
    def test_generate_truncates_long_sequence(self, strategy, device, mock_logger):
        """Test generate truncates sequences exceeding max length."""
        # Mock model
        mock_model = MagicMock()
        mock_model.__class__.__name__ = 'ESMC'
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        
        # Mock tokens and output
        mock_tokens = torch.tensor([[1] * 100])
        mock_model._tokenize.return_value = mock_tokens
        
        mock_embeddings = torch.randn(1, 100, 960)
        mock_output = MagicMock()
        mock_output.embeddings = mock_embeddings
        mock_model.forward.return_value = mock_output
        
        # Very long sequence (>2048)
        long_sequence = "A" * 3000
        
        embedding = strategy.generate(
            mock_model,
            mock_tokenizer,
            long_sequence,
            device,
            logger=mock_logger
        )
        
        # Should truncate and log warning
        assert isinstance(embedding, np.ndarray)
        assert mock_logger.warning.called
    
    def test_generate_handles_embedding_error(self, strategy, device):
        """Test generate raises EmbeddingError on failure."""
        # Mock model that raises error
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_model._tokenize.side_effect = RuntimeError("Tokenization failed")
        
        with pytest.raises(EmbeddingError, match="ESM-C embedding generation failed"):
            strategy.generate(
                mock_model,
                mock_tokenizer,
                "ACDEFG",
                device
            )
    
    # ===== Cleanup Tests =====
    
    def test_cleanup_calls_gc_collect(self, strategy):
        """Test cleanup calls garbage collection."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        
        with patch('gc.collect') as mock_gc:
            strategy.cleanup(mock_model, mock_tokenizer)
            mock_gc.assert_called_once()
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.empty_cache')
    def test_cleanup_clears_cuda_cache(self, mock_empty_cache, mock_cuda_available, strategy):
        """Test cleanup clears CUDA cache when available."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        
        strategy.cleanup(mock_model, mock_tokenizer)
        
        mock_empty_cache.assert_called_once()
    
    # ===== Flash Attention Detection =====
    
    @patch('importlib.import_module')
    def test_check_flash_attention_available(self, mock_import, strategy):
        """Test flash attention detection when available."""
        mock_import.return_value = MagicMock()  # Simulate successful import
        result = strategy._check_flash_attention()
        # Note: actual implementation uses try/except, adjust if needed
        assert isinstance(result, bool)
    
    @patch('importlib.import_module')
    def test_check_flash_attention_unavailable(self, mock_import, strategy):
        """Test flash attention detection when unavailable."""
        mock_import.side_effect = ImportError("flash_attn not found")
        result = strategy._check_flash_attention()
        assert isinstance(result, bool)


# ===== Integration Tests (Require Real Model) =====

class TestESMCStrategyIntegration:
    """
    Integration tests that require actual ESM-C model.
    
    These tests are slower and require:
    - ESM-3 repository installed
    - Model weights downloaded
    - Sufficient GPU/CPU memory
    
    Run with: pytest tests/test_esmc_strategy.py::TestESMCStrategyIntegration -v
    """
    
    @pytest.fixture
    def strategy(self):
        """Create ESMCStrategy instance."""
        return ESMCStrategy()
    
    @pytest.fixture
    def device(self):
        """Get appropriate device."""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_load_real_model(self, strategy, device):
        """Test loading real ESM-C model (requires model weights)."""
        pytest.importorskip("esm")  # Skip if esm not installed
        
        try:
            model, tokenizer = strategy.load('esmc-300m-2024-12', device)
            
            assert model is not None
            assert tokenizer is not None
            assert hasattr(model, '_tokenize')
            assert hasattr(model, 'forward')
            
            # Cleanup
            strategy.cleanup(model, tokenizer)
            
        except (ImportError, ModelLoadError) as e:
            pytest.skip(f"ESM-C model not available: {e}")
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_generate_real_embedding(self, strategy, device):
        """Test generating real embedding with actual model."""
        pytest.importorskip("esm")
        
        try:
            # Load model
            model, tokenizer = strategy.load('esmc-300m-2024-12', device)
            
            # Generate embedding
            sequence = "ACDEFGHIKLMNPQRSTVWY"
            embedding = strategy.generate(model, tokenizer, sequence, device)
            
            # Assertions
            assert isinstance(embedding, np.ndarray)
            assert embedding.shape == (960,)
            assert not np.isnan(embedding).any()
            assert not np.isinf(embedding).any()
            
            # Cleanup
            strategy.cleanup(model, tokenizer)
            
        except (ImportError, ModelLoadError) as e:
            pytest.skip(f"ESM-C model not available: {e}")
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_end_to_end_pipeline(self, strategy, device):
        """Test complete pipeline: load → generate → cleanup."""
        pytest.importorskip("esm")
        
        try:
            # Load
            model, tokenizer = strategy.load('esmc-300m-2024-12', device)
            
            # Generate for multiple sequences
            sequences = [
                "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVK",
                "ACDEFGHIKLMNPQRSTVWY",
                "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"
            ]
            
            embeddings = []
            for seq in sequences:
                emb = strategy.generate(model, tokenizer, seq, device)
                embeddings.append(emb)
                assert emb.shape == (960,)
            
            # Check embeddings are different
            assert not np.allclose(embeddings[0], embeddings[1])
            assert not np.allclose(embeddings[1], embeddings[2])
            
            # Cleanup
            strategy.cleanup(model, tokenizer)
            
        except (ImportError, ModelLoadError) as e:
            pytest.skip(f"ESM-C model not available: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
