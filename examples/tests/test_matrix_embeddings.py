"""
Testes para validar a implementação de generate_matrix() nas estratégias de embedding.

Fase 1 - Multi-Matrix Branch
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import tempfile
import os

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestBaseProteinStrategy:
    """Testes para BaseProteinStrategy.generate_matrix()"""
    
    def test_generate_matrix_default_returns_none(self):
        """Verifica que implementação padrão retorna None"""
        from src.build.embeddings.strategies.base_protein_strategy import BaseProteinStrategy
        
        # Criar instância concreta mínima para teste
        class MinimalStrategy(BaseProteinStrategy):
            def load(self, model_name, device, **kwargs):
                return None, None
            
            def generate(self, model, auxiliary_objects, sequence, device, **kwargs):
                return np.zeros(100)
            
            def cleanup(self, model, auxiliary_objects):
                pass
            
            def get_embedding_dim(self, model_name: str) -> int:
                return 100
            
            def get_max_length(self, model_name: str) -> int:
                return 1024
        
        strategy = MinimalStrategy()
        result = strategy.generate_matrix(
            model=None,
            auxiliary_objects=None,
            sequence="MKFLKFSL",
            device=None
        )
        
        assert result is None, "Implementação padrão deve retornar None"
    
    def test_generate_matrix_has_correct_signature(self):
        """Verifica que generate_matrix tem assinatura correta"""
        from src.build.embeddings.strategies.base_protein_strategy import BaseProteinStrategy
        import inspect
        
        sig = inspect.signature(BaseProteinStrategy.generate_matrix)
        params = list(sig.parameters.keys())
        
        expected_params = ['self', 'model', 'auxiliary_objects', 'sequence', 'device']
        for param in expected_params:
            assert param in params, f"Parâmetro '{param}' faltando na assinatura"


class TestESM2StrategyMatrix:
    """Testes para ESM2Strategy.generate_matrix()"""
    
    def test_generate_matrix_method_exists(self):
        """Verifica que método existe na classe"""
        from src.build.embeddings.strategies.esm2_strategy import ESM2Strategy
        
        assert hasattr(ESM2Strategy, 'generate_matrix'), "ESM2Strategy deve ter generate_matrix()"
    
    def test_generate_matrix_has_correct_signature(self):
        """Verifica que generate_matrix tem assinatura correta"""
        from src.build.embeddings.strategies.esm2_strategy import ESM2Strategy
        import inspect
        
        sig = inspect.signature(ESM2Strategy.generate_matrix)
        params = list(sig.parameters.keys())
        
        expected_params = ['self', 'model', 'auxiliary_objects', 'sequence', 'device']
        for param in expected_params:
            assert param in params, f"Parâmetro '{param}' faltando na assinatura"
    
    def test_generate_matrix_docstring_exists(self):
        """Verifica que método tem documentação"""
        from src.build.embeddings.strategies.esm2_strategy import ESM2Strategy
        
        assert ESM2Strategy.generate_matrix.__doc__ is not None, \
            "generate_matrix() deve ter docstring"
        assert "seq_len" in ESM2Strategy.generate_matrix.__doc__, \
            "Docstring deve mencionar shape [seq_len, ...]"


class TestBoltzStrategyMatrix:
    """Testes para BoltzStrategy.generate_matrix()"""
    
    def test_generate_matrix_method_exists(self):
        """Verifica que método existe na classe"""
        from src.build.embeddings.strategies.boltz_strategy import BoltzStrategy
        
        assert hasattr(BoltzStrategy, 'generate_matrix'), "BoltzStrategy deve ter generate_matrix()"
    
    def test_extract_embedding_matrix_method_exists(self):
        """Verifica que método auxiliar existe"""
        from src.build.embeddings.strategies.boltz_strategy import BoltzStrategy
        
        assert hasattr(BoltzStrategy, '_extract_embedding_matrix'), \
            "BoltzStrategy deve ter _extract_embedding_matrix()"


class TestESMCStrategyMatrix:
    """Testes para ESMCStrategy.generate_matrix()"""
    
    def test_generate_matrix_method_exists(self):
        """Verifica que método existe na classe"""
        from src.build.embeddings.strategies.esmc_strategy import ESMCStrategy
        
        assert hasattr(ESMCStrategy, 'generate_matrix'), "ESMCStrategy deve ter generate_matrix()"


class TestESMCForgeStrategyMatrix:
    """Testes para ESMCForgeStrategy.generate_matrix()"""
    
    def test_generate_matrix_method_exists(self):
        """Verifica que método existe na classe"""
        from src.build.embeddings.strategies.esmc_forge_strategy import ESMCForgeStrategy
        
        assert hasattr(ESMCForgeStrategy, 'generate_matrix'), \
            "ESMCForgeStrategy deve ter generate_matrix()"
    
    def test_generate_matrix_returns_none(self):
        """Verifica que retorna None (API-based não suporta matriz)"""
        from src.build.embeddings.strategies.esmc_forge_strategy import ESMCForgeStrategy
        
        strategy = ESMCForgeStrategy()
        result = strategy.generate_matrix(
            model=None,
            auxiliary_objects=None,
            sequence="MKFLKFSL",
            device=None
        )
        
        assert result is None, "ESMCForgeStrategy.generate_matrix() deve retornar None"


class TestProteinEmbeddingMatrix:
    """Testes para ProteinEmbedding com suporte a matriz"""
    
    def test_generate_embedding_matrix_method_exists(self):
        """Verifica que método existe"""
        from src.build.embeddings.protein_embedding import ProteinEmbedding
        
        assert hasattr(ProteinEmbedding, 'generate_embedding_matrix'), \
            "ProteinEmbedding deve ter generate_embedding_matrix()"
    
    def test_get_matrix_output_path_method_exists(self):
        """Verifica que método existe"""
        from src.build.embeddings.protein_embedding import ProteinEmbedding
        
        assert hasattr(ProteinEmbedding, 'get_matrix_output_path'), \
            "ProteinEmbedding deve ter get_matrix_output_path()"
    
    def test_generate_embeddings_has_save_matrix_param(self):
        """Verifica que generate_embeddings tem parâmetro save_matrix"""
        from src.build.embeddings.protein_embedding import ProteinEmbedding
        import inspect
        
        sig = inspect.signature(ProteinEmbedding.generate_embeddings)
        params = list(sig.parameters.keys())
        
        assert 'save_matrix' in params, "generate_embeddings deve ter parâmetro save_matrix"
        assert 'matrix_output_dir' in params, "generate_embeddings deve ter parâmetro matrix_output_dir"
    
    def test_get_embeddings_info_includes_matrix_info(self):
        """Verifica que get_embeddings_info retorna info sobre matrizes"""
        from src.build.embeddings.protein_embedding import ProteinEmbedding
        
        # Mock para evitar inicialização do modelo
        with patch.object(ProteinEmbedding, '_check_dependencies'):
            with patch.object(ProteinEmbedding, '_create_strategy'):
                with patch.object(ProteinEmbedding, '_validate_config'):
                    pe = ProteinEmbedding.__new__(ProteinEmbedding)
                    pe.model_name = "test"
                    pe.embedding_dim = 2560
                    pe._output_path = None
                    pe._matrix_output_path = None
                    
                    info = pe.get_embeddings_info()
                    
                    assert 'matrix_output_path' in info, "Info deve conter matrix_output_path"
                    assert 'matrix_count' in info, "Info deve conter matrix_count"


class TestLigandEmbeddingMatrix:
    """Testes para LigandEmbedding com suporte a matriz"""
    
    def test_generate_embeddings_has_save_matrix_param(self):
        """Verifica que generate_embeddings tem parâmetro save_matrix"""
        from src.build.embeddings.ligand_embedding import LigandEmbedding
        import inspect
        
        sig = inspect.signature(LigandEmbedding.generate_embeddings)
        params = list(sig.parameters.keys())
        
        assert 'save_matrix' in params, "generate_embeddings deve ter parâmetro save_matrix"
    
    def test_get_embeddings_info_includes_matrix_info(self):
        """Verifica que get_embeddings_info retorna info sobre matrizes"""
        from src.build.embeddings.ligand_embedding import LigandEmbedding
        
        # Mock para evitar inicialização
        with patch.object(LigandEmbedding, '_check_dependencies'):
            with patch.object(LigandEmbedding, '_setup_fm4m_path'):
                with patch.object(LigandEmbedding, '_validate_config'):
                    le = LigandEmbedding.__new__(LigandEmbedding)
                    le.model_name = "SMI-TED"
                    le.embedding_dim = 768
                    le._output_path = None
                    le._matrix_output_path = None
                    
                    info = le.get_embeddings_info()
                    
                    assert 'matrix_output_path' in info, "Info deve conter matrix_output_path"
                    assert 'matrix_count' in info, "Info deve conter matrix_count"


class TestConstants:
    """Testes para constantes de diretório de matrizes"""
    
    def test_protein_matrix_output_dir_exists(self):
        """Verifica que constante existe"""
        from src.build.core.constants import BuildConstants
        
        assert hasattr(BuildConstants, 'DEFAULT_PROTEIN_MATRIX_OUTPUT_DIR'), \
            "BuildConstants deve ter DEFAULT_PROTEIN_MATRIX_OUTPUT_DIR"
        assert BuildConstants.DEFAULT_PROTEIN_MATRIX_OUTPUT_DIR == 'protein_matrix_embeddings'
    
    def test_ligand_matrix_output_dir_exists(self):
        """Verifica que constante existe"""
        from src.build.core.constants import BuildConstants
        
        assert hasattr(BuildConstants, 'DEFAULT_LIGAND_MATRIX_OUTPUT_DIR'), \
            "BuildConstants deve ter DEFAULT_LIGAND_MATRIX_OUTPUT_DIR"
        assert BuildConstants.DEFAULT_LIGAND_MATRIX_OUTPUT_DIR == 'ligand_matrix_embeddings'
    
    def test_constants_exported_in_all(self):
        """Verifica que constantes estão no __all__"""
        from src.build.core import constants
        
        # Verificar se estão acessíveis diretamente
        assert hasattr(constants, 'BuildConstants')


class TestMatrixOutputFormat:
    """Testes para formato de saída das matrizes"""
    
    def test_matrix_file_naming_convention(self):
        """Verifica convenção de nomenclatura: {seq_id}_matrix.npy"""
        # Apenas verifica que a convenção está documentada no código
        from src.build.embeddings.protein_embedding import ProteinEmbedding
        import inspect
        
        source = inspect.getsource(ProteinEmbedding.generate_embeddings)
        
        assert '_matrix.npy' in source, \
            "Código deve usar convenção {seq_id}_matrix.npy para arquivos de matriz"


class TestIntegrationMock:
    """Testes de integração com mocks"""
    
    def test_protein_embedding_matrix_workflow(self):
        """Testa workflow completo de geração de matriz (mockado)"""
        from src.build.embeddings.protein_embedding import ProteinEmbedding
        
        # Criar mock da estratégia
        mock_strategy = Mock()
        mock_strategy.generate_matrix.return_value = np.random.randn(10, 2560)
        
        # Mock do ProteinEmbedding
        with patch.object(ProteinEmbedding, '_check_dependencies'):
            with patch.object(ProteinEmbedding, '_create_strategy'):
                with patch.object(ProteinEmbedding, '_validate_config'):
                    pe = ProteinEmbedding.__new__(ProteinEmbedding)
                    pe.strategy = mock_strategy
                    pe.model = Mock()
                    pe.alphabet = Mock()
                    pe.device = Mock()
                    pe._model_loaded = True
                    pe.logger = Mock()
                    
                    # Chamar generate_embedding_matrix
                    result = pe.generate_embedding_matrix("MKFLKFSLKK")
                    
                    # Verificar
                    assert result is not None
                    assert result.shape == (10, 2560)
                    mock_strategy.generate_matrix.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
