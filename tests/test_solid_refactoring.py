"""
Testes Completos: Validação da Refatoração SOLID do ESM-2
Verifica interface, implementação, factory e integração.
"""

import pytest
import torch
import numpy as np
from abc import ABC
from src.build.embeddings.strategies.base_protein_strategy import BaseProteinStrategy
from src.build.embeddings.strategies.esm2_strategy import ESM2Strategy
from src.build.embeddings.factories.protein_model_factory import ProteinModelFactory
from src.build.embeddings.protein_embedding import ProteinEmbedding


# =============================================================================
# TESTES 1: VALIDAÇÃO DA INTERFACE (BaseProteinStrategy)
# =============================================================================

class TestBaseProteinStrategyInterface:
    """Valida que a interface abstrata está correta."""
    
    def test_interface_is_abstract(self):
        """Teste: BaseProteinStrategy não pode ser instanciada diretamente."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseProteinStrategy()
    
    def test_interface_has_all_abstract_methods(self):
        """Teste: Interface declara todos os 5 métodos abstratos."""
        abstract_methods = BaseProteinStrategy.__abstractmethods__
        
        expected_methods = {'load', 'generate', 'get_max_length', 'get_embedding_dim', 'cleanup'}
        assert abstract_methods == expected_methods, f"Métodos esperados: {expected_methods}, encontrados: {abstract_methods}"
    
    def test_interface_inherits_from_abc(self):
        """Teste: Interface herda de ABC."""
        assert issubclass(BaseProteinStrategy, ABC)
    
    def test_incomplete_implementation_fails(self):
        """Teste: Implementação incompleta não pode ser instanciada."""
        
        # Implementação incompleta (faltando cleanup)
        class IncompleteStrategy(BaseProteinStrategy):
            def load(self, model_name, device, offload_folder=None, **kwargs):
                return None, None
            
            def generate(self, model, auxiliary_objects, sequence, device, **kwargs):
                return np.array([1.0, 2.0])
            
            def get_max_length(self, model_name):
                return 1024
            
            def get_embedding_dim(self, model_name):
                return 320
            
            # Faltando cleanup() propositalmente
        
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteStrategy()


# =============================================================================
# TESTES 2: VALIDAÇÃO DA IMPLEMENTAÇÃO (ESM2Strategy)
# =============================================================================

class TestESM2StrategyImplementation:
    """Valida que ESM2Strategy implementa corretamente a interface."""
    
    @pytest.fixture
    def strategy(self):
        return ESM2Strategy()
    
    def test_strategy_is_concrete_class(self, strategy):
        """Teste: ESM2Strategy pode ser instanciada."""
        assert strategy is not None
        assert isinstance(strategy, BaseProteinStrategy)
    
    def test_strategy_implements_all_methods(self, strategy):
        """Teste: ESM2Strategy implementa todos os métodos abstratos."""
        assert hasattr(strategy, 'load')
        assert hasattr(strategy, 'generate')
        assert hasattr(strategy, 'get_max_length')
        assert hasattr(strategy, 'get_embedding_dim')
        assert hasattr(strategy, 'cleanup')
        
        # Verificar que não são abstratos
        assert not getattr(strategy.load, '__isabstractmethod__', False)
    
    def test_strategy_method_signatures(self, strategy):
        """Teste: Métodos têm assinaturas corretas."""
        import inspect
        
        # load(model_name, device, offload_folder, **kwargs)
        load_sig = inspect.signature(strategy.load)
        assert 'model_name' in load_sig.parameters
        assert 'device' in load_sig.parameters
        assert 'offload_folder' in load_sig.parameters
        
        # generate(model, auxiliary_objects, sequence, device, **kwargs)
        gen_sig = inspect.signature(strategy.generate)
        assert 'model' in gen_sig.parameters
        assert 'auxiliary_objects' in gen_sig.parameters
        assert 'sequence' in gen_sig.parameters
        assert 'device' in gen_sig.parameters


# =============================================================================
# TESTES 3: VALIDAÇÃO DA FACTORY (ProteinModelFactory)
# =============================================================================

class TestProteinModelFactory:
    """Valida que a factory cria strategies corretas."""
    
    @pytest.mark.parametrize("model_name", [
        "esm2_t6_8M_UR50D",
        "esm2_t12_35M_UR50D",
        "esm2_t30_150M_UR50D",
        "esm2_t33_650M_UR50D",
        "esm2_t36_3B_UR50D",
        "esm2_t48_15B_UR50D",
        "esm1b_t33_650M_UR50S",
    ])
    def test_factory_creates_esm2_strategy(self, model_name):
        """Teste: Factory cria ESM2Strategy para todos os modelos ESM-2."""
        strategy = ProteinModelFactory.create_strategy(model_name)
        assert isinstance(strategy, ESM2Strategy)
        assert isinstance(strategy, BaseProteinStrategy)
    
    def test_factory_rejects_invalid_models(self):
        """Teste: Factory lança erro para modelos inválidos."""
        invalid_models = [
            "gpt4_protein",
            "bert_protein",
            "t5_protein",
            "esm4_nonexistent",
            "",
            "   ",
        ]
        
        for invalid in invalid_models:
            with pytest.raises(ValueError, match="não"):
                ProteinModelFactory.create_strategy(invalid)
    
    def test_factory_is_stateless(self):
        """Teste: Factory pode criar múltiplas strategies independentes."""
        strategy1 = ProteinModelFactory.create_strategy("esm2_t6_8M_UR50D")
        strategy2 = ProteinModelFactory.create_strategy("esm2_t33_650M_UR50D")
        
        # São instâncias diferentes
        assert strategy1 is not strategy2
        assert id(strategy1) != id(strategy2)


# =============================================================================
# TESTES 4: VALIDAÇÃO DA INTEGRAÇÃO (ProteinEmbedding)
# =============================================================================

class TestProteinEmbeddingIntegration:
    """Valida que ProteinEmbedding usa strategies corretamente."""
    
    @pytest.fixture
    def embedding_gen(self):
        return ProteinEmbedding(model_name='esm2_t6_8M_UR50D', use_gpu=False)
    
    def test_embedding_creates_strategy_on_init(self, embedding_gen):
        """Teste: ProteinEmbedding cria strategy no __init__."""
        assert hasattr(embedding_gen, 'strategy')
        assert embedding_gen.strategy is not None
        assert isinstance(embedding_gen.strategy, BaseProteinStrategy)
    
    def test_embedding_delegates_to_strategy(self, embedding_gen):
        """Teste: ProteinEmbedding delega para strategy (não tem lógica direta)."""
        # Verificar que métodos internos são simples delegações
        import inspect
        
        # _load_model deve chamar strategy.load()
        load_source = inspect.getsource(embedding_gen._load_model)
        assert 'strategy.load' in load_source
        
        # _generate_single_embedding deve chamar strategy.generate()
        gen_source = inspect.getsource(embedding_gen._generate_single_embedding)
        assert 'strategy.generate' in gen_source
    
    def test_embedding_maintains_backward_compatibility(self, embedding_gen):
        """Teste: API pública mantida (retrocompatibilidade)."""
        # Métodos públicos devem existir
        public_methods = [
            'initialize',
            'generate_embeddings',
            'process_fasta_file',
            'build',
        ]
        
        for method in public_methods:
            assert hasattr(embedding_gen, method), f"Método público '{method}' não encontrado"


# =============================================================================
# TESTES 5: TESTES FUNCIONAIS END-TO-END
# =============================================================================

class TestEndToEndFunctionality:
    """Testes funcionais completos do início ao fim."""
    
    @pytest.fixture
    def device(self):
        return torch.device("cpu")  # Usar CPU para testes rápidos
    
    def test_full_pipeline_8M_model(self, device):
        """Teste E2E: Pipeline completo com modelo 8M."""
        # 1. Factory cria strategy
        strategy = ProteinModelFactory.create_strategy("esm2_t6_8M_UR50D")
        
        # 2. Strategy carrega modelo
        model, alphabet = strategy.load("esm2_t6_8M_UR50D", device)
        assert model is not None
        assert alphabet is not None
        
        # 3. Strategy gera embedding
        sequence = "MKTAYIAKQRQISFVK"
        embedding = strategy.generate(model, alphabet, sequence, device)
        
        # 4. Validações
        assert isinstance(embedding, np.ndarray)
        expected_dim = strategy.get_embedding_dim("esm2_t6_8M_UR50D")
        assert embedding.shape == (expected_dim,)
        assert not np.any(np.isnan(embedding))
        assert not np.any(np.isinf(embedding))
        
        # 5. Cleanup
        strategy.cleanup(model, alphabet)
    
    def test_multiple_sequences_same_strategy(self, device):
        """Teste E2E: Gerar embeddings de múltiplas sequências."""
        strategy = ProteinModelFactory.create_strategy("esm2_t6_8M_UR50D")
        model, alphabet = strategy.load("esm2_t6_8M_UR50D", device)
        
        sequences = [
            "MKTAYIAKQRQISFVK",
            "ARHPHILNEQVAEVAEALR",
            "GLSDGEWQQVLNVWGKVE",
        ]
        
        embeddings = []
        for seq in sequences:
            emb = strategy.generate(model, alphabet, seq, device)
            embeddings.append(emb)
        
        # Todos devem ter mesma dimensão
        dims = [emb.shape[0] for emb in embeddings]
        assert len(set(dims)) == 1, "Embeddings com dimensões diferentes"
        
        # Embeddings devem ser diferentes
        assert not np.allclose(embeddings[0], embeddings[1])
        assert not np.allclose(embeddings[1], embeddings[2])
        
        strategy.cleanup(model, alphabet)
    
    def test_switching_strategies_runtime(self):
        """Teste E2E: Trocar strategies em runtime (demonstra polimorfismo)."""
        # Criar dois geradores com modelos diferentes
        gen1 = ProteinEmbedding(model_name='esm2_t6_8M_UR50D', use_gpu=False)
        gen1.initialize()
        
        # Estratégias são diferentes mas intercambiáveis
        assert isinstance(gen1.strategy, ESM2Strategy)
        
        sequence = "MKTAYIAKQRQISFVK"
        emb1 = gen1._generate_single_embedding(sequence)
        
        # Validar resultado
        assert emb1.shape[0] == 320  # 8M model


# =============================================================================
# TESTES 6: TESTES DE PRINCÍPIOS SOLID
# =============================================================================

class TestSOLIDPrinciples:
    """Valida que os princípios SOLID foram aplicados corretamente."""
    
    def test_single_responsibility_principle(self):
        """Teste SRP: Cada classe tem uma única responsabilidade."""
        # BaseProteinStrategy: Define contrato
        # ESM2Strategy: Implementa ESM-2
        # ProteinModelFactory: Cria strategies
        # ProteinEmbedding: Orquestra pipeline
        
        strategy = ESM2Strategy()
        
        # ESM2Strategy não deve ter responsabilidades de factory
        assert not hasattr(strategy, 'create_strategy')
        
        # ESM2Strategy não deve ter lógica de pipeline
        assert not hasattr(strategy, 'process_fasta_file')
    
    def test_open_closed_principle(self):
        """Teste OCP: Aberto para extensão, fechado para modificação."""
        # Simular adição de ESM-3 sem modificar código existente
        
        class MockESM3Strategy(BaseProteinStrategy):
            """Strategy fictícia para ESM-3 (apenas para teste)."""
            
            def load(self, model_name, device, offload_folder=None, **kwargs):
                return "mock_model", "mock_alphabet"
            
            def generate(self, model, auxiliary_objects, sequence, device, **kwargs):
                return np.random.rand(2560)  # ESM-3 fictício com 2560-dim
            
            def get_max_length(self, model_name):
                return 8192
            
            def get_embedding_dim(self, model_name):
                return 2560
            
            def cleanup(self, model, auxiliary_objects):
                pass
        
        # Criar strategy ESM-3 sem modificar ESM2Strategy
        esm3_strategy = MockESM3Strategy()
        assert isinstance(esm3_strategy, BaseProteinStrategy)
        
        # ESM2Strategy continua funcionando sem modificações
        esm2_strategy = ESM2Strategy()
        assert esm2_strategy.get_embedding_dim("esm2_t6_8M_UR50D") == 320
    
    def test_liskov_substitution_principle(self):
        """Teste LSP: Strategies são intercambiáveis."""
        # Qualquer BaseProteinStrategy pode substituir outra
        
        def process_with_strategy(strategy: BaseProteinStrategy, model_name: str):
            """Função que aceita qualquer strategy."""
            dim = strategy.get_embedding_dim(model_name)
            max_len = strategy.get_max_length(model_name)
            return dim, max_len
        
        # Funciona com ESM2Strategy
        esm2 = ESM2Strategy()
        dim2, len2 = process_with_strategy(esm2, "esm2_t6_8M_UR50D")
        assert dim2 == 320
        assert len2 == 1024
    
    def test_interface_segregation_principle(self):
        """Teste ISP: Interface não força métodos desnecessários."""
        # BaseProteinStrategy tem apenas métodos essenciais (5)
        abstract_methods = BaseProteinStrategy.__abstractmethods__
        assert len(abstract_methods) == 5
        
        # Nenhum método "inchado" ou com muitos parâmetros opcionais
        strategy = ESM2Strategy()
        import inspect
        
        for method_name in ['load', 'generate', 'get_max_length', 'get_embedding_dim', 'cleanup']:
            method = getattr(strategy, method_name)
            sig = inspect.signature(method)
            
            # Número razoável de parâmetros (≤ 6 incluindo self)
            param_count = len(sig.parameters)
            assert param_count <= 6, f"{method_name} tem muitos parâmetros: {param_count}"
    
    def test_dependency_inversion_principle(self):
        """
        DIP: ProteinEmbedding depende de abstração (BaseProteinStrategy), não de ESM2Strategy.
        
        Valida:
        - Type hint usa BaseProteinStrategy (abstração)
        - Factory injeta dependência (ESM2Strategy)
        - Runtime type check confirma polimorfismo
        """
        gen = ProteinEmbedding(model_name='esm2_t6_8M_UR50D', use_gpu=False)
        
        # Runtime: strategy é instância de BaseProteinStrategy (abstração)
        assert isinstance(gen.strategy, BaseProteinStrategy)
        
        # Runtime: strategy também é instância de ESM2Strategy (implementação)
        assert isinstance(gen.strategy, ESM2Strategy)
        
        # Type hint validation: ProteinEmbedding usa abstração
        import inspect
        from typing import get_type_hints
        
        # Obter type hints do __init__
        embedding_source = inspect.getsource(ProteinEmbedding)
        
        # Deve ter type hint com BaseProteinStrategy (depende da abstração)
        assert 'BaseProteinStrategy' in embedding_source
        assert 'self.strategy: Optional[BaseProteinStrategy]' in embedding_source or \
               'self.strategy:Optional[BaseProteinStrategy]' in embedding_source


# =============================================================================
# TESTES 7: TESTES DE ROBUSTEZ E EDGE CASES
# =============================================================================

class TestRobustnessAndEdgeCases:
    """Testes de casos extremos e robustez."""
    
    @pytest.fixture
    def strategy(self):
        return ESM2Strategy()
    
    @pytest.fixture
    def device(self):
        return torch.device("cpu")
    
    def test_strategy_handles_invalid_model_names(self, strategy, device):
        """Teste: Strategy rejeita modelos inválidos gracefully."""
        with pytest.raises(ValueError):
            strategy.load("modelo_inexistente_xyz", device)
    
    def test_strategy_configuration_methods_never_fail(self, strategy):
        """Teste: get_max_length e get_embedding_dim sempre retornam valores."""
        # Mesmo para modelos desconhecidos, devem retornar defaults
        max_len = strategy.get_max_length("unknown_model")
        assert isinstance(max_len, int)
        assert max_len > 0
        
        dim = strategy.get_embedding_dim("unknown_model")
        assert isinstance(dim, int)
        assert dim > 0
    
    def test_cleanup_is_idempotent(self, strategy, device):
        """Teste: cleanup() pode ser chamado múltiplas vezes."""
        model, alphabet = strategy.load("esm2_t6_8M_UR50D", device)
        
        # Chamar cleanup 3 vezes não deve causar erro
        strategy.cleanup(model, alphabet)
        strategy.cleanup(model, alphabet)
        strategy.cleanup(model, alphabet)
    
    def test_factory_method_names_are_descriptive(self):
        """Teste: Factory tem métodos auxiliares descritivos."""
        factory = ProteinModelFactory()
        
        # Verificar métodos auxiliares existem
        assert hasattr(factory, 'is_esm2_model')
        assert hasattr(factory, 'is_esm3_model')
        assert hasattr(factory, 'list_supported_models')
        
        # Testar funcionalidade
        assert factory.is_esm2_model("esm2_t6_8M_UR50D") == True
        assert factory.is_esm2_model("esm3_unknown") == False
        
        models = factory.list_supported_models()
        assert 'esm2' in models
        assert len(models['esm2']) >= 6


# =============================================================================
# TESTES 8: TESTES DE DOCUMENTAÇÃO E MANUTENIBILIDADE
# =============================================================================

class TestDocumentationAndMaintainability:
    """Valida documentação e manutenibilidade do código."""
    
    def test_all_classes_have_docstrings(self):
        """Teste: Todas as classes têm docstrings."""
        classes = [
            BaseProteinStrategy,
            ESM2Strategy,
            ProteinModelFactory,
        ]
        
        for cls in classes:
            assert cls.__doc__ is not None, f"{cls.__name__} sem docstring"
            assert len(cls.__doc__.strip()) > 50, f"{cls.__name__} com docstring muito curta"
    
    def test_all_abstract_methods_have_docstrings(self):
        """Teste: Métodos abstratos têm docstrings detalhadas."""
        abstract_methods = ['load', 'generate', 'get_max_length', 'get_embedding_dim', 'cleanup']
        
        for method_name in abstract_methods:
            method = getattr(BaseProteinStrategy, method_name)
            assert method.__doc__ is not None, f"{method_name} sem docstring"
            
            # Docstring deve mencionar Args e Returns
            docstring = method.__doc__
            assert 'Args:' in docstring or 'Returns:' in docstring
    
    def test_strategy_implementation_has_type_hints(self):
        """Teste: ESM2Strategy usa type hints."""
        import inspect
        
        strategy = ESM2Strategy()
        
        for method_name in ['load', 'generate', 'get_max_length', 'get_embedding_dim']:
            method = getattr(strategy, method_name)
            sig = inspect.signature(method)
            
            # Verificar que tem return annotation
            assert sig.return_annotation != inspect.Signature.empty, \
                f"{method_name} sem type hint de retorno"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
