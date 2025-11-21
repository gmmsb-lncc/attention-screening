"""
Factory para criação de estratégias de modelos de proteína.
Implementa Factory Pattern para desacoplar criação de objetos.
"""

from src.build.embeddings.strategies.base_protein_strategy import BaseProteinStrategy
from src.build.embeddings.strategies.esm2_strategy import ESM2Strategy
from src.build.embeddings.strategies.esmc_strategy import ESMCStrategy
from src.build.embeddings.strategies.openfold_strategy import OpenFoldStrategy


class ProteinModelFactory:
    """
    Fábrica para criação de estratégias de modelos de proteína.
    
    Responsabilidades:
    - Detectar tipo de modelo baseado no nome
    - Instanciar estratégia apropriada (ESM2Strategy, ESM3Strategy, etc.)
    - Validar modelos suportados
    
    Padrão de Design: Factory Pattern
    - Centraliza criação de objetos
    - Isola lógica de detecção de modelo
    - Facilita adição de novos modelos (Open/Closed Principle)
    
    Exemplo de Uso:
        >>> factory = ProteinModelFactory()
        >>> strategy = factory.create_strategy("esm2_t48_15B_UR50D")
        >>> isinstance(strategy, ESM2Strategy)
        True
    """
    
    # Modelos ESM-2 suportados (Meta AI / Fair-ESM)
    ESM2_MODELS = {
        'esm2_t48_15B_UR50D',   # 15B parâmetros
        'esm2_t36_3B_UR50D',    # 3B parâmetros
        'esm2_t33_650M_UR50D',  # 650M parâmetros
        'esm2_t30_150M_UR50D',  # 150M parâmetros
        'esm2_t12_35M_UR50D',   # 35M parâmetros
        'esm2_t6_8M_UR50D',     # 8M parâmetros
        'esm1b_t33_650M_UR50S', # ESM-1b (legacy)
    }
    
    # Modelos ESM-C suportados (EvolutionaryScale Cambrian)
    ESMC_MODELS = {
        'esmc-300m-2024-12',    # 300M parâmetros, 960-dim (PRIORITÁRIO)
        'esmc-600m-2024-12',    # 600M parâmetros, 1152-dim
        'esmc-6b-2024-12',      # 7B parâmetros, 3072-dim
    }
    
    # Modelos OpenFold suportados (AlphaFold3 reproduction)
    OPENFOLD_MODELS = {
        'openfold3',            # OpenFold3 - structure-aware embeddings (384-dim)
    }
    
    # Modelos ESM-3 suportados (Meta AI / EvolutionaryScale)
    # FUTURO: Adicionar quando ESM-3 estiver disponível
    ESM3_MODELS = {
        # 'esm3_sm_open_v1',    # ESM-3 Small (open source)
        # 'esm3_medium',        # ESM-3 Medium
        # 'esm3_large',         # ESM-3 Large
    }
    
    @staticmethod
    def create_strategy(model_name: str) -> BaseProteinStrategy:
        """
        Cria estratégia apropriada baseada no nome do modelo.
        
        Args:
            model_name: Nome do modelo (ex: "esm2_t48_15B_UR50D")
            
        Returns:
            Instância de BaseProteinStrategy (ESM2Strategy, ESM3Strategy, etc.)
            
        Raises:
            ValueError: Se modelo não for suportado
            
        Exemplo:
            >>> factory = ProteinModelFactory()
            >>> strategy = factory.create_strategy("esm2_t33_650M_UR50D")
            >>> strategy.get_embedding_dim("esm2_t33_650M_UR50D")
            1280
        """
        # Detectar ESM-2
        if model_name in ProteinModelFactory.ESM2_MODELS:
            return ESM2Strategy()
        
        # Detectar ESM-C
        if model_name in ProteinModelFactory.ESMC_MODELS:
            return ESMCStrategy()
        
        # Detectar OpenFold
        if model_name in ProteinModelFactory.OPENFOLD_MODELS:
            return OpenFoldStrategy()
        
        # FUTURO: Detectar ESM-3
        # if model_name in ProteinModelFactory.ESM3_MODELS:
        #     from src.build.embeddings.strategies.esm3_strategy import ESM3Strategy
        #     return ESM3Strategy()
        
        # Modelo não suportado
        supported_esm2 = sorted(ProteinModelFactory.ESM2_MODELS)
        supported_esmc = sorted(ProteinModelFactory.ESMC_MODELS)
        supported_openfold = sorted(ProteinModelFactory.OPENFOLD_MODELS)
        
        raise ValueError(
            f"Modelo de proteína '{model_name}' não é suportado.\n\n"
            f"Modelos ESM-2 disponíveis:\n"
            + "\n".join(f"  • {m}" for m in supported_esm2)
            + "\n\nModelos ESM-C disponíveis:\n"
            + "\n".join(f"  • {m}" for m in supported_esmc)
            + "\n\nModelos OpenFold disponíveis:\n"
            + "\n".join(f"  • {m}" for m in supported_openfold)
            + "\n\nPara adicionar novos modelos:\n"
            + "1. Crie uma nova estratégia (ex: ESM3Strategy)\n"
            + "2. Adicione à factory em protein_model_factory.py\n"
            + "3. Registre na constante ESM_MODELS em constants.py"
        )
    
    @staticmethod
    def is_esm2_model(model_name: str) -> bool:
        """
        Verifica se o modelo é ESM-2.
        
        Args:
            model_name: Nome do modelo
            
        Returns:
            True se for ESM-2, False caso contrário
            
        Exemplo:
            >>> ProteinModelFactory.is_esm2_model("esm2_t48_15B_UR50D")
            True
            >>> ProteinModelFactory.is_esm2_model("esm3_sm_open_v1")
            False
        """
        return model_name in ProteinModelFactory.ESM2_MODELS
    
    @staticmethod
    def is_esmc_model(model_name: str) -> bool:
        """
        Verifica se o modelo é ESM-C.
        
        Args:
            model_name: Nome do modelo
            
        Returns:
            True se for ESM-C, False caso contrário
            
        Exemplo:
            >>> ProteinModelFactory.is_esmc_model("esmc-300m-2024-12")
            True
        """
        return model_name in ProteinModelFactory.ESMC_MODELS
    
    @staticmethod
    def is_openfold_model(model_name: str) -> bool:
        """
        Verifica se o modelo é OpenFold.
        
        Args:
            model_name: Nome do modelo
            
        Returns:
            True se for OpenFold, False caso contrário
            
        Exemplo:
            >>> ProteinModelFactory.is_openfold_model("openfold3")
            True
        """
        return model_name in ProteinModelFactory.OPENFOLD_MODELS
    
    @staticmethod
    def is_esm3_model(model_name: str) -> bool:
        """
        Verifica se o modelo é ESM-3.
        
        Args:
            model_name: Nome do modelo
            
        Returns:
            True se for ESM-3, False caso contrário
            
        Exemplo:
            >>> ProteinModelFactory.is_esm3_model("esm3_sm_open_v1")
            False  # Ainda não implementado
        """
        return model_name in ProteinModelFactory.ESM3_MODELS
    
    @staticmethod
    def list_supported_models() -> dict[str, list[str]]:
        """
        Lista todos os modelos suportados por tipo.
        
        Returns:
            Dicionário {tipo: [modelos]}
            
        Exemplo:
            >>> factory = ProteinModelFactory()
            >>> models = factory.list_supported_models()
            >>> 'esm2' in models
            True
            >>> len(models['esm2'])
            7
        """
        return {
            'esm2': sorted(ProteinModelFactory.ESM2_MODELS),
            'esmc': sorted(ProteinModelFactory.ESMC_MODELS),
            'openfold': sorted(ProteinModelFactory.OPENFOLD_MODELS),
            'esm3': sorted(ProteinModelFactory.ESM3_MODELS),
        }
