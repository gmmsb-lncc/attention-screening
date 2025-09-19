"""
Interface base para geração de embeddings.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union, Optional, Tuple
import numpy as np
from pathlib import Path

from ..core.base_builder import BaseBuilder
from ..core.exceptions import EmbeddingError, ModelLoadError, DependencyError
from ..utils import ProgressLogger, memory_monitor

class BaseEmbedding(BaseBuilder):
    """Classe base abstrata para geração de embeddings."""
    
    def __init__(self, model_name: str = None, config=None, **kwargs):
        """
        Inicializa gerador de embeddings.
        
        Args:
            model_name: Nome do modelo a usar
            config: Configuração do build
            **kwargs: Argumentos de configuração
        """
        # Garantir que temos um model_name ANTES de chamar super
        if model_name is None:
            model_name = "default"
        
        self.model_name = model_name
        self.model = None
        self.embedding_dim = None
        self._model_loaded = False
        
        # Agora chamar super com tudo definido
        super().__init__(config=config, **kwargs)
    
    def _validate_config(self) -> None:
        """Valida configuração específica de embeddings."""
        if not self.model_name:
            raise EmbeddingError("Nome do modelo é obrigatório")
    
    def build(self) -> Any:
        """
        Build embeddings.
        
        Returns:
            Generated embeddings or model
        """
        try:
            return self._load_model()
        except Exception as e:
            self.logger.error(f"Embedding build failed: {e}")
            raise
    
    @abstractmethod
    def _load_model(self) -> Any:
        """
        Carrega modelo específico.
        Deve ser implementado por cada subclasse.
        
        Returns:
            Modelo carregado
        """
        pass
    
    @abstractmethod
    def _generate_single_embedding(self, input_data: str) -> np.ndarray:
        """
        Gera embedding para uma única entrada.
        
        Args:
            input_data: Dados de entrada (sequência, SMILES, etc.)
            
        Returns:
            Array NumPy com embedding
        """
        pass
    
    @abstractmethod
    def get_supported_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Retorna modelos suportados.
        
        Returns:
            Dicionário com modelos e suas propriedades
        """
        pass
    
    def _do_initialize(self) -> None:
        """Inicialização específica de embeddings."""
        super()._do_initialize()
        
        # Verificar se modelo é suportado
        supported_models = self.get_supported_models()
        if self.model_name not in supported_models:
            raise EmbeddingError(
                f"Modelo não suportado: {self.model_name}. "
                f"Disponíveis: {list(supported_models.keys())}"
            )
        
        # Carregar modelo
        try:
            self.logger.info(f"Carregando modelo: {self.model_name}")
            self.model = self._load_model()
            self.embedding_dim = supported_models[self.model_name]['dim']
            self._model_loaded = True
            self.logger.info(f"Modelo carregado - Dimensão: {self.embedding_dim}")
            
        except Exception as e:
            raise ModelLoadError(f"Erro ao carregar modelo {self.model_name}: {e}")
    
    def _do_cleanup(self) -> None:
        """Limpeza específica de embeddings."""
        if self.model is not None:
            try:
                # Tentar limpar modelo da memória
                del self.model
                self.model = None
                self._model_loaded = False
                
                # Forçar garbage collection
                import gc
                gc.collect()
                
                self.logger.info("Modelo removido da memória")
            except Exception as e:
                self.logger.warning(f"Erro na limpeza do modelo: {e}")
        
        super()._do_cleanup()
    
    def generate_embedding(self, input_data: str) -> np.ndarray:
        """
        Gera embedding para entrada individual.
        
        Args:
            input_data: Dados de entrada
            
        Returns:
            Array NumPy com embedding
        """
        if not self._model_loaded:
            raise EmbeddingError("Modelo não carregado. Execute initialize() primeiro.")
        
        if not input_data or not input_data.strip():
            raise EmbeddingError("Dados de entrada vazios")
        
        try:
            return self._generate_single_embedding(input_data.strip())
        except Exception as e:
            raise EmbeddingError(f"Erro ao gerar embedding: {e}")
    
    @memory_monitor(threshold_percent=85.0)
    def generate_batch_embeddings(self, 
                                 input_list: List[str],
                                 batch_size: Optional[int] = None,
                                 show_progress: bool = True) -> List[np.ndarray]:
        """
        Gera embeddings para múltiplas entradas.
        
        Args:
            input_list: Lista de dados de entrada
            batch_size: Tamanho do batch (usa configuração se None)
            show_progress: Se deve mostrar progresso
            
        Returns:
            Lista de arrays NumPy com embeddings
        """
        if not self._model_loaded:
            raise EmbeddingError("Modelo não carregado. Execute initialize() primeiro.")
        
        if not input_list:
            return []
        
        # Usar batch size da configuração se não especificado
        if batch_size is None:
            batch_size = self.get_config('batch_size', 32)
        
        # Otimizar batch size baseado na memória disponível
        from ..utils import optimize_batch_size
        batch_size = optimize_batch_size(batch_size)
        
        embeddings = []
        total_items = len(input_list)
        
        # Logger de progresso
        if show_progress:
            progress_logger = ProgressLogger(
                self.logger, 
                total_items, 
                f"Gerando embeddings ({self.model_name})"
            )
        
        try:
            # Processar em batches
            for i in range(0, total_items, batch_size):
                batch = input_list[i:i + batch_size]
                
                # Gerar embeddings do batch
                batch_embeddings = []
                for item in batch:
                    try:
                        embedding = self.generate_embedding(item)
                        batch_embeddings.append(embedding)
                    except Exception as e:
                        self.logger.warning(f"Erro ao gerar embedding para item {i}: {e}")
                        # Usar embedding zero em caso de erro
                        zero_embedding = np.zeros(self.embedding_dim)
                        batch_embeddings.append(zero_embedding)
                
                embeddings.extend(batch_embeddings)
                
                # Atualizar progresso
                if show_progress:
                    progress_logger.update(len(batch))
                
                # Limpeza de memória entre batches
                if i % (batch_size * 10) == 0:  # A cada 10 batches
                    import gc
                    gc.collect()
            
            if show_progress:
                progress_logger.finish()
            
            return embeddings
            
        except Exception as e:
            raise EmbeddingError(f"Erro no processamento em batch: {e}")
    
    def process_file(self, 
                    input_file: Union[str, Path],
                    output_dir: Union[str, Path],
                    id_column: str = 'id',
                    data_column: str = 'sequence',
                    batch_size: Optional[int] = None) -> Tuple[int, int]:
        """
        Processa arquivo com dados para embeddings.
        
        Args:
            input_file: Arquivo de entrada (TSV)
            output_dir: Diretório de saída
            id_column: Nome da coluna com IDs
            data_column: Nome da coluna com dados
            batch_size: Tamanho do batch
            
        Returns:
            Tupla (sucessos, falhas)
        """
        from ..utils import load_tsv, ensure_directory, save_numpy
        
        # Carregar dados
        try:
            df = load_tsv(input_file)
            self.logger.info(f"Carregados {len(df)} registros de {input_file}")
        except Exception as e:
            raise EmbeddingError(f"Erro ao carregar arquivo {input_file}: {e}")
        
        # Verificar colunas obrigatórias
        if id_column not in df.columns:
            raise EmbeddingError(f"Coluna '{id_column}' não encontrada")
        if data_column not in df.columns:
            raise EmbeddingError(f"Coluna '{data_column}' não encontrada")
        
        # Preparar saída
        output_path = ensure_directory(output_dir)
        
        # Gerar embeddings
        data_list = df[data_column].tolist()
        id_list = df[id_column].tolist()
        
        embeddings = self.generate_batch_embeddings(
            data_list, 
            batch_size=batch_size
        )
        
        # Salvar embeddings individuais
        sucessos = 0
        falhas = 0
        
        for embedding_id, embedding in zip(id_list, embeddings):
            try:
                output_file = output_path / f"{embedding_id}.npy"
                save_numpy(embedding, output_file)
                sucessos += 1
            except Exception as e:
                self.logger.error(f"Erro ao salvar embedding {embedding_id}: {e}")
                falhas += 1
        
        self.logger.info(f"Processamento concluído: {sucessos} sucessos, {falhas} falhas")
        return sucessos, falhas
    
    def is_model_loaded(self) -> bool:
        """Verifica se modelo está carregado."""
        return self._model_loaded
    
    def get_embedding_dimension(self) -> Optional[int]:
        """Obtém dimensão dos embeddings."""
        return self.embedding_dim
    
    def get_model_info(self) -> Dict[str, Any]:
        """Obtém informações do modelo atual."""
        supported_models = self.get_supported_models()
        if self.model_name in supported_models:
            return supported_models[self.model_name].copy()
        return {}
    
    def build(self) -> Dict[str, Any]:
        """
        Implementa método build da classe base.
        Para embeddings, retorna informações do modelo.
        """
        return {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'model_loaded': self._model_loaded,
            'model_info': self.get_model_info()
        }
