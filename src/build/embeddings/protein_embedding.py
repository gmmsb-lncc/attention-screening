"""
Geração de embeddings de proteínas usando modelos ESM (Meta AI).
Utiliza código fonte local do ESM incluído no repositório.

REFATORADO: Agora usa Strategy Pattern para suportar múltiplos modelos.
"""

import os
import sys
import gc
from typing import Dict, Any, List, Tuple, Optional, TYPE_CHECKING, Union
import numpy as np
from pathlib import Path

# Adicionar ESM local ao path
ESM_LOCAL_PATH = Path(__file__).parent.parent.parent.parent / "ESM"
if str(ESM_LOCAL_PATH) not in sys.path:
    sys.path.insert(0, str(ESM_LOCAL_PATH))

if TYPE_CHECKING:
    from src.build.core import BuildConfig

from src.build.embeddings.base_embedding import BaseEmbedding
from src.build.core.constants import ESM_MODELS, DEFAULT_ESM_MODEL
from src.build.core.exceptions import DependencyError, EmbeddingError, ModelLoadError
from src.build.utils import ProgressLogger, ensure_directory

# Importar factory e estratégias
from src.build.embeddings.factories.protein_model_factory import ProteinModelFactory
from src.build.embeddings.strategies.base_protein_strategy import BaseProteinStrategy

class ProteinEmbedding(BaseEmbedding):
    """
    Gerador de embeddings de proteínas usando ESM.
    
    REFATORADO: Atua como orchestrator que delega para estratégias específicas.
    - Usa ProteinModelFactory para criar estratégias (ESM2Strategy, ESM3Strategy)
    - Delega carregamento e inferência para a estratégia
    - Mantém API pública retrocompatível
    """
    
    def __init__(self, 
                 config: Optional['BuildConfig'] = None,
                 model_name: str = DEFAULT_ESM_MODEL,
                 use_gpu: bool = False,
                 **kwargs):
        """
        Inicializa gerador de embeddings de proteínas.
        
        Args:
            config: Configuração do sistema build
            model_name: Nome do modelo ESM
            use_gpu: Se deve usar GPU quando disponível
            **kwargs: Argumentos adicionais
        """
        # Definir atributos antes da inicialização do pai
        self.use_gpu = use_gpu
        self.device = None
        self.alphabet = None  # Será configurado pela estratégia
        self.batch_converter = None
        self.strategy: Optional[BaseProteinStrategy] = None  # Estratégia de modelo
            
        super().__init__(model_name=model_name, config=config, **kwargs)
        
        # Verificar dependências
        self._check_dependencies()
        
        # Criar estratégia apropriada usando factory
        self._create_strategy()
    
    def _check_dependencies(self) -> None:
        """Verifica se dependências estão disponíveis."""
        try:
            import torch
            self.torch = torch
            self.torch_available = True
        except ImportError:
            raise DependencyError(
                "PyTorch não está disponível. Instale com: pip install torch"
            )
        
        try:
            # Importar ESM do código fonte local
            import esm
            self.esm = esm
            self.esm_available = True
            self.logger.info(f"ESM carregado do código fonte local: {ESM_LOCAL_PATH}")
        except ImportError as e:
            raise DependencyError(
                f"ESM não está disponível no repositório local ({ESM_LOCAL_PATH}). "
                f"Verifique se a pasta ESM/ existe e contém o código fonte. Erro: {e}"
            )
    
    def _create_strategy(self) -> None:
        """
        Cria estratégia apropriada usando factory.
        
        Raises:
            ValueError: Se modelo não for suportado
        """
        try:
            factory = ProteinModelFactory()
            self.strategy = factory.create_strategy(self.model_name)
            self.logger.info(f"✅ Estratégia criada: {self.strategy.__class__.__name__}")
        except ValueError as e:
            raise EmbeddingError(f"Falha ao criar estratégia: {e}")
    
    def _validate_config(self) -> None:
        """Valida configuração específica para embeddings de proteínas."""
        super()._validate_config()
        
        # Validar modelo
        if self.model_name not in ESM_MODELS:
            raise EmbeddingError(f"Modelo ESM inválido: {self.model_name}. Modelos disponíveis: {list(ESM_MODELS.keys())}")
        
        # Verificar configuração de GPU (CUDA ou MPS)
        if self.use_gpu:
            import torch
            has_cuda = torch.cuda.is_available()
            has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
            
            if not (has_cuda or has_mps):
                self.logger.warning("GPU solicitada mas não disponível. Usando CPU.")
                self.use_gpu = False
            elif has_mps and not has_cuda:
                self.logger.info("✨ Usando MPS (Metal Performance Shaders) para aceleração GPU")
    
    def get_supported_models(self) -> Dict[str, Dict[str, Any]]:
        """Retorna modelos ESM suportados."""
        return ESM_MODELS.copy()
    
    def _load_model(self) -> Any:
        """
        Carrega modelo usando estratégia apropriada.
        
        REFATORADO: Delega para strategy.load() ao invés de código direto.
        """
        self.logger.info(f"Configurando dispositivo...")
        
        # Configurar cache local para modelos ESM
        import os
        cache_dir = Path(__file__).parent.parent.parent.parent / "models_cache" / "ESM"
        cache_dir.mkdir(parents=True, exist_ok=True)
        offload_folder = cache_dir / "offload"
        
        # Configurar dispositivo (prioridade: CUDA > MPS > CPU)
        if self.use_gpu:
            if self.torch.cuda.is_available():
                self.device = self.torch.device("cuda")
                self.logger.info(f"Usando GPU CUDA: {self.torch.cuda.get_device_name()}")
            elif hasattr(self.torch.backends, 'mps') and self.torch.backends.mps.is_available():
                self.device = self.torch.device("mps")
                self.logger.info("✨ Usando GPU MPS (Metal Performance Shaders - Apple Silicon)")
            else:
                self.device = self.torch.device("cpu")
                self.logger.info("Usando CPU")
        else:
            self.device = self.torch.device("cpu")
            self.logger.info("Usando CPU")
        
        # Delegar carregamento para estratégia
        model, self.alphabet = self.strategy.load(
            model_name=self.model_name,
            device=self.device,
            offload_folder=str(offload_folder),
            logger=self.logger
        )
        
        # Configurar conversor de batch (compatibilidade retroativa)
        self.batch_converter = self.alphabet.get_batch_converter()
        
        return model
    
    def _generate_single_embedding(self, sequence: str) -> np.ndarray:
        """
        Gera embedding para uma única sequência de proteína.
        
        REFATORADO: Delega para strategy.generate() ao invés de código direto.
        
        Args:
            sequence: Sequência de aminoácidos
            
        Returns:
            Array NumPy com embedding
        """
        # Delegar geração para estratégia
        return self.strategy.generate(
            model=self.model,
            auxiliary_objects=self.alphabet,
            sequence=sequence,
            device=self.device,
            logger=self.logger
        )
    
    def process_fasta_file(self,
                          fasta_file: Path,
                          output_dir: Path,
                          batch_size: Optional[int] = None) -> Tuple[int, int]:
        """
        Processa arquivo FASTA com sequências de proteínas.
        
        Args:
            fasta_file: Arquivo FASTA de entrada
            output_dir: Diretório de saída
            batch_size: Tamanho do batch
            
        Returns:
            Tupla (sucessos, falhas)
        """
        sequences = self._read_fasta_file(fasta_file)
        
        if not sequences:
            raise EmbeddingError(f"Nenhuma sequência encontrada em {fasta_file}")
        
        self.logger.info(f"Processando {len(sequences)} sequências de {fasta_file}")
        
        # Preparar saída
        output_path = ensure_directory(output_dir)
        
        sucessos = 0
        falhas = 0
        
        # Progress logger
        progress_logger = ProgressLogger(
            self.logger,
            len(sequences),
            "Processando sequências FASTA"
        )
        
        for seq_id, sequence in sequences:
            try:
                # Verificar se já existe
                output_file = output_path / f"{seq_id}_embedding.npy"
                if output_file.exists():
                    self.logger.debug(f"Embedding já existe, pulando: {seq_id}")
                    sucessos += 1
                    progress_logger.update()
                    continue
                
                # Gerar embedding
                embedding = self.generate_embedding(sequence)
                
                # Salvar
                np.save(output_file, embedding)
                sucessos += 1
                
            except Exception as e:
                self.logger.error(f"Erro ao processar sequência {seq_id}: {e}")
                falhas += 1
            
            progress_logger.update()
        
        progress_logger.finish()
        self.logger.info(f"Processamento FASTA concluído: {sucessos} sucessos, {falhas} falhas")
        
        return sucessos, falhas
    
    def _read_fasta_file(self, fasta_file: Path) -> List[Tuple[str, str]]:
        """
        Lê arquivo FASTA e retorna lista de (ID, sequência).
        
        Args:
            fasta_file: Arquivo FASTA
            
        Returns:
            Lista de tuplas (ID, sequência)
        """
        sequences = []
        current_id = None
        current_sequence = []
        
        try:
            with open(fasta_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    
                    if line.startswith('>'):
                        # Salvar sequência anterior se existir
                        if current_id and current_sequence:
                            sequences.append((current_id, ''.join(current_sequence)))
                        
                        # Nova sequência
                        current_id = line[1:].split()[0]  # Primeira palavra após >
                        current_sequence = []
                        
                    elif line and current_id:
                        current_sequence.append(line)
                
                # Salvar última sequência
                if current_id and current_sequence:
                    sequences.append((current_id, ''.join(current_sequence)))
                    
        except Exception as e:
            raise EmbeddingError(f"Erro ao ler arquivo FASTA {fasta_file}: {e}")
        
        return sequences
    
    def process_sequence_directory(self,
                                  seq_input_dir: Path,
                                  output_dir: Path,
                                  batch_size: Optional[int] = None) -> Tuple[int, int]:
        """
        Processa diretório com arquivos de sequências.
        
        Args:
            seq_input_dir: Diretório com arquivos de sequência
            output_dir: Diretório de saída
            batch_size: Tamanho do batch
            
        Returns:
            Tupla (sucessos, falhas)
        """
        seq_dir = Path(seq_input_dir)
        
        if not seq_dir.exists():
            raise EmbeddingError(f"Diretório não encontrado: {seq_input_dir}")
        
        # Encontrar arquivos de sequência
        sequence_files = list(seq_dir.glob("*.fasta")) + list(seq_dir.glob("*.txt"))
        
        if not sequence_files:
            raise EmbeddingError(f"Nenhum arquivo de sequência encontrado em {seq_input_dir}")
        
        self.logger.info(f"Encontrados {len(sequence_files)} arquivos de sequência")
        
        total_sucessos = 0
        total_falhas = 0
        
        for seq_file in sequence_files:
            try:
                if seq_file.suffix.lower() == '.fasta':
                    sucessos, falhas = self.process_fasta_file(seq_file, output_dir, batch_size)
                else:
                    # Processar como arquivo de texto simples
                    sucessos, falhas = self._process_text_file(seq_file, output_dir)
                
                total_sucessos += sucessos
                total_falhas += falhas
                
            except Exception as e:
                self.logger.error(f"Erro ao processar arquivo {seq_file}: {e}")
                total_falhas += 1
        
        return total_sucessos, total_falhas
    
    def _process_text_file(self, text_file: Path, output_dir: Path) -> Tuple[int, int]:
        """Processa arquivo de texto simples com sequência."""
        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Juntar linhas que não começam com >
            sequence = ''.join(line.strip() for line in lines if not line.startswith('>'))
            
            if not sequence:
                self.logger.warning(f"Arquivo vazio ou sem sequência: {text_file}")
                return 0, 1
            
            # Nome do arquivo sem extensão
            seq_id = text_file.stem
            
            # Gerar embedding
            embedding = self.generate_embedding(sequence)
            
            # Salvar
            output_path = ensure_directory(output_dir)
            output_file = output_path / f"{seq_id}_embedding.npy"
            np.save(output_file, embedding)
            
            return 1, 0
            
        except Exception as e:
            self.logger.error(f"Erro ao processar arquivo de texto {text_file}: {e}")
            return 0, 1
    
    def get_memory_usage_estimate(self, sequence_length: int) -> float:
        """
        Estima uso de memória para sequência.
        
        Args:
            sequence_length: Comprimento da sequência
            
        Returns:
            Estimativa de uso em GB
        """
        # Estimativa baseada no modelo e comprimento da sequência
        base_model_memory = 3.0  # GB para modelo base
        sequence_memory = sequence_length * 0.001  # Aproximação
        
        return base_model_memory + sequence_memory
    
    def generate_embeddings(self, 
                          tsv_path: Path, 
                          output_dir: Optional[Path] = None) -> bool:
        """
        Gera embeddings a partir de arquivo TSV (interface para pipeline).
        
        Args:
            tsv_path: Arquivo TSV com dados
            output_dir: Diretório de saída (usa config se None)
            
        Returns:
            True se sucesso
        """
        import pandas as pd
        from src.build.utils import ensure_directory
        
        try:
            # Garantir que modelo está inicializado
            if not self._model_loaded:
                self.logger.info("Inicializando modelo ESM...")
                self._do_initialize()
            
            # Determinar diretório de saída
            if output_dir is None:
                output_dir = Path(self.get_config('protein_output_dir', 'protein_embeddings'))
            
            output_dir = Path(output_dir)
            output_dir = ensure_directory(output_dir)
            
            # Carregar TSV
            self.logger.info(f"Carregando dados de {tsv_path}")
            df = pd.read_csv(tsv_path, sep='\t')
            
            # Verificar colunas obrigatórias
            if 'seq_id' not in df.columns or 'seq' not in df.columns:
                raise EmbeddingError("TSV deve conter colunas 'seq_id' e 'seq'")
            
            # Obter sequências únicas
            unique_seqs = df.groupby('seq_id')['seq'].first()
            self.logger.info(f"Processando {len(unique_seqs)} sequências únicas")
            
            # Processar cada sequência
            sucessos = 0
            falhas = 0
            
            progress_logger = ProgressLogger(
                self.logger,
                len(unique_seqs),
                "Gerando embeddings de proteínas"
            )
            
            for seq_id, sequence in unique_seqs.items():
                try:
                    # Verificar se já existe
                    output_file = output_dir / f"{seq_id}_embedding.npy"
                    if output_file.exists():
                        self.logger.debug(f"Embedding já existe: {seq_id}")
                        sucessos += 1
                        progress_logger.update()
                        continue
                    
                    # Gerar embedding
                    embedding = self.generate_embedding(sequence)
                    
                    # Salvar
                    np.save(output_file, embedding)
                    sucessos += 1
                    
                except Exception as e:
                    self.logger.error(f"Erro ao processar {seq_id}: {e}")
                    falhas += 1
                
                progress_logger.update()
            
            progress_logger.finish()
            
            # Salvar path de saída
            self._output_path = output_dir
            
            self.logger.info(f"Embeddings de proteínas: {sucessos} sucessos, {falhas} falhas")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao gerar embeddings: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def get_output_path(self) -> Optional[Path]:
        """Retorna path de saída dos embeddings."""
        return getattr(self, '_output_path', None)
    
    def get_embeddings_info(self) -> Dict[str, Any]:
        """Retorna informações sobre embeddings gerados."""
        output_path = self.get_output_path()
        
        if output_path and output_path.exists():
            embedding_files = list(output_path.glob("*_embedding.npy"))
            return {
                'output_path': str(output_path),
                'count': len(embedding_files),
                'model': self.model_name,
                'dimension': self.embedding_dim
            }
        
        return {
            'output_path': None,
            'count': 0,
            'model': self.model_name,
            'dimension': self.embedding_dim
        }
    
    def build(self) -> Dict[str, Any]:
        """Constrói resumo do processamento."""
        result = super().build()
        
        result.update({
            'device': str(self.device) if self.device else None,
            'use_gpu': self.use_gpu,
            'torch_available': self.torch_available,
            'esm_available': self.esm_available,
        })
        
        return result
    
    def __del__(self):
        """
        Cleanup ao destruir objeto.
        
        REFATORADO: Delega para strategy.cleanup() ao invés de código direto.
        """
        if hasattr(self, 'strategy') and self.strategy:
            try:
                self.strategy.cleanup(self.model, self.alphabet)
            except:
                pass  # Ignorar erros no destrutor
