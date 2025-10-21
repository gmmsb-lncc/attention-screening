"""
Geração de embeddings de ligantes usando modelos FM4M (IBM).
"""

import os
import sys
import time
import random
from typing import Dict, Any, List, Tuple, Optional, Union, TYPE_CHECKING
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

if TYPE_CHECKING:
    from build.core import BuildConfig

from build.embeddings.base_embedding import BaseEmbedding
from build.core.constants import FM4M_MODELS
from build.core.exceptions import DependencyError, EmbeddingError
from build.utils import ProgressLogger, ensure_directory, optimize_batch_size, memory_monitor

class LigandEmbedding(BaseEmbedding):
    """Gerador de embeddings de ligantes usando FM4M."""
    
    def __init__(self, 
                 config: Optional['BuildConfig'] = None,
                 model_name: str = "SMI-TED",
                 use_parallel: bool = True,
                 checkpoint_enabled: bool = True,
                 **kwargs):
        """
        Inicializa gerador de embeddings de ligantes.
        
        Args:
            config: Configuração do sistema build
            model_name: Nome do modelo FM4M
            use_parallel: Se deve usar processamento paralelo
            checkpoint_enabled: Se deve usar sistema de checkpoint
            **kwargs: Argumentos adicionais
        """
        # Definir atributos antes da inicialização do pai
        self.use_parallel = use_parallel
        self.checkpoint_enabled = checkpoint_enabled
        self.processed_files = set()
        self.checkpoint_file = None
        
        # Cache do modelo SMI-TED
        self._smited_model = None
        self._smited_model_loaded = False
            
        super().__init__(model_name=model_name, config=config, **kwargs)
        
        # Verificar dependências
        self._check_dependencies()
        self._setup_fm4m_path()
    
    def _check_dependencies(self) -> None:
        """Verifica se dependências estão disponíveis."""
        try:
            import pandas as pd
            self.pd = pd
        except ImportError:
            raise DependencyError("Pandas não disponível. Instale com: pip install pandas")
        
        # FM4M será verificado durante inicialização
        self.fm4m = None
        self.fm4m_available = False
    
    def _validate_config(self) -> None:
        """Valida configuração específica para embeddings de ligantes."""
        super()._validate_config()
        
        # Validar modelo FM4M
        if self.model_name not in FM4M_MODELS:
            raise EmbeddingError(f"Modelo FM4M inválido: {self.model_name}. Modelos disponíveis: {list(FM4M_MODELS.keys())}")
        
        # Verificar configuração de processamento paralelo
        if self.use_parallel:
            try:
                import multiprocessing
                cpu_count = multiprocessing.cpu_count()
                if cpu_count < 2:
                    self.logger.warning("Processamento paralelo solicitado mas apenas 1 CPU disponível")
                    self.use_parallel = False
            except Exception:
                self.logger.warning("Não foi possível verificar CPUs disponíveis. Desabilitando processamento paralelo.")
                self.use_parallel = False
    
    def _setup_fm4m_path(self) -> None:
        """Configura caminho para FM4M."""
        # Adicionar FM4M ao path se necessário
        current_dir = Path(__file__).parent.parent.parent.parent  # Volta para raiz
        fm4m_path = current_dir / "FM4M"
        
        if fm4m_path.exists():
            fm4m_str = str(fm4m_path)
            models_path = str(fm4m_path / "models")
            
            if fm4m_str not in sys.path:
                sys.path.append(fm4m_str)
            if models_path not in sys.path:
                sys.path.append(models_path)
    
    def get_supported_models(self) -> Dict[str, Dict[str, Any]]:
        """Retorna modelos FM4M suportados."""
        return FM4M_MODELS.copy()
    
    def _load_model(self) -> Any:
        """Carrega módulo FM4M e pré-carrega o modelo SMI-TED."""
        try:
            import models.fm4m as fm4m
            self.fm4m = fm4m
            self.fm4m_available = True
            
            self.logger.info(f"Módulo FM4M carregado para modelo: {self.model_name}")
            
            # Pré-carregar modelo SMI-TED se for o modelo escolhido
            if self.model_name == "SMI-TED":
                import os
                import torch
                from models.smi_ted.smi_ted_light.load import load_smi_ted
                
                # Localizar arquivos do modelo
                materials_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
                model_files_path = os.path.join(materials_path, "FM4M", "model_files")
                
                self.logger.info(f"Pré-carregando modelo SMI-TED de: {model_files_path}")
                
                # Verificar se arquivos existem
                vocab_file = os.path.join(model_files_path, "bert_vocab_curated.txt")
                ckpt_file = os.path.join(model_files_path, "smi-ted-Light_40.pt")
                
                if not os.path.exists(vocab_file) or not os.path.exists(ckpt_file):
                    raise DependencyError(
                        f"SMI-TED model files not found at {model_files_path}. "
                        f"Please run: cd FM4M && python download_model_files.py"
                    )
                
                # Carregar modelo UMA VEZ e cachear
                self._smited_model = load_smi_ted(folder=model_files_path, ckpt_filename='smi-ted-Light_40.pt')
                self._smited_model_loaded = True
                self.logger.info("✅ Modelo SMI-TED pré-carregado e cacheado!")
            else:
                self._smited_model = None
                self._smited_model_loaded = False
            
            return fm4m
            
        except ImportError as e:
            raise DependencyError(
                f"FM4M não está disponível: {e}. "
                "Certifique-se de que o diretório FM4M está presente e as dependências instaladas."
            )
    
    def _do_initialize(self) -> None:
        """Inicialização específica de ligantes."""
        super()._do_initialize()
        
        # Configurar checkpoint se habilitado
        if self.checkpoint_enabled:
            self.checkpoint_file = Path(self.get_config('base_dir', '.')) / 'processed_ligands.log'
            self._load_checkpoint()
    
    def _load_checkpoint(self) -> None:
        """Carrega arquivos já processados do checkpoint."""
        if self.checkpoint_file and self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    self.processed_files = {line.strip() for line in f}
                self.logger.info(f"Checkpoint carregado: {len(self.processed_files)} arquivos já processados")
            except Exception as e:
                self.logger.warning(f"Erro ao carregar checkpoint: {e}")
                self.processed_files = set()
    
    def _update_checkpoint(self, file_name: str) -> None:
        """Atualiza checkpoint com arquivo processado."""
        if self.checkpoint_enabled and self.checkpoint_file:
            try:
                with open(self.checkpoint_file, 'a') as f:
                    f.write(f"{file_name}\\n")
                self.processed_files.add(file_name)
            except Exception as e:
                self.logger.warning(f"Erro ao atualizar checkpoint: {e}")
    
    def _generate_single_embedding(self, smiles: str) -> np.ndarray:
        """
        Gera embedding para uma única string SMILES.
        
        Args:
            smiles: String SMILES do ligante
            
        Returns:
            Array NumPy com embedding
        """
        return self._generate_batch_embeddings([smiles])[0]
    
    def _generate_batch_embeddings(self, smiles_list: List[str]) -> List[np.ndarray]:
        """
        Gera embeddings para batch de SMILES com retry logic.
        
        Args:
            smiles_list: Lista de strings SMILES
            
        Returns:
            Lista de arrays NumPy com embeddings
        """
        if not self.fm4m_available:
            raise EmbeddingError("FM4M não está carregado")
        
        # Filtrar SMILES vazios
        valid_smiles = [s for s in smiles_list if s and s.strip()]
        if not valid_smiles:
            raise EmbeddingError("Nenhum SMILES válido fornecido")
        
        try:
            # Se modelo está cacheado (SMI-TED), usar diretamente
            if self._smited_model_loaded and self.model_name == "SMI-TED":
                import torch
                with torch.no_grad():
                    # Usar modelo cacheado ao invés de recarregar
                    representations = self._smited_model.encode(valid_smiles, return_torch=False)
                    
                    # Converter para lista de arrays numpy
                    if hasattr(representations, 'values'):
                        embeddings = [representations.values[i] for i in range(len(representations.values))]
                    else:
                        embeddings = [np.array(row) for row in representations]
                    
                    return embeddings
            else:
                # Fallback: usar método original (para outros modelos)
                representations = self._get_representation_with_retry(valid_smiles)
                
                # Converter para lista de arrays numpy
                if hasattr(representations, 'values'):
                    embeddings = [representations.values[i] for i in range(len(representations.values))]
                else:
                    embeddings = [np.array(row) for row in representations]
                
                return embeddings
            
        except Exception as e:
            raise EmbeddingError(f"Erro ao gerar embeddings FM4M: {e}")
    
    def _get_representation_with_retry(self, smiles_list: List[str], max_retries: int = 3):
        """Gera representações com retry logic para rate limiting."""
        
        def _get_representation():
            return self.fm4m.get_representation(
                train_data=smiles_list,
                test_data=smiles_list,  # Reutilizar para evitar erro de test_data vazio
                model_type=self.model_name,
                return_tensor=False
            )
        
        for attempt in range(max_retries):
            try:
                return _get_representation()
                
            except Exception as e:
                if "429" in str(e) or "Too Many Requests" in str(e):
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) + random.uniform(0, 1)
                        self.logger.warning(f"Rate limited. Aguardando {wait_time:.2f}s antes de tentar novamente...")
                        time.sleep(wait_time)
                    else:
                        raise EmbeddingError(f"Rate limit excedido após {max_retries} tentativas")
                else:
                    raise e
    
    @memory_monitor(threshold_percent=85.0)
    def generate_batch_embeddings(self, 
                                 input_list: List[str],
                                 batch_size: Optional[int] = None,
                                 show_progress: bool = True) -> List[np.ndarray]:
        """
        Sobrescreve método da classe base para usar implementação FM4M.
        
        Args:
            input_list: Lista de SMILES
            batch_size: Tamanho do batch
            show_progress: Se deve mostrar progresso
            
        Returns:
            Lista de embeddings
        """
        if not self.fm4m_available:
            raise EmbeddingError("FM4M não está carregado")
        
        if not input_list:
            return []
        
        # Otimizar batch size
        if batch_size is None:
            batch_size = self.get_config('batch_size', 32)
        batch_size = optimize_batch_size(batch_size)
        
        embeddings = []
        total_items = len(input_list)
        
        # Logger de progresso
        if show_progress:
            progress_logger = ProgressLogger(
                self.logger,
                total_items,
                f"Gerando embeddings FM4M ({self.model_name})"
            )
        
        # Processar em batches
        for i in range(0, total_items, batch_size):
            batch = input_list[i:i + batch_size]
            
            try:
                batch_embeddings = self._generate_batch_embeddings(batch)
                embeddings.extend(batch_embeddings)
                
                if show_progress:
                    progress_logger.update(len(batch))
                    
            except Exception as e:
                self.logger.error(f"Erro no batch {i//batch_size + 1}: {e}")
                # Adicionar embeddings zero para manter consistência
                zero_embeddings = [np.zeros(self.embedding_dim) for _ in batch]
                embeddings.extend(zero_embeddings)
                
                if show_progress:
                    progress_logger.update(len(batch))
        
        if show_progress:
            progress_logger.finish()
        
        return embeddings
    
    def process_smi_file(self, 
                        smi_file: Path,
                        output_dir: Path,
                        batch_size: Optional[int] = None) -> Tuple[int, int]:
        """
        Processa arquivo .smi com SMILES.
        
        Args:
            smi_file: Arquivo .smi de entrada
            output_dir: Diretório de saída
            batch_size: Tamanho do batch
            
        Returns:
            Tupla (sucessos, falhas)
        """
        file_name = smi_file.name
        
        # Verificar checkpoint
        if self.checkpoint_enabled and file_name in self.processed_files:
            self.logger.info(f"Arquivo já processado (checkpoint): {file_name}")
            return 1, 0  # Assumir sucesso
        
        try:
            # Ler SMILES
            with open(smi_file, 'r') as f:
                smiles_list = [line.strip() for line in f if line.strip()]
            
            if not smiles_list:
                self.logger.warning(f"Arquivo vazio: {file_name}")
                return 0, 1
            
            self.logger.info(f"Processando {len(smiles_list)} SMILES de {file_name}")
            
            # Gerar embeddings
            embeddings = self.generate_batch_embeddings(smiles_list, batch_size)
            
            # Preparar saída
            output_path = ensure_directory(output_dir)
            output_file = output_path / f"{smi_file.stem}.npy"
            
            # Converter para array numpy e salvar
            embeddings_array = np.array(embeddings)
            np.save(output_file, embeddings_array)
            
            # Atualizar checkpoint
            self._update_checkpoint(file_name)
            
            self.logger.info(f"Embeddings salvos: {output_file}")
            return 1, 0
            
        except Exception as e:
            self.logger.error(f"Erro ao processar {file_name}: {e}")
            return 0, 1
    
    def process_smiles_directory(self,
                               input_dir: Path,
                               output_dir: Path,
                               use_parallel: Optional[bool] = None,
                               max_workers: Optional[int] = None) -> Tuple[int, int]:
        """
        Processa diretório com arquivos .smi.
        
        Args:
            input_dir: Diretório de entrada
            output_dir: Diretório de saída
            use_parallel: Se deve usar processamento paralelo
            max_workers: Número máximo de workers
            
        Returns:
            Tupla (sucessos, falhas)
        """
        input_path = Path(input_dir)
        
        if not input_path.exists():
            raise EmbeddingError(f"Diretório não encontrado: {input_dir}")
        
        # Encontrar arquivos .smi
        smi_files = list(input_path.glob("*.smi"))
        
        if not smi_files:
            raise EmbeddingError(f"Nenhum arquivo .smi encontrado em {input_dir}")
        
        self.logger.info(f"Encontrados {len(smi_files)} arquivos .smi")
        
        # Usar configuração de classe se não especificado
        if use_parallel is None:
            use_parallel = self.use_parallel
        
        total_sucessos = 0
        total_falhas = 0
        
        if use_parallel and len(smi_files) > 1:
            # Processamento paralelo
            if max_workers is None:
                max_workers = min(os.cpu_count(), len(smi_files))
            
            self.logger.info(f"Processamento paralelo com {max_workers} workers")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit todos os jobs
                futures = []
                for smi_file in smi_files:
                    future = executor.submit(self.process_smi_file, smi_file, output_dir)
                    futures.append(future)
                
                # Coletar resultados
                for future in futures:
                    try:
                        sucessos, falhas = future.result()
                        total_sucessos += sucessos
                        total_falhas += falhas
                    except Exception as e:
                        self.logger.error(f"Erro em worker paralelo: {e}")
                        total_falhas += 1
        else:
            # Processamento sequencial
            self.logger.info("Processamento sequencial")
            
            for smi_file in smi_files:
                sucessos, falhas = self.process_smi_file(smi_file, output_dir)
                total_sucessos += sucessos
                total_falhas += falhas
        
        self.logger.info(f"Processamento de diretório concluído: {total_sucessos} sucessos, {total_falhas} falhas")
        return total_sucessos, total_falhas
    
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
        from build.utils import ensure_directory
        
        try:
            # Garantir que modelo está inicializado
            if not self._model_loaded:
                self.logger.info("Inicializando modelo FM4M...")
                self._do_initialize()
            
            # Determinar diretório de saída
            if output_dir is None:
                output_dir = Path(self.get_config('ligand_output_dir', 'ligand_embeddings'))
            
            output_dir = Path(output_dir)
            output_dir = ensure_directory(output_dir)
            
            # Carregar TSV
            self.logger.info(f"Carregando dados de {tsv_path}")
            df = pd.read_csv(tsv_path, sep='\t')
            
            # Verificar colunas obrigatórias
            if 'chembl_id' not in df.columns or 'canonical_smiles' not in df.columns:
                raise EmbeddingError("TSV deve conter colunas 'chembl_id' e 'canonical_smiles'")
            
            # Obter SMILES únicos
            unique_smiles = df.groupby('chembl_id')['canonical_smiles'].first()
            self.logger.info(f"Processando {len(unique_smiles)} ligantes únicos")
            
            # Processar cada ligante
            sucessos = 0
            falhas = 0
            
            progress_logger = ProgressLogger(
                self.logger,
                len(unique_smiles),
                "Gerando embeddings de ligantes"
            )
            
            for chembl_id, smiles in unique_smiles.items():
                try:
                    # Verificar se já existe
                    output_file = output_dir / f"{chembl_id}_embedding.npy"
                    if output_file.exists():
                        self.logger.debug(f"Embedding já existe: {chembl_id}")
                        sucessos += 1
                        progress_logger.update()
                        continue
                    
                    # Gerar embedding
                    embedding = self.generate_embedding(smiles)
                    
                    # Salvar
                    np.save(output_file, embedding)
                    sucessos += 1
                    
                except Exception as e:
                    self.logger.error(f"Erro ao processar {chembl_id}: {e}")
                    falhas += 1
                
                progress_logger.update()
            
            progress_logger.finish()
            
            # Salvar path de saída
            self._output_path = output_dir
            
            self.logger.info(f"Embeddings de ligantes: {sucessos} sucessos, {falhas} falhas")
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
            'fm4m_available': self.fm4m_available,
            'use_parallel': self.use_parallel,
            'checkpoint_enabled': self.checkpoint_enabled,
            'processed_files_count': len(self.processed_files),
        })
        
        return result
