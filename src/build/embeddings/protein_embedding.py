"""
Geração de embeddings de proteínas usando modelos ESM (Meta AI).
"""

import os
from typing import Dict, Any, List, Tuple, Optional, TYPE_CHECKING, Union
import numpy as np
from pathlib import Path

if TYPE_CHECKING:
    from build.core import BuildConfig

from build.embeddings.base_embedding import BaseEmbedding
from build.core.constants import ESM_MODELS, DEFAULT_ESM_MODEL
from build.core.exceptions import DependencyError, EmbeddingError, ModelLoadError
from build.utils import ProgressLogger, ensure_directory

class ProteinEmbedding(BaseEmbedding):
    """Gerador de embeddings de proteínas usando ESM."""
    
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
        self.alphabet = None
        self.batch_converter = None
            
        super().__init__(model_name=model_name, config=config, **kwargs)        # Verificar dependências
        self._check_dependencies()
    
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
            import esm
            self.esm = esm
            self.esm_available = True
        except ImportError:
            raise DependencyError(
                "ESM não está disponível. Instale com: pip install fair-esm"
            )
    
    def _validate_config(self) -> None:
        """Valida configuração específica para embeddings de proteínas."""
        super()._validate_config()
        
        # Validar modelo
        if self.model_name not in ESM_MODELS:
            raise EmbeddingError(f"Modelo ESM inválido: {self.model_name}. Modelos disponíveis: {list(ESM_MODELS.keys())}")
        
        # Verificar configuração de GPU
        if self.use_gpu:
            if not hasattr(self, 'torch') or not self.torch.cuda.is_available():
                self.logger.warning("GPU solicitada mas não disponível. Usando CPU.")
                self.use_gpu = False
    
    def get_supported_models(self) -> Dict[str, Dict[str, Any]]:
        """Retorna modelos ESM suportados."""
        return ESM_MODELS.copy()
    
    def _load_model(self) -> Any:
        """Carrega modelo ESM."""
        self.logger.info(f"Configurando dispositivo...")
        
        # Configurar dispositivo
        if self.use_gpu and self.torch.cuda.is_available():
            self.device = self.torch.device("cuda")
            self.logger.info(f"Usando GPU: {self.torch.cuda.get_device_name()}")
        else:
            self.device = self.torch.device("cpu")
            self.logger.info("Usando CPU")
        
        # Carregar modelo
        try:
            self.logger.info(f"Carregando modelo ESM: {self.model_name}")
            model, alphabet = self.esm.pretrained.load_model_and_alphabet(self.model_name)
            
            # Mover para dispositivo e configurar para avaliação
            model = model.to(self.device).eval()
            
            # Configurar conversor de batch
            self.alphabet = alphabet
            self.batch_converter = alphabet.get_batch_converter()
            
            self.logger.info("Modelo ESM carregado com sucesso")
            return model
            
        except Exception as e:
            # Tentar modelo alternativo se falhar
            if self.model_name != "esm2_t33_650M_UR50D":
                self.logger.warning(f"Falha ao carregar {self.model_name}, tentando modelo alternativo...")
                
                try:
                    alternative_model = "esm2_t33_650M_UR50D"
                    model, alphabet = self.esm.pretrained.load_model_and_alphabet(alternative_model)
                    model = model.to(self.device).eval()
                    
                    self.alphabet = alphabet
                    self.batch_converter = alphabet.get_batch_converter()
                    self.model_name = alternative_model  # Atualizar nome do modelo
                    
                    self.logger.info(f"Modelo alternativo carregado: {alternative_model}")
                    return model
                    
                except Exception as e2:
                    raise ModelLoadError(f"Falha ao carregar modelo ESM: {e2}")
            else:
                raise ModelLoadError(f"Falha ao carregar modelo ESM: {e}")
    
    def _generate_single_embedding(self, sequence: str) -> np.ndarray:
        """
        Gera embedding para uma única sequência de proteína.
        
        Args:
            sequence: Sequência de aminoácidos
            
        Returns:
            Array NumPy com embedding
        """
        if not sequence or not sequence.strip():
            raise EmbeddingError("Sequência vazia")
        
        # Limpar sequência (remover caracteres não-aminoácidos)
        clean_sequence = ''.join(c for c in sequence.upper() if c in 'ACDEFGHIKLMNPQRSTVWY')
        
        if not clean_sequence:
            raise EmbeddingError("Sequência não contém aminoácidos válidos")
        
        if len(clean_sequence) > 1024:  # Limite de sequência para evitar problemas de memória
            self.logger.warning(f"Sequência muito longa ({len(clean_sequence)} aminoácidos), truncando para 1024")
            clean_sequence = clean_sequence[:1024]
        
        try:
            # Preparar dados para o modelo
            data = [("sequence", clean_sequence)]
            batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
            batch_tokens = batch_tokens.to(self.device)
            
            # Gerar embedding
            with self.torch.no_grad():
                results = self.model(
                    batch_tokens, 
                    repr_layers=[self.model.num_layers], 
                    return_contacts=False
                )
                
                # Extrair embedding da última camada (representação CLS)
                # Remove tokens especiais (primeiro e último)
                embedding = results["representations"][self.model.num_layers][0, 1:-1]
                
                # Usar média da sequência como representação final
                sequence_embedding = embedding.mean(dim=0)
                
            return sequence_embedding.cpu().numpy()
            
        except Exception as e:
            raise EmbeddingError(f"Erro ao gerar embedding ESM: {e}")
    
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
