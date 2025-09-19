"""
Interface base para construção de matrizes de embeddings.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from pathlib import Path

from ..core.base_builder import BaseBuilder
from ..core.exceptions import MatrixError, BuildFileNotFoundError
from ..utils import ProgressLogger, memory_monitor, load_numpy, save_numpy, load_tsv

class BaseMatrix(BaseBuilder):
    """Classe base abstrata para construção de matrizes."""
    
    def __init__(self, 
                 original_tsv_path: str = None,
                 output_dir: str = "concatenated_embeddings",
                 config=None,
                 **kwargs):
        """
        Inicializa construtor de matriz.
        
        Args:
            original_tsv_path: Caminho para arquivo TSV original
            output_dir: Diretório de saída
            config: Configuração do build
            **kwargs: Argumentos de configuração
        """
        # Garantir que temos um caminho TSV ANTES de chamar super
        if original_tsv_path is None:
            original_tsv_path = "/dev/null"
        
        self.original_tsv_path = Path(original_tsv_path)
        self.output_dir = Path(output_dir)
        
        # Estado da matriz
        self.matrix = None
        self.matrix_shape = None
        self.original_data = None
        
        # Agora chamar super com tudo definido
        super().__init__(config=config, **kwargs)
    
    def _validate_config(self) -> None:
        """Valida configuração de matriz."""
        if not self.original_tsv_path.exists():
            self.logger.warning(f"Arquivo TSV não encontrado: {self.original_tsv_path}")
    
    def build(self) -> np.ndarray:
        """
        Build the matrix.
        
        Returns:
            Constructed matrix
        """
        try:
            return self._build_matrix()
        except Exception as e:
            self.logger.error(f"Matrix build failed: {e}")
            raise
    
    def _do_initialize(self) -> None:
        """Inicialização específica de matriz."""
        super()._do_initialize()
        
        # Carregar dados originais
        try:
            self.original_data = load_tsv(self.original_tsv_path)
            self.logger.info(f"Carregados {len(self.original_data)} registros de {self.original_tsv_path}")
        except Exception as e:
            raise MatrixError(f"Erro ao carregar dados originais: {e}")
        
        # Criar diretório de saída
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def _build_matrix(self) -> np.ndarray:
        """
        Constrói a matriz específica.
        Deve ser implementado por cada subclasse.
        
        Returns:
            Array NumPy com a matriz construída
        """
        pass
    
    @abstractmethod
    def get_expected_dimensions(self) -> Tuple[int, int]:
        """
        Retorna dimensões esperadas da matriz.
        
        Returns:
            Tupla (linhas, colunas)
        """
        pass
    
    def build(self) -> np.ndarray:
        """
        Constrói a matriz principal.
        
        Returns:
            Matriz construída
        """
        if not self.is_initialized():
            raise MatrixError("Matrix builder não foi inicializado")
        
        try:
            self.logger.info("Iniciando construção da matriz...")
            self.matrix = self._build_matrix()
            self.matrix_shape = self.matrix.shape
            
            # Validar matriz
            self._validate_matrix()
            
            self.logger.info(f"Matriz construída com sucesso: {self.matrix_shape}")
            return self.matrix
            
        except Exception as e:
            raise MatrixError(f"Erro na construção da matriz: {e}")
    
    def _validate_matrix(self) -> None:
        """Valida matriz construída."""
        if self.matrix is None:
            raise MatrixError("Matriz não foi construída")
        
        # Verificar dimensões
        expected_rows, expected_cols = self.get_expected_dimensions()
        actual_rows, actual_cols = self.matrix.shape
        
        if expected_rows > 0 and actual_rows != expected_rows:
            self.logger.warning(f"Número de linhas difere do esperado: {actual_rows} vs {expected_rows}")
        
        if expected_cols > 0 and actual_cols != expected_cols:
            self.logger.warning(f"Número de colunas difere do esperado: {actual_cols} vs {expected_cols}")
        
        # Verificar valores
        if np.isnan(self.matrix).any():
            nan_count = np.isnan(self.matrix).sum()
            self.logger.warning(f"Matriz contém {nan_count} valores NaN")
        
        if np.isinf(self.matrix).any():
            inf_count = np.isinf(self.matrix).sum()
            self.logger.warning(f"Matriz contém {inf_count} valores infinitos")
    
    def save_matrix(self, 
                   filename: str = "matrix.npy",
                   normalize: bool = False) -> Path:
        """
        Salva matriz no diretório de saída.
        
        Args:
            filename: Nome do arquivo
            normalize: Se deve normalizar antes de salvar
            
        Returns:
            Caminho do arquivo salvo
        """
        if self.matrix is None:
            raise MatrixError("Matriz não foi construída")
        
        # Normalizar se solicitado
        matrix_to_save = self.matrix
        if normalize:
            matrix_to_save = self.normalize_matrix(self.matrix)
            self.logger.info("Matriz normalizada antes de salvar")
        
        # Salvar
        output_path = self.output_dir / filename
        save_numpy(matrix_to_save, output_path)
        
        self.logger.info(f"Matriz salva em: {output_path}")
        return output_path
    
    def normalize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """
        Normaliza matriz usando z-score.
        
        Args:
            matrix: Matriz para normalizar
            
        Returns:
            Matriz normalizada
        """
        try:
            mean = np.mean(matrix, axis=0, keepdims=True)
            std = np.std(matrix, axis=0, keepdims=True)
            
            # Evitar divisão por zero
            std = np.where(std == 0, 1, std)
            
            normalized = (matrix - mean) / std
            
            self.logger.info("Normalização z-score aplicada")
            return normalized
            
        except Exception as e:
            raise MatrixError(f"Erro na normalização: {e}")
    
    def get_matrix_statistics(self) -> Dict[str, Any]:
        """
        Obtém estatísticas da matriz.
        
        Returns:
            Dicionário com estatísticas
        """
        if self.matrix is None:
            return {}
        
        stats = {
            'shape': self.matrix.shape,
            'dtype': str(self.matrix.dtype),
            'mean': float(np.mean(self.matrix)),
            'std': float(np.std(self.matrix)),
            'min': float(np.min(self.matrix)),
            'max': float(np.max(self.matrix)),
            'nan_count': int(np.isnan(self.matrix).sum()),
            'inf_count': int(np.isinf(self.matrix).sum()),
            'memory_usage_mb': self.matrix.nbytes / (1024 * 1024)
        }
        
        return stats
    
    def load_embedding(self, file_path: Path, is_protein: bool = False) -> Optional[np.ndarray]:
        """
        Carrega embedding individual de arquivo.
        
        Args:
            file_path: Caminho do arquivo de embedding
            is_protein: Se é embedding de proteína (para logs)
            
        Returns:
            Array NumPy com embedding ou None se falhar
        """
        try:
            if not file_path.exists():
                self.logger.debug(f"Arquivo de embedding não encontrado: {file_path}")
                return None
            
            embedding = load_numpy(file_path)
            
            # Validação básica
            if embedding.size == 0:
                self.logger.warning(f"Embedding vazio: {file_path}")
                return None
            
            # Tratar embeddings de proteína (podem ter dimensão extra de sequência)
            if is_protein and embedding.ndim > 1:
                # Usar média se for matriz (sequência x dimensões)
                embedding = np.mean(embedding, axis=0)
            
            return embedding
            
        except Exception as e:
            self.logger.warning(f"Erro ao carregar embedding {file_path}: {e}")
            return None
    
    def find_embedding_files(self, 
                           base_dir: Path, 
                           identifier: str, 
                           extensions: List[str] = ['.npy']) -> List[Path]:
        """
        Encontra arquivos de embedding para um identificador.
        
        Args:
            base_dir: Diretório base para busca
            identifier: Identificador para buscar
            extensions: Extensões de arquivo aceitas
            
        Returns:
            Lista de caminhos encontrados
        """
        found_files = []
        
        if not base_dir.exists():
            return found_files
        
        # Buscar padrões comuns
        patterns = [
            f"{identifier}*",
            f"*{identifier}*",
            f"{identifier}_embedding*",
            f"*{identifier}_embedding*"
        ]
        
        for pattern in patterns:
            for ext in extensions:
                search_pattern = f"{pattern}{ext}"
                matches = list(base_dir.glob(search_pattern))
                found_files.extend(matches)
        
        # Remover duplicatas mantendo ordem
        unique_files = []
        seen = set()
        for file_path in found_files:
            if file_path not in seen:
                unique_files.append(file_path)
                seen.add(file_path)
        
        return unique_files
    
    @memory_monitor(threshold_percent=85.0)
    def process_embeddings_batch(self,
                                identifiers: List[str],
                                ligand_dir: Path,
                                protein_dir: Path,
                                batch_size: int = 1000) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Processa batch de embeddings de ligantes e proteínas.
        
        Args:
            identifiers: Lista de identificadores
            ligand_dir: Diretório de embeddings de ligantes
            protein_dir: Diretório de embeddings de proteínas
            batch_size: Tamanho do batch para processamento
            
        Returns:
            Tupla (embeddings_ligantes, embeddings_proteinas)
        """
        ligand_embeddings = []
        protein_embeddings = []
        
        # Processar em batches para controle de memória
        for i in range(0, len(identifiers), batch_size):
            batch_ids = identifiers[i:i + batch_size]
            
            # Processar batch
            batch_ligands = []
            batch_proteins = []
            
            for identifier in batch_ids:
                # Buscar embedding de ligante
                ligand_files = self.find_embedding_files(ligand_dir, identifier)
                if ligand_files:
                    ligand_emb = self.load_embedding(ligand_files[0], is_protein=False)
                    batch_ligands.append(ligand_emb)
                else:
                    batch_ligands.append(None)
                
                # Buscar embedding de proteína
                protein_files = self.find_embedding_files(protein_dir, identifier)
                if protein_files:
                    protein_emb = self.load_embedding(protein_files[0], is_protein=True)
                    batch_proteins.append(protein_emb)
                else:
                    batch_proteins.append(None)
            
            ligand_embeddings.extend(batch_ligands)
            protein_embeddings.extend(batch_proteins)
            
            # Limpeza de memória entre batches
            if i % (batch_size * 5) == 0:
                import gc
                gc.collect()
        
        return ligand_embeddings, protein_embeddings
    
    def get_matrix(self) -> Optional[np.ndarray]:
        """Retorna matriz construída."""
        return self.matrix
    
    def get_matrix_shape(self) -> Optional[Tuple[int, int]]:
        """Retorna shape da matriz."""
        return self.matrix_shape
    
    def is_matrix_built(self) -> bool:
        """Verifica se matriz foi construída."""
        return self.matrix is not None
