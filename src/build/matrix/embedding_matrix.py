"""
Construção de matriz de embeddings concatenados (ligantes + proteínas).
"""

import os
from typing import Dict, Any, Tuple, Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from build.core import BuildConfig
import numpy as np
from pathlib import Path
from tqdm import tqdm

from build.matrix.base_matrix import BaseMatrix
from build.core.exceptions import MatrixError
from build.utils import ProgressLogger, memory_monitor

class EmbeddingMatrix(BaseMatrix):
    """Construtor de matriz de embeddings concatenados."""
    
    def __init__(self,
                 config: Optional['BuildConfig'] = None,
                 ligand_embeddings_dir: str = 'ligand_embeddings',
                 protein_embeddings_dir: str = 'protein_embeddings',
                 embedding_type: str = 'cls',
                 ligand_dim: int = 768,
                 protein_dim: int = 2560,
                 original_tsv_path: str = '/dev/null',
                 **kwargs):
        """
        Inicializa construtor de matriz de embeddings.
        
        Args:
            config: Configuração do sistema build
            ligand_embeddings_dir: Diretório com embeddings de ligantes
            protein_embeddings_dir: Diretório com embeddings de proteínas
            embedding_type: Tipo de embedding ('cls' ou 'mean')
            ligand_dim: Dimensão dos embeddings de ligantes
            protein_dim: Dimensão dos embeddings de proteínas
            original_tsv_path: Caminho do arquivo TSV original
            **kwargs: Argumentos adicionais
        """
        # Definir atributos ANTES de chamar super
        self.ligand_embeddings_dir = Path(ligand_embeddings_dir)
        self.protein_embeddings_dir = Path(protein_embeddings_dir)
        self.embedding_type = embedding_type
        self.ligand_dim = ligand_dim
        self.protein_dim = protein_dim
        
        # Backward compatibility properties
        self.ligand_dir = str(self.ligand_embeddings_dir)
        self.protein_dir = str(self.protein_embeddings_dir)
        
        # Cache para embeddings carregados
        self.ligand_cache = {}
        self.protein_cache = {}
        
        # Log de embeddings ausentes
        self.missing_log_file = None
        
        # Agora chamar super com tudo definido
        super().__init__(
            original_tsv_path=original_tsv_path,
            config=config,
            **kwargs
        )
    
    def _validate_config(self) -> None:
        """Validação específica para matriz de embeddings."""
        super()._validate_config()
        
        # Validar tipo de embedding
        if self.embedding_type not in ['cls', 'mean']:
            raise MatrixError(f"Tipo de embedding inválido: {self.embedding_type}. Use 'cls' ou 'mean'.")
        
        # Verificar se diretórios de embeddings existem
        if not self.ligand_embeddings_dir.exists():
            self.logger.warning(f"Diretório de ligantes não encontrado: {self.ligand_embeddings_dir}")
        
        if not self.protein_embeddings_dir.exists():
            self.logger.warning(f"Diretório de proteínas não encontrado: {self.protein_embeddings_dir}")
    
    def _do_initialize(self) -> None:
        """Inicialização específica da matriz de embeddings."""
        super()._do_initialize()
        
        # Configurar arquivo de log para embeddings ausentes
        self.missing_log_file = self.output_dir / "missing_embeddings.log"
        
        # Determinar dimensões automaticamente se possível
        if self.ligand_dim <= 0 or self.protein_dim <= 0:
            self._determine_embedding_dimensions()
    
    def _determine_embedding_dimensions(self, force: bool = False) -> None:
        """
        Determina dimensões dos embeddings automaticamente.
        
        Args:
            force: Se True, força re-detecção mesmo se dimensões já definidas
        """
        self.logger.info("Determinando dimensões dos embeddings automaticamente...")
        
        # Verificar embeddings de ligantes
        if self.ligand_dim <= 0 or force:
            # Ligantes começam com CHEMBL
            ligand_files = list(self.ligand_embeddings_dir.glob("CHEMBL*.npy"))
            if ligand_files:
                sample_ligand = self.load_embedding(ligand_files[0], is_protein=False)
                if sample_ligand is not None:
                    self.ligand_dim = len(sample_ligand)
                    self.logger.info(f"Dimensão de ligantes determinada: {self.ligand_dim}")
        
        # Verificar embeddings de proteínas
        if self.protein_dim <= 0 or force:
            # Proteínas são arquivos numéricos (seq_id_embedding.npy)
            # Excluir arquivos que começam com CHEMBL
            all_files = list(self.protein_embeddings_dir.glob("*.npy"))
            protein_files = [f for f in all_files if not f.name.startswith("CHEMBL")]
            if protein_files:
                sample_protein = self.load_embedding(protein_files[0], is_protein=True)
                if sample_protein is not None:
                    self.protein_dim = len(sample_protein)
                    self.logger.info(f"Dimensão de proteínas determinada: {self.protein_dim}")
        
        # Validar dimensões finais
        if self.ligand_dim <= 0 or self.protein_dim <= 0:
            raise MatrixError(
                f"Não foi possível determinar dimensões válidas: "
                f"ligand_dim={self.ligand_dim}, protein_dim={self.protein_dim}"
            )
    
    def get_expected_dimensions(self) -> Tuple[int, int]:
        """Retorna dimensões esperadas da matriz."""
        num_rows = len(self.original_data) if self.original_data is not None else 0
        num_cols = self.ligand_dim + self.protein_dim
        return num_rows, num_cols
    
    def _load_embedding_with_type(self, file_path: Path, is_protein: bool = False) -> Optional[np.ndarray]:
        """
        Carrega embedding aplicando o tipo especificado.
        
        Args:
            file_path: Caminho do arquivo
            is_protein: Se é embedding de proteína
            
        Returns:
            Array NumPy com embedding processado
        """
        try:
            embedding = self.load_embedding(file_path, is_protein)
            if embedding is None:
                return None
            
            # Aplicar tipo de embedding
            if self.embedding_type == 'cls':
                # Para embeddings que são matrizes (sequência x dim)
                if embedding.ndim > 1:
                    # Para proteínas, usar primeiro token (CLS)
                    if is_protein:
                        return embedding[0, :] if embedding.shape[0] > 0 else embedding.flatten()
                    # Para ligantes, usar primeiro embedding
                    else:
                        return embedding[0, :] if embedding.shape[0] > 0 else embedding.flatten()
                else:
                    return embedding
                    
            elif self.embedding_type == 'mean':
                # Usar média ao longo do eixo de sequência
                if embedding.ndim > 1:
                    return np.mean(embedding, axis=0)
                else:
                    return embedding
            
            else:
                raise MatrixError(f"Tipo de embedding inválido: {self.embedding_type}")
                
        except Exception as e:
            self.logger.warning(f"Erro ao processar embedding {file_path}: {e}")
            return None
    
    def _get_cached_embedding(self, 
                            identifier: str, 
                            base_dir: Path, 
                            cache: Dict[str, np.ndarray],
                            is_protein: bool = False) -> Optional[np.ndarray]:
        """
        Obtém embedding do cache ou carrega do disco.
        
        Args:
            identifier: Identificador do embedding
            base_dir: Diretório base
            cache: Cache para usar
            is_protein: Se é embedding de proteína
            
        Returns:
            Array NumPy com embedding ou None
        """
        # Verificar cache primeiro
        if identifier in cache:
            return cache[identifier]
        
        # Buscar arquivos de embedding
        embedding_files = self.find_embedding_files(base_dir, identifier)
        
        if not embedding_files:
            return None
        
        # Carregar primeiro arquivo encontrado
        embedding = self._load_embedding_with_type(embedding_files[0], is_protein)
        
        # Adicionar ao cache se carregado com sucesso
        if embedding is not None:
            cache[identifier] = embedding
        
        return embedding
    
    @memory_monitor(threshold_percent=80.0)
    def _build_matrix(self) -> np.ndarray:
        """Constrói matriz de embeddings concatenados."""
        # Validar colunas necessárias
        required_columns = ['seq_id']
        # Para ligantes, aceitar chembl_id ou molregno
        if 'chembl_id' not in self.original_data.columns and 'molregno' not in self.original_data.columns:
            raise MatrixError("TSV deve conter coluna 'chembl_id' ou 'molregno' para identificar ligantes")
        
        if 'seq_id' not in self.original_data.columns:
            raise MatrixError("TSV deve conter coluna 'seq_id' para identificar proteínas")
        
        # Converter colunas para string para garantir consistência
        df = self.original_data.copy()
        
        # Preferir chembl_id sobre molregno para ligantes (embeddings são salvos com chembl_id)
        if 'chembl_id' in df.columns:
            df['ligand_id'] = df['chembl_id'].astype(str)
            ligand_id_col = 'chembl_id'
        else:
            df['ligand_id'] = df['molregno'].astype(str)
            ligand_id_col = 'molregno'
        
        df['seq_id'] = df['seq_id'].astype(str)
        
        self.logger.info(f"Construindo matriz para {len(df)} pares ligante+proteína")
        self.logger.info(f"Usando '{ligand_id_col}' para identificar ligantes")
        self.logger.info(f"Dimensões: ligantes={self.ligand_dim}, proteínas={self.protein_dim}")
        
        # Listas para armazenar dados
        concatenated_embeddings = []
        missing_entries = []
        
        # Progress tracking
        progress_logger = ProgressLogger(
            self.logger,
            len(df),
            "Processando pares ligante+proteína"
        )
        
        # Processar cada linha do TSV
        for _, row in df.iterrows():
            ligand_id = str(row['ligand_id'])
            seq_id = str(row['seq_id'])
            
            # Buscar embedding de ligante (usando ligand_id que pode ser chembl_id ou molregno)
            ligand_emb = self._get_cached_embedding(
                ligand_id, 
                self.ligand_embeddings_dir, 
                self.ligand_cache,
                is_protein=False
            )
            
            # Buscar embedding de proteína
            protein_emb = self._get_cached_embedding(
                seq_id,
                self.protein_embeddings_dir,
                self.protein_cache,
                is_protein=True
            )
            
            # Verificar se ambos foram encontrados
            if ligand_emb is None or protein_emb is None:
                missing_info = []
                if ligand_emb is None:
                    missing_info.append(f"ligante:{ligand_id}")
                if protein_emb is None:
                    missing_info.append(f"proteina:{seq_id}")
                
                missing_entries.append(f"ligand_id:{ligand_id}, seq_id:{seq_id} - Missing: {', '.join(missing_info)}")
                
                # Usar embeddings zero para entradas ausentes
                final_embedding = np.zeros(self.protein_dim + self.ligand_dim)
            else:
                # Concatenar embeddings: [protein_embedding, ligand_embedding]
                final_embedding = np.concatenate([protein_emb, ligand_emb])
            
            concatenated_embeddings.append(final_embedding)
            progress_logger.update()
        
        progress_logger.finish()
        
        # Log de embeddings ausentes
        if missing_entries:
            self._save_missing_log(missing_entries)
        
        # Converter para matriz numpy
        matrix = np.vstack(concatenated_embeddings)
        
        self.logger.info(f"Matriz construída: shape={matrix.shape}")
        self.logger.info(f"Embeddings ausentes: {len(missing_entries)}")
        
        return matrix
    
    def _save_missing_log(self, missing_entries: List[str]) -> None:
        """Salva log de embeddings ausentes."""
        try:
            with open(self.missing_log_file, 'w', encoding='utf-8') as f:
                f.write("# Log de embeddings ausentes\\n")
                f.write(f"# Total de entradas ausentes: {len(missing_entries)}\\n\\n")
                
                for entry in missing_entries:
                    f.write(f"{entry}\\n")
            
            self.logger.warning(f"Log de embeddings ausentes salvo em: {self.missing_log_file}")
            
        except Exception as e:
            self.logger.error(f"Erro ao salvar log de ausentes: {e}")
    
    def run(self) -> np.ndarray:
        """
        Executa construção completa da matriz.
        
        Returns:
            Matriz de embeddings construída
        """
        with self:
            # Construir matriz
            matrix = self.build()
            
            # Salvar matriz
            output_file = self.save_matrix("concatenated_embeddings.npy")
            
            # Salvar matriz normalizada
            normalized_file = self.save_matrix("concatenated_embeddings_normalized.npy", normalize=True)
            
            self.logger.info(f"Matriz salva em: {output_file}")
            self.logger.info(f"Matriz normalizada salva em: {normalized_file}")
            
            return matrix
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Obtém estatísticas dos caches."""
        return {
            'ligand_cache_size': len(self.ligand_cache),
            'protein_cache_size': len(self.protein_cache),
            'ligand_cache_memory_mb': sum(emb.nbytes for emb in self.ligand_cache.values()) / (1024 * 1024),
            'protein_cache_memory_mb': sum(emb.nbytes for emb in self.protein_cache.values()) / (1024 * 1024)
        }
    
    def clear_caches(self) -> None:
        """Limpa caches de embeddings."""
        self.ligand_cache.clear()
        self.protein_cache.clear()
        
        import gc
        gc.collect()
        
        self.logger.info("Caches de embeddings limpos")
    
    def get_matrix_info(self) -> Dict[str, Any]:
        """
        Retorna informações sobre a matriz construída.
        
        Returns:
            Dict com informações da matriz
        """
        matrix_file = self.output_dir / "embedding_matrix.npy"
        matrix_shape = (0, 0)
        
        if matrix_file.exists():
            try:
                matrix = np.load(matrix_file)
                matrix_shape = matrix.shape
            except Exception as e:
                self.logger.warning(f"Erro ao carregar matriz para info: {e}")
        
        return {
            'ligand_dim': self.ligand_dim,
            'protein_dim': self.protein_dim,
            'total_dim': self.ligand_dim + self.protein_dim,
            'matrix_shape': matrix_shape,
            'num_samples': matrix_shape[0] if matrix_shape[0] > 0 else 0,
            'matrix_file': str(matrix_file)
        }
    
    def get_output_path(self) -> Path:
        """
        Retorna o caminho do diretório de saída.
        
        Returns:
            Path do diretório de saída
        """
        return self.output_dir

    
    def build(self) -> Dict[str, Any]:
        """Constrói resultado com informações da matriz."""
        matrix = super().build()
        
        result = {
            'matrix': matrix,
            'embedding_type': self.embedding_type,
            'ligand_dim': self.ligand_dim,
            'protein_dim': self.protein_dim,
            'cache_stats': self.get_cache_statistics(),
            'missing_log_file': str(self.missing_log_file)
        }
        
        return result
    
    def build_matrix(self, 
                    protein_embeddings_path: Path,
                    ligand_embeddings_path: Path,
                    output_dir: Optional[Path] = None,
                    data_path: Optional[Path] = None) -> bool:
        """
        Constrói matriz de embeddings concatenados (interface do pipeline).
        
        Args:
            protein_embeddings_path: Diretório com embeddings de proteínas
            ligand_embeddings_path: Diretório com embeddings de ligantes
            output_dir: Diretório de saída
            data_path: Caminho para arquivo TSV original (necessário para carregar dados)
            
        Returns:
            True se sucesso
        """
        try:
            # Atualizar caminhos
            self.protein_embeddings_dir = Path(protein_embeddings_path)
            self.ligand_embeddings_dir = Path(ligand_embeddings_path)
            
            if output_dir:
                self.output_dir = Path(output_dir)
            
            # Atualizar data path se fornecido
            if data_path:
                self.original_tsv_path = Path(data_path)
            
            # Garantir inicialização
            if not self._initialized:
                self.initialize()
            
            # Forçar re-detecção de dimensões com os novos caminhos
            self.logger.info("Re-detectando dimensões com caminhos atualizados...")
            self._determine_embedding_dimensions(force=True)
            
            # Construir matriz
            self.logger.info("Construindo matriz de embeddings concatenados...")
            matrix = self._build_matrix()
            
            # Salvar matriz
            matrix_file = self.output_dir / "embedding_matrix.npy"
            np.save(matrix_file, matrix)
            self.logger.info(f"Matriz salva: {matrix_file} - Shape: {matrix.shape}")
            
            # Atualizar output path
            self._output_path = self.output_dir
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao construir matriz: {e}")
            return False
    
    def reconstruct_matrix(self) -> np.ndarray:
        """
        Método de compatibilidade com scripts originais.
        
        Returns:
            Matriz concatenada de embeddings
        """
        return self._build_matrix()

