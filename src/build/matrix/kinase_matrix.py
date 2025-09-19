"""
Construção de matriz específica para dados de kinase.
"""

from typing import Dict, Any, Tuple
import numpy as np
from pathlib import Path

from build.matrix.embedding_matrix import EmbeddingMatrix
from build.utils import ProgressLogger

class KinaseMatrix(EmbeddingMatrix):
    """Construtor de matriz específico para dados de kinase."""
    
    def __init__(self,
                 config_or_path=None,
                 ligand_embeddings_dir: str = 'ligand_embeddings', 
                 protein_embeddings_dir: str = 'protein_embeddings',
                 kinase_specific: bool = True,
                 **kwargs):
        """
        Inicializa construtor de matriz de kinase.
        
        Args:
            config_or_path: BuildConfig ou caminho TSV
            ligand_embeddings_dir: Diretório com embeddings de ligantes
            protein_embeddings_dir: Diretório com embeddings de proteínas
            kinase_specific: Se deve aplicar processamento específico para kinases
            **kwargs: Argumentos adicionais
        """
        # Definir atributos específicos ANTES de chamar super
        self.kinase_specific = kinase_specific
        
        # Chamar construtor da classe pai
        super().__init__(
            config_or_path=config_or_path,
            ligand_embeddings_dir=ligand_embeddings_dir,
            protein_embeddings_dir=protein_embeddings_dir,
            **kwargs
        )
    
    def _validate_config(self) -> None:
        """Validação específica para matriz de kinase."""
        super()._validate_config()
        
        # Validações específicas para kinase podem ser adicionadas aqui
        # Por exemplo, verificar se o TSV contém colunas específicas de kinase
        pass
    
    def _preprocess_kinase_data(self) -> None:
        """Pré-processa dados específicos de kinase."""
        if not self.kinase_specific:
            return
        
        self.logger.info("Aplicando pré-processamento específico para kinases...")
        
        # Exemplo: filtrar apenas dados válidos de kinase
        if 'kinase_name' in self.original_data.columns:
            original_size = len(self.original_data)
            
            # Remover linhas com kinase_name vazio
            self.original_data = self.original_data.dropna(subset=['kinase_name'])
            
            filtered_size = len(self.original_data)
            self.logger.info(f"Dados filtrados: {original_size} → {filtered_size} registros")
        
        # Exemplo: normalizar identificadores de kinase
        if 'seq_id' in self.original_data.columns:
            # Garantir que seq_id tenha formato consistente
            self.original_data['seq_id'] = self.original_data['seq_id'].astype(str).str.strip()
    
    def _build_matrix(self) -> np.ndarray:
        """Constrói matriz com processamento específico para kinase."""
        # Aplicar pré-processamento
        self._preprocess_kinase_data()
        
        # Usar implementação da classe pai
        matrix = super()._build_matrix()
        
        # Aplicar pós-processamento específico se necessário
        if self.kinase_specific:
            matrix = self._postprocess_kinase_matrix(matrix)
        
        return matrix
    
    def _postprocess_kinase_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Aplica pós-processamento específico para dados de kinase."""
        self.logger.info("Aplicando pós-processamento específico para kinases...")
        
        # Exemplo: normalização específica para kinases
        # Isso é um exemplo - ajuste conforme necessário
        
        # Verificar se há valores extremos que precisam ser tratados
        extreme_values = np.abs(matrix) > 10
        if np.any(extreme_values):
            extreme_count = np.sum(extreme_values)
            self.logger.warning(f"Encontrados {extreme_count} valores extremos na matriz")
            
            # Clipping de valores extremos
            matrix = np.clip(matrix, -10, 10)
            self.logger.info("Valores extremos limitados ao intervalo [-10, 10]")
        
        return matrix
    
    def get_kinase_statistics(self) -> Dict[str, Any]:
        """Obtém estatísticas específicas para dados de kinase."""
        if self.original_data is None:
            return {}
        
        stats = {}
        
        # Estatísticas de kinases únicas
        if 'seq_id' in self.original_data.columns:
            unique_kinases = self.original_data['seq_id'].nunique()
            stats['unique_kinases'] = unique_kinases
        
        # Estatísticas de compostos únicos
        if 'molregno' in self.original_data.columns:
            unique_compounds = self.original_data['molregno'].nunique()
            stats['unique_compounds'] = unique_compounds
        
        # Estatísticas de pares kinase-composto
        if 'seq_id' in self.original_data.columns and 'molregno' in self.original_data.columns:
            unique_pairs = len(self.original_data.drop_duplicates(subset=['seq_id', 'molregno']))
            stats['unique_kinase_compound_pairs'] = unique_pairs
        
        # Estatísticas de atividade (se disponível)
        activity_columns = ['pchembl_value', 'activity_value', 'ic50', 'ki', 'kd']
        for col in activity_columns:
            if col in self.original_data.columns:
                activity_data = self.original_data[col].dropna()
                if len(activity_data) > 0:
                    stats[f'{col}_mean'] = float(activity_data.mean())
                    stats[f'{col}_std'] = float(activity_data.std())
                    stats[f'{col}_min'] = float(activity_data.min())
                    stats[f'{col}_max'] = float(activity_data.max())
                    stats[f'{col}_count'] = int(len(activity_data))
        
        return stats
    
    def run(self) -> np.ndarray:
        """
        Executa construção completa da matriz de kinase.
        
        Returns:
            Matriz de kinase construída
        """
        with self:
            # Construir matriz
            matrix = self.build()
            
            # Salvar matriz principal
            output_file = self.save_matrix("kinase_embeddings.npy")
            
            # Salvar matriz normalizada
            normalized_file = self.save_matrix("kinase_embeddings_normalized.npy", normalize=True)
            
            # Salvar estatísticas de kinase
            kinase_stats = self.get_kinase_statistics()
            if kinase_stats:
                stats_file = self.output_dir / "kinase_statistics.json"
                import json
                with open(stats_file, 'w', encoding='utf-8') as f:
                    json.dump(kinase_stats, f, indent=2, ensure_ascii=False)
                
                self.logger.info(f"Estatísticas de kinase salvas em: {stats_file}")
            
            self.logger.info(f"Matriz de kinase salva em: {output_file}")
            self.logger.info(f"Matriz normalizada salva em: {normalized_file}")
            
            return matrix
    
    def build(self) -> Dict[str, Any]:
        """Constrói resultado com informações específicas de kinase."""
        result = super().build()
        
        # Adicionar estatísticas específicas de kinase
        result.update({
            'kinase_specific': self.kinase_specific,
            'kinase_statistics': self.get_kinase_statistics()
        })
        
        return result
