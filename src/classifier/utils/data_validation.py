"""
Utilidades para validação e análise de qualidade de dados.
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
@dataclass
class DataQualityReport:
    """Relatório de qualidade dos dados."""
    issues: List[str]
    stats: Dict[str, Any]
    warnings: List[str]
    passed_validation: bool
    
    @property
    def is_valid(self) -> bool:
        """Alias para passed_validation para compatibilidade."""
        return self.passed_validation


class DataValidator:
    """Validador de qualidade para embeddings e labels."""
    
    def __init__(self, 
                 max_sample_size: int = 10000,
                 imbalance_threshold: float = 10.0,
                 duplicate_threshold: float = 0.05):
        self.max_sample_size = max_sample_size
        self.imbalance_threshold = imbalance_threshold
        self.duplicate_threshold = duplicate_threshold
    
    def validate_data_files(self, embeddings_path: str, labels_path: str) -> DataQualityReport:
        """
        Validação completa de arquivos de dados.
        
        Args:
            embeddings_path: Caminho para arquivo .npy de embeddings
            labels_path: Caminho para arquivo .npy de labels
            
        Returns:
            DataQualityReport com resultados da validação
        """
        logger.info("🔍 Iniciando validação de qualidade dos dados...")
        
        issues = []
        warnings = []
        stats = {}
    
    def validate_arrays(self, X: np.ndarray, y: np.ndarray) -> DataQualityReport:
        """
        Validação de arrays NumPy (versão simplificada).
        
        Args:
            X: Array de features
            y: Array de labels
            
        Returns:
            DataQualityReport com resultados da validação
        """
        issues = []
        warnings = []
        stats = {}
        
        # Validação básica de dimensões
        self._validate_dimensions(X, y, issues, stats)
        
        # Validação de conteúdo
        self._validate_content(X, y, issues, warnings, stats)
        
        # Análises adicionais
        self._analyze_distribution(y, issues, warnings, stats)
        
        is_valid = len(issues) == 0
        
        return DataQualityReport(
            issues=issues,
            warnings=warnings,
            stats=stats,
            passed_validation=is_valid
        )
        
        try:
            # Validação de existência de arquivos
            self._validate_file_existence(embeddings_path, labels_path, issues)
            
            if issues:  # Se arquivos não existem, para aqui
                return DataQualityReport(
                    issues=issues, 
                    stats=stats, 
                    warnings=warnings, 
                    passed_validation=False
                )
            
            # Carregamento de dados
            embeddings_np, labels_np = self._load_data(embeddings_path, labels_path)
            
            # Validações dimensionais
            self._validate_dimensions(embeddings_np, labels_np, issues, stats)
            
            # Validações de conteúdo
            self._validate_content(embeddings_np, labels_np, issues, warnings, stats)
            
            # Análise de distribuição
            self._analyze_distribution(labels_np, issues, warnings, stats)
            
            # Detecção de duplicatas
            self._detect_duplicates(embeddings_np, warnings, stats)
            
            # Análise de outliers
            self._analyze_outliers(embeddings_np, warnings, stats)
            
        except Exception as e:
            issues.append(f"Erro durante validação: {str(e)}")
            logger.error(f"❌ Erro na validação: {e}")
        
        # Resultado final
        passed = len(issues) == 0
        
        if passed:
            logger.info("✅ Dados passaram na validação básica")
        else:
            logger.warning(f"⚠️  Encontrados {len(issues)} problemas críticos")
        
        if warnings:
            logger.info(f"📋 {len(warnings)} avisos identificados")
        
        # Log das estatísticas principais
        if 'class_distribution' in stats:
            logger.info(f"📊 Distribuição de classes: {stats['class_distribution']}")
        if 'duplicate_rate' in stats:
            logger.info(f"📊 Taxa de duplicatas: {stats['duplicate_rate']*100:.1f}%")
        
        return DataQualityReport(
            issues=issues,
            stats=stats,
            warnings=warnings,
            passed_validation=passed
        )
    
    def _validate_file_existence(self, embeddings_path: str, labels_path: str, issues: List[str]):
        """Valida se os arquivos existem."""
        if not Path(embeddings_path).exists():
            issues.append(f"Arquivo de embeddings não encontrado: {embeddings_path}")
        
        if not Path(labels_path).exists():
            issues.append(f"Arquivo de labels não encontrado: {labels_path}")
    
    def _load_data(self, embeddings_path: str, labels_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Carrega dados com memory mapping para eficiência."""
        try:
            embeddings_np = np.load(embeddings_path, mmap_mode="r", allow_pickle=False)
            labels_np = np.load(labels_path, mmap_mode="r", allow_pickle=False).astype(np.float32).flatten()
            return embeddings_np, labels_np
        except Exception as e:
            raise ValueError(f"Erro ao carregar dados: {e}")
    
    def _validate_dimensions(self, embeddings: np.ndarray, labels: np.ndarray, 
                           issues: List[str], stats: Dict[str, Any]):
        """Valida dimensões dos dados."""
        stats['embeddings_shape'] = embeddings.shape
        stats['labels_shape'] = labels.shape
        stats['total_samples'] = len(labels)
        
        if len(embeddings) != len(labels):
            issues.append(
                f"Mismatch de tamanhos: embeddings {len(embeddings)}, labels {len(labels)}"
            )
        
        if len(embeddings.shape) != 2:
            issues.append(
                f"Embeddings devem ter 2 dimensões, encontrado: {len(embeddings.shape)}"
            )
        
        if embeddings.shape[1] < 10:
            issues.append(
                f"Dimensão de embedding muito pequena: {embeddings.shape[1]} < 10"
            )
    
    def _validate_content(self, embeddings: np.ndarray, labels: np.ndarray,
                         issues: List[str], warnings: List[str], stats: Dict[str, Any]):
        """Valida conteúdo dos dados."""
        # Valores NaN/infinitos nos embeddings
        nan_embeddings = np.isnan(embeddings).sum()
        inf_embeddings = np.isinf(embeddings).sum()
        
        if nan_embeddings > 0:
            issues.append(f"Embeddings contêm {nan_embeddings} valores NaN")
        
        if inf_embeddings > 0:
            issues.append(f"Embeddings contêm {inf_embeddings} valores infinitos")
        
        stats['nan_count'] = int(nan_embeddings)
        stats['inf_count'] = int(inf_embeddings)
        
        # Validação de labels
        unique_labels = np.unique(labels)
        if not np.array_equal(unique_labels, [0., 1.]) and not np.array_equal(unique_labels, [0]) and not np.array_equal(unique_labels, [1]):
            issues.append(f"Labels devem ser 0 e 1, encontrado: {unique_labels}")
        
        # Verificar se todos os labels são da mesma classe
        if len(unique_labels) == 1:
            warnings.append(f"Todos os labels são da mesma classe: {unique_labels[0]}")
    
    def _analyze_distribution(self, labels: np.ndarray, issues: List[str], 
                            warnings: List[str], stats: Dict[str, Any]):
        """Analisa distribuição de classes."""
        unique, counts = np.unique(labels, return_counts=True)
        class_distribution = dict(zip(unique.astype(int), counts.astype(int)))
        
        if len(counts) > 1:
            imbalance_ratio = max(counts) / min(counts)
        else:
            imbalance_ratio = 1.0
        
        stats['class_distribution'] = class_distribution
        stats['imbalance_ratio'] = float(imbalance_ratio)
        
        if imbalance_ratio > self.imbalance_threshold:
            warnings.append(
                f"Desbalanceamento severo de classes: {imbalance_ratio:.1f}:1"
            )
    
    def _detect_duplicates(self, embeddings: np.ndarray, warnings: List[str], 
                          stats: Dict[str, Any]):
        """Detecta embeddings duplicados."""
        # Amostragem para datasets grandes
        if len(embeddings) > self.max_sample_size:
            logger.info(f"📊 Analisando duplicatas em amostra de {self.max_sample_size} exemplos...")
            idx_sample = np.random.choice(len(embeddings), self.max_sample_size, replace=False)
            sample_embeddings = embeddings[idx_sample]
        else:
            sample_embeddings = embeddings
        
        # Detecção de duplicatas
        unique_embeddings = np.unique(sample_embeddings, axis=0)
        duplicate_rate = 1.0 - (len(unique_embeddings) / len(sample_embeddings))
        
        stats['duplicate_rate'] = float(duplicate_rate)
        stats['unique_samples'] = len(unique_embeddings)
        stats['analyzed_samples'] = len(sample_embeddings)
        
        if duplicate_rate > self.duplicate_threshold:
            warnings.append(f"Alta taxa de duplicatas: {duplicate_rate*100:.1f}%")
    
    def _analyze_outliers(self, embeddings: np.ndarray, warnings: List[str], 
                         stats: Dict[str, Any]):
        """Analisa outliers nos embeddings."""
        # Análise estatística básica
        mean_vals = np.mean(embeddings, axis=0)
        std_vals = np.std(embeddings, axis=0)
        
        # Detecta features com desvio padrão muito baixo (quase constantes)
        low_variance_features = np.sum(std_vals < 1e-6)
        if low_variance_features > 0:
            warnings.append(
                f"{low_variance_features} features têm variância muito baixa (quase constantes)"
            )
        
        # Detecta amostras extremas (além de 3 sigma)
        z_scores = np.abs((embeddings - mean_vals) / (std_vals + 1e-8))
        extreme_samples = np.sum(np.any(z_scores > 3, axis=1))
        extreme_rate = extreme_samples / len(embeddings)
        
        stats['low_variance_features'] = int(low_variance_features)
        stats['extreme_samples'] = int(extreme_samples)
        stats['extreme_rate'] = float(extreme_rate)
        
        if extreme_rate > 0.05:  # >5% de amostras extremas
            warnings.append(f"Alta taxa de outliers: {extreme_rate*100:.1f}%")


def quick_data_check(embeddings_path: str, labels_path: str) -> bool:
    """
    Verificação rápida se os dados são válidos para treinamento.
    
    Returns:
        True se dados são válidos, False caso contrário
    """
    validator = DataValidator()
    report = validator.validate_data_files(embeddings_path, labels_path)
    
    return report.passed_validation


def get_data_statistics(embeddings_path: str, labels_path: str) -> Dict[str, Any]:
    """
    Retorna estatísticas básicas dos dados sem validação completa.
    """
    try:
        embeddings = np.load(embeddings_path, mmap_mode="r")
        labels = np.load(labels_path, mmap_mode="r").flatten()
        
        unique, counts = np.unique(labels, return_counts=True)
        
        return {
            'total_samples': len(labels),
            'embedding_dim': embeddings.shape[1],
            'class_distribution': dict(zip(unique.astype(int), counts.astype(int))),
            'imbalance_ratio': float(max(counts) / min(counts)) if len(counts) > 1 else 1.0,
            'embedding_stats': {
                'mean': float(np.mean(embeddings)),
                'std': float(np.std(embeddings)),
                'min': float(np.min(embeddings)),
                'max': float(np.max(embeddings))
            }
        }
    except Exception as e:
        return {'error': str(e)}
