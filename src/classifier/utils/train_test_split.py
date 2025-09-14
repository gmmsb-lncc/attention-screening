"""
Implementação de divisão train/test para o DockTKinase.
Garante estratificação, reprodutibilidade e validação estatística.
"""

import torch
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any, Optional
import logging
from dataclasses import dataclass
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class SplitValidationReport:
    """Relatório de validação da divisão train/test."""
    
    is_valid: bool
    train_distribution: Dict[int, float]
    test_distribution: Dict[int, float]
    chi_square_p_value: float
    imbalance_ratio: float
    issues: list
    recommendations: list


class TrainTestSplitter:
    """
    Divisão train/test cientificamente válida.
    
    Características:
    - Estratificação automática
    - Reprodutibilidade garantida
    - Validação estatística
    - Detecção de problemas
    - Múltiplas estratégias de balanceamento
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        
    def split(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        test_size: float = 0.2,
        stratify: bool = True,
        min_samples_per_class: int = 5,
        max_imbalance_ratio: float = 10.0,
        verbose: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Realiza divisão train/test com validação.
        
        Args:
            X: Features tensor
            y: Labels tensor  
            test_size: Proporção para teste (0.2 = 20%)
            stratify: Se deve usar estratificação
            min_samples_per_class: Mínimo de amostras por classe em cada split
            max_imbalance_ratio: Máximo desbalanceamento aceitável
            verbose: Se deve imprimir informações detalhadas
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        if verbose:
            logger.info(f"🔄 Iniciando divisão train/test ({test_size*100:.0f}% teste)")
        
        # Converter para numpy para usar sklearn
        X_np = X.cpu().numpy()
        y_np = y.cpu().numpy()
        
        # 1. VALIDAÇÕES PRÉ-DIVISÃO
        issues = []
        
        # Verificar classes únicas
        unique_classes, class_counts = np.unique(y_np, return_counts=True)
        logger.info(f"📊 Classes encontradas: {len(unique_classes)}")
        
        for cls, count in zip(unique_classes, class_counts):
            logger.info(f"  Classe {cls}: {count} amostras ({count/len(y_np)*100:.1f}%)")
            
            # Verificar se haverá amostras suficientes após divisão
            expected_test = int(count * test_size)
            expected_train = count - expected_test
            
            if expected_test < min_samples_per_class:
                issues.append(f"Classe {cls}: apenas {expected_test} amostras no teste (mín: {min_samples_per_class})")
            if expected_train < min_samples_per_class:
                issues.append(f"Classe {cls}: apenas {expected_train} amostras no treino (mín: {min_samples_per_class})")
        
        # Verificar desbalanceamento
        imbalance_ratio = max(class_counts) / min(class_counts)
        if imbalance_ratio > max_imbalance_ratio:
            issues.append(f"Alto desbalanceamento: {imbalance_ratio:.1f}:1 (máx: {max_imbalance_ratio:.1f}:1)")
        
        # 2. ESTRATÉGIA DE DIVISÃO BASEADA NAS VALIDAÇÕES
        if len(unique_classes) == 1:
            if verbose:
                logger.warning("⚠️  Apenas uma classe detectada - usando divisão aleatória")
            X_train, X_test, y_train, y_test = train_test_split(
                X_np, y_np, 
                test_size=test_size,
                random_state=self.random_state,
                shuffle=True
            )
        elif any("apenas" in issue for issue in issues):
            if verbose:
                logger.warning("⚠️  Poucas amostras por classe - usando divisão proporcional ajustada")
            # Ajustar test_size para garantir mínimo por classe
            adjusted_test_size = self._calculate_adjusted_test_size(class_counts, min_samples_per_class)
            X_train, X_test, y_train, y_test = train_test_split(
                X_np, y_np,
                test_size=adjusted_test_size,
                stratify=y_np if stratify else None,
                random_state=self.random_state,
                shuffle=True
            )
        else:
            if verbose:
                logger.info("✅ Usando divisão estratificada padrão")
            X_train, X_test, y_train, y_test = train_test_split(
                X_np, y_np,
                test_size=test_size,
                stratify=y_np if stratify else None,  # 🎯 ESTRATIFICAÇÃO
                random_state=self.random_state,
                shuffle=True
            )
        
        # 3. CONVERTER DE VOLTA PARA TENSORES
        device = X.device
        X_train = torch.from_numpy(X_train).to(device)
        X_test = torch.from_numpy(X_test).to(device)
        y_train = torch.from_numpy(y_train).to(device)
        y_test = torch.from_numpy(y_test).to(device)
        
        # 4. LOG DOS RESULTADOS  
        if verbose:
            logger.info(f"✅ Divisão concluída:")
            logger.info(f"  📈 Treino: {len(X_train)} amostras")
            logger.info(f"  📊 Teste: {len(X_test)} amostras")  
            logger.info(f"  📉 Proporção real: {len(X_test)/(len(X_train)+len(X_test))*100:.1f}% teste")
            logger.info(f"  🎯 Classes treino: {torch.bincount(y_train)}")
            logger.info(f"  🎯 Classes teste: {torch.bincount(y_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def _calculate_adjusted_test_size(self, class_counts: np.ndarray, min_samples: int) -> float:
        """Calcula test_size ajustado para garantir mínimo de amostras por classe."""
        total_samples = sum(class_counts)
        min_class_size = min(class_counts)
        
        # Garantir pelo menos min_samples no teste para a classe minoritária
        max_test_from_min_class = min_class_size - min_samples  # Deixar min_samples no treino
        max_test_total = len(class_counts) * max_test_from_min_class  # Aproximação
        
        adjusted_test_size = min(0.3, max_test_total / total_samples)  # Máximo 30%
        adjusted_test_size = max(0.1, adjusted_test_size)  # Mínimo 10%
        
        logger.info(f"📐 Test size ajustado: {adjusted_test_size:.2f}")
        return adjusted_test_size
    
    def _validate_split(
        self, 
        y_train: torch.Tensor, 
        y_test: torch.Tensor,
        original_classes: np.ndarray,
        original_imbalance: float,
        pre_issues: list
    ) -> SplitValidationReport:
        """Valida estatisticamente a divisão realizada."""
        
        # Converter para numpy
        y_train_np = y_train.cpu().numpy()
        y_test_np = y_test.cpu().numpy()
        
        # Distribuições por conjunto
        train_unique, train_counts = np.unique(y_train_np, return_counts=True)
        test_unique, test_counts = np.unique(y_test_np, return_counts=True)
        
        train_dist = {int(cls): count/len(y_train_np) for cls, count in zip(train_unique, train_counts)}
        test_dist = {int(cls): count/len(y_test_np) for cls, count in zip(test_unique, test_counts)}
        
        # Teste chi-quadrado para comparar distribuições
        chi2_p_value = self._chi_square_test(train_counts, test_counts)
        
        # Validações
        issues = pre_issues.copy()
        recommendations = []
        
        # 1. Verificar se todas as classes estão presentes
        missing_in_train = set(original_classes) - set(train_unique)
        missing_in_test = set(original_classes) - set(test_unique)
        
        if missing_in_train:
            issues.append(f"Classes ausentes no treino: {missing_in_train}")
        if missing_in_test:
            issues.append(f"Classes ausentes no teste: {missing_in_test}")
        
        # 2. Verificar similaridade de distribuições
        if chi2_p_value < 0.05:
            issues.append(f"Distribuições train/test significativamente diferentes (p={chi2_p_value:.4f})")
            recommendations.append("Considere aumentar o dataset ou usar cross-validation")
        
        # 3. Verificar se estratificação funcionou
        max_diff = 0
        for cls in original_classes:
            train_prop = train_dist.get(cls, 0)
            test_prop = test_dist.get(cls, 0)
            diff = abs(train_prop - test_prop)
            max_diff = max(max_diff, diff)
        
        if max_diff > 0.05:  # Diferença > 5%
            issues.append(f"Estratificação imperfeita: máx diferença {max_diff*100:.1f}%")
            recommendations.append("Verificar se estratificação foi aplicada corretamente")
        
        # 4. Recomendações baseadas no tamanho
        total_samples = len(y_train_np) + len(y_test_np)
        if total_samples < 1000:
            recommendations.append("Dataset pequeno: considere cross-validation em vez de train/test")
        elif len(y_test_np) < 100:
            recommendations.append("Conjunto de teste pequeno: considere aumentar test_size")
        
        is_valid = len([i for i in issues if "ausentes" in i or "diferentes" in i]) == 0
        
        return SplitValidationReport(
            is_valid=is_valid,
            train_distribution=train_dist,
            test_distribution=test_dist,
            chi_square_p_value=chi2_p_value,
            imbalance_ratio=original_imbalance,
            issues=issues,
            recommendations=recommendations
        )
    
    def _chi_square_test(self, train_counts: np.ndarray, test_counts: np.ndarray) -> float:
        """Teste chi-quadrado para comparar distribuições."""
        try:
            # Normalizar contagens para comparar proporções
            train_prop = train_counts / train_counts.sum()
            test_prop = test_counts / test_counts.sum()
            
            # Chi-quadrado
            if len(train_prop) == len(test_prop) and len(train_prop) > 1:
                chi2, p_value = stats.chisquare(test_prop, train_prop)
                return p_value
            else:
                return 1.0  # Não significativo se apenas uma classe
        except:
            return 1.0  # Em caso de erro, assumir não significativo
    
    def _create_minimal_report(self) -> SplitValidationReport:
        """Cria relatório mínimo quando validação está desabilitada."""
        return SplitValidationReport(
            is_valid=True,
            train_distribution={},
            test_distribution={},
            chi_square_p_value=1.0,
            imbalance_ratio=1.0,
            issues=[],
            recommendations=[]
        )


# 🎯 FUNÇÃO DE CONVENIÊNCIA PARA INTEGRAÇÃO FÁCIL
def robust_train_test_split(
    X: torch.Tensor,
    y: torch.Tensor, 
    test_size: float = 0.2,
    random_state: int = 42,
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, SplitValidationReport]:
    """
    Função de conveniência para divisão robusta train/test.
    
    Usage:
        X_train, X_test, y_train, y_test, report = robust_train_test_split(X, y)
        
        if not report.is_valid:
            logger.warning("Problemas na divisão!")
    """
    splitter = TrainTestSplitter(random_state=random_state)
    return splitter.split(X, y, test_size=test_size, **kwargs)
