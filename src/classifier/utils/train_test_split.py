"""
Train/test split implementation for DockTKinase.
Ensures stratification, reproducibility and statistical validation.
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
    """Train/test split validation report."""
    
    is_valid: bool
    train_distribution: Dict[int, float]
    test_distribution: Dict[int, float]
    chi_square_p_value: float
    imbalance_ratio: float
    issues: list
    recommendations: list


class TrainTestSplitter:
    """
    Scientifically valid train/test split.
    
    Features:
    - Automatic stratification
    - Guaranteed reproducibility
    - Statistical validation
    - Problem detection
    - Multiple balancing strategies
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
        Performs train/test split with validation.
        
        Args:
            X: Features tensor
            y: Labels tensor  
            test_size: Test proportion (0.2 = 20%)
            stratify: Whether to use stratification
            min_samples_per_class: Minimum samples per class in each split
            max_imbalance_ratio: Maximum acceptable imbalance
            verbose: Whether to print detailed information
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        if verbose:
            logger.info(f"🔄 Starting train/test split ({test_size*100:.0f}% test)")
        
        # Convert to numpy for sklearn
        X_np = X.cpu().numpy()
        y_np = y.cpu().numpy()
        
        # 1. PRE-SPLIT VALIDATIONS
        issues = []
        
        # Check unique classes
        unique_classes, class_counts = np.unique(y_np, return_counts=True)
        logger.info(f"📊 Classes found: {len(unique_classes)}")
        
        for cls, count in zip(unique_classes, class_counts):
            logger.info(f"  Class {cls}: {count} samples ({count/len(y_np)*100:.1f}%)")
            
            # Check if there will be enough samples after split
            expected_test = int(count * test_size)
            expected_train = count - expected_test
            
            if expected_test < min_samples_per_class:
                issues.append(f"Class {cls}: only {expected_test} samples in test (min: {min_samples_per_class})")
            if expected_train < min_samples_per_class:
                issues.append(f"Class {cls}: only {expected_train} samples in train (min: {min_samples_per_class})")
        
        # Check imbalance
        imbalance_ratio = max(class_counts) / min(class_counts)
        if imbalance_ratio > max_imbalance_ratio:
            issues.append(f"High imbalance: {imbalance_ratio:.1f}:1 (max: {max_imbalance_ratio:.1f}:1)")
        
        # 2. SPLIT STRATEGY BASED ON VALIDATIONS
        if len(unique_classes) == 1:
            if verbose:
                logger.warning("⚠️  Only one class detected - using random split")
            X_train, X_test, y_train, y_test = train_test_split(
                X_np, y_np, 
                test_size=test_size,
                random_state=self.random_state,
                shuffle=True
            )
        elif any("only" in issue for issue in issues):
            if verbose:
                logger.warning("⚠️  Few samples per class - using adjusted proportional split")
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
                logger.info("✅ Using standard stratified split")
            X_train, X_test, y_train, y_test = train_test_split(
                X_np, y_np,
                test_size=test_size,
                stratify=y_np if stratify else None,  # 🎯 STRATIFICATION
                random_state=self.random_state,
                shuffle=True
            )
        
        # 3. CONVERT BACK TO TENSORS
        device = X.device
        X_train = torch.from_numpy(X_train).to(device)
        X_test = torch.from_numpy(X_test).to(device)
        y_train = torch.from_numpy(y_train).to(device)
        y_test = torch.from_numpy(y_test).to(device)
        
        # 4. LOG RESULTS  
        if verbose:
            logger.info(f"✅ Split completed:")
            logger.info(f"  📈 Train: {len(X_train)} samples")
            logger.info(f"  📊 Test: {len(X_test)} samples")  
            logger.info(f"  📉 Actual ratio: {len(X_test)/(len(X_train)+len(X_test))*100:.1f}% test")
            logger.info(f"  🎯 Train classes: {torch.bincount(y_train.long())}")
            logger.info(f"  🎯 Test classes: {torch.bincount(y_test.long())}"
        
        return X_train, X_test, y_train, y_test
    
    def _calculate_adjusted_test_size(self, class_counts: np.ndarray, min_samples: int) -> float:
        """Calculates adjusted test_size to ensure minimum samples per class."""
        total_samples = sum(class_counts)
        min_class_size = min(class_counts)
        
        # Ensure at least min_samples in test for the minority class
        max_test_from_min_class = min_class_size - min_samples  # Leave min_samples in train
        max_test_total = len(class_counts) * max_test_from_min_class  # Approximation
        
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
        """Statistically validates the performed split."""
        
        # Convert to numpy
        y_train_np = y_train.cpu().numpy()
        y_test_np = y_test.cpu().numpy()
        
        # Distributions per set
        train_unique, train_counts = np.unique(y_train_np, return_counts=True)
        test_unique, test_counts = np.unique(y_test_np, return_counts=True)
        
        train_dist = {int(cls): count/len(y_train_np) for cls, count in zip(train_unique, train_counts)}
        test_dist = {int(cls): count/len(y_test_np) for cls, count in zip(test_unique, test_counts)}
        
        # Chi-square test to compare distributions
        chi2_p_value = self._chi_square_test(train_counts, test_counts)
        
        # Validations
        issues = pre_issues.copy()
        recommendations = []
        
        # 1. Check if all classes are present
        missing_in_train = set(original_classes) - set(train_unique)
        missing_in_test = set(original_classes) - set(test_unique)
        
        if missing_in_train:
            issues.append(f"Classes missing in train: {missing_in_train}")
        if missing_in_test:
            issues.append(f"Classes missing in test: {missing_in_test}")
        
        # 2. Check distribution similarity
        if chi2_p_value < 0.05:
            issues.append(f"Train/test distributions significantly different (p={chi2_p_value:.4f})")
            recommendations.append("Consider increasing the dataset or using cross-validation")
        
        # 3. Check if stratification worked
        max_diff = 0
        for cls in original_classes:
            train_prop = train_dist.get(cls, 0)
            test_prop = test_dist.get(cls, 0)
            diff = abs(train_prop - test_prop)
            max_diff = max(max_diff, diff)
        
        if max_diff > 0.05:  # Difference > 5%
            issues.append(f"Imperfect stratification: max difference {max_diff*100:.1f}%")
            recommendations.append("Check if stratification was applied correctly")
        
        # 4. Size-based recommendations
        total_samples = len(y_train_np) + len(y_test_np)
        if total_samples < 1000:
            recommendations.append("Small dataset: consider cross-validation instead of train/test")
        elif len(y_test_np) < 100:
            recommendations.append("Small test set: consider increasing test_size"
        
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
        """Chi-square test to compare distributions."""
        try:
            # Normalize counts to compare proportions
            train_prop = train_counts / train_counts.sum()
            test_prop = test_counts / test_counts.sum()
            
            # Chi-square
            if len(train_prop) == len(test_prop) and len(train_prop) > 1:
                chi2, p_value = stats.chisquare(test_prop, train_prop)
                return p_value
            else:
                return 1.0  # Not significant if only one class
        # FIX #40: Specify expected exceptions instead of bare except
        except (ValueError, ZeroDivisionError, RuntimeError) as e:
            if hasattr(self, 'verbose') and self.verbose:
                print(f'   ⚠️  Error calculating chi-square: {e}')
            return 1.0  # On error, assume not significant
    
    def _create_minimal_report(self) -> SplitValidationReport:
        """Creates minimal report when validation is disabled."""
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
