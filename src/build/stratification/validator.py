"""
Split validator for assessing the quality of train/test/validation splits.

This module provides comprehensive metrics to evaluate whether splits are
balanced, representative, and appropriate for molecular machine learning tasks.
"""

import numpy as np
from typing import Union, Optional, Tuple, List, Dict, Any
from pathlib import Path
import logging
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.model_selection import cross_val_score
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

from build.core.base_builder import BaseBuilder
from build.core.config import BuildConfig
from build.core.exceptions import BuildException
from .cluster_analyzer import ClusterAnalyzer


class SplitValidator(BaseBuilder):
    """
    Validator for assessing the quality of data splits.
    
    Provides comprehensive validation metrics for train/test/validation splits
    in molecular machine learning contexts.
    """
    
    def __init__(self, 
                 config: Optional[BuildConfig] = None,
                 **kwargs):
        """
        Initialize split validator.

        Args:
            config: Build configuration
            **kwargs: Additional configuration options
        """
        super().__init__(config, **kwargs)
        self.cluster_analyzer = ClusterAnalyzer(config)
        
    def _validate_config(self) -> None:
        """Validate configuration."""
        # No specific validation needed for validator
        pass
    
    def validate_split_distribution(self, 
                                  labels: np.ndarray,
                                  train_idx: np.ndarray,
                                  val_idx: np.ndarray,
                                  test_idx: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Validate that label distributions are balanced across splits.

        Args:
            labels: Target labels for the dataset
            train_idx: Indices for training set
            val_idx: Indices for validation set
            test_idx: Indices for test set

        Returns:
            Dictionary with distribution metrics for each split
        """
        result = {}
        
        # Get labels for each split
        train_labels = labels[train_idx] if len(train_idx) > 0 else np.array([])
        val_labels = labels[val_idx] if len(val_idx) > 0 else np.array([])
        test_labels = labels[test_idx] if len(test_idx) > 0 else np.array([])
        
        # Calculate distribution for each split
        splits = {
            'train': train_labels,
            'validation': val_labels,
            'test': test_labels
        }
        
        for split_name, split_labels in splits.items():
            if len(split_labels) == 0:
                result[split_name] = {'empty': True}
                continue
            
            # Calculate distribution metrics
            unique, counts = np.unique(split_labels, return_counts=True)
            proportions = counts / len(split_labels)
            
            result[split_name] = {
                'size': len(split_labels),
                'n_unique_labels': len(unique),
                'label_distribution': dict(zip(unique, counts)),
                'label_proportions': dict(zip(unique, proportions)),
                'entropy': -np.sum(proportions * np.log2(proportions + 1e-10))  # Add small value to avoid log(0)
            }
        
        # Calculate distribution similarity across splits (for classification)
        if len(np.unique(labels)) < len(labels):  # If it's a classification task
            all_distributions = []
            for split_name in ['train', 'validation', 'test']:
                if split_name in result and 'label_proportions' in result[split_name]:
                    all_distributions.append(result[split_name]['label_proportions'])
            
            if len(all_distributions) >= 2:
                # Calculate distribution similarity (Jensen-Shannon divergence approximation)
                # Using simple absolute difference between proportions
                dist_similarities = []
                for i in range(len(all_distributions)):
                    for j in range(i+1, len(all_distributions)):
                        common_labels = set(all_distributions[i].keys()) & set(all_distributions[j].keys())
                        if len(common_labels) > 0:
                            total_diff = 0
                            for label in common_labels:
                                total_diff += abs(all_distributions[i][label] - all_distributions[j][label])
                            avg_diff = total_diff / len(common_labels)
                            dist_similarities.append(avg_diff)
                
                if dist_similarities:
                    avg_distribution_diff = np.mean(dist_similarities)
                    result['label_distribution_similarity'] = 1 - avg_distribution_diff  # Convert to similarity (higher is better)
        
        return result
    
    def validate_similarity_preservation(self,
                                       embeddings: np.ndarray,
                                       train_idx: np.ndarray,
                                       val_idx: np.ndarray,
                                       test_idx: np.ndarray) -> Dict[str, float]:
        """
        Validate that similarity relationships are preserved appropriately across splits.

        Args:
            embeddings: Embedding matrix for the dataset
            train_idx: Indices for training set
            val_idx: Indices for validation set
            test_idx: Indices for test set

        Returns:
            Dictionary with similarity preservation metrics
        """
        result = {}
        
        # Calculate inter-split similarities
        if len(train_idx) > 0 and len(test_idx) > 0:
            train_embeddings = embeddings[train_idx]
            test_embeddings = embeddings[test_idx]
            
            # Calculate average similarity between train and test
            cross_similarities = []
            for i in range(min(len(train_embeddings), 500)):  # Sample for efficiency
                train_emb = train_embeddings[i].reshape(1, -1)
                # Calculate similarities with a sample of test embeddings
                test_sample_idx = np.random.choice(len(test_embeddings), 
                                                  min(100, len(test_embeddings)), 
                                                  replace=False)
                test_sample = test_embeddings[test_sample_idx]
                
                sims = np.dot(train_emb, test_sample.T).flatten()
                cross_similarities.extend(sims)
            
            result['avg_train_test_similarity'] = float(np.mean(cross_similarities))
            result['std_train_test_similarity'] = float(np.std(cross_similarities))
        
        # Calculate intra-split similarities
        splits = {
            'train': train_idx,
            'validation': val_idx,
            'test': test_idx
        }
        
        for split_name, indices in splits.items():
            if len(indices) > 1:
                split_embeddings = embeddings[indices]
                
                # Calculate average intra-split similarity (sample for efficiency)
                if len(split_embeddings) > 100:
                    # Sample for efficiency
                    sample_indices = np.random.choice(len(split_embeddings), 
                                                     min(100, len(split_embeddings)), 
                                                     replace=False)
                    sample_embeddings = split_embeddings[sample_indices]
                else:
                    sample_embeddings = split_embeddings
                
                if len(sample_embeddings) > 1:
                    # Calculate pairwise similarities within split
                    similarities = []
                    for i in range(len(sample_embeddings)):
                        for j in range(i+1, len(sample_embeddings)):
                            sim = np.dot(sample_embeddings[i], sample_embeddings[j])
                            similarities.append(sim)
                    
                    if similarities:
                        result[f'avg_{split_name}_similarity'] = float(np.mean(similarities))
                        result[f'std_{split_name}_similarity'] = float(np.std(similarities))
        
        return result
    
    def validate_split_diversity(self,
                               embeddings: np.ndarray,
                               train_idx: np.ndarray,
                               val_idx: np.ndarray,
                               test_idx: np.ndarray) -> Dict[str, float]:
        """
        Validate the diversity of each split using embedding characteristics.

        Args:
            embeddings: Embedding matrix for the dataset
            train_idx: Indices for training set
            val_idx: Indices for validation set
            test_idx: Indices for test set

        Returns:
            Dictionary with diversity metrics for each split
        """
        result = {}
        
        splits = {
            'train': train_idx,
            'validation': val_idx,
            'test': test_idx
        }
        
        for split_name, indices in splits.items():
            if len(indices) > 0:
                split_embeddings = embeddings[indices]
                
                # Calculate diversity metrics
                # 1. Average distance from centroid
                centroid = np.mean(split_embeddings, axis=0)
                distances_from_centroid = np.sqrt(np.sum((split_embeddings - centroid) ** 2, axis=1))
                avg_distance = np.mean(distances_from_centroid)
                
                # 2. Coverage of embedding space (using PCA variance)
                if split_embeddings.shape[1] > 1:  # More than 1 feature
                    centered = split_embeddings - centroid
                    # Calculate covariance matrix
                    cov_matrix = np.cov(centered.T)
                    # Calculate eigenvalues to assess spread
                    eigenvals = np.linalg.eigvals(cov_matrix)
                    total_variance = np.sum(eigenvals)
                else:
                    total_variance = np.var(split_embeddings)
                
                result[f'{split_name}_diversity_avg_distance'] = float(avg_distance)
                result[f'{split_name}_diversity_variance'] = float(np.real(total_variance))  # Take real part if complex
                result[f'{split_name}_size'] = len(indices)
        
        return result
    
    def validate_novelty(self,
                        embeddings: np.ndarray,
                        train_idx: np.ndarray,
                        test_idx: np.ndarray,
                        novelty_threshold: float = 0.8) -> Dict[str, float]:
        """
        Validate that test set contains appropriately novel compounds/samples.

        Args:
            embeddings: Embedding matrix for the dataset
            train_idx: Indices for training set
            test_idx: Indices for test set
            novelty_threshold: Similarity threshold for considering samples as similar

        Returns:
            Dictionary with novelty metrics
        """
        if len(train_idx) == 0 or len(test_idx) == 0:
            return {'novelty_assessment': 'insufficient_data'}
        
        train_embeddings = embeddings[train_idx]
        test_embeddings = embeddings[test_idx]
        
        result = {}
        
        # Calculate novelty metrics
        highly_similar_count = 0  # Count of test samples highly similar to any training sample
        min_similarities = []  # Minimum similarity of each test sample to training set
        
        # Sample for efficiency
        test_sample_size = min(len(test_embeddings), 500)
        test_sample_indices = np.random.choice(len(test_embeddings), test_sample_size, replace=False)
        sample_test_embeddings = test_embeddings[test_sample_indices]
        
        for test_emb in sample_test_embeddings:
            test_emb = test_emb.reshape(1, -1)
            
            # Calculate similarities with training set (sample for efficiency)
            train_sample_size = min(len(train_embeddings), 500)
            train_sample_indices = np.random.choice(len(train_embeddings), train_sample_size, replace=False)
            sample_train_embeddings = train_embeddings[train_sample_indices]
            
            similarities = np.dot(test_emb, sample_train_embeddings.T).flatten()
            min_sim = np.min(similarities)
            max_sim = np.max(similarities)
            
            min_similarities.append(min_sim)
            
            if max_sim >= novelty_threshold:
                highly_similar_count += 1
        
        result['highly_similar_test_ratio'] = highly_similar_count / len(sample_test_embeddings)
        result['avg_min_similarity_to_train'] = float(np.mean(min_similarities))
        result['std_min_similarity_to_train'] = float(np.std(min_similarities))
        result['test_set_novelty_score'] = 1 - result['highly_similar_test_ratio']  # Higher is more novel
        
        # Assess if test set is appropriately challenging
        if result['test_set_novelty_score'] > 0.9:
            result['challenge_level'] = 'very_high'  # Very novel, potentially too hard
        elif result['test_set_novelty_score'] > 0.7:
            result['challenge_level'] = 'high'  # Novel but reasonable
        elif result['test_set_novelty_score'] > 0.4:
            result['challenge_level'] = 'moderate'  # Balanced
        else:
            result['challenge_level'] = 'low'  # Too similar to training set
        
        return result
    
    def validate_splits_comprehensively(self,
                                      embeddings: np.ndarray,
                                      labels: np.ndarray,
                                      train_idx: np.ndarray,
                                      val_idx: np.ndarray,
                                      test_idx: np.ndarray) -> Dict[str, Any]:
        """
        Perform comprehensive validation of splits.

        Args:
            embeddings: Embedding matrix for the dataset
            labels: Target labels for the dataset
            train_idx: Indices for training set
            val_idx: Indices for validation set
            test_idx: Indices for test set

        Returns:
            Comprehensive validation report
        """
        self.logger.info("Performing comprehensive split validation...")
        
        # Combine all validation metrics
        validation_report = {
            'split_sizes': {
                'train': len(train_idx),
                'validation': len(val_idx),
                'test': len(test_idx),
                'total': len(train_idx) + len(val_idx) + len(test_idx)
            }
        }
        
        # Add distribution validation
        validation_report['label_distribution'] = self.validate_split_distribution(
            labels, train_idx, val_idx, test_idx
        )
        
        # Add similarity preservation validation
        validation_report['similarity_preservation'] = self.validate_similarity_preservation(
            embeddings, train_idx, val_idx, test_idx
        )
        
        # Add diversity validation
        validation_report['diversity_metrics'] = self.validate_split_diversity(
            embeddings, train_idx, val_idx, test_idx
        )
        
        # Add novelty validation (train vs test)
        validation_report['novelty_metrics'] = self.validate_novelty(
            embeddings, train_idx, test_idx
        )
        
        # Overall assessment
        overall_score = self._calculate_overall_split_quality(validation_report)
        validation_report['overall_quality_score'] = overall_score
        
        # Identify potential issues
        validation_report['issues'] = self._identify_split_issues(validation_report)
        
        return validation_report
    
    def _calculate_overall_split_quality(self, validation_report: Dict[str, Any]) -> float:
        """
        Calculate an overall quality score for the splits.

        Args:
            validation_report: Comprehensive validation report

        Returns:
            Overall quality score (0-1, higher is better)
        """
        score = 0.0
        weight_sum = 0.0
        
        # Weight different aspects
        # Distribution similarity (for classification)
        if 'label_distribution_similarity' in validation_report['label_distribution']:
            distribution_score = validation_report['label_distribution']['label_distribution_similarity']
            score += 0.3 * distribution_score
            weight_sum += 0.3
        
        # Novelty (want balance - not too similar but not too different)
        if 'test_set_novelty_score' in validation_report['novelty_metrics']:
            novelty_score = validation_report['novelty_metrics']['test_set_novelty_score']
            # Ideal novelty is around 0.6-0.8 (challenging but not impossible)
            if 0.4 <= novelty_score <= 0.8:
                # Score based on proximity to ideal range
                ideal_score = 1.0 - abs(novelty_score - 0.6) / 0.4
                ideal_score = max(0.0, min(1.0, ideal_score))  # Clamp to [0,1]
            else:
                # Lower score for too similar or too dissimilar
                ideal_score = max(0.0, 1.0 - abs(novelty_score - 0.6))
            
            score += 0.4 * ideal_score
            weight_sum += 0.4
        
        # Diversity (higher is better)
        for split_name in ['train', 'validation', 'test']:
            diversity_key = f'{split_name}_diversity_variance'
            if diversity_key in validation_report['diversity_metrics']:
                # Normalize diversity metric
                div_score = min(1.0, validation_report['diversity_metrics'][diversity_key] / 10.0)  # Arbitrary normalization
                score += 0.1 * div_score
                weight_sum += 0.1
                break  # Just use one split for diversity scoring
        
        # Normalize by weight sum
        if weight_sum > 0:
            score = score / weight_sum
        else:
            score = 0.5  # Default score if no metrics available
        
        return min(1.0, max(0.0, score))  # Clamp to [0,1]
    
    def _identify_split_issues(self, validation_report: Dict[str, Any]) -> List[str]:
        """
        Identify potential issues with the splits.

        Args:
            validation_report: Comprehensive validation report

        Returns:
            List of identified issues
        """
        issues = []
        
        # Check for imbalanced splits
        sizes = validation_report['split_sizes']
        total = sizes['total']
        if total > 0:
            train_ratio = sizes['train'] / total
            val_ratio = sizes['validation'] / total
            test_ratio = sizes['test'] / total
            
            if test_ratio < 0.1:
                issues.append("Test set too small (<10%)")
            if val_ratio < 0.05:
                issues.append("Validation set too small (<5%)")
            if train_ratio < 0.7:
                issues.append("Training set too small (<70%)")
        
        # Check for label distribution issues
        label_dist = validation_report['label_distribution']
        if 'label_distribution_similarity' in label_dist:
            if label_dist['label_distribution_similarity'] < 0.7:
                issues.append("Label distributions differ significantly across splits")
        
        # Check for novelty issues
        novelty = validation_report['novelty_metrics']
        if 'challenge_level' in novelty:
            if novelty['challenge_level'] == 'very_high':
                issues.append("Test set too challenging (very novel compounds)")
            elif novelty['challenge_level'] == 'low':
                issues.append("Test set too similar to training set")
        
        # Check for diversity issues
        diversity = validation_report['diversity_metrics']
        for split_name in ['train', 'validation', 'test']:
            diversity_key = f'{split_name}_diversity_variance'
            if diversity_key in diversity and diversity[diversity_key] < 0.01:
                issues.append(f"{split_name.capitalize()} set has low diversity")
        
        return issues
    
    def generate_validation_report(self, 
                                 validation_report: Dict[str, Any],
                                 output_path: Optional[Union[str, Path]] = None) -> str:
        """
        Generate a human-readable validation report.

        Args:
            validation_report: Comprehensive validation report
            output_path: Optional path to save the report

        Returns:
            Formatted validation report as string
        """
        report_lines = [
            "COMPREHENSIVE SPLIT VALIDATION REPORT",
            "=" * 50,
            "",
            "Split Sizes:",
            f"  Train: {validation_report['split_sizes']['train']} ({validation_report['split_sizes']['train']/validation_report['split_sizes']['total']*100:.1f}%)",
            f"  Validation: {validation_report['split_sizes']['validation']} ({validation_report['split_sizes']['validation']/validation_report['split_sizes']['total']*100:.1f}%)",
            f"  Test: {validation_report['split_sizes']['test']} ({validation_report['split_sizes']['test']/validation_report['split_sizes']['total']*100:.1f}%)",
            "",
            f"Overall Quality Score: {validation_report['overall_quality_score']:.3f}",
            ""
        ]
        
        # Add label distribution info
        label_dist = validation_report['label_distribution']
        if 'label_distribution_similarity' in label_dist:
            report_lines.append(f"Label Distribution Similarity: {label_dist['label_distribution_similarity']:.3f}")
        
        # Add novelty info
        novelty = validation_report['novelty_metrics']
        if 'test_set_novelty_score' in novelty:
            report_lines.append(f"Test Set Novelty Score: {novelty['test_set_novelty_score']:.3f}")
            report_lines.append(f"Challenge Level: {novelty['challenge_level']}")
        
        # Add issues
        if validation_report['issues']:
            report_lines.extend([
                "",
                "Identified Issues:",
            ])
            for issue in validation_report['issues']:
                report_lines.append(f"  - {issue}")
        else:
            report_lines.extend([
                "",
                "No significant issues identified!"
            ])
        
        report = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            self.logger.info(f"Validation report saved to {output_path}")
        
        return report
    
    def build(self) -> dict:
        """
        Build method for BaseBuilder compatibility.

        Returns:
            Dictionary with validator information
        """
        return {
            'validator_initialized': True
        }