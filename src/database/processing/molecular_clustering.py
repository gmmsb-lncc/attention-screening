"""
Molecular Clustering Module.

Provides functionality for clustering molecules based on structural similarity
using RDKit fingerprints and Tanimoto similarity.
"""

import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from sklearn.manifold import TSNE
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem, DataStructs
from matplotlib.colors import Normalize

import sys
import os
from pathlib import Path

# Add the database directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.base_analyzer import BaseAnalyzer
from core.config import DatabaseConfig
from core.exceptions import ProcessingError


class MolecularClusterer(BaseAnalyzer):
    """
    Molecular clustering using fingerprint-based similarity.
    
    This class provides functionality to cluster molecules based on their
    structural similarity using Morgan fingerprints and Tanimoto similarity.
    Maintains compatibility with the original cluster.py functionality.
    """

    def __init__(self, config: DatabaseConfig = None, smiles_file_path: str = None):
        """
        Initialize the molecular clusterer.
        
        Args:
            config: DatabaseConfig instance
            smiles_file_path: Path to file containing SMILES data
        """
        super().__init__(config)
        self.smiles_file_path = smiles_file_path
        self.fingerprints = []
        self._clusters = None

    def load_smiles_data(self, smile_column: str = 'canonical_smiles') -> pd.DataFrame:
        """
        Load SMILES data from file.
        
        Args:
            smile_column: Name of column containing SMILES strings
            
        Returns:
            Loaded DataFrame with SMILES data
        """
        if not self.smiles_file_path:
            raise ProcessingError("No SMILES file path provided")
        
        self.load_data(self.smiles_file_path)
        self._data = self._data.dropna(subset=[smile_column])
        return self._data

    @staticmethod
    def smiles_to_fingerprint(smiles: str):
        """
        Convert SMILES string to Morgan fingerprint.

        Args:
            smiles: SMILES string

        Returns:
            Morgan fingerprint or None if conversion fails
        """
        mol = Chem.MolFromSmiles(smiles)
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) if mol else None

    def generate_fingerprints(self, smile_column: str = 'canonical_smiles', 
                            use_parallel: bool = True, batch_size: int = None) -> list:
        """
        Generate fingerprints for all molecules.
        
        Args:
            smile_column: Column containing SMILES strings
            use_parallel: Whether to use parallel processing
            batch_size: Batch size for parallel processing
            
        Returns:
            List of fingerprints
        """
        if self._data is None:
            raise ProcessingError("No data loaded. Call load_smiles_data() first.")
        
        smiles_list = self._data[smile_column].tolist()
        batch_size = batch_size or self.config.batch_size
        
        if use_parallel and self.config.get('use_parallel', True):
            return self._parallel_generate_fingerprints(smiles_list, batch_size)
        else:
            return self._sequential_generate_fingerprints(smiles_list)

    def _parallel_generate_fingerprints(self, smiles_list: list, batch_size: int) -> list:
        """Generate fingerprints using parallel processing."""
        num_cpus = self.config.num_workers
        
        with ProcessPoolExecutor(max_workers=num_cpus) as executor:
            results = list(tqdm(
                executor.map(self.smiles_to_fingerprint, smiles_list, chunksize=batch_size),
                total=len(smiles_list),
                desc="Generating fingerprints"
            ))

        # Filter out None results
        self.fingerprints = [fp for fp in results if fp is not None]
        return self.fingerprints

    def _sequential_generate_fingerprints(self, smiles_list: list) -> list:
        """Generate fingerprints sequentially."""
        self.fingerprints = []
        for smiles in tqdm(smiles_list, desc="Generating fingerprints"):
            fp = self.smiles_to_fingerprint(smiles)
            if fp is not None:
                self.fingerprints.append(fp)
        return self.fingerprints

    def cluster_by_similarity(self, threshold: float = None) -> list:
        """
        Cluster molecules by Tanimoto similarity.
        
        Args:
            threshold: Similarity threshold for clustering
            
        Returns:
            List of clusters (each cluster is a list of indices)
        """
        if not self.fingerprints:
            raise ProcessingError("No fingerprints available. Call generate_fingerprints() first.")
        
        threshold = threshold or self.config.similarity_threshold
        num_fps = len(self.fingerprints)
        clusters = []
        visited = set()

        for i in tqdm(range(num_fps), desc="Clustering molecules"):
            if i in visited:
                continue

            cluster = [i]
            visited.add(i)
            
            for j in range(i + 1, num_fps):
                if j not in visited:
                    similarity = DataStructs.TanimotoSimilarity(
                        self.fingerprints[i], 
                        self.fingerprints[j]
                    )
                    if similarity >= threshold:
                        cluster.append(j)
                        visited.add(j)

            clusters.append(cluster)

        self._clusters = clusters
        self._results['clusters'] = clusters
        self._results['num_clusters'] = len(clusters)
        self._results['threshold'] = threshold
        
        return clusters

    def perform_tsne(self, perplexity: int = 30, n_components: int = 2) -> np.ndarray:
        """
        Perform t-SNE dimensionality reduction on fingerprints.
        
        Args:
            perplexity: t-SNE perplexity parameter
            n_components: Number of dimensions for output
            
        Returns:
            t-SNE coordinates
        """
        if not self.fingerprints:
            raise ProcessingError("No fingerprints available. Call generate_fingerprints() first.")
        
        # Convert fingerprints to numpy array
        fp_array = np.array([list(fp) for fp in self.fingerprints])
        
        # Perform t-SNE
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
        tsne_coords = tsne.fit_transform(fp_array)
        
        self._results['tsne_coords'] = tsne_coords
        return tsne_coords

    def plot_clusters(self, output_path: str = None, figsize: tuple = None) -> None:
        """
        Plot cluster visualization using t-SNE coordinates.
        
        Args:
            output_path: Path to save plot
            figsize: Figure size tuple
        """
        if 'tsne_coords' not in self._results or 'clusters' not in self._results:
            raise ProcessingError("Need both t-SNE coordinates and clusters. Run perform_tsne() and cluster_by_similarity() first.")
        
        figsize = figsize or self.config.get('figure_size', (12, 8))
        tsne_coords = self._results['tsne_coords']
        clusters = self._results['clusters']
        
        plt.figure(figsize=figsize)
        
        # Assign colors to clusters
        colors = plt.cm.tab10(np.linspace(0, 1, len(clusters)))
        
        for i, cluster in enumerate(clusters):
            if len(cluster) > 1:  # Only plot clusters with more than one molecule
                cluster_coords = tsne_coords[cluster]
                plt.scatter(cluster_coords[:, 0], cluster_coords[:, 1], 
                          c=[colors[i]], label=f'Cluster {i} (n={len(cluster)})', 
                          alpha=0.7, s=50)
        
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.title(f'Molecular Clusters (threshold={self._results["threshold"]:.2f})')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.config.get('dpi', 300), bbox_inches='tight')
        
        if self.config.get('save_plots', True):
            plt.show()

    def get_cluster_statistics(self) -> dict:
        """
        Get statistics about the clustering results.
        
        Returns:
            Dictionary with clustering statistics
        """
        if not self._clusters:
            raise ProcessingError("No clusters available. Run cluster_by_similarity() first.")
        
        cluster_sizes = [len(cluster) for cluster in self._clusters]
        
        stats = {
            'total_molecules': len(self.fingerprints),
            'num_clusters': len(self._clusters),
            'singleton_clusters': sum(1 for size in cluster_sizes if size == 1),
            'largest_cluster_size': max(cluster_sizes),
            'average_cluster_size': np.mean(cluster_sizes),
            'median_cluster_size': np.median(cluster_sizes)
        }
        
        self._results['statistics'] = stats
        return stats

    def analyze(self) -> dict:
        """
        Perform complete clustering analysis.
        
        Returns:
            Dictionary with analysis results
        """
        if self._data is None:
            raise ProcessingError("No data loaded. Call load_smiles_data() first.")
        
        # Generate fingerprints
        self.generate_fingerprints()
        
        # Perform clustering
        self.cluster_by_similarity()
        
        # Perform t-SNE
        self.perform_tsne()
        
        # Get statistics
        self.get_cluster_statistics()
        
        return self._results

    def save_clusters(self, output_path: str) -> None:
        """
        Save clustering results to file.
        
        Args:
            output_path: Path to save results
        """
        if not self._clusters:
            raise ProcessingError("No clusters to save. Run cluster_by_similarity() first.")
        
        with open(output_path, 'wb') as f:
            pickle.dump({
                'clusters': self._clusters,
                'fingerprints': self.fingerprints,
                'results': self._results
            }, f)

    def load_clusters(self, input_path: str) -> None:
        """
        Load clustering results from file.
        
        Args:
            input_path: Path to load results from
        """
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
            self._clusters = data['clusters']
            self.fingerprints = data['fingerprints']
            self._results = data['results']
