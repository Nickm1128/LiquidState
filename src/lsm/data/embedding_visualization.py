#!/usr/bin/env python3
"""
Embedding Visualization Tools for Sinusoidal Embeddings.

This module provides comprehensive visualization utilities for analyzing
sinusoidal embedding patterns, frequency characteristics, and clustering behavior.
"""

import os
import numpy as np
import tensorflow as tf
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
import warnings

from ..utils.lsm_logging import get_logger

logger = get_logger(__name__)

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import LinearSegmentedColormap
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib and seaborn not available - visualization features disabled")

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available - clustering and dimensionality reduction disabled")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("plotly not available - interactive visualizations disabled")


@dataclass
class VisualizationConfig:
    """Configuration for embedding visualization."""
    
    # Figure settings
    figure_size: Tuple[int, int] = (15, 10)
    dpi: int = 300
    style: str = 'seaborn-v0_8'
    color_palette: str = 'viridis'
    
    # Analysis settings
    max_positions: int = 200
    max_dimensions: int = 128
    sample_tokens: int = 1000
    
    # Clustering settings
    n_clusters: int = 8
    clustering_method: str = 'kmeans'  # 'kmeans', 'hierarchical'
    
    # Dimensionality reduction settings
    pca_components: int = 2
    tsne_perplexity: float = 30.0
    tsne_n_iter: int = 1000
    
    # Interactive settings
    interactive: bool = False
    save_html: bool = False


class EmbeddingVisualizer:
    """
    Comprehensive visualization tools for sinusoidal embeddings.
    
    This class provides various visualization methods for analyzing
    sinusoidal embedding patterns, frequency characteristics, and
    clustering behavior.
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Initialize the embedding visualizer.
        
        Args:
            config: Visualization configuration
        """
        self.config = config or VisualizationConfig()
        
        # Check dependencies
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib not available - static visualizations disabled")
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available - clustering analysis disabled")
        if not PLOTLY_AVAILABLE:
            logger.warning("plotly not available - interactive visualizations disabled")
        
        # Set matplotlib style if available
        if MATPLOTLIB_AVAILABLE:
            try:
                plt.style.use(self.config.style)
            except OSError:
                logger.warning(f"Style '{self.config.style}' not available, using default")
                plt.style.use('default')
    
    def visualize_embedding_patterns(
        self, 
        embedder, 
        save_path: Optional[str] = None,
        show_plot: bool = True
    ) -> Dict[str, Any]:
        """
        Create comprehensive visualization of embedding patterns.
        
        Args:
            embedder: ConfigurableSinusoidalEmbedder instance
            save_path: Optional path to save the visualization
            show_plot: Whether to display the plot
            
        Returns:
            Dictionary containing analysis results
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("matplotlib required for pattern visualization")
            return {}
        
        # Get embedding patterns
        patterns = embedder.get_embedding_patterns(self.config.max_positions)
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Positional encoding heatmap
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_positional_heatmap(ax1, patterns)
        
        # 2. Frequency spectrum analysis
        ax2 = fig.add_subplot(gs[0, 1])
        freq_analysis = self._plot_frequency_spectrum(ax2, patterns)
        
        # 3. Phase relationships
        ax3 = fig.add_subplot(gs[0, 2])
        phase_analysis = self._plot_phase_relationships(ax3, patterns)
        
        # 4. Embedding evolution over positions
        ax4 = fig.add_subplot(gs[1, 0])
        evolution_analysis = self._plot_embedding_evolution(ax4, patterns)
        
        # 5. Similarity matrix
        ax5 = fig.add_subplot(gs[1, 1])
        similarity_analysis = self._plot_similarity_matrix(ax5, patterns)
        
        # 6. Frequency distribution
        ax6 = fig.add_subplot(gs[1, 2])
        distribution_analysis = self._plot_frequency_distribution(ax6, patterns)
        
        # 7. Embedding variance analysis
        ax7 = fig.add_subplot(gs[2, 0])
        variance_analysis = self._plot_variance_analysis(ax7, patterns)
        
        # 8. Periodicity analysis
        ax8 = fig.add_subplot(gs[2, 1])
        periodicity_analysis = self._plot_periodicity_analysis(ax8, patterns)
        
        # 9. Energy distribution
        ax9 = fig.add_subplot(gs[2, 2])
        energy_analysis = self._plot_energy_distribution(ax9, patterns)
        
        plt.suptitle('Comprehensive Sinusoidal Embedding Analysis', fontsize=16, y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            logger.info(f"Pattern visualization saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        # Compile analysis results
        analysis_results = {
            'frequency_analysis': freq_analysis,
            'phase_analysis': phase_analysis,
            'evolution_analysis': evolution_analysis,
            'similarity_analysis': similarity_analysis,
            'distribution_analysis': distribution_analysis,
            'variance_analysis': variance_analysis,
            'periodicity_analysis': periodicity_analysis,
            'energy_analysis': energy_analysis
        }
        
        return analysis_results
    
    def _plot_positional_heatmap(self, ax, patterns):
        """Plot positional encoding heatmap."""
        encoding = patterns['positional_encoding']
        max_dims = min(self.config.max_dimensions, encoding.shape[1])
        max_pos = min(100, encoding.shape[0])
        
        im = ax.imshow(
            encoding[:max_pos, :max_dims].T,
            cmap=self.config.color_palette,
            aspect='auto',
            interpolation='nearest'
        )
        
        ax.set_title('Positional Encoding Heatmap')
        ax.set_xlabel('Position')
        ax.set_ylabel('Embedding Dimension')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Embedding Value')
    
    def _plot_frequency_spectrum(self, ax, patterns):
        """Plot frequency spectrum analysis."""
        frequencies = patterns['frequencies']
        
        # Plot frequency spectrum
        ax.plot(frequencies, 'o-', linewidth=2, markersize=4)
        ax.set_title('Frequency Spectrum')
        ax.set_xlabel('Dimension Pair Index')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        # Add frequency statistics
        freq_stats = {
            'mean_frequency': np.mean(frequencies),
            'std_frequency': np.std(frequencies),
            'min_frequency': np.min(frequencies),
            'max_frequency': np.max(frequencies),
            'frequency_range': np.max(frequencies) - np.min(frequencies)
        }
        
        # Add text box with statistics
        stats_text = f"Mean: {freq_stats['mean_frequency']:.4f}\n"
        stats_text += f"Std: {freq_stats['std_frequency']:.4f}\n"
        stats_text += f"Range: {freq_stats['frequency_range']:.4f}"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        return freq_stats
    
    def _plot_phase_relationships(self, ax, patterns):
        """Plot phase relationships between dimensions."""
        encoding = patterns['positional_encoding']
        
        # Calculate phase relationships for first few positions
        positions_to_analyze = min(50, encoding.shape[0])
        dims_to_analyze = min(16, encoding.shape[1])
        
        # Compute phase differences
        phase_matrix = np.zeros((dims_to_analyze, dims_to_analyze))
        
        for i in range(dims_to_analyze):
            for j in range(dims_to_analyze):
                if i != j:
                    # Compute cross-correlation to find phase difference
                    signal1 = encoding[:positions_to_analyze, i]
                    signal2 = encoding[:positions_to_analyze, j]
                    correlation = np.correlate(signal1, signal2, mode='full')
                    phase_diff = np.argmax(correlation) - len(signal1) + 1
                    phase_matrix[i, j] = phase_diff
        
        im = ax.imshow(phase_matrix, cmap='RdBu_r', vmin=-10, vmax=10)
        ax.set_title('Phase Relationships Between Dimensions')
        ax.set_xlabel('Dimension Index')
        ax.set_ylabel('Dimension Index')
        plt.colorbar(im, ax=ax, label='Phase Difference')
        
        phase_analysis = {
            'phase_matrix': phase_matrix,
            'mean_phase_diff': np.mean(np.abs(phase_matrix)),
            'max_phase_diff': np.max(np.abs(phase_matrix))
        }
        
        return phase_analysis
    
    def _plot_embedding_evolution(self, ax, patterns):
        """Plot how embeddings evolve over positions."""
        encoding = patterns['positional_encoding']
        positions = patterns['positions']
        
        # Select representative dimensions
        dims_to_plot = [0, encoding.shape[1]//4, encoding.shape[1]//2, encoding.shape[1]-1]
        colors = plt.cm.tab10(np.linspace(0, 1, len(dims_to_plot)))
        
        max_pos = min(100, len(positions))
        
        for i, dim in enumerate(dims_to_plot):
            ax.plot(positions[:max_pos], encoding[:max_pos, dim], 
                   color=colors[i], label=f'Dim {dim}', alpha=0.8, linewidth=1.5)
        
        ax.set_title('Embedding Evolution Over Positions')
        ax.set_xlabel('Position')
        ax.set_ylabel('Embedding Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Calculate evolution statistics
        evolution_stats = {
            'position_variance': np.var(encoding[:max_pos], axis=0),
            'mean_amplitude': np.mean(np.abs(encoding[:max_pos]), axis=0),
            'position_correlation': np.corrcoef(encoding[:max_pos].T)
        }
        
        return evolution_stats
    
    def _plot_similarity_matrix(self, ax, patterns):
        """Plot position similarity matrix."""
        encoding = patterns['positional_encoding']
        max_pos = min(50, encoding.shape[0])
        
        # Compute cosine similarity matrix
        embeddings = encoding[:max_pos]
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / (norms + 1e-8)
        similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
        
        im = ax.imshow(similarity_matrix, cmap='viridis', vmin=-1, vmax=1)
        ax.set_title('Position Similarity Matrix')
        ax.set_xlabel('Position')
        ax.set_ylabel('Position')
        plt.colorbar(im, ax=ax, label='Cosine Similarity')
        
        # Calculate similarity statistics
        similarity_stats = {
            'mean_similarity': np.mean(similarity_matrix),
            'similarity_std': np.std(similarity_matrix),
            'diagonal_similarity': np.diag(similarity_matrix),
            'off_diagonal_mean': np.mean(similarity_matrix[~np.eye(similarity_matrix.shape[0], dtype=bool)])
        }
        
        return similarity_stats
    
    def _plot_frequency_distribution(self, ax, patterns):
        """Plot frequency distribution analysis."""
        frequencies = patterns['frequencies']
        
        # Create histogram
        ax.hist(frequencies, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(np.mean(frequencies), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(frequencies):.4f}')
        ax.axvline(np.median(frequencies), color='green', linestyle='--', 
                  label=f'Median: {np.median(frequencies):.4f}')
        
        ax.set_title('Frequency Distribution')
        ax.set_xlabel('Frequency Value')
        ax.set_ylabel('Count')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        distribution_stats = {
            'frequency_histogram': np.histogram(frequencies, bins=20),
            'skewness': self._calculate_skewness(frequencies),
            'kurtosis': self._calculate_kurtosis(frequencies)
        }
        
        return distribution_stats
    
    def _plot_variance_analysis(self, ax, patterns):
        """Plot embedding variance analysis."""
        encoding = patterns['positional_encoding']
        
        # Calculate variance across positions for each dimension
        dimension_variance = np.var(encoding, axis=0)
        
        ax.plot(dimension_variance, 'o-', linewidth=2, markersize=3)
        ax.set_title('Embedding Variance by Dimension')
        ax.set_xlabel('Dimension Index')
        ax.set_ylabel('Variance')
        ax.grid(True, alpha=0.3)
        
        # Add mean variance line
        mean_variance = np.mean(dimension_variance)
        ax.axhline(mean_variance, color='red', linestyle='--', 
                  label=f'Mean Variance: {mean_variance:.4f}')
        ax.legend()
        
        variance_stats = {
            'dimension_variance': dimension_variance,
            'mean_variance': mean_variance,
            'variance_std': np.std(dimension_variance),
            'max_variance_dim': np.argmax(dimension_variance),
            'min_variance_dim': np.argmin(dimension_variance)
        }
        
        return variance_stats
    
    def _plot_periodicity_analysis(self, ax, patterns):
        """Plot periodicity analysis of embeddings."""
        encoding = patterns['positional_encoding']
        
        # Analyze periodicity using FFT for first few dimensions
        dims_to_analyze = min(8, encoding.shape[1])
        max_pos = min(200, encoding.shape[0])
        
        periodicities = []
        
        for dim in range(dims_to_analyze):
            signal = encoding[:max_pos, dim]
            fft = np.fft.fft(signal)
            freqs = np.fft.fftfreq(len(signal))
            
            # Find dominant frequency
            dominant_freq_idx = np.argmax(np.abs(fft[1:len(fft)//2])) + 1
            dominant_freq = freqs[dominant_freq_idx]
            period = 1 / abs(dominant_freq) if dominant_freq != 0 else float('inf')
            periodicities.append(period)
        
        ax.bar(range(dims_to_analyze), periodicities, alpha=0.7)
        ax.set_title('Periodicity Analysis by Dimension')
        ax.set_xlabel('Dimension Index')
        ax.set_ylabel('Period Length')
        ax.grid(True, alpha=0.3)
        
        periodicity_stats = {
            'periodicities': periodicities,
            'mean_period': np.mean([p for p in periodicities if p != float('inf')]),
            'period_variance': np.var([p for p in periodicities if p != float('inf')])
        }
        
        return periodicity_stats
    
    def _plot_energy_distribution(self, ax, patterns):
        """Plot energy distribution across dimensions."""
        encoding = patterns['positional_encoding']
        
        # Calculate energy (sum of squares) for each dimension
        energy_per_dim = np.sum(encoding**2, axis=0)
        
        ax.plot(energy_per_dim, 'o-', linewidth=2, markersize=3, color='orange')
        ax.set_title('Energy Distribution Across Dimensions')
        ax.set_xlabel('Dimension Index')
        ax.set_ylabel('Energy (Sum of Squares)')
        ax.grid(True, alpha=0.3)
        
        # Add cumulative energy
        ax2 = ax.twinx()
        cumulative_energy = np.cumsum(energy_per_dim) / np.sum(energy_per_dim)
        ax2.plot(cumulative_energy, 's-', color='red', alpha=0.6, 
                label='Cumulative Energy %')
        ax2.set_ylabel('Cumulative Energy Fraction')
        ax2.legend()
        
        energy_stats = {
            'energy_per_dimension': energy_per_dim,
            'total_energy': np.sum(energy_per_dim),
            'energy_concentration': np.sum(energy_per_dim[:10]) / np.sum(energy_per_dim),  # First 10 dims
            'energy_entropy': -np.sum((energy_per_dim / np.sum(energy_per_dim)) * 
                                    np.log(energy_per_dim / np.sum(energy_per_dim) + 1e-8))
        }
        
        return energy_stats
    
    def visualize_embedding_clustering(
        self, 
        embedder, 
        sample_tokens: Optional[List[int]] = None,
        save_path: Optional[str] = None,
        show_plot: bool = True
    ) -> Dict[str, Any]:
        """
        Visualize embedding clustering and similarity patterns.
        
        Args:
            embedder: ConfigurableSinusoidalEmbedder instance
            sample_tokens: Optional list of token IDs to analyze
            save_path: Optional path to save the visualization
            show_plot: Whether to display the plot
            
        Returns:
            Dictionary containing clustering analysis results
        """
        if not SKLEARN_AVAILABLE:
            logger.error("scikit-learn required for clustering analysis")
            return {}
        
        if not MATPLOTLIB_AVAILABLE:
            logger.error("matplotlib required for clustering visualization")
            return {}
        
        # Generate sample token embeddings
        if sample_tokens is None:
            sample_tokens = list(range(min(self.config.sample_tokens, embedder.config.vocab_size)))
        
        # Get token embeddings
        token_ids = tf.constant([sample_tokens])
        embeddings = embedder(token_ids, training=False)
        embeddings_np = embeddings.numpy()[0]  # Remove batch dimension
        
        # Perform clustering analysis
        clustering_results = self._perform_clustering_analysis(embeddings_np)
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Embedding Clustering Analysis', fontsize=16)
        
        # 1. PCA visualization
        pca_results = self._plot_pca_clustering(axes[0, 0], embeddings_np, clustering_results)
        
        # 2. t-SNE visualization
        tsne_results = self._plot_tsne_clustering(axes[0, 1], embeddings_np, clustering_results)
        
        # 3. Cluster centers heatmap
        centers_analysis = self._plot_cluster_centers(axes[0, 2], clustering_results)
        
        # 4. Silhouette analysis
        silhouette_analysis = self._plot_silhouette_analysis(axes[1, 0], embeddings_np, clustering_results)
        
        # 5. Cluster size distribution
        size_analysis = self._plot_cluster_sizes(axes[1, 1], clustering_results)
        
        # 6. Intra-cluster distances
        distance_analysis = self._plot_cluster_distances(axes[1, 2], embeddings_np, clustering_results)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            logger.info(f"Clustering visualization saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        # Compile results
        analysis_results = {
            'clustering_results': clustering_results,
            'pca_results': pca_results,
            'tsne_results': tsne_results,
            'centers_analysis': centers_analysis,
            'silhouette_analysis': silhouette_analysis,
            'size_analysis': size_analysis,
            'distance_analysis': distance_analysis
        }
        
        return analysis_results
    
    def _perform_clustering_analysis(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """Perform clustering analysis on embeddings."""
        # K-means clustering
        kmeans = KMeans(n_clusters=self.config.n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(embeddings, cluster_labels)
        
        # Calculate inertia (within-cluster sum of squares)
        inertia = kmeans.inertia_
        
        return {
            'kmeans_model': kmeans,
            'cluster_labels': cluster_labels,
            'cluster_centers': kmeans.cluster_centers_,
            'silhouette_score': silhouette_avg,
            'inertia': inertia,
            'n_clusters': self.config.n_clusters
        }
    
    def _plot_pca_clustering(self, ax, embeddings, clustering_results):
        """Plot PCA visualization with clustering."""
        pca = PCA(n_components=self.config.pca_components)
        embeddings_pca = pca.fit_transform(embeddings)
        
        # Plot clusters
        scatter = ax.scatter(
            embeddings_pca[:, 0], embeddings_pca[:, 1],
            c=clustering_results['cluster_labels'],
            cmap='tab10',
            alpha=0.6,
            s=20
        )
        
        # Plot cluster centers
        centers_pca = pca.transform(clustering_results['cluster_centers'])
        ax.scatter(
            centers_pca[:, 0], centers_pca[:, 1],
            c='red', marker='x', s=200, linewidths=3,
            label='Centroids'
        )
        
        ax.set_title(f'PCA Clustering (Silhouette: {clustering_results["silhouette_score"]:.3f})')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return {
            'pca_model': pca,
            'embeddings_pca': embeddings_pca,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'total_variance_explained': np.sum(pca.explained_variance_ratio_)
        }
    
    def _plot_tsne_clustering(self, ax, embeddings, clustering_results):
        """Plot t-SNE visualization with clustering."""
        # Use PCA for preprocessing if embeddings are high-dimensional
        if embeddings.shape[1] > 50:
            pca = PCA(n_components=50)
            embeddings_preprocessed = pca.fit_transform(embeddings)
        else:
            embeddings_preprocessed = embeddings
        
        tsne = TSNE(
            n_components=2,
            perplexity=min(self.config.tsne_perplexity, len(embeddings) - 1),
            n_iter=self.config.tsne_n_iter,
            random_state=42
        )
        embeddings_tsne = tsne.fit_transform(embeddings_preprocessed)
        
        # Plot clusters
        scatter = ax.scatter(
            embeddings_tsne[:, 0], embeddings_tsne[:, 1],
            c=clustering_results['cluster_labels'],
            cmap='tab10',
            alpha=0.6,
            s=20
        )
        
        ax.set_title('t-SNE Clustering')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.grid(True, alpha=0.3)
        
        return {
            'tsne_model': tsne,
            'embeddings_tsne': embeddings_tsne
        }
    
    def _plot_cluster_centers(self, ax, clustering_results):
        """Plot cluster centers heatmap."""
        centers = clustering_results['cluster_centers']
        
        im = ax.imshow(centers, cmap='RdBu_r', aspect='auto')
        ax.set_title('Cluster Centers Heatmap')
        ax.set_xlabel('Embedding Dimension')
        ax.set_ylabel('Cluster Index')
        plt.colorbar(im, ax=ax, label='Center Value')
        
        return {
            'cluster_centers': centers,
            'center_norms': np.linalg.norm(centers, axis=1),
            'center_similarities': np.corrcoef(centers)
        }
    
    def _plot_silhouette_analysis(self, ax, embeddings, clustering_results):
        """Plot silhouette analysis."""
        from sklearn.metrics import silhouette_samples
        
        cluster_labels = clustering_results['cluster_labels']
        silhouette_scores = silhouette_samples(embeddings, cluster_labels)
        
        y_lower = 10
        for i in range(self.config.n_clusters):
            cluster_silhouette_scores = silhouette_scores[cluster_labels == i]
            cluster_silhouette_scores.sort()
            
            size_cluster_i = cluster_silhouette_scores.shape[0]
            y_upper = y_lower + size_cluster_i
            
            color = plt.cm.tab10(i / self.config.n_clusters)
            ax.fill_betweenx(
                np.arange(y_lower, y_upper),
                0, cluster_silhouette_scores,
                facecolor=color, edgecolor=color, alpha=0.7
            )
            
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10
        
        ax.axvline(x=clustering_results['silhouette_score'], color="red", linestyle="--")
        ax.set_title('Silhouette Analysis')
        ax.set_xlabel('Silhouette Coefficient Values')
        ax.set_ylabel('Cluster Label')
        
        return {
            'silhouette_scores': silhouette_scores,
            'average_silhouette': clustering_results['silhouette_score']
        }
    
    def _plot_cluster_sizes(self, ax, clustering_results):
        """Plot cluster size distribution."""
        cluster_labels = clustering_results['cluster_labels']
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        
        bars = ax.bar(unique_labels, counts, alpha=0.7)
        ax.set_title('Cluster Size Distribution')
        ax.set_xlabel('Cluster Index')
        ax.set_ylabel('Number of Points')
        ax.grid(True, alpha=0.3)
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   str(count), ha='center', va='bottom')
        
        return {
            'cluster_sizes': counts,
            'size_variance': np.var(counts),
            'size_balance': np.min(counts) / np.max(counts)
        }
    
    def _plot_cluster_distances(self, ax, embeddings, clustering_results):
        """Plot intra-cluster distance analysis."""
        cluster_labels = clustering_results['cluster_labels']
        centers = clustering_results['cluster_centers']
        
        intra_distances = []
        for i in range(self.config.n_clusters):
            cluster_points = embeddings[cluster_labels == i]
            if len(cluster_points) > 0:
                distances = np.linalg.norm(cluster_points - centers[i], axis=1)
                intra_distances.append(distances)
        
        # Create box plot
        ax.boxplot(intra_distances, labels=range(self.config.n_clusters))
        ax.set_title('Intra-cluster Distance Distribution')
        ax.set_xlabel('Cluster Index')
        ax.set_ylabel('Distance to Centroid')
        ax.grid(True, alpha=0.3)
        
        return {
            'intra_distances': intra_distances,
            'mean_distances': [np.mean(dist) for dist in intra_distances],
            'distance_variance': [np.var(dist) for dist in intra_distances]
        }
    
    def create_interactive_visualization(
        self,
        embedder,
        save_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Create interactive visualization using plotly.
        
        Args:
            embedder: ConfigurableSinusoidalEmbedder instance
            save_path: Optional path to save HTML file
            
        Returns:
            HTML string if plotly available, None otherwise
        """
        if not PLOTLY_AVAILABLE:
            logger.error("plotly required for interactive visualization")
            return None
        
        # Get embedding patterns
        patterns = embedder.get_embedding_patterns(self.config.max_positions)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Positional Encoding Heatmap', 'Frequency Spectrum',
                          'Embedding Evolution', 'Similarity Matrix'),
            specs=[[{"type": "heatmap"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "heatmap"}]]
        )
        
        # 1. Positional encoding heatmap
        encoding = patterns['positional_encoding']
        max_dims = min(64, encoding.shape[1])
        max_pos = min(100, encoding.shape[0])
        
        fig.add_trace(
            go.Heatmap(
                z=encoding[:max_pos, :max_dims].T,
                colorscale='Viridis',
                name='Positional Encoding'
            ),
            row=1, col=1
        )
        
        # 2. Frequency spectrum
        frequencies = patterns['frequencies']
        fig.add_trace(
            go.Scatter(
                x=list(range(len(frequencies))),
                y=frequencies,
                mode='lines+markers',
                name='Frequencies'
            ),
            row=1, col=2
        )
        
        # 3. Embedding evolution
        positions = patterns['positions'][:max_pos]
        dims_to_plot = [0, encoding.shape[1]//4, encoding.shape[1]//2, encoding.shape[1]-1]
        
        for dim in dims_to_plot:
            fig.add_trace(
                go.Scatter(
                    x=positions,
                    y=encoding[:max_pos, dim],
                    mode='lines',
                    name=f'Dim {dim}'
                ),
                row=2, col=1
            )
        
        # 4. Similarity matrix
        embeddings = encoding[:50]
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / (norms + 1e-8)
        similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
        
        fig.add_trace(
            go.Heatmap(
                z=similarity_matrix,
                colorscale='Viridis',
                name='Similarity'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title='Interactive Sinusoidal Embedding Analysis',
            height=800,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Interactive visualization saved to {save_path}")
        
        return fig.to_html() if self.config.save_html else None
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def generate_comprehensive_report(
        self,
        embedder,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive analysis report.
        
        Args:
            embedder: ConfigurableSinusoidalEmbedder instance
            save_path: Optional path to save report
            
        Returns:
            Dictionary containing complete analysis results
        """
        logger.info("Generating comprehensive embedding analysis report...")
        
        # Perform all analyses
        pattern_analysis = self.visualize_embedding_patterns(
            embedder, show_plot=False
        )
        
        clustering_analysis = self.visualize_embedding_clustering(
            embedder, show_plot=False
        )
        
        # Compile comprehensive report
        report = {
            'embedder_config': embedder.get_adaptation_info(),
            'pattern_analysis': pattern_analysis,
            'clustering_analysis': clustering_analysis,
            'summary_statistics': self._generate_summary_statistics(
                pattern_analysis, clustering_analysis
            )
        }
        
        if save_path:
            import json
            with open(save_path, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                json_report = self._convert_numpy_for_json(report)
                json.dump(json_report, f, indent=2)
            logger.info(f"Comprehensive report saved to {save_path}")
        
        return report
    
    def _generate_summary_statistics(
        self,
        pattern_analysis: Dict[str, Any],
        clustering_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate summary statistics from analyses."""
        summary = {
            'embedding_quality_score': 0.0,
            'frequency_diversity': 0.0,
            'clustering_quality': 0.0,
            'pattern_regularity': 0.0
        }
        
        # Calculate embedding quality score
        if 'variance_analysis' in pattern_analysis:
            variance_stats = pattern_analysis['variance_analysis']
            summary['embedding_quality_score'] = float(variance_stats['mean_variance'])
        
        # Calculate frequency diversity
        if 'frequency_analysis' in pattern_analysis:
            freq_stats = pattern_analysis['frequency_analysis']
            summary['frequency_diversity'] = float(freq_stats['std_frequency'])
        
        # Calculate clustering quality
        if 'clustering_results' in clustering_analysis:
            clustering_results = clustering_analysis['clustering_results']
            summary['clustering_quality'] = float(clustering_results['silhouette_score'])
        
        # Calculate pattern regularity
        if 'periodicity_analysis' in pattern_analysis:
            period_stats = pattern_analysis['periodicity_analysis']
            if 'period_variance' in period_stats:
                summary['pattern_regularity'] = 1.0 / (1.0 + float(period_stats['period_variance']))
        
        return summary
    
    def _convert_numpy_for_json(self, obj):
        """Convert numpy arrays to lists for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj


# Convenience functions for quick visualization
def quick_pattern_visualization(embedder, save_path: Optional[str] = None):
    """Quick pattern visualization with default settings."""
    visualizer = EmbeddingVisualizer()
    return visualizer.visualize_embedding_patterns(embedder, save_path)


def quick_clustering_analysis(embedder, save_path: Optional[str] = None):
    """Quick clustering analysis with default settings."""
    visualizer = EmbeddingVisualizer()
    return visualizer.visualize_embedding_clustering(embedder, save_path)


def generate_embedding_report(embedder, save_path: Optional[str] = None):
    """Generate comprehensive embedding analysis report."""
    visualizer = EmbeddingVisualizer()
    return visualizer.generate_comprehensive_report(embedder, save_path)