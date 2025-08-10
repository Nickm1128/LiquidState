#!/usr/bin/env python3
"""
Tests for Embedding Visualization Tools

This module contains comprehensive tests for the sinusoidal embedding
visualization utilities, including pattern analysis, clustering, and
interactive visualization features.
"""

import os
import sys
import unittest
import tempfile
import shutil
import numpy as np
import tensorflow as tf
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from lsm.data.configurable_sinusoidal_embedder import (
    ConfigurableSinusoidalEmbedder, SinusoidalConfig
)
from lsm.data.embedding_visualization import (
    EmbeddingVisualizer, VisualizationConfig,
    quick_pattern_visualization, quick_clustering_analysis,
    generate_embedding_report
)


class TestVisualizationConfig(unittest.TestCase):
    """Test VisualizationConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = VisualizationConfig()
        
        self.assertEqual(config.figure_size, (15, 10))
        self.assertEqual(config.dpi, 300)
        self.assertEqual(config.style, 'seaborn-v0_8')
        self.assertEqual(config.color_palette, 'viridis')
        self.assertEqual(config.max_positions, 200)
        self.assertEqual(config.max_dimensions, 128)
        self.assertEqual(config.sample_tokens, 1000)
        self.assertEqual(config.n_clusters, 8)
        self.assertEqual(config.clustering_method, 'kmeans')
        self.assertEqual(config.pca_components, 2)
        self.assertEqual(config.tsne_perplexity, 30.0)
        self.assertEqual(config.tsne_n_iter, 1000)
        self.assertFalse(config.interactive)
        self.assertFalse(config.save_html)
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = VisualizationConfig(
            figure_size=(20, 12),
            dpi=150,
            max_positions=500,
            n_clusters=12,
            interactive=True
        )
        
        self.assertEqual(config.figure_size, (20, 12))
        self.assertEqual(config.dpi, 150)
        self.assertEqual(config.max_positions, 500)
        self.assertEqual(config.n_clusters, 12)
        self.assertTrue(config.interactive)


class TestEmbeddingVisualizer(unittest.TestCase):
    """Test EmbeddingVisualizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample embedder
        config = SinusoidalConfig(
            embedding_dim=32,
            vocab_size=100,
            max_sequence_length=50,
            learnable_frequencies=True
        )
        
        self.embedder = ConfigurableSinusoidalEmbedder(config)
        
        # Build embedder with sample input
        sample_input = tf.constant([[1, 2, 3, 4, 5]])
        _ = self.embedder(sample_input)
        
        # Create visualizer
        self.visualizer = EmbeddingVisualizer()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_visualizer_initialization(self):
        """Test visualizer initialization."""
        # Test with default config
        visualizer = EmbeddingVisualizer()
        self.assertIsInstance(visualizer.config, VisualizationConfig)
        
        # Test with custom config
        custom_config = VisualizationConfig(max_positions=100)
        visualizer = EmbeddingVisualizer(custom_config)
        self.assertEqual(visualizer.config.max_positions, 100)
    
    @patch('lsm.data.embedding_visualization.MATPLOTLIB_AVAILABLE', True)
    @patch('lsm.data.embedding_visualization.plt')
    def test_visualize_embedding_patterns(self, mock_plt):
        """Test embedding pattern visualization."""
        # Mock matplotlib components
        mock_fig = MagicMock()
        mock_axes = [[MagicMock() for _ in range(3)] for _ in range(3)]
        mock_plt.figure.return_value = mock_fig
        mock_fig.add_gridspec.return_value.add_subplot.side_effect = [
            ax for row in mock_axes for ax in row
        ]
        
        # Test visualization
        results = self.visualizer.visualize_embedding_patterns(
            self.embedder,
            show_plot=False
        )
        
        # Check that results contain expected keys
        expected_keys = [
            'frequency_analysis', 'phase_analysis', 'evolution_analysis',
            'similarity_analysis', 'distribution_analysis', 'variance_analysis',
            'periodicity_analysis', 'energy_analysis'
        ]
        
        for key in expected_keys:
            self.assertIn(key, results)
        
        # Check that matplotlib functions were called
        mock_plt.figure.assert_called_once()
        mock_plt.suptitle.assert_called_once()
    
    @patch('lsm.data.embedding_visualization.MATPLOTLIB_AVAILABLE', False)
    def test_visualize_patterns_no_matplotlib(self):
        """Test pattern visualization without matplotlib."""
        results = self.visualizer.visualize_embedding_patterns(
            self.embedder,
            show_plot=False
        )
        
        # Should return empty dict when matplotlib not available
        self.assertEqual(results, {})
    
    @patch('lsm.data.embedding_visualization.SKLEARN_AVAILABLE', True)
    @patch('lsm.data.embedding_visualization.MATPLOTLIB_AVAILABLE', True)
    @patch('lsm.data.embedding_visualization.plt')
    @patch('lsm.data.embedding_visualization.KMeans')
    @patch('lsm.data.embedding_visualization.silhouette_score')
    def test_visualize_embedding_clustering(self, mock_silhouette, mock_kmeans, mock_plt):
        """Test embedding clustering visualization."""
        # Mock sklearn components
        mock_kmeans_instance = MagicMock()
        mock_kmeans_instance.fit_predict.return_value = np.array([0, 1, 0, 1, 0])
        mock_kmeans_instance.cluster_centers_ = np.random.randn(2, 32)
        mock_kmeans_instance.inertia_ = 10.5
        mock_kmeans.return_value = mock_kmeans_instance
        mock_silhouette.return_value = 0.75
        
        # Mock matplotlib components
        mock_fig = MagicMock()
        mock_axes = np.array([[MagicMock() for _ in range(3)] for _ in range(2)])
        mock_plt.subplots.return_value = (mock_fig, mock_axes)
        
        # Test clustering visualization
        results = self.visualizer.visualize_embedding_clustering(
            self.embedder,
            show_plot=False
        )
        
        # Check that results contain expected keys
        expected_keys = [
            'clustering_results', 'pca_results', 'tsne_results',
            'centers_analysis', 'silhouette_analysis', 'size_analysis',
            'distance_analysis'
        ]
        
        for key in expected_keys:
            self.assertIn(key, results)
        
        # Check clustering results
        clustering_results = results['clustering_results']
        self.assertIn('silhouette_score', clustering_results)
        self.assertIn('cluster_labels', clustering_results)
        self.assertIn('cluster_centers', clustering_results)
    
    @patch('lsm.data.embedding_visualization.SKLEARN_AVAILABLE', False)
    def test_clustering_no_sklearn(self):
        """Test clustering visualization without sklearn."""
        results = self.visualizer.visualize_embedding_clustering(
            self.embedder,
            show_plot=False
        )
        
        # Should return empty dict when sklearn not available
        self.assertEqual(results, {})
    
    @patch('lsm.data.embedding_visualization.PLOTLY_AVAILABLE', True)
    @patch('lsm.data.embedding_visualization.make_subplots')
    def test_create_interactive_visualization(self, mock_make_subplots):
        """Test interactive visualization creation."""
        # Mock plotly components
        mock_fig = MagicMock()
        mock_make_subplots.return_value = mock_fig
        mock_fig.to_html.return_value = "<html>test</html>"
        
        # Test interactive visualization
        html_content = self.visualizer.create_interactive_visualization(
            self.embedder
        )
        
        # Check that HTML content is returned
        self.assertEqual(html_content, "<html>test</html>")
        
        # Check that plotly functions were called
        mock_make_subplots.assert_called_once()
        mock_fig.add_trace.assert_called()
        mock_fig.update_layout.assert_called_once()
    
    @patch('lsm.data.embedding_visualization.PLOTLY_AVAILABLE', False)
    def test_interactive_visualization_no_plotly(self):
        """Test interactive visualization without plotly."""
        html_content = self.visualizer.create_interactive_visualization(
            self.embedder
        )
        
        # Should return None when plotly not available
        self.assertIsNone(html_content)
    
    def test_generate_comprehensive_report(self):
        """Test comprehensive report generation."""
        with patch.object(self.visualizer, 'visualize_embedding_patterns') as mock_patterns, \
             patch.object(self.visualizer, 'visualize_embedding_clustering') as mock_clustering:
            
            # Mock analysis results
            mock_patterns.return_value = {
                'frequency_analysis': {'mean_frequency': 0.1, 'std_frequency': 0.05},
                'variance_analysis': {'mean_variance': 0.2}
            }
            mock_clustering.return_value = {'clustering_results': {'silhouette_score': 0.8}}
            
            # Generate report
            report = self.visualizer.generate_comprehensive_report(
                self.embedder
            )
            
            # Check report structure
            self.assertIn('embedder_config', report)
            self.assertIn('pattern_analysis', report)
            self.assertIn('clustering_analysis', report)
            self.assertIn('summary_statistics', report)
            
            # Check summary statistics
            summary = report['summary_statistics']
            self.assertIn('embedding_quality_score', summary)
            self.assertIn('frequency_diversity', summary)
            self.assertIn('clustering_quality', summary)
            self.assertIn('pattern_regularity', summary)
    
    def test_save_report_to_file(self):
        """Test saving report to file."""
        with patch.object(self.visualizer, 'visualize_embedding_patterns') as mock_patterns, \
             patch.object(self.visualizer, 'visualize_embedding_clustering') as mock_clustering:
            
            # Mock analysis results
            mock_patterns.return_value = {'test': 'data'}
            mock_clustering.return_value = {'test': 'data'}
            
            # Generate report with file save
            report_path = os.path.join(self.temp_dir, 'test_report.json')
            report = self.visualizer.generate_comprehensive_report(
                self.embedder,
                save_path=report_path
            )
            
            # Check that file was created
            self.assertTrue(os.path.exists(report_path))
            
            # Check file content
            import json
            with open(report_path, 'r') as f:
                saved_report = json.load(f)
            
            self.assertIn('embedder_config', saved_report)
            self.assertIn('summary_statistics', saved_report)
    
    def test_calculate_skewness(self):
        """Test skewness calculation."""
        # Test normal distribution (should have low skewness)
        normal_data = np.random.normal(0, 1, 1000)
        skewness = self.visualizer._calculate_skewness(normal_data)
        self.assertLess(abs(skewness), 0.5)  # Should be close to 0
        
        # Test skewed distribution
        skewed_data = np.array([1, 1, 1, 1, 10])
        skewness = self.visualizer._calculate_skewness(skewed_data)
        self.assertGreater(skewness, 0)  # Should be positive (right-skewed)
        
        # Test constant data (should be 0)
        constant_data = np.array([5, 5, 5, 5, 5])
        skewness = self.visualizer._calculate_skewness(constant_data)
        self.assertEqual(skewness, 0)
    
    def test_calculate_kurtosis(self):
        """Test kurtosis calculation."""
        # Test normal distribution (should have kurtosis close to 0)
        normal_data = np.random.normal(0, 1, 1000)
        kurtosis = self.visualizer._calculate_kurtosis(normal_data)
        self.assertLess(abs(kurtosis), 1.0)  # Should be close to 0
        
        # Test constant data (should be -3, which becomes 0 after adjustment)
        constant_data = np.array([5, 5, 5, 5, 5])
        kurtosis = self.visualizer._calculate_kurtosis(constant_data)
        self.assertEqual(kurtosis, 0)
    
    def test_convert_numpy_for_json(self):
        """Test numpy to JSON conversion."""
        # Test with nested structure containing numpy arrays
        test_data = {
            'array': np.array([1, 2, 3]),
            'nested': {
                'matrix': np.array([[1, 2], [3, 4]]),
                'scalar': np.float64(3.14),
                'list': [np.int32(1), np.int32(2)]
            },
            'regular': 'string'
        }
        
        converted = self.visualizer._convert_numpy_for_json(test_data)
        
        # Check conversions
        self.assertEqual(converted['array'], [1, 2, 3])
        self.assertEqual(converted['nested']['matrix'], [[1, 2], [3, 4]])
        self.assertIsInstance(converted['nested']['scalar'], float)
        self.assertEqual(converted['nested']['list'], [1, 2])
        self.assertEqual(converted['regular'], 'string')


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions for quick visualization."""
    
    def setUp(self):
        """Set up test fixtures."""
        config = SinusoidalConfig(
            embedding_dim=16,
            vocab_size=50,
            learnable_frequencies=False
        )
        
        self.embedder = ConfigurableSinusoidalEmbedder(config)
        
        # Build embedder
        sample_input = tf.constant([[1, 2, 3]])
        _ = self.embedder(sample_input)
    
    @patch('lsm.data.embedding_visualization.EmbeddingVisualizer')
    def test_quick_pattern_visualization(self, mock_visualizer_class):
        """Test quick pattern visualization function."""
        mock_visualizer = MagicMock()
        mock_visualizer_class.return_value = mock_visualizer
        mock_visualizer.visualize_embedding_patterns.return_value = {'test': 'result'}
        
        result = quick_pattern_visualization(self.embedder, save_path='test.png')
        
        # Check that visualizer was created and method called
        mock_visualizer_class.assert_called_once()
        mock_visualizer.visualize_embedding_patterns.assert_called_once_with(
            self.embedder, 'test.png'
        )
        self.assertEqual(result, {'test': 'result'})
    
    @patch('lsm.data.embedding_visualization.EmbeddingVisualizer')
    def test_quick_clustering_analysis(self, mock_visualizer_class):
        """Test quick clustering analysis function."""
        mock_visualizer = MagicMock()
        mock_visualizer_class.return_value = mock_visualizer
        mock_visualizer.visualize_embedding_clustering.return_value = {'test': 'result'}
        
        result = quick_clustering_analysis(self.embedder, save_path='test.png')
        
        # Check that visualizer was created and method called
        mock_visualizer_class.assert_called_once()
        mock_visualizer.visualize_embedding_clustering.assert_called_once_with(
            self.embedder, 'test.png'
        )
        self.assertEqual(result, {'test': 'result'})
    
    @patch('lsm.data.embedding_visualization.EmbeddingVisualizer')
    def test_generate_embedding_report(self, mock_visualizer_class):
        """Test generate embedding report function."""
        mock_visualizer = MagicMock()
        mock_visualizer_class.return_value = mock_visualizer
        mock_visualizer.generate_comprehensive_report.return_value = {'test': 'report'}
        
        result = generate_embedding_report(self.embedder, save_path='report.json')
        
        # Check that visualizer was created and method called
        mock_visualizer_class.assert_called_once()
        mock_visualizer.generate_comprehensive_report.assert_called_once_with(
            self.embedder, 'report.json'
        )
        self.assertEqual(result, {'test': 'report'})


class TestVisualizationIntegration(unittest.TestCase):
    """Integration tests for visualization with real embedder."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create embedder with different configurations
        self.configs = [
            SinusoidalConfig(
                embedding_dim=16, vocab_size=50, learnable_frequencies=False
            ),
            SinusoidalConfig(
                embedding_dim=32, vocab_size=100, learnable_frequencies=True
            )
        ]
        
        self.embedders = []
        for config in self.configs:
            embedder = ConfigurableSinusoidalEmbedder(config)
            sample_input = tf.constant([[1, 2, 3, 4, 5]])
            _ = embedder(sample_input)
            self.embedders.append(embedder)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_pattern_analysis_consistency(self):
        """Test that pattern analysis produces consistent results."""
        visualizer = EmbeddingVisualizer()
        
        for i, embedder in enumerate(self.embedders):
            # Get embedding patterns directly
            patterns = embedder.get_embedding_patterns(50)
            
            # Check pattern structure
            self.assertIn('positional_encoding', patterns)
            self.assertIn('frequencies', patterns)
            self.assertIn('positions', patterns)
            
            # Check dimensions
            encoding = patterns['positional_encoding']
            self.assertEqual(encoding.shape[0], 50)  # positions
            self.assertEqual(encoding.shape[1], embedder.config.embedding_dim)
            
            # Check frequencies
            frequencies = patterns['frequencies']
            expected_freq_pairs = embedder.config.embedding_dim // 2
            self.assertEqual(len(frequencies), expected_freq_pairs)
    
    def test_embedding_adaptation_effects(self):
        """Test visualization of embedding adaptation effects."""
        embedder = self.embedders[0]  # Use first embedder
        visualizer = EmbeddingVisualizer()
        
        # Get initial patterns
        initial_patterns = embedder.get_embedding_patterns(30)
        initial_frequencies = initial_patterns['frequencies'].copy()
        
        # Adapt vocabulary size
        embedder.adapt_to_vocabulary(200)
        
        # Get patterns after adaptation
        adapted_patterns = embedder.get_embedding_patterns(30)
        adapted_frequencies = adapted_patterns['frequencies']
        
        # Frequencies should remain the same for non-learnable frequencies
        if not embedder.config.learnable_frequencies:
            np.testing.assert_array_almost_equal(
                initial_frequencies, adapted_frequencies, decimal=6
            )
        
        # Embedding dimensions should remain the same
        self.assertEqual(
            initial_patterns['positional_encoding'].shape[1],
            adapted_patterns['positional_encoding'].shape[1]
        )
    
    def test_different_config_comparisons(self):
        """Test visualization differences between different configurations."""
        visualizer = EmbeddingVisualizer()
        
        patterns_list = []
        for embedder in self.embedders:
            patterns = embedder.get_embedding_patterns(50)
            patterns_list.append(patterns)
        
        # Compare frequency characteristics
        freq1 = patterns_list[0]['frequencies']
        freq2 = patterns_list[1]['frequencies']
        
        # Different embedding dimensions should have different frequency arrays
        if len(freq1) != len(freq2):
            self.assertNotEqual(len(freq1), len(freq2))
        
        # Compare encoding patterns
        enc1 = patterns_list[0]['positional_encoding']
        enc2 = patterns_list[1]['positional_encoding']
        
        # Different configurations should produce different patterns
        self.assertNotEqual(enc1.shape[1], enc2.shape[1])  # Different embedding dims


if __name__ == '__main__':
    # Set up test environment
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
    
    # Run tests
    unittest.main(verbosity=2)