#!/usr/bin/env python3
"""
Embedding Visualization Demo

This script demonstrates the comprehensive visualization tools for
sinusoidal embeddings, including pattern analysis, frequency analysis,
and clustering visualization.
"""

import os
import sys
import numpy as np
import tensorflow as tf

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lsm.data.configurable_sinusoidal_embedder import (
    ConfigurableSinusoidalEmbedder, SinusoidalConfig
)
from lsm.data.embedding_visualization import (
    EmbeddingVisualizer, VisualizationConfig,
    quick_pattern_visualization, quick_clustering_analysis,
    generate_embedding_report
)


def create_sample_embedder():
    """Create a sample sinusoidal embedder for demonstration."""
    config = SinusoidalConfig(
        embedding_dim=128,
        vocab_size=5000,
        max_sequence_length=256,
        base_frequency=10000.0,
        frequency_scaling=1.0,
        learnable_frequencies=True,
        use_absolute_position=True,
        use_relative_position=False
    )
    
    embedder = ConfigurableSinusoidalEmbedder(config)
    
    # Build the embedder with sample input
    sample_input = tf.constant([[1, 2, 3, 4, 5]])
    _ = embedder(sample_input)
    
    return embedder


def demo_basic_visualization():
    """Demonstrate basic visualization capabilities."""
    print("=== Basic Visualization Demo ===")
    
    # Create sample embedder
    embedder = create_sample_embedder()
    
    # Quick pattern visualization
    print("Generating pattern visualization...")
    pattern_results = quick_pattern_visualization(
        embedder, 
        save_path="embedding_patterns_demo.png"
    )
    
    print(f"Pattern analysis completed. Results keys: {list(pattern_results.keys())}")
    
    # Quick clustering analysis
    print("Generating clustering analysis...")
    clustering_results = quick_clustering_analysis(
        embedder,
        save_path="embedding_clustering_demo.png"
    )
    
    print(f"Clustering analysis completed. Results keys: {list(clustering_results.keys())}")


def demo_advanced_visualization():
    """Demonstrate advanced visualization with custom configuration."""
    print("\n=== Advanced Visualization Demo ===")
    
    # Create custom visualization configuration
    viz_config = VisualizationConfig(
        figure_size=(20, 15),
        dpi=300,
        color_palette='plasma',
        max_positions=300,
        max_dimensions=64,
        sample_tokens=2000,
        n_clusters=12,
        interactive=True
    )
    
    # Create visualizer with custom config
    visualizer = EmbeddingVisualizer(viz_config)
    
    # Create sample embedder
    embedder = create_sample_embedder()
    
    # Comprehensive pattern analysis
    print("Performing comprehensive pattern analysis...")
    pattern_analysis = visualizer.visualize_embedding_patterns(
        embedder,
        save_path="advanced_patterns_demo.png",
        show_plot=False
    )
    
    # Print some analysis results
    if 'frequency_analysis' in pattern_analysis:
        freq_stats = pattern_analysis['frequency_analysis']
        print(f"Frequency Statistics:")
        print(f"  Mean frequency: {freq_stats['mean_frequency']:.6f}")
        print(f"  Frequency std: {freq_stats['std_frequency']:.6f}")
        print(f"  Frequency range: {freq_stats['frequency_range']:.6f}")
    
    if 'variance_analysis' in pattern_analysis:
        var_stats = pattern_analysis['variance_analysis']
        print(f"Variance Statistics:")
        print(f"  Mean variance: {var_stats['mean_variance']:.6f}")
        print(f"  Max variance dimension: {var_stats['max_variance_dim']}")
        print(f"  Min variance dimension: {var_stats['min_variance_dim']}")
    
    # Advanced clustering analysis
    print("Performing advanced clustering analysis...")
    clustering_analysis = visualizer.visualize_embedding_clustering(
        embedder,
        save_path="advanced_clustering_demo.png",
        show_plot=False
    )
    
    if 'clustering_results' in clustering_analysis:
        cluster_results = clustering_analysis['clustering_results']
        print(f"Clustering Results:")
        print(f"  Silhouette score: {cluster_results['silhouette_score']:.4f}")
        print(f"  Inertia: {cluster_results['inertia']:.2f}")
        print(f"  Number of clusters: {cluster_results['n_clusters']}")


def demo_interactive_visualization():
    """Demonstrate interactive visualization capabilities."""
    print("\n=== Interactive Visualization Demo ===")
    
    try:
        # Create visualizer
        visualizer = EmbeddingVisualizer()
        
        # Create sample embedder
        embedder = create_sample_embedder()
        
        # Generate interactive visualization
        print("Generating interactive visualization...")
        html_content = visualizer.create_interactive_visualization(
            embedder,
            save_path="interactive_embedding_demo.html"
        )
        
        if html_content:
            print("Interactive visualization created successfully!")
            print("Open 'interactive_embedding_demo.html' in your browser to view.")
        else:
            print("Interactive visualization requires plotly. Install with: pip install plotly")
            
    except Exception as e:
        print(f"Interactive visualization failed: {e}")
        print("This may be due to missing plotly dependency.")


def demo_comprehensive_report():
    """Demonstrate comprehensive analysis report generation."""
    print("\n=== Comprehensive Report Demo ===")
    
    # Create visualizer
    visualizer = EmbeddingVisualizer()
    
    # Create sample embedder
    embedder = create_sample_embedder()
    
    # Generate comprehensive report
    print("Generating comprehensive analysis report...")
    report = generate_embedding_report(
        embedder,
        save_path="embedding_analysis_report.json"
    )
    
    # Print summary statistics
    if 'summary_statistics' in report:
        summary = report['summary_statistics']
        print("Summary Statistics:")
        print(f"  Embedding quality score: {summary['embedding_quality_score']:.4f}")
        print(f"  Frequency diversity: {summary['frequency_diversity']:.4f}")
        print(f"  Clustering quality: {summary['clustering_quality']:.4f}")
        print(f"  Pattern regularity: {summary['pattern_regularity']:.4f}")
    
    # Print embedder configuration
    if 'embedder_config' in report:
        config = report['embedder_config']
        print("Embedder Configuration:")
        print(f"  Vocabulary size: {config['vocab_size']}")
        print(f"  Embedding dimension: {config['embedding_dim']}")
        print(f"  Learnable frequencies: {config['learnable_frequencies']}")
        print(f"  Base frequency: {config['base_frequency']}")


def demo_frequency_analysis():
    """Demonstrate detailed frequency analysis."""
    print("\n=== Frequency Analysis Demo ===")
    
    # Create embedders with different configurations
    configs = [
        SinusoidalConfig(
            embedding_dim=64, base_frequency=1000.0, 
            learnable_frequencies=False, frequency_scaling=1.0
        ),
        SinusoidalConfig(
            embedding_dim=64, base_frequency=10000.0, 
            learnable_frequencies=False, frequency_scaling=1.0
        ),
        SinusoidalConfig(
            embedding_dim=64, base_frequency=10000.0, 
            learnable_frequencies=True, frequency_scaling=2.0
        )
    ]
    
    config_names = ["Low Base Freq", "High Base Freq", "Learnable Freq"]
    
    for i, (config, name) in enumerate(zip(configs, config_names)):
        print(f"\nAnalyzing {name} configuration...")
        
        embedder = ConfigurableSinusoidalEmbedder(config)
        sample_input = tf.constant([[1, 2, 3, 4, 5]])
        _ = embedder(sample_input)
        
        # Get embedding patterns
        patterns = embedder.get_embedding_patterns(100)
        
        print(f"  Frequency statistics:")
        frequencies = patterns['frequencies']
        print(f"    Mean: {np.mean(frequencies):.6f}")
        print(f"    Std: {np.std(frequencies):.6f}")
        print(f"    Min: {np.min(frequencies):.6f}")
        print(f"    Max: {np.max(frequencies):.6f}")
        
        # Save individual visualization
        visualizer = EmbeddingVisualizer()
        visualizer.visualize_embedding_patterns(
            embedder,
            save_path=f"frequency_analysis_{i}_{name.replace(' ', '_').lower()}.png",
            show_plot=False
        )


def demo_adaptation_visualization():
    """Demonstrate visualization of embedding adaptation."""
    print("\n=== Adaptation Visualization Demo ===")
    
    # Create initial embedder
    config = SinusoidalConfig(
        embedding_dim=64,
        vocab_size=1000,
        learnable_frequencies=True
    )
    
    embedder = ConfigurableSinusoidalEmbedder(config)
    sample_input = tf.constant([[1, 2, 3, 4, 5]])
    _ = embedder(sample_input)
    
    visualizer = EmbeddingVisualizer()
    
    # Visualize before adaptation
    print("Visualizing before adaptation...")
    visualizer.visualize_embedding_patterns(
        embedder,
        save_path="before_adaptation.png",
        show_plot=False
    )
    
    # Adapt to larger vocabulary
    print("Adapting to larger vocabulary...")
    embedder.adapt_to_vocabulary(5000)
    
    # Visualize after vocabulary adaptation
    print("Visualizing after vocabulary adaptation...")
    visualizer.visualize_embedding_patterns(
        embedder,
        save_path="after_vocab_adaptation.png",
        show_plot=False
    )
    
    # Adapt embedding dimension
    print("Adapting embedding dimension...")
    embedder.adapt_embedding_dimension(128, preserve_properties=True)
    
    # Visualize after dimension adaptation
    print("Visualizing after dimension adaptation...")
    visualizer.visualize_embedding_patterns(
        embedder,
        save_path="after_dim_adaptation.png",
        show_plot=False
    )
    
    print("Adaptation visualization complete!")


def main():
    """Run all visualization demos."""
    print("Sinusoidal Embedding Visualization Demo")
    print("=" * 50)
    
    try:
        # Run basic demos
        demo_basic_visualization()
        demo_advanced_visualization()
        
        # Run specialized demos
        demo_frequency_analysis()
        demo_adaptation_visualization()
        
        # Run comprehensive analysis
        demo_comprehensive_report()
        
        # Try interactive visualization
        demo_interactive_visualization()
        
        print("\n" + "=" * 50)
        print("All visualization demos completed successfully!")
        print("Check the generated files:")
        print("  - embedding_patterns_demo.png")
        print("  - embedding_clustering_demo.png")
        print("  - advanced_patterns_demo.png")
        print("  - advanced_clustering_demo.png")
        print("  - frequency_analysis_*.png")
        print("  - *_adaptation.png")
        print("  - embedding_analysis_report.json")
        print("  - interactive_embedding_demo.html (if plotly available)")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()