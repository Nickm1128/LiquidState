#!/usr/bin/env python3
"""
Reservoir Manager Demo.

This script demonstrates the capabilities of the ReservoirManager class,
including strategy decision making, instance management, output coordination,
and support for different reservoir types.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from typing import List, Dict, Any

# Add src to path for imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.lsm.inference.reservoir_manager import (
    ReservoirManager,
    ReservoirStrategy,
    ReservoirType,
    ReservoirOutput,
    create_reservoir_manager,
    create_high_performance_reservoir_manager
)
from src.lsm.utils.lsm_logging import get_logger

logger = get_logger(__name__)


def demonstrate_strategy_decision():
    """Demonstrate reservoir strategy decision making."""
    print("=" * 60)
    print("RESERVOIR STRATEGY DECISION DEMONSTRATION")
    print("=" * 60)
    
    manager = create_reservoir_manager(strategy="adaptive")
    
    # Test different input scenarios
    scenarios = [
        {
            "name": "Simple Short Sequence",
            "input": np.ones((1, 20)) * 0.1,
            "system_context": None,
            "description": "Low complexity, short sequence"
        },
        {
            "name": "Complex Long Sequence", 
            "input": np.random.random((1, 1000)) * 5,
            "system_context": None,
            "description": "High complexity, long sequence"
        },
        {
            "name": "Medium Sequence with System Context",
            "input": np.random.random((1, 200)) * 2,
            "system_context": {"message": "You are a helpful assistant"},
            "description": "Medium complexity with system message"
        },
        {
            "name": "High Variance Sequence",
            "input": np.random.normal(0, 3, (1, 500)),
            "system_context": None,
            "description": "High variance, medium length"
        }
    ]
    
    print(f"{'Scenario':<35} {'Strategy':<15} {'Factors'}")
    print("-" * 80)
    
    for scenario in scenarios:
        strategy = manager.decide_reservoir_strategy(
            scenario["input"],
            system_context=scenario["system_context"]
        )
        
        # Get decision factors for analysis
        factors = manager._calculate_decision_factors(
            scenario["input"],
            scenario["system_context"],
            None
        )
        
        factor_summary = f"len:{factors['sequence_length']:.2f}, " \
                        f"comp:{factors['embedding_complexity']:.2f}, " \
                        f"sys:{factors['system_context']:.2f}"
        
        print(f"{scenario['name']:<35} {strategy.value:<15} {factor_summary}")
        print(f"  → {scenario['description']}")
        print()


def demonstrate_instance_management():
    """Demonstrate reservoir instance management."""
    print("=" * 60)
    print("RESERVOIR INSTANCE MANAGEMENT DEMONSTRATION")
    print("=" * 60)
    
    manager = create_reservoir_manager(max_instances=3)
    
    print("Creating reservoir instances...")
    
    # Create different types of reservoirs
    reservoir_types = ["standard", "hierarchical", "attentive"]
    instances = []
    
    for i, res_type in enumerate(reservoir_types):
        try:
            print(f"\nCreating {res_type} reservoir...")
            instance = manager.get_or_create_reservoir(
                reservoir_type=res_type,
                strategy="separate",
                config={"input_dim": 128}
            )
            instances.append(instance)
            print(f"  ✓ Created instance: {instance.instance_id}")
            print(f"  ✓ Type: {instance.reservoir_type.value}")
            print(f"  ✓ Usage count: {instance.usage_count}")
        except Exception as e:
            print(f"  ✗ Failed to create {res_type} reservoir: {e}")
    
    print(f"\nTotal active instances: {len(manager._instances)}")
    
    # Demonstrate instance reuse
    print("\nTesting instance reuse...")
    reused_instance = manager.get_or_create_reservoir(
        reservoir_type="standard",
        strategy="reuse"
    )
    
    if reused_instance.instance_id in [i.instance_id for i in instances]:
        print("  ✓ Successfully reused existing instance")
        print(f"  ✓ Usage count increased to: {reused_instance.usage_count}")
    else:
        print("  → Created new instance (no reusable instance found)")
    
    # Test max instances limit
    print(f"\nTesting max instances limit (current limit: {manager.max_instances})...")
    try:
        extra_instance = manager.get_or_create_reservoir(
            reservoir_type="deep",
            strategy="separate"
        )
        print(f"  ✓ Created additional instance: {extra_instance.instance_id}")
        print(f"  ✓ Total instances maintained at: {len(manager._instances)}")
    except Exception as e:
        print(f"  ✗ Failed to create additional instance: {e}")
    
    # Show instance management statistics
    print("\nRunning instance maintenance...")
    stats = manager.manage_reservoir_instances()
    print("  Maintenance statistics:")
    for key, value in stats.items():
        print(f"    {key}: {value}")


def demonstrate_output_coordination():
    """Demonstrate reservoir output coordination."""
    print("=" * 60)
    print("RESERVOIR OUTPUT COORDINATION DEMONSTRATION")
    print("=" * 60)
    
    manager = create_reservoir_manager()
    
    # Create mock reservoir outputs with different characteristics
    outputs = [
        ReservoirOutput(
            output=np.array([[1.0, 2.0, 3.0, 4.0]]),
            instance_id="high_confidence_reservoir",
            reservoir_type="standard",
            processing_time=0.05,
            confidence=0.9,
            metadata={"description": "High confidence output"}
        ),
        ReservoirOutput(
            output=np.array([[2.0, 3.0, 4.0, 5.0]]),
            instance_id="medium_confidence_reservoir", 
            reservoir_type="hierarchical",
            processing_time=0.08,
            confidence=0.7,
            metadata={"description": "Medium confidence output"}
        ),
        ReservoirOutput(
            output=np.array([[0.5, 1.5, 2.5, 3.5]]),
            instance_id="low_confidence_reservoir",
            reservoir_type="attentive", 
            processing_time=0.12,
            confidence=0.4,
            metadata={"description": "Low confidence output"}
        )
    ]
    
    print("Individual reservoir outputs:")
    for i, output in enumerate(outputs):
        print(f"  Output {i+1}: {output.output.flatten()}")
        print(f"    Confidence: {output.confidence:.2f}")
        print(f"    Type: {output.reservoir_type}")
        print(f"    Processing time: {output.processing_time:.3f}s")
        print()
    
    # Test different coordination strategies
    strategies = ["weighted_average", "max_confidence", "ensemble_voting", "hierarchical_merge"]
    
    print("Coordination results:")
    print("-" * 50)
    
    for strategy in strategies:
        try:
            result = manager.coordinate_multiple_outputs(outputs, strategy)
            
            print(f"\n{strategy.upper()}:")
            print(f"  Coordinated output: {result.coordinated_output.flatten()}")
            print(f"  Processing time: {result.total_processing_time:.3f}s")
            print(f"  Individual confidences: {[f'{c:.2f}' for c in result.confidence_scores]}")
            
            # Show coordination effect
            individual_mean = np.mean([o.output for o in outputs], axis=0)
            print(f"  Simple average: {individual_mean.flatten()}")
            print(f"  Coordination difference: {(result.coordinated_output - individual_mean).flatten()}")
            
        except Exception as e:
            print(f"  ✗ {strategy} failed: {e}")


def demonstrate_performance_tracking():
    """Demonstrate performance tracking and statistics."""
    print("=" * 60)
    print("PERFORMANCE TRACKING DEMONSTRATION")
    print("=" * 60)
    
    manager = create_reservoir_manager()
    
    # Simulate various operations to generate performance data
    print("Simulating reservoir operations...")
    
    # Generate test data
    test_inputs = [
        np.random.random((1, 64)) for _ in range(10)
    ]
    
    strategies_to_test = ["reuse", "separate", "adaptive"]
    
    for strategy in strategies_to_test:
        print(f"\nTesting {strategy} strategy...")
        
        for i, test_input in enumerate(test_inputs):
            try:
                # Decide strategy
                decided_strategy = manager.decide_reservoir_strategy(test_input)
                
                # Create/get reservoir
                instance = manager.get_or_create_reservoir(
                    reservoir_type="standard",
                    strategy=strategy
                )
                
                # Simulate processing (mock output)
                mock_output = np.random.random((1, 32))
                
                # Create mock result
                result = ReservoirOutput(
                    output=mock_output,
                    instance_id=instance.instance_id,
                    reservoir_type="standard",
                    processing_time=np.random.uniform(0.01, 0.1),
                    confidence=np.random.uniform(0.5, 1.0)
                )
                
                # Update performance tracking manually
                manager._strategy_usage_stats[strategy] += 1
                manager._performance_history["standard"].append({
                    "processing_time": result.processing_time,
                    "confidence": result.confidence,
                    "input_size": test_input.size
                })
                
                if i % 3 == 0:  # Simulate some cache hits
                    manager._cache_hits += 1
                else:
                    manager._cache_misses += 1
                
            except Exception as e:
                print(f"    ✗ Operation {i+1} failed: {e}")
    
    # Display performance statistics
    print("\nPerformance Statistics:")
    print("-" * 30)
    
    stats = manager.get_performance_statistics()
    
    print("Strategy Usage:")
    for strategy, count in stats["strategy_usage"].items():
        print(f"  {strategy}: {count} times")
    
    print(f"\nActive Instances: {stats['active_instances']}")
    print(f"Cache Hit Rate: {stats['cache_hit_rate']:.2%}")
    
    print("\nPerformance by Reservoir Type:")
    for res_type, perf_data in stats["performance_by_type"].items():
        print(f"  {res_type}:")
        print(f"    Average processing time: {perf_data['avg_processing_time']:.3f}s")
        print(f"    Average confidence: {perf_data['avg_confidence']:.3f}")
        print(f"    Total uses: {perf_data['total_uses']}")
        print(f"    Time range: {perf_data['min_processing_time']:.3f}s - {perf_data['max_processing_time']:.3f}s")


def demonstrate_advanced_scenarios():
    """Demonstrate advanced usage scenarios."""
    print("=" * 60)
    print("ADVANCED SCENARIOS DEMONSTRATION")
    print("=" * 60)
    
    # High-performance manager
    print("1. High-Performance Manager:")
    hp_manager = create_high_performance_reservoir_manager()
    print(f"   Max instances: {hp_manager.max_instances}")
    print(f"   Instance timeout: {hp_manager.instance_timeout}s")
    print(f"   Coordination strategy: {hp_manager.coordination_strategy}")
    print(f"   Caching enabled: {hp_manager.enable_caching}")
    
    # Batch processing simulation
    print("\n2. Batch Processing Simulation:")
    manager = create_reservoir_manager()
    
    # Create multiple outputs for batch coordination
    batch_outputs = []
    for i in range(5):
        output = ReservoirOutput(
            output=np.random.random((1, 16)),
            instance_id=f"batch_reservoir_{i}",
            reservoir_type="standard",
            processing_time=np.random.uniform(0.01, 0.05),
            confidence=np.random.uniform(0.6, 0.95)
        )
        batch_outputs.append(output)
    
    print(f"   Processing batch of {len(batch_outputs)} outputs...")
    
    try:
        batch_result = manager.coordinate_multiple_outputs(
            batch_outputs,
            coordination_strategy="ensemble_voting"
        )
        print(f"   ✓ Batch coordination successful")
        print(f"   ✓ Coordinated output shape: {batch_result.coordinated_output.shape}")
        print(f"   ✓ Average confidence: {np.mean(batch_result.confidence_scores):.3f}")
    except Exception as e:
        print(f"   ✗ Batch coordination failed: {e}")
    
    # Resource management
    print("\n3. Resource Management:")
    print("   Testing resource cleanup and optimization...")
    
    # Create many instances to test cleanup
    temp_manager = ReservoirManager(max_instances=2, instance_timeout=0.1)
    
    for i in range(5):
        try:
            instance = temp_manager.get_or_create_reservoir("standard", strategy="separate")
            print(f"   Created instance {i+1}: {instance.instance_id}")
        except Exception as e:
            print(f"   Instance {i+1} creation handled: {type(e).__name__}")
    
    print(f"   Active instances before cleanup: {len(temp_manager._instances)}")
    
    # Wait for timeout
    time.sleep(0.2)
    
    # Run maintenance
    cleanup_stats = temp_manager.manage_reservoir_instances()
    print(f"   Active instances after cleanup: {len(temp_manager._instances)}")
    print(f"   Cleanup statistics: {cleanup_stats}")


def create_performance_visualization():
    """Create visualizations of reservoir manager performance."""
    print("=" * 60)
    print("PERFORMANCE VISUALIZATION")
    print("=" * 60)
    
    try:
        import matplotlib.pyplot as plt
        
        manager = create_reservoir_manager()
        
        # Simulate performance data
        strategies = ["reuse", "separate", "adaptive"]
        processing_times = {strategy: [] for strategy in strategies}
        confidences = {strategy: [] for strategy in strategies}
        
        # Generate mock performance data
        for strategy in strategies:
            for _ in range(20):
                # Simulate different performance characteristics
                if strategy == "reuse":
                    time_val = np.random.normal(0.02, 0.005)  # Fast but variable
                    conf_val = np.random.normal(0.75, 0.1)   # Medium confidence
                elif strategy == "separate":
                    time_val = np.random.normal(0.05, 0.01)  # Slower but consistent
                    conf_val = np.random.normal(0.85, 0.05)  # High confidence
                else:  # adaptive
                    time_val = np.random.normal(0.035, 0.008) # Balanced
                    conf_val = np.random.normal(0.80, 0.08)   # Good confidence
                
                processing_times[strategy].append(max(0.001, time_val))
                confidences[strategy].append(np.clip(conf_val, 0, 1))
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Processing time comparison
        ax1.boxplot([processing_times[s] for s in strategies], labels=strategies)
        ax1.set_title('Processing Time by Strategy')
        ax1.set_ylabel('Processing Time (seconds)')
        ax1.grid(True, alpha=0.3)
        
        # Confidence comparison
        ax2.boxplot([confidences[s] for s in strategies], labels=strategies)
        ax2.set_title('Confidence by Strategy')
        ax2.set_ylabel('Confidence Score')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(os.path.dirname(__file__), 'reservoir_manager_performance.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"   ✓ Performance visualization saved to: {plot_path}")
        
        # Show statistics
        print("\n   Performance Summary:")
        for strategy in strategies:
            avg_time = np.mean(processing_times[strategy])
            avg_conf = np.mean(confidences[strategy])
            print(f"     {strategy}: {avg_time:.3f}s avg time, {avg_conf:.3f} avg confidence")
        
        plt.show()
        
    except ImportError:
        print("   Matplotlib not available - skipping visualization")
    except Exception as e:
        print(f"   Visualization failed: {e}")


def main():
    """Run all demonstrations."""
    print("RESERVOIR MANAGER COMPREHENSIVE DEMONSTRATION")
    print("=" * 80)
    print("This demo showcases the ReservoirManager's capabilities including:")
    print("• Strategy decision making")
    print("• Instance management and cleanup")
    print("• Output coordination strategies")
    print("• Performance tracking and statistics")
    print("• Advanced usage scenarios")
    print("=" * 80)
    print()
    
    try:
        # Run demonstrations
        demonstrate_strategy_decision()
        print("\n")
        
        demonstrate_instance_management()
        print("\n")
        
        demonstrate_output_coordination()
        print("\n")
        
        demonstrate_performance_tracking()
        print("\n")
        
        demonstrate_advanced_scenarios()
        print("\n")
        
        create_performance_visualization()
        
        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("The ReservoirManager provides comprehensive reservoir management")
        print("capabilities for the LSM training pipeline enhancement.")
        print("Key features demonstrated:")
        print("• Intelligent strategy selection based on input characteristics")
        print("• Efficient instance management with automatic cleanup")
        print("• Multiple output coordination strategies")
        print("• Performance monitoring and optimization")
        print("• Support for different reservoir types")
        
    except Exception as e:
        logger.exception("Demo failed")
        print(f"\n✗ Demo failed with error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())