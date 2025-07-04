"""
Main experiment runner for Santa Claus Problem algorithm comparisons.
Records experiment results in CSV format and generates visualizations.
"""

import sys
import os
import time
import logging
import csv
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Tuple
import pandas as pd

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from csv_experiments import experiment
except ImportError:
    print("Error: csv-experiments library not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "csv-experiments"])
    from csv_experiments import experiment

# Import our modules
from simulations.algorithm_variants import ALGORITHMS
from simulations.instance_generator import generate_random_instances, SMALL_SIZES, MEDIUM_SIZES, LARGE_SIZES
from simulations.metrics import measure_solution_quality, run_with_timeout
from backend.santa_claus.core import Instance

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_experiment(algorithm_name: str, instance: Instance, timeout: float = 60.0) -> Dict[str, Any]:
    """
    Run a single experiment for an algorithm on an instance.
    
    Args:
        algorithm_name: The name of the algorithm to run
        instance: The problem instance
        timeout: Maximum execution time in seconds
        
    Returns:
        Dictionary of results
    """
    algorithm_func = ALGORITHMS[algorithm_name]
    
    # Run the algorithm with timeout
    allocation, execution_time, timed_out = run_with_timeout(
        algorithm_func, args=(instance,), timeout=timeout
    )
    
    # Prepare results
    results = {
        "algorithm": algorithm_name,
        "instance_size": len(instance.items),
        "num_kids": len(instance.agents),
        "runtime": execution_time,
        "timed_out": timed_out,
    }
    
    # If the algorithm didn't time out, evaluate solution quality
    if not timed_out and allocation is not None:
        solution_metrics = measure_solution_quality(instance, allocation)
        results.update(solution_metrics)
    else:
        # If timed out, set solution metrics to None
        results.update({
            "total_happiness": None,
            "min_happiness": None,
            "max_happiness": None,
            "avg_happiness": None,
            "happiness_stddev": None,
            "fairness_ratio": None,
            "min_allocation_size": None,
            "max_allocation_size": None,
            "avg_allocation_size": None,
        })
    
    return results


def run_all_experiments(sizes: List[int], algorithms: List[str] = None, 
                       timeout: float = 60.0, results_dir: str = "results") -> pd.DataFrame:
    """
    Run experiments for all specified algorithms on instances of different sizes.
    
    Args:
        sizes: List of instance sizes to test
        algorithms: List of algorithm names to test (all if None)
        timeout: Maximum execution time per algorithm in seconds
        results_dir: Directory to store results
        
    Returns:
        DataFrame with experiment results
    """
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Use all algorithms if none specified
    if algorithms is None:
        algorithms = list(ALGORITHMS.keys())
    
    # Generate random instances
    logger.info(f"Generating {len(sizes)} random instances...")
    instances = generate_random_instances(sizes)
    
    # Run experiments for each instance and algorithm
    all_results = []
    
    # Setup CSV file for results
    csv_file = os.path.join(results_dir, f"santa_claus_results_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    with open(csv_file, 'w', newline='') as f:
        writer = None  # Will be initialized after first result
        
        for instance_name, instance in instances.items():
            logger.info(f"Testing instance {instance_name} with {len(instance.items)} presents and {len(instance.agents)} kids")
            
            for algorithm_name in algorithms:
                logger.info(f"  Running algorithm: {algorithm_name}")
                
                # Run the experiment
                result = run_experiment(algorithm_name, instance, timeout=timeout)
                result["instance_name"] = instance_name
                all_results.append(result)
                
                # Initialize CSV writer if first result
                if writer is None:
                    fieldnames = result.keys()
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                
                # Write result to CSV
                writer.writerow(result)
                f.flush()  # Ensure data is written immediately
                
                # Log result summary
                if result["timed_out"]:
                    logger.info(f"    Timed out after {result['runtime']:.2f} seconds")
                else:
                    logger.info(f"    Completed in {result['runtime']:.2f} seconds")
                    logger.info(f"    Min happiness: {result['min_happiness']:.2f}")
                    logger.info(f"    Total happiness: {result['total_happiness']:.2f}")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)
    logger.info(f"All experiments completed. Results saved to {csv_file}")
    return results_df


def plot_results(results_df: pd.DataFrame, metric: str, results_dir: str = "results") -> None:
    """
    Plot results for a specific metric.
    
    Args:
        results_df: DataFrame with experiment results
        metric: Metric to plot
        results_dir: Directory to store plots
    """
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Filter out timed out experiments
    filtered_df = results_df[~results_df["timed_out"]]
    
    # If all experiments timed out for a metric, skip plotting
    if metric not in filtered_df.columns or filtered_df[metric].isna().all():
        logger.warning(f"No valid data for metric {metric}. Skipping plot.")
        return
    
    # Get unique algorithms and instance sizes
    algorithms = filtered_df["algorithm"].unique()
    sizes = sorted(filtered_df["instance_size"].unique())
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot each algorithm
    for algorithm in algorithms:
        algo_data = filtered_df[filtered_df["algorithm"] == algorithm]
        if len(algo_data) == 0:
            continue
            
        # Group by instance size and calculate mean
        grouped = algo_data.groupby("instance_size")[metric].mean()
        
        # Plot line
        sizes_present = [size for size in sizes if size in grouped.index]
        values = [grouped.loc[size] if size in grouped.index else np.nan for size in sizes_present]
        
        # Skip if no valid data points
        if all(np.isnan(values)):
            continue
            
        plt.plot(sizes_present, values, marker='o', label=algorithm)
    
    # Set plot labels and title
    plt.xlabel('Instance Size (Number of Presents)')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(f'{metric.replace("_", " ").title()} vs Instance Size')
    
    # Add legend
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save plot
    plot_file = os.path.join(results_dir, f"{metric}_comparison.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Plot saved to {plot_file}")


def plot_all_metrics(results_df: pd.DataFrame, results_dir: str = "results") -> None:
    """
    Plot all metrics from the experiment results.
    
    Args:
        results_df: DataFrame with experiment results
        results_dir: Directory to store plots
    """
    # Metrics to plot
    metrics = [
        "runtime",
        "min_happiness", 
        "total_happiness",
        "fairness_ratio",
        "avg_allocation_size"
    ]
    
    for metric in metrics:
        plot_results(results_df, metric, results_dir)


def main():
    """
    Main function to run the experiments.
    """
    # Create results directory
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Define sizes to test
    # Start with smaller sizes and increase gradually
    # sizes = SMALL_SIZES + MEDIUM_SIZES  # Add LARGE_SIZES for more extensive tests
    sizes = [5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100]
    
    # Define algorithms to test
    # We include all algorithm variants defined in algorithm_variants.py
    algorithms = list(ALGORITHMS.keys())
    
    logger.info(f"Starting experiments with sizes: {sizes}")
    logger.info(f"Testing algorithms: {algorithms}")
    
    # Set timeout (30-60 seconds per algorithm as suggested)
    timeout = 45.0  # seconds
    
    # Run the experiments
    try:
        with experiment.Experiment(description="Santa Claus Problem Algorithm Comparison") as exp:
            # Run experiments and get results
            results_df = run_all_experiments(
                sizes=sizes, 
                algorithms=algorithms, 
                timeout=timeout, 
                results_dir=results_dir
            )
            
            # Plot the results
            plot_all_metrics(results_df, results_dir)
            
            # Save the experiment details
            exp.log_parameter("Instance Sizes", sizes)
            exp.log_parameter("Algorithms", algorithms)
            exp.log_parameter("Timeout (seconds)", timeout)
            exp.log_artifact(os.path.join(results_dir, "santa_claus_results_*.csv"))
            
            # Record metrics summary
            for metric in ["runtime", "min_happiness", "total_happiness", "fairness_ratio"]:
                if metric in results_df.columns:
                    exp.log_metric(f"Mean {metric}", results_df[metric].mean())
                    
    except Exception as e:
        logger.error(f"Error running experiments: {e}")
        raise


if __name__ == "__main__":
    main()
