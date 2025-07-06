"""
Script for comparing the Santa Claus algorithm with other allocation algorithms.
"""

import sys
import os
import time
import logging
import csv
import matplotlib.pyplot as plt
import numpy as np
import random
from typing import Dict, List, Any, Tuple
import pandas as pd
from datetime import datetime
import traceback

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulations.instance_generator import generate_random_instance, SMALL_SIZES, MEDIUM_SIZES
from simulations.algorithm_variants import ALGORITHMS
from simulations.metrics import measure_solution_quality, run_with_timeout

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # גם לקונסול
        logging.FileHandler('algorithm_comparison.log')  # גם לקובץ
    ]
)
logger = logging.getLogger(__name__)

# הדפסה ישירה לקונסול בנוסף ללוגר
print("Starting algorithm comparison script...")

# Input sizes to test
COMPARISON_SIZES = [5, 10, 15, 20]

def run_comparison_experiment(timeout: float = 60.0, seed: int = 42) -> None:
    """
    Run comparison experiments between Santa Claus algorithm and other allocation algorithms.
    
    Args:
        timeout: Maximum runtime allowed in seconds
        seed: Random seed for instance generation
    """
    print(f"Starting comparison experiment with timeout={timeout}s, seed={seed}")
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join("simulations", "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Create CSV file for results
    csv_file = os.path.join(results_dir, "algorithm_comparison.csv")
    
    # Set random seed
    random.seed(seed)
    
    # Create and write headers to CSV
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "algorithm_name", "instance_size", "runtime", "timed_out",
            "min_happiness", "total_happiness", "fairness_ratio", "seed"
        ])
    
    # Run experiments for each algorithm and instance size
    for size in COMPARISON_SIZES:
        logger.info(f"Running experiments for instance size {size}")
        
        # Generate instance once for fair comparison
        instance = generate_random_instance(
            num_kids=max(2, size // 2),
            num_presents=size,
            max_value=100,
            kid_capacity=3
        )
        
        for algo_name, algo_func in ALGORITHMS.items():
            logger.info(f"  Running {algo_name}...")
            
            try:
                # Run algorithm with timeout
                start_time = time.time()
                allocation, execution_time, timed_out = run_with_timeout(
                    algo_func, args=(instance,), timeout=timeout
                )
                
                # Calculate metrics if algorithm completed
                if not timed_out:
                    try:
                        metrics = measure_solution_quality(instance, allocation)
                        min_happiness = metrics["min_happiness"]
                        total_happiness = metrics["total_happiness"]
                        fairness_ratio = metrics["fairness_ratio"]
                    except Exception as e:
                        logger.error(f"Error calculating metrics: {e}")
                        logger.error(traceback.format_exc())
                        min_happiness = total_happiness = fairness_ratio = float('nan')
                else:
                    min_happiness = total_happiness = fairness_ratio = float('nan')
                
                # Write results to CSV
                with open(csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        algo_name, size, execution_time, 1 if timed_out else 0,
                        min_happiness, total_happiness, fairness_ratio, seed
                    ])
                
                logger.info(f"  {algo_name} on size {size}: runtime={execution_time:.3f}s, timed_out={timed_out}")
                
            except Exception as e:
                logger.error(f"Error running {algo_name} on instance size {size}: {e}")
                logger.error(traceback.format_exc())
                
                # Still record the failure in CSV
                with open(csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        algo_name, size, timeout, 1, # Timed out
                        float('nan'), float('nan'), float('nan'), seed
                    ])


def create_comparison_plots(csv_file: str, results_dir: str) -> None:
    """
    Create plots comparing different allocation algorithms.
    
    Args:
        csv_file: Path to CSV file with results
        results_dir: Directory to save plots
    """
    # Read data
    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded data with columns: {df.columns}")
        print(f"Algorithms in data: {df['algorithm_name'].unique()}")
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        return
    
    metrics = ["runtime", "min_happiness", "total_happiness", "fairness_ratio"]
    titles = {
        "runtime": "Runtime Comparison of Allocation Algorithms",
        "min_happiness": "Minimum Happiness Comparison of Allocation Algorithms",
        "total_happiness": "Total Happiness Comparison of Allocation Algorithms",
        "fairness_ratio": "Fairness Ratio Comparison of Allocation Algorithms"
    }
    
    # Ensure the columns exist in the dataframe
    for metric in metrics:
        if metric not in df.columns:
            logger.error(f"Column {metric} not found in CSV data")
            return
    
    # Create plots
    for metric in metrics:
        plt.figure(figsize=(12, 8))
        
        # Plot data by algorithm
        for algo in df["algorithm_name"].unique():
            algo_data = df[df["algorithm_name"] == algo]
            
            # For runtime, include timed out runs at the timeout value
            if metric == "runtime":
                # Mark timed out points differently
                timed_out_data = algo_data[algo_data["timed_out"] == 1]
                if len(timed_out_data) > 0:
                    plt.scatter(
                        timed_out_data["instance_size"], 
                        timed_out_data["runtime"],
                        marker='x', s=100
                    )
            
            # Regular plot for non-timed-out data
            valid_data = algo_data[~pd.isna(algo_data[metric])]
            if len(valid_data) > 0:
                plt.plot(
                    valid_data["instance_size"], 
                    valid_data[metric], 
                    marker='o', 
                    label=algo
                )
        
        # Add title and labels
        plt.title(titles[metric])
        plt.xlabel("Instance Size (Number of Presents)")
        
        if metric == "runtime":
            plt.ylabel("Runtime (seconds)")
            plt.yscale("log")  # Log scale for runtime
        elif metric == "min_happiness":
            plt.ylabel("Minimum Happiness")
        elif metric == "total_happiness":
            plt.ylabel("Total Happiness")
        elif metric == "fairness_ratio":
            plt.ylabel("Fairness Ratio")
            plt.ylim(0, 1.1)  # Fairness ratio is between 0 and 1
        
        # Add legend
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plot_file = os.path.join(results_dir, f"algorithm_comparison_{metric}.png")
        plt.savefig(plot_file)
        logger.info(f"Saved plot to {plot_file}")


def find_best_algorithms(csv_file: str) -> None:
    """
    Analyze the results to find which algorithms perform best for different metrics.
    
    Args:
        csv_file: Path to CSV file with results
    """
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        return
    
    # Filter out timed out or failed runs
    df_valid = df[df["timed_out"] == 0]
    
    # Find best algorithm for minimum happiness (MaxiMin objective)
    print("\n==== Best Algorithms by Metric ====")
    
    metrics = ["min_happiness", "total_happiness", "fairness_ratio", "runtime"]
    better_funcs = {
        "min_happiness": max,    # Higher is better
        "total_happiness": max,  # Higher is better
        "fairness_ratio": max,   # Higher is better
        "runtime": min           # Lower is better
    }
    
    for metric in metrics:
        print(f"\n--- {metric} ---")
        for size in df["instance_size"].unique():
            size_data = df_valid[df_valid["instance_size"] == size]
            
            if len(size_data) == 0:
                print(f"  Size {size}: No valid data")
                continue
                
            # Find best algorithm(s)
            better_func = better_funcs[metric]
            best_value = better_func(size_data[metric])
            best_algos = size_data[size_data[metric] == best_value]["algorithm_name"].tolist()
            
            print(f"  Size {size}: {', '.join(best_algos)} ({best_value:.2f})")


def main():
    """
    Main function to run algorithm comparison experiments.
    """
    print("Starting main function...")
    
    logger.info("Starting algorithm comparison experiments")
    
    # Create results directory
    results_dir = os.path.join("simulations", "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    print(f"Results directory: {results_dir}")
    
    # Path to the CSV file with results
    csv_file = os.path.join(results_dir, "algorithm_comparison.csv")
    
    # Force new experiments to include the improved algorithm
    print("Running new experiments with the improved algorithm...")
    run_comparison_experiment(timeout=60.0)
    create_comparison_plots(csv_file, results_dir)
    find_best_algorithms(csv_file)
    
    print("Algorithm comparison completed.")


if __name__ == "__main__":
    main()
