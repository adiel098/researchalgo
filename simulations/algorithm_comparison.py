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

# Input sizes to test - עם מספרים מותאמים כדי שהאלגוריתם המקורי יוכל להשלים
COMPARISON_SIZES = [20, 25, 30, 35, 40]

# Filter the algorithms to only include the original and improved Santa Claus algorithms
FILTERED_ALGORITHMS = {
    "Santa Claus (Original)": ALGORITHMS["Santa Claus (Original)"],
    "Santa Claus (Improved)": ALGORITHMS["Santa Claus (Improved)"]
}

def run_comparison_experiment(timeout: float = 60.0, seed: int = 42) -> None:
    """
    Run comparison experiments between Santa Claus algorithm and other allocation algorithms.
    
    Args:
        timeout: Maximum runtime allowed in seconds
        seed: Random seed for instance generation
    """
    print(f"Starting comparison experiment with timeout={timeout}s, seed={seed}")
    print(f"Testing sizes: {COMPARISON_SIZES}")
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join("simulations", "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Create CSV file for results
    csv_file = os.path.join(results_dir, "algorithm_comparison.csv")
    
    # Remove existing file to start fresh
    if os.path.exists(csv_file):
        os.remove(csv_file)
        print(f"Removed existing results file {csv_file}")
    
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
        print(f"\n========== Running experiments for instance size {size} ==========")
        logger.info(f"Running experiments for instance size {size}")
        
        # Generate instance once for fair comparison
        # הקטנת מספר הילדים עוד יותר - קבוע 3 ילדים לכל גודל קלט
        instance = generate_random_instance(
            num_kids=3,  # מספר קבוע של ילדים
            num_presents=size,
            max_value=100,
            kid_capacity=3
        )
        
        for algo_name, algo_func in FILTERED_ALGORITHMS.items():
            print(f"  Running {algo_name} with instance size {size}...")
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
                
                result_str = f"  {algo_name} on size {size}: runtime={execution_time:.3f}s, timed_out={timed_out}"
                if not timed_out:
                    result_str += f", min_happiness={min_happiness:.2f}, total={total_happiness:.2f}"
                print(result_str)
                logger.info(result_str)
                
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
    
    # הסרת fairness_ratio כפי שהתבקשנו
    metrics = ["runtime", "min_happiness", "total_happiness"]
    titles = {
        "runtime": "Runtime Comparison of Allocation Algorithms",
        "min_happiness": "Minimum Happiness Comparison of Allocation Algorithms",
        "total_happiness": "Total Happiness Comparison of Allocation Algorithms"
    }
    
    # Ensure the columns exist in the dataframe
    for metric in metrics:
        if metric not in df.columns:
            logger.error(f"Column {metric} not found in CSV data")
            return
    
    # Create simulations directory if it doesn't exist
    simulations_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "simulations")
    if not os.path.exists(simulations_dir):
        os.makedirs(simulations_dir)
        
    # יצירת גרף מאוחד עם 3 תת-גרפים
    fig, axes = plt.subplots(nrows=len(metrics), ncols=1, figsize=(14, 18))
    
    # Filter data to only include the original and improved algorithms
    filtered_algos = ["Santa Claus (Original)", "Santa Claus (Improved)"]
    df_filtered = df[df["algorithm_name"].isin(filtered_algos)]
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Plot data by algorithm
        for algo in filtered_algos:
            if algo not in df_filtered["algorithm_name"].unique():
                logger.warning(f"Algorithm {algo} not found in data")
                continue
                
            algo_data = df_filtered[df_filtered["algorithm_name"] == algo]
            
            # For runtime, include timed out runs at the timeout value
            if metric == "runtime":
                # Mark timed out points differently
                timed_out_data = algo_data[algo_data["timed_out"] == 1]
                if len(timed_out_data) > 0:
                    ax.scatter(
                        timed_out_data["instance_size"], 
                        timed_out_data["runtime"],
                        marker='x', s=100
                    )
            
            # Regular plot for non-timed-out data
            valid_data = algo_data[~pd.isna(algo_data[metric])]
            if len(valid_data) > 0:
                # הגדרת סגנון שונה לכל אלגוריתם
                line_style = '-'
                line_width = 2
                
                # לאלגוריתם המקורי נשתמש בקו מקווקו ורחב יותר
                if "Original" in algo:
                    line_style = '--'
                    line_width = 3
                
                # סיטות (היסטים) אם הערכים זהים בין האלגוריתמים
                jitter = 0
                if "Original" in algo and metric != "runtime":
                    jitter = 0.3  # הזזת קו האלגוריתם המקורי ימינה כדי שלא יהיה מוסתר
                
                # Plot the line
                ax.plot(
                    valid_data['instance_size'] + jitter,  # הוספת היסט למיקום הקו בציר X
                    valid_data[metric], 
                    marker='o',
                    linestyle=line_style,
                    linewidth=line_width,
                    label=algo
                )
        
        # Add title and labels
        ax.set_title(titles[metric], fontsize=16)
        ax.set_xlabel("Instance Size (Number of Presents)", fontsize=12)
        
        if metric == "runtime":
            ax.set_ylabel("Runtime (seconds)", fontsize=12)
            ax.set_yscale("log")  # Log scale for runtime
        elif metric == "min_happiness":
            ax.set_ylabel("Minimum Happiness", fontsize=12)
        elif metric == "total_happiness":
            ax.set_ylabel("Total Happiness", fontsize=12)
        
        ax.legend(fontsize=12)
        ax.grid(True)
    
    plt.tight_layout(pad=3.0)
    
    # Save combined plot
    combined_plot_file = os.path.join(simulations_dir, "algorithm_comparison_combined.png")
    plt.savefig(combined_plot_file)
    logger.info(f"Saved combined plot to {combined_plot_file}")
    
    # סגירת הגרף כדי לשחרר זיכרון
    plt.close(fig)


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
    # הגדלת זמן ה-timeout ל-600 שניות (10 דקות) כדי לאפשר לאלגוריתם המקורי לעבוד על קלטים גדולים יותר
    run_comparison_experiment(timeout=600.0)
    create_comparison_plots(csv_file, results_dir)
    find_best_algorithms(csv_file)
    
    print("Algorithm comparison completed.")


if __name__ == "__main__":
    main()
