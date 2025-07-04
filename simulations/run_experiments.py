"""
Main experiment runner for Santa Claus Problem algorithm comparisons.
Uses the experiments-csv library to run experiments and generate visualizations.
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

from simulations.instance_generator import generate_random_instance, SMALL_SIZES, MEDIUM_SIZES, LARGE_SIZES
from simulations.algorithm_variants import ALGORITHMS
from simulations.metrics import measure_solution_quality, run_with_timeout


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Import the experiments-csv library
import experiments_csv


def run_santa_claus_experiment(algorithm_name: str, instance_size: int, timeout: float = 60.0, seed: int = 42) -> Dict[str, Any]:
    """
    Run a single experiment with specified algorithm and instance size.
    This function is designed to be used with experiments_csv.
    
    Args:
        algorithm_name: Name of algorithm to run
        instance_size: Size of problem instance
        timeout: Maximum runtime allowed in seconds
        seed: Random seed for instance generation
        
    Returns:
        Dictionary with experiment results
    """
    try:
        # Get the algorithm function
        algorithm = ALGORITHMS.get(algorithm_name)
        if not algorithm:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
        
        # Generate a random instance
        instance = generate_random_instance(
            num_kids=max(2, instance_size // 2),  # At least 2 kids
            num_presents=instance_size,
            max_value=100,
            kid_capacity=3
        )
        # Set random seed
        random.seed(seed)
        
        start_time = time.time()
        
        try:
            # Run the algorithm with timeout
            allocation, execution_time, timed_out = run_with_timeout(
                algorithm, args=(instance,), timeout=timeout
            )
            
            # Calculate metrics if algorithm completed successfully
            if not timed_out:
                try:
                    # Use measure_solution_quality to get all metrics
                    metrics = measure_solution_quality(instance, allocation)
                    min_happiness = metrics["min_happiness"]
                    total_happiness = metrics["total_happiness"]
                    fairness_ratio = metrics["fairness_ratio"]
                except Exception as metric_error:
                    logger.error(f"Error calculating metrics: {metric_error}")
                    logger.error(traceback.format_exc())
                    min_happiness = total_happiness = fairness_ratio = float('nan')
            else:
                min_happiness = total_happiness = fairness_ratio = float('nan')
            
            return {
                "runtime": execution_time,
                "timed_out": 1 if timed_out else 0,  # Convert to 1/0 for CSV
                "min_happiness": min_happiness,
                "total_happiness": total_happiness,
                "fairness_ratio": fairness_ratio
            }
            
        except Exception as e:
            logger.error(f"Error running {algorithm_name} on instance size {instance_size}: {e}")
            logger.error(traceback.format_exc())
            return {
                "runtime": time.time() - start_time,
                "timed_out": 1,  # Indicates error
                "min_happiness": float('nan'),
                "total_happiness": float('nan'),
                "fairness_ratio": float('nan'),
                "error": str(e)
            }
    except Exception as outer_e:
        logger.error(f"Error setting up experiment: {outer_e}")
        logger.error(traceback.format_exc())
        return {
            "runtime": 0.0,
            "timed_out": 1,
            "min_happiness": float('nan'),
            "total_happiness": float('nan'),
            "fairness_ratio": float('nan'),
            "error": str(outer_e)
        }


def generate_parameter_ranges(sizes: List[int], algorithms: List[str] = None, 
                            timeout: float = 60.0, seeds: List[int] = None) -> Dict[str, List]:
    """
    Generate parameter ranges for the experiments.
    
    Args:
        sizes: List of instance sizes to test
        algorithms: List of algorithm names to test
        timeout: Maximum runtime allowed per algorithm
        seeds: List of random seeds for instance generation
        
    Returns:
        Dictionary with parameter ranges
    """
    if algorithms is None:
        algorithms = list(ALGORITHMS.keys())
        
    if seeds is None:
        seeds = [42]  # Default seed
        
    # Create parameter ranges dictionary
    parameter_ranges = {
        "algorithm_name": algorithms,
        "instance_size": sizes,
        "timeout": [timeout],
        "seed": seeds
    }
    
    return parameter_ranges


def create_plots_with_library(csv_file: str, results_dir: str) -> List[str]:
    """
    Create plots using the experiments-csv library and matplotlib
    
    Args:
        csv_file: Path to CSV file with results
        results_dir: Directory to save plots
        
    Returns:
        List of plot file paths
    """
    metrics = ["runtime", "min_happiness", "total_happiness", "fairness_ratio"]
    plot_files = []
    
    try:
        logger.info(f"Starting plot creation with {len(metrics)} metrics")
        logger.info(f"Checking if experiments_csv module has plot attribute: {hasattr(experiments_csv, 'plot')}")
        
        # בדוק מהן הפונקציות הזמינות ב-experiments_csv
        logger.info(f"Dir of experiments_csv: {dir(experiments_csv)}")
        
        if hasattr(experiments_csv, 'plot'):
            logger.info(f"Dir of experiments_csv.plot: {dir(experiments_csv.plot)}")
        
        # נסה ליצור את הגרפים בצורה ידנית
        for metric in metrics:
            try:
                # Create plot file path
                plot_file = os.path.join(results_dir, f"{metric}_vs_size_by_algorithm.png")
                logger.info(f"Creating plot for {metric}, saving to {plot_file}")
                
                # קרא את הנתונים באופן ידני
                df = pd.read_csv(csv_file)
                
                # פילטור נתונים לא תקינים
                if metric in ["min_happiness", "total_happiness", "fairness_ratio"]:
                    df = df[df["timed_out"] == 0]  # רק הרצות שהסתיימו בהצלחה
                
                # יצירת פלוט ידני עם matplotlib
                plt.figure(figsize=(10, 6))
                
                # צור פלוט לכל אלגוריתם
                for algo in df["algorithm_name"].unique():
                    algo_data = df[df["algorithm_name"] == algo]
                    plt.plot(algo_data["instance_size"], algo_data[metric], marker='o', label=algo)
                
                plt.title(f"{metric.replace('_', ' ').title()} vs Instance Size by Algorithm")
                plt.xlabel("Instance Size (Number of Presents)")
                plt.ylabel(metric.replace('_', ' ').title())
                plt.legend()
                plt.grid(True)
                
                # הוסף סקלה לוגריתמית לזמני ריצה
                if metric == "runtime":
                    plt.yscale("log")
                
                # Save figure
                plt.savefig(plot_file, dpi=300, bbox_inches="tight")
                plt.close()
                
                logger.info(f"Successfully created and saved plot to {plot_file}")
                plot_files.append(plot_file)
            
            except Exception as e:
                logger.error(f"Error creating plot for {metric}: {str(e)}")
                logger.error(traceback.format_exc())
        
    except Exception as e:
        logger.error(f"Error creating plots: {str(e)}")
        logger.error(traceback.format_exc())
    
    return plot_files


def main():
    """
    Main function to run the experiments using experiments-csv library.
    """
    # Set experiment parameters
    sizes = [5, 10, 15, 20]  # Starting with smaller sizes for testing
    algorithms = list(ALGORITHMS.keys())  # All algorithms
    timeout = 60.0  # Maximum runtime allowed per algorithm (seconds)
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    
    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    try:
        # Generate parameter ranges
        parameter_ranges = generate_parameter_ranges(
            sizes=sizes,
            algorithms=algorithms,
            timeout=timeout
        )
        
        # Create experiment object
        exp = experiments_csv.Experiment(
            results_dir,  # Results folder
            "santa_claus_results.csv",  # Results filename
            os.path.join(results_dir, "backups")  # Backup folder
        )
        
        # Enable logging
        exp.logger.setLevel(logging.INFO)
        
        # Run experiments
        logger.info("Starting Santa Claus experiments with experiments-csv library")
        logger.info(f"Parameter ranges: {parameter_ranges}")
        
        exp.run(run_santa_claus_experiment, parameter_ranges)
        
        # Create plots
        csv_file = os.path.join(results_dir, "santa_claus_results.csv")
        plot_files = create_plots_with_library(csv_file, results_dir)
        
        logger.info(f"Experiments completed. Results saved to {csv_file}")
        logger.info(f"Generated {len(plot_files)} plots in {results_dir}")
                    
    except Exception as e:
        logger.error(f"Error running experiments: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()