"""
Metrics for evaluating Santa Claus Problem solutions.
"""

import time
from typing import Dict, List, Callable, Any, Tuple
import statistics

from backend.santa_claus.core import Instance


def measure_solution_quality(instance: Instance, allocation: Dict[str, List[str]]) -> Dict[str, float]:
    """
    Evaluate the quality of a solution to the Santa Claus problem.
    
    Args:
        instance: The problem instance
        allocation: A mapping from kids to the presents they received
        
    Returns:
        Dictionary containing metrics about the solution
    """
    metrics = {}
    
    # Check if all kids received a valid allocation
    for kid in instance.agents:
        if kid not in allocation:
            allocation[kid] = []
    
    # Calculate happiness for each kid
    happiness_values = {}
    for kid, presents in allocation.items():
        # Sum the value of all presents assigned to this kid
        kid_happiness = sum(instance.agent_item_value(kid, present) for present in presents)
        happiness_values[kid] = kid_happiness
    
    # Calculate metrics
    metrics["total_happiness"] = sum(happiness_values.values())
    metrics["min_happiness"] = min(happiness_values.values()) if happiness_values else 0
    metrics["max_happiness"] = max(happiness_values.values()) if happiness_values else 0
    metrics["avg_happiness"] = statistics.mean(happiness_values.values()) if happiness_values else 0
    metrics["happiness_stddev"] = statistics.stdev(happiness_values.values()) if len(happiness_values) > 1 else 0
    
    # Calculate fairness metrics
    if metrics["max_happiness"] > 0:
        metrics["fairness_ratio"] = metrics["min_happiness"] / metrics["max_happiness"]
    else:
        metrics["fairness_ratio"] = 0
    
    # Calculate allocation size metrics
    allocation_sizes = [len(presents) for kid, presents in allocation.items()]
    metrics["min_allocation_size"] = min(allocation_sizes) if allocation_sizes else 0
    metrics["max_allocation_size"] = max(allocation_sizes) if allocation_sizes else 0
    metrics["avg_allocation_size"] = statistics.mean(allocation_sizes) if allocation_sizes else 0
    
    return metrics


def run_with_timeout(func: Callable, args: Tuple = (), kwargs: Dict = None, 
                    timeout: float = 60.0) -> Tuple[Any, float, bool]:
    """
    Run a function with a timeout.
    
    Args:
        func: Function to run
        args: Arguments to pass to the function
        kwargs: Keyword arguments to pass to the function
        timeout: Maximum execution time in seconds
        
    Returns:
        Tuple of (result, execution_time, timed_out)
    """
    import threading
    import time
    
    if kwargs is None:
        kwargs = {}
        
    result = [None]
    exception = [None]
    finished = [False]
    
    def target():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            exception[0] = e
        finally:
            finished[0] = True
    
    # Start the thread
    thread = threading.Thread(target=target)
    start_time = time.time()
    thread.start()
    
    # Wait for the thread to finish or timeout
    thread.join(timeout)
    execution_time = time.time() - start_time
    
    # Check if the thread finished or timed out
    if thread.is_alive():
        # Thread is still running, so we timed out
        return None, execution_time, True
    else:
        # Thread finished
        if exception[0] is not None:
            # Function raised an exception
            raise exception[0]
        return result[0], execution_time, False
