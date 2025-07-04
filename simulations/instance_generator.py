"""
Instance generator for Santa Claus Problem experiments.
Generates random problem instances with different sizes for benchmarking.
"""

import random
from typing import Dict, List, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.santa_claus.core import Instance


def generate_random_instance(num_kids: int, num_presents: int, 
                            min_value: float = 1.0, max_value: float = 10.0,
                            kid_capacity: int = None, present_capacity: int = 1) -> Instance:
    """
    Generate a random Santa Claus Problem instance.
    
    Args:
        num_kids: Number of kids (agents)
        num_presents: Number of presents (items)
        min_value: Minimum value of a present to a kid
        max_value: Maximum value of a present to a kid
        kid_capacity: Maximum number of presents a kid can receive (None for unlimited)
        present_capacity: Maximum number of kids that can receive each present (usually 1)
        
    Returns:
        A Santa Claus Problem instance
    """
    # Create kid and present names
    kid_names = [f"kid_{i+1}" for i in range(num_kids)]
    present_names = [f"present_{i+1}" for i in range(num_presents)]
    
    # Generate random valuations
    valuations = {}
    for kid in kid_names:
        valuations[kid] = {}
        for present in present_names:
            # Each kid values each present randomly between min_value and max_value
            valuations[kid][present] = random.uniform(min_value, max_value)
    
    # Create agent and item capacities
    agent_capacities = {kid: kid_capacity for kid in kid_names} if kid_capacity else {}
    item_capacities = {present: present_capacity for present in present_names}
    
    # Create and return the instance
    return Instance(kid_names, present_names, valuations, agent_capacities, item_capacities)


def generate_random_instances(sizes: List[int], min_kids: int = 3, 
                             min_value: float = 1.0, max_value: float = 10.0) -> Dict[str, Instance]:
    """
    Generate a set of random instances with increasing sizes.
    
    Args:
        sizes: List of sizes (number of presents) to generate
        min_kids: Minimum number of kids (agents)
        min_value: Minimum value of a present to a kid
        max_value: Maximum value of a present to a kid
        
    Returns:
        Dictionary mapping instance names to instances
    """
    instances = {}
    
    for size in sizes:
        # Calculate number of kids based on size
        # We use sqrt(size) as a heuristic to have a reasonable number of kids
        # with at least 2*sqrt(size) kids
        num_kids = max(min_kids, int(2 * (size ** 0.5)))
        
        # Determine kid capacity based on number of presents and kids
        # Each kid should get roughly the same number of presents
        kid_capacity = max(1, int(1.5 * size / num_kids))
        
        # Generate and store the instance
        instance_name = f"n{size}_k{num_kids}"
        instances[instance_name] = generate_random_instance(
            num_kids=num_kids,
            num_presents=size,
            min_value=min_value,
            max_value=max_value,
            kid_capacity=kid_capacity
        )
    
    return instances


# Example sizes for experiments
SMALL_SIZES = [10, 20, 30, 40, 50]
MEDIUM_SIZES = [60, 80, 100, 120, 150]
LARGE_SIZES = [200, 300, 400, 500]
VERY_LARGE_SIZES = [600, 800, 1000]

# Combined list of sizes
ALL_SIZES = SMALL_SIZES + MEDIUM_SIZES + LARGE_SIZES + VERY_LARGE_SIZES
