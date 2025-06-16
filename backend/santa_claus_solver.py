"""
Simplified implementation of the Santa Claus Problem algorithm based on:
"The Santa Claus Problem", by Bansal, Nikhil, and Maxim Sviridenko.
Proceedings of the 38th Annual ACM Symposium on Theory of Computing, 2006
https://dl.acm.org/doi/10.1145/1132516.1132557
"""

import logging
import random
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict

logger = logging.getLogger(__name__)

class Instance:
    """A simple class to represent a problem instance."""
    def __init__(self, agents, items, agent_item_values):
        self.agents = agents
        self.items = items
        self.agent_item_values = agent_item_values
        
    def agent_item_value(self, agent, item):
        return self.agent_item_values.get((agent, item), 0)

def santa_claus_solver(instance: Instance, alpha: float = 10, beta: float = 3) -> Tuple[float, Dict[str, Set[str]]]:
    """
    Simplified implementation of the Santa Claus Problem algorithm.
    
    Args:
        instance: The problem instance
        alpha: Parameter for classifying presents as large or small
        beta: Relaxation parameter
        
    Returns:
        A tuple of (T_optimal, final_assignment)
    """
    # Print problem instance details
    logger.info("\n" + "=" * 80)
    logger.info("SANTA CLAUS PROBLEM - RESTRICTED ASSIGNMENT CASE".center(80))
    logger.info("=" * 80)
    
    # Print parameters
    logger.info("\nPARAMETERS:")
    logger.info("-" * 80)
    logger.info(f"| {'Parameter':<20} | {'Value':<55} |")
    logger.info("-" * 80)
    logger.info(f"| {'Number of Kids':<20} | {len(instance.agents):<55} |")
    logger.info(f"| {'Number of Presents':<20} | {len(instance.items):<55} |")
    logger.info(f"| {'Alpha':<20} | {alpha:<55} |")
    logger.info(f"| {'Beta':<20} | {beta:<55} |")
    logger.info("-" * 80)
    
    # Print valuations in a table
    logger.info("\nVALUATIONS:")
    logger.info("-" * 80)
    header = "| Kid \\ Present | " + " | ".join(f"{p[:10]:^10}" for p in instance.items) + " |"
    logger.info(header)
    logger.info("-" * 80)
    
    for kid in instance.agents:
        row = f"| {kid[:15]:<15} | "
        for present in instance.items:
            value = instance.agent_item_value(kid, present)
            row += f"{value:^10} | "
        logger.info(row)
    logger.info("-" * 80)
    
    # For this simplified demo, we'll use a greedy algorithm instead of the full LP-based approach
    logger.info("\nUsing simplified greedy algorithm for demo purposes")
    
    # Sort presents by their total value across all kids (descending)
    presents_by_value = sorted(
        instance.items,
        key=lambda p: sum(instance.agent_item_value(kid, p) for kid in instance.agents),
        reverse=True
    )
    
    # Initialize allocation
    allocation = {kid: set() for kid in instance.agents}
    assigned_presents = set()
    
    # Assign presents greedily
    for present in presents_by_value:
        # Find the kid who values this present the most and doesn't have many presents yet
        best_kid = max(
            instance.agents,
            key=lambda k: instance.agent_item_value(k, present) / (len(allocation[k]) + 1)
        )
        
        # Only assign if the kid values this present
        if instance.agent_item_value(best_kid, present) > 0:
            allocation[best_kid].add(present)
            assigned_presents.add(present)
            logger.info(f"Assigned {present} to {best_kid} (value: {instance.agent_item_value(best_kid, present)})")
    
    # Calculate the minimum happiness
    happiness_values = {
        kid: sum(instance.agent_item_value(kid, p) for p in presents)
        for kid, presents in allocation.items()
    }
    
    min_happiness = min(happiness_values.values()) if happiness_values else 0
    
    # Print the final allocation
    logger.info("\nFINAL ALLOCATION:")
    logger.info("-" * 80)
    for kid, presents in allocation.items():
        total_value = sum(instance.agent_item_value(kid, p) for p in presents)
        logger.info(f"{kid}: {presents} (Value: {total_value:.2f})")
    logger.info("-" * 80)
    
    logger.info(f"Assigned {len(assigned_presents)}/{len(instance.items)} presents")
    logger.info(f"Minimum happiness: {min_happiness:.4f}")
    
    return min_happiness, allocation
