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
        self.agent_item_values = agent_item_values # Keep original for compatibility if needed
        
        # Pre-process for faster lookups
        self.agent_to_item_values = defaultdict(dict)
        self.item_to_agent_values = defaultdict(dict)
        for (agent, item), value in agent_item_values.items():
            self.agent_to_item_values[agent][item] = value
            self.item_to_agent_values[item][agent] = value

        # Pre-calculate total value for each present for faster sorting
        self.present_total_values = {
            item: sum(self.item_to_agent_values[item].values())
            for item in self.items
        }

    def agent_item_value(self, agent, item):
        # Use defaultdict's behavior: if agent is not found, it creates an empty dict
        # Then, .get(item, 0) handles cases where the item is not in that agent's dict
        return self.agent_to_item_values[agent].get(item, 0)

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
    # Use pre-calculated present_total_values from the Instance
    presents_by_value = sorted(
        instance.items,
        key=lambda p: instance.present_total_values[p],
        reverse=True
    )
    
    # Initialize allocation, kid present counts, and total assigned values for optimized greedy assignment
    allocation = {kid: set() for kid in instance.agents}
    kid_present_counts = {kid: 0 for kid in instance.agents} # New: track count of presents per kid
    kid_total_assigned_value = {kid: 0 for kid in instance.agents} # New: track total value assigned to each kid
    assigned_presents = set()
    
    # Assign presents greedily
    for present in presents_by_value:
        # Find the kid who values this present the most and doesn't have many presents yet
        # Pre-calculate scores for each kid for the current present
        kid_scores = []
        # Get all agent values for the current present directly
        present_agent_values = instance.item_to_agent_values[present]
        for k in instance.agents:
            # Use the pre-fetched values from present_agent_values
            value = present_agent_values.get(k, 0) # Use .get(k, 0) in case an agent doesn't value this specific item
            score = value / (kid_present_counts[k] + 1)
            kid_scores.append((score, k, value))

        # Find the kid with the maximum score and its corresponding value
        best_score, best_kid, best_kid_value = max(kid_scores, key=lambda x: x[0])
        
        # Only assign if the kid values this present
        if best_kid_value > 0:
            allocation[best_kid].add(present)
            assigned_presents.add(present)
            kid_present_counts[best_kid] += 1 # Update tracked count
            kid_total_assigned_value[best_kid] += best_kid_value # Update total assigned value
            logger.info(f"Assigned {present} to {best_kid} (value: {best_kid_value})")
    
    # Calculate the minimum happiness using pre-calculated totals
    happiness_values = kid_total_assigned_value
    
    min_happiness = min(happiness_values.values()) if happiness_values else 0
    total_happiness = sum(happiness_values.values()) if happiness_values else 0
    
    # Print the final allocation
    logger.info("\nFINAL ALLOCATION:")
    logger.info("-" * 80)
    for kid, presents in allocation.items():
        total_value = kid_total_assigned_value[kid] # Use pre-calculated total value
        logger.info(f"{kid}: {presents} (Value: {total_value:.2f})")
    logger.info("-" * 80)
    
    logger.info(f"Assigned {len(assigned_presents)}/{len(instance.items)} presents")
    logger.info(f"Minimum happiness: {min_happiness:.4f}")
    logger.info(f"Total happiness: {total_happiness:.4f}")
    
    return min_happiness, total_happiness, allocation