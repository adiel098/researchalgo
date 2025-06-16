"""
Simplified implementation of the Santa Claus Problem algorithm based on:
"The Santa Claus Problem", by Bansal, Nikhil, and Maxim Sviridenko.
Proceedings of the 38th Annual ACM Symposium on Theory of Computing, 2006
https://dl.acm.org/doi/10.1145/1132516.1132557
"""

import logging
import random
from typing import Dict, List, Set, Tuple, Optional, Any
import itertools

logger = logging.getLogger(__name__)

class Instance:
    """A simple class to represent a problem instance."""
    def __init__(self, valuations, agent_capacities=None, item_capacities=None):
        self.valuations = valuations
        self.agents = list(valuations.keys())
        
        # Get all unique items from valuations
        self.items = set()
        for agent_vals in valuations.values():
            self.items.update(agent_vals.keys())
        self.items = list(self.items)
        
        # Set default capacities if not provided
        self.agent_capacities = agent_capacities or {agent: 1 for agent in self.agents}
        self.item_capacities = item_capacities or {item: 1 for item in self.items}
    
    def agent_item_value(self, agent, item):
        """Get the value of an item for an agent."""
        return self.valuations.get(agent, {}).get(item, 0)

class AllocationBuilder:
    """A class to build and track allocations."""
    def __init__(self, instance):
        self.instance = instance
        self.allocation = {agent: [] for agent in instance.agents}
    
    def allocate(self, agent, item):
        """Allocate an item to an agent."""
        if agent in self.allocation:
            self.allocation[agent].append(item)
    
    def get_allocation(self):
        """Get the current allocation."""
        return self.allocation

def santa_claus(alloc: AllocationBuilder) -> Dict[str, List[str]]:
    """
    Main entry point for the Santa Claus Problem algorithm.
    
    This implements a simplified greedy algorithm for the Santa Claus Problem.
    
    Args:
        alloc: The allocation builder that tracks the allocation process
        
    Returns:
        A dictionary mapping each agent (kid) to their allocated items (presents)
    """
    instance = alloc.instance
    
    # Log the problem details
    logger.info("\n" + "=" * 80)
    logger.info("SANTA CLAUS PROBLEM - SIMPLIFIED GREEDY IMPLEMENTATION".center(80))
    logger.info("=" * 80)
    
    # Log parameters
    logger.info("\nPARAMETERS:")
    logger.info("-" * 80)
    logger.info(f"| {'Parameter':<20} | {'Value':<55} |")
    logger.info("-" * 80)
    logger.info(f"| {'Number of Kids':<20} | {len(instance.agents):<55} |")
    logger.info(f"| {'Number of Presents':<20} | {len(instance.items):<55} |")
    logger.info("-" * 80)
    
    # Log valuations
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
    
    # Sort presents by their total value across all kids (descending)
    presents_by_value = sorted(
        instance.items,
        key=lambda p: sum(instance.agent_item_value(kid, p) for kid in instance.agents),
        reverse=True
    )
    
    # Keep track of assigned presents
    assigned_presents = set()
    
    # Assign presents greedily to maximize the minimum happiness
    while len(assigned_presents) < len(instance.items):
        # Find the kid with the minimum current happiness
        current_happiness = {
            kid: sum(instance.agent_item_value(kid, p) for p in alloc.allocation[kid])
            for kid in instance.agents
        }
        
        min_happiness_kid = min(
            instance.agents, 
            key=lambda k: current_happiness[k]
        )
        
        # Find the best unassigned present for this kid
        best_present = None
        best_value = -1
        
        for present in presents_by_value:
            if present not in assigned_presents:
                value = instance.agent_item_value(min_happiness_kid, present)
                if value > best_value:
                    best_value = value
                    best_present = present
        
        # If we found a present with positive value, assign it
        if best_present and best_value > 0:
            alloc.allocate(min_happiness_kid, best_present)
            assigned_presents.add(best_present)
            logger.info(f"Assigned {best_present} to {min_happiness_kid} (value: {best_value})")
        else:
            # No more valuable presents for this kid, move to the next one
            break
    
    # Calculate final happiness values
    final_happiness = {
        kid: sum(instance.agent_item_value(kid, p) for p in alloc.allocation[kid])
        for kid in instance.agents
    }
    
    # Log the final allocation
    logger.info("\nFINAL ALLOCATION:")
    logger.info("-" * 80)
    for kid, presents in alloc.allocation.items():
        logger.info(f"{kid}: {presents} (Value: {final_happiness[kid]})")
    logger.info("-" * 80)
    
    min_happiness = min(final_happiness.values()) if final_happiness else 0
    logger.info(f"Minimum happiness: {min_happiness}")
    
    return alloc.allocation

def divide(algorithm, instance):
    """
    Run the algorithm on the instance and return the allocation.
    
    Args:
        algorithm: The algorithm to run
        instance: The problem instance
        
    Returns:
        A dictionary mapping each agent to their allocated items
    """
    alloc = AllocationBuilder(instance)
    algorithm(alloc)
    return alloc.get_allocation()
