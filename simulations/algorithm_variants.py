"""
Different algorithm implementations for the Santa Claus problem to be compared in experiments.

This file contains:
1. The main O(log log m / log log log m) approximation algorithm with different parameters
2. A greedy algorithm as a simpler alternative
3. A random allocation algorithm as a baseline
4. A round-robin allocation approach
"""

import sys
import os
import random
import time
import logging
from typing import Dict, List, Set, Tuple, Any, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the main algorithm
from backend.santa_claus.core import AllocationBuilder, Instance
from backend.santa_claus.algorithm import santa_claus_algorithm

# Setup flag for fairpyx availability
FAIRPYX_AVAILABLE = False
FairpyxInstance = None
FairpyxAllocationBuilder = None
fairpyx_santa_claus = None
fairpyx_maximin = None

# Setup logger
logger = logging.getLogger(__name__)

# Setup null handler to suppress output during benchmarks
null_handler = logging.NullHandler()
logger.addHandler(null_handler)
logger.setLevel(logging.CRITICAL)  # Only critical messages will be shown


def main_algorithm(instance: Instance, alpha: float = 3.0) -> Dict[str, List[str]]:
    """
    Run the main O(log log m / log log log m) approximation algorithm with the given alpha parameter.
    
    Args:
        instance: The Santa Claus problem instance
        alpha: Parameter for classifying gifts as large or small
        
    Returns:
        A dictionary mapping each agent to their allocated items
    """
    # Create a new allocation builder
    alloc = AllocationBuilder(instance)
    
    # Run the algorithm with the specified alpha parameter
    santa_claus_algorithm(alloc, alpha=alpha)
    
    # Return the allocation
    return alloc.allocation


def greedy_max_value_algorithm(instance: Instance) -> Dict[str, List[str]]:
    """
    A greedy algorithm that assigns each present to the kid who values it most.
    
    Args:
        instance: The Santa Claus problem instance
        
    Returns:
        A dictionary mapping each kid to their allocated presents
    """
    # Create a new allocation builder
    alloc = AllocationBuilder(instance)
    
    # Sort items by maximum value across all agents (in descending order)
    items_by_max_value = sorted(
        list(instance.items),
        key=lambda item: max(instance.agent_item_value(agent, item) for agent in instance.agents),
        reverse=True
    )
    
    # For each item, assign it to the agent who values it most
    for item in items_by_max_value:
        # Find the agent who values this item most
        best_agent = max(
            instance.agents, 
            key=lambda agent: instance.agent_item_value(agent, item)
        )
        
        # Get the agent's current allocation
        current_allocation = alloc.get_allocation().get(best_agent, [])
        
        # Check if the agent has capacity for one more item
        if len(current_allocation) < instance.agent_capacities.get(best_agent, float('inf')):
            # Allocate the item to this agent
            alloc.give(best_agent, item)
    
    return alloc.allocation


def greedy_min_value_algorithm(instance: Instance) -> Dict[str, List[str]]:
    """
    A greedy algorithm that focuses on maximizing the minimum value.
    It assigns each present to the kid with the lowest total value so far.
    
    Args:
        instance: The Santa Claus problem instance
        
    Returns:
        A dictionary mapping each kid to their allocated presents
    """
    # Create a new allocation builder
    alloc = AllocationBuilder(instance)
    
    # Track total value for each agent
    agent_values = {agent: 0.0 for agent in instance.agents}
    
    # Sort items by maximum value (in descending order)
    items_by_max_value = sorted(
        list(instance.items),
        key=lambda item: max(instance.agent_item_value(agent, item) for agent in instance.agents),
        reverse=True
    )
    
    # For each item, assign it to the agent with the lowest total value
    for item in items_by_max_value:
        # Find eligible agents (those who haven't reached capacity)
        eligible_agents = [
            agent for agent in instance.agents
            if len(alloc.get_allocation().get(agent, [])) < instance.agent_capacities.get(agent, float('inf'))
        ]
        
        if not eligible_agents:
            break  # All agents are at capacity
        
        # Find the eligible agent with the lowest total value so far
        best_agent = min(
            eligible_agents, 
            key=lambda agent: agent_values[agent]
        )
        
        # Allocate the item and update total value
        alloc.give(best_agent, item)
        agent_values[best_agent] += instance.agent_item_value(best_agent, item)
    
    return alloc.allocation


def random_allocation_algorithm(instance: Instance) -> Dict[str, List[str]]:
    """
    A baseline algorithm that randomly assigns presents to kids.
    
    Args:
        instance: The Santa Claus problem instance
        
    Returns:
        A dictionary mapping each kid to their allocated presents
    """
    # Create a new allocation builder
    alloc = AllocationBuilder(instance)
    
    # Create a list of all items
    all_items = list(instance.items)
    random.shuffle(all_items)
    
    # Create a list of agents in random order
    all_agents = list(instance.agents)
    random.shuffle(all_agents)
    
    # Keep track of how many items each agent has received
    agent_counts = {agent: 0 for agent in all_agents}
    
    # Allocate each item to a random agent that hasn't reached capacity
    for item in all_items:
        # Find eligible agents (those who haven't reached capacity)
        eligible_agents = [
            agent for agent in all_agents
            if agent_counts[agent] < instance.agent_capacities.get(agent, float('inf'))
        ]
        
        if not eligible_agents:
            break  # All agents are at capacity
        
        # Choose a random eligible agent
        chosen_agent = random.choice(eligible_agents)
        
        # Allocate the item
        alloc.give(chosen_agent, item)
        agent_counts[chosen_agent] += 1
    
    return alloc.allocation


def round_robin_algorithm(instance: Instance) -> Dict[str, List[str]]:
    """
    A simple round-robin algorithm that allocates presents to kids in a cyclic manner.
    
    Args:
        instance: The Santa Claus problem instance
        
    Returns:
        A dictionary mapping each kid to their allocated presents
    """
    # Create a new allocation builder
    alloc = AllocationBuilder(instance)
    
    # Sort items by maximum value across all agents (in descending order)
    items_by_max_value = sorted(
        list(instance.items),
        key=lambda item: max(instance.agent_item_value(agent, item) for agent in instance.agents),
        reverse=True
    )
    
    # Create an ordered list of agents
    all_agents = sorted(list(instance.agents))
    
    # Keep track of how many items each agent has received
    agent_counts = {agent: 0 for agent in all_agents}
    
    # Allocate each item in a round-robin fashion
    agent_index = 0
    for item in items_by_max_value:
        # Find the next agent that hasn't reached capacity
        while True:
            agent = all_agents[agent_index]
            agent_index = (agent_index + 1) % len(all_agents)
            
            if agent_counts[agent] < instance.agent_capacities.get(agent, float('inf')):
                # Allocate the item
                alloc.give(agent, item)
                agent_counts[agent] += 1
                break
                
            # If we've gone through all agents and none have capacity, break
            if agent_index == 0:
                break
        
        # If we've gone through all agents and none have capacity, break
        if agent_index == 0 and all(agent_counts[a] >= instance.agent_capacities.get(a, float('inf')) for a in all_agents):
            break
    
    return alloc.allocation


# Implement Leximin allocation algorithm
def leximin_allocation_algorithm(instance: Instance) -> Dict[str, List[str]]:
    """
    Leximin allocation algorithm for fair division. This algorithm tries to maximize
    the minimum happiness, then the second minimum, and so on.
    
    This is a different approach to solving the Santa Claus problem, focusing on
    iterative improvement rather than configuration LP.
    
    Args:
        instance: Problem instance
        
    Returns:
        Dictionary mapping agent names to their allocated items
    """
    logger.info("Starting Leximin allocation algorithm...")
    
    # Create a new allocation builder
    alloc = AllocationBuilder(instance)
    
    # Keep track of agent values
    agent_values = {agent: 0.0 for agent in instance.agents}
    unallocated_items = list(instance.items)
    
    # Sort items by maximum value any agent has for them (descending)
    item_max_values = {item: max(instance.valuations[agent][item] for agent in instance.agents) 
                      for item in instance.items}
    unallocated_items.sort(key=lambda item: item_max_values[item], reverse=True)
    
    # Assign each item to the agent who has minimum value so far
    # This is a greedy approach to the leximin solution
    for item in unallocated_items:
        # Find agent with minimum value so far
        min_agent = min(instance.agents, key=lambda agent: agent_values[agent])
        
        # If multiple agents have the same minimum value, choose the one who values the item most
        min_value = agent_values[min_agent]
        min_agents = [agent for agent in instance.agents if agent_values[agent] == min_value]
        
        if len(min_agents) > 1:
            min_agent = max(min_agents, key=lambda agent: instance.valuations[agent][item])
        
        # Assign item to this agent
        alloc.give(min_agent, item)
        agent_values[min_agent] += instance.valuations[min_agent][item]
    
    logger.info(f"Leximin allocation complete. Final agent values: {agent_values}")
    return alloc.allocation


# Implement Envy-Free Up To One Good (EF1) allocation algorithm
def ef1_allocation_algorithm(instance: Instance) -> Dict[str, List[str]]:
    """
    Envy-Free Up To One Good (EF1) allocation algorithm for fair division.
    This algorithm tries to create an allocation where each agent does not envy another
    agent after removing at most one item from the other agent's bundle.
    
    Args:
        instance: Problem instance
        
    Returns:
        Dictionary mapping agent names to their allocated items
    """
    logger.info("Starting EF1 allocation algorithm...")
    
    # Create a new allocation builder
    alloc = AllocationBuilder(instance)
    
    # Keep track of each agent's bundle value to them
    agent_values = {agent: 0.0 for agent in instance.agents}
    
    # Sort items by maximum value any agent has for them (descending)
    item_values = [(item, max(instance.valuations[agent][item] for agent in instance.agents))
                  for item in instance.items]
    item_values.sort(key=lambda x: x[1], reverse=True)
    
    # Use a round-robin approach with envy minimization
    agents_order = list(instance.agents)
    random.shuffle(agents_order)  # Random order for initial allocation
    
    # First pass: give each agent their most valued item from remaining items
    for agent in agents_order:
        if not item_values:
            break
            
        # Find item that this agent values most
        best_item_idx = max(range(len(item_values)), 
                           key=lambda i: instance.valuations[agent][item_values[i][0]])
        best_item = item_values.pop(best_item_idx)[0]
        
        alloc.give(agent, best_item)
        agent_values[agent] += instance.valuations[agent][best_item]
    
    # Remaining items: give to agent who has least value so far
    while item_values:
        item, _ = item_values.pop(0)
        
        # Find agent who currently has minimum value
        min_agent = min(instance.agents, key=lambda a: agent_values[a])
        alloc.give(min_agent, item)
        agent_values[min_agent] += instance.valuations[min_agent][item]
    
    logger.info(f"EF1 allocation complete. Final agent values: {agent_values}")
    return alloc.allocation


# נסיונות להשתמש ב-fairpyx נכשלו, אז נשאיר את הקוד כאן כהערה בלבד

# # Convert our Instance to fairpyx Instance
# def convert_to_fairpyx_instance(instance: Instance):
#     if not FAIRPYX_AVAILABLE:
#         raise ImportError("fairpyx library is not available")
#         
#     # Create a new fairpyx Instance
#     return FairpyxInstance(
#         agents=instance.agents,
#         items=instance.items,
#         valuations=instance.valuations,
#         agent_capacities=instance.agent_capacities,
#         item_capacities=instance.item_capacities
#     )
# 
# # Wrapper for fairpyx santa_claus algorithm
# def fairpyx_santa_claus_algorithm(instance: Instance) -> Dict[str, List[str]]:
#     if not FAIRPYX_AVAILABLE:
#         logger.warning("fairpyx library is not available")
#         return {agent: [] for agent in instance.agents}
#     
#     # Convert our Instance to fairpyx Instance
#     fairpyx_inst = convert_to_fairpyx_instance(instance)
#     
#     # Create a fairpyx AllocationBuilder
#     fairpyx_alloc = FairpyxAllocationBuilder(fairpyx_inst)
#     
#     # Run the fairpyx santa_claus algorithm
#     fairpyx_santa_claus(fairpyx_alloc)
#     
#     # Return the allocation
#     return fairpyx_alloc.allocation
# 
# # Wrapper for fairpyx maximin algorithm
# def fairpyx_maximin_algorithm(instance: Instance) -> Dict[str, List[str]]:
#     if not FAIRPYX_AVAILABLE:
#         logger.warning("fairpyx library is not available")
#         return {agent: [] for agent in instance.agents}
#     
#     # Convert our Instance to fairpyx Instance
#     fairpyx_inst = convert_to_fairpyx_instance(instance)
#     
#     # Create a fairpyx AllocationBuilder
#     fairpyx_alloc = FairpyxAllocationBuilder(fairpyx_inst)
#     
#     # Run the fairpyx maximin algorithm

# Dictionary mapping algorithm names to their implementations
ALGORITHMS = {
    "Santa Claus": lambda instance: main_algorithm(instance, alpha=3.0),
    "Leximin": leximin_allocation_algorithm,
    "EF1": ef1_allocation_algorithm,
    "Greedy (Max Value)": greedy_max_value_algorithm,
    "Greedy (Min Value)": greedy_min_value_algorithm,
    "Round Robin": round_robin_algorithm,
    "Random": random_allocation_algorithm
}
# # Add fairpyx algorithms if available
# if FAIRPYX_AVAILABLE:
#     ALGORITHMS.update({
#         "Fairpyx Santa Claus": fairpyx_santa_claus_algorithm,
#         "Fairpyx Maximin": fairpyx_maximin_algorithm
#     })
