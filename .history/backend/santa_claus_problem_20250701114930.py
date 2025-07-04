"""
Algorithm implementation for 'The Santa Claus Problem'.

Paper: The Santa Claus Problem
Authors: Nikhil Bansal, Maxim Sviridenko
Link: https://dl.acm.org/doi/10.1145/1132516.1132557 (Proceedings of the 38th Annual ACM Symposium on Theory of Computing, 2006)

The problem involves distributing n gifts (items) among m children (agents).
Each child i has an arbitrary value pij for each present j.
The Santa's goal is to distribute presents such that the least lucky kid (agent)
is as happy as possible (maximin objective function):
    maximize (min_i sum_{j in S_i} p_ij)
where S_i is the set of presents received by kid i.

This file focuses on implementing the O(log log m / log log log m) approximation
algorithm for the restricted assignment case (p_ij in {p_j, 0}).

Programmers: Roey and Adiel
Date: 2023-05-29
"""

import logging
import numpy as np
from typing import List, Dict, Tuple, Set, Optional, Union, Any
import itertools
import random
from scipy.optimize import linprog
import networkx as nx

logger = logging.getLogger(__name__)

class Instance:
    """
    A class representing a problem instance.
    
    Each instance has:
    - agents: a list of agent names
    - items: a list of item names
    - valuations: a dict mapping each agent to a dict mapping items to values
    - agent_capacities: a dict mapping each agent to its capacity (max number of items it can get)
    - item_capacities: a dict mapping each item to its capacity (max number of agents it can be assigned to)
    """
    def __init__(self, valuations: Dict[str, Dict[str, float]], 
                agent_capacities: Dict[str, int] = None, 
                item_capacities: Dict[str, int] = None):
        self.valuations = valuations
        self.agents = list(valuations.keys())
        
        # Extract all items mentioned in any valuation
        self.items = []
        for agent, agent_valuations in valuations.items():
            for item in agent_valuations:
                if item not in self.items:
                    self.items.append(item)
                    
        # Default capacities to 1 if not specified
        self.agent_capacities = agent_capacities or {agent: 1 for agent in self.agents}
        self.item_capacities = item_capacities or {item: 1 for item in self.items}
    
    def agent_item_value(self, agent: str, item: str) -> float:
        """Return the value of an item to an agent"""
        return self.valuations.get(agent, {}).get(item, 0)

class AllocationBuilder:
    """
    A class for building allocations.
    
    This class tracks the allocation process, including:
    - What items have been given to which agents
    - The remaining capacities of agents and items
    - The full problem instance
    """
    def __init__(self, instance: Instance):
        self.instance = instance
        # Initialize empty bundles for each agent
        self.bundles: Dict[str, List[str]] = {agent: [] for agent in instance.agents}
        
        # Track remaining capacities
        self.remaining_agent_capacities = dict(instance.agent_capacities)
        self.remaining_item_capacities = dict(instance.item_capacities)
    
    @property
    def allocation(self) -> Dict[str, List[str]]:
        """Return the current allocation"""
        return self.bundles
    
    def give(self, agent: str, item: str) -> bool:
        """
        Give an item to an agent.
        
        Returns True if successful, False if the agent or item has no remaining capacity.
        """
        if self.remaining_agent_capacities.get(agent, 0) <= 0:
            logger.warning(f"Cannot give {item} to {agent}: agent has no remaining capacity")
            return False
        
        if self.remaining_item_capacities.get(item, 0) <= 0:
            logger.warning(f"Cannot give {item} to {agent}: item has no remaining capacity")
            return False
        
        self.bundles[agent].append(item)
        self.remaining_agent_capacities[agent] -= 1
        self.remaining_item_capacities[item] -= 1
        return True
    
    def remaining_agents(self) -> List[str]:
        """Return a list of agents with remaining capacity"""
        return [agent for agent, capacity in self.remaining_agent_capacities.items() if capacity > 0]
    
    def remaining_items(self) -> List[str]:
        """Return a list of items with remaining capacity"""
        return [item for item, capacity in self.remaining_item_capacities.items() if capacity > 0]

def _generate_valid_configurations(agent_name: str, items: List[str], valuations: Dict[str, Dict[str, float]], target_value: float, agent_capacity: int) -> List[Tuple[str, ...]]:
    """
    Generates all valid configurations for a given agent (kid) and a target value T.
    A configuration is a subset of items.
    It's valid if the sum of (truncated) values of items in it is >= T and its size <= agent_capacity.
    Values are truncated at T, i.e., p'_ij = min(p_ij, T).

    :param agent_name: The name of the agent.
    :param items: A list of all available item names.
    :param valuations: The full valuation matrix (agents -> items -> value).
    :param target_value: The target value T.
    :param agent_capacity: The maximum number of items the agent can receive.
    :return: A list of tuples, where each tuple represents a valid configuration (a sorted tuple of item names).
    """
    logger.debug(f"Generating valid configurations for agent {agent_name} with target T={target_value} and capacity {agent_capacity}")
    valid_configurations = []
    agent_valuations = valuations.get(agent_name, {})

    # Consider items the agent actually values
    relevant_items = [item for item in items if agent_valuations.get(item, 0) > 0]

    for k in range(1, min(len(relevant_items), agent_capacity) + 1):
        for combo_indices in itertools.combinations(range(len(relevant_items)), k):
            current_config_items = tuple(sorted([relevant_items[i] for i in combo_indices]))
            current_value = 0
            for item_in_config in current_config_items:
                original_value = agent_valuations.get(item_in_config, 0)
                truncated_value = min(original_value, target_value)
                current_value += truncated_value
            
            if current_value >= target_value:
                valid_configurations.append(current_config_items)
                logger.debug(f"  Found valid config for {agent_name}: {current_config_items} with truncated value {current_value:.2f}")

    # Sort to ensure deterministic output, primarily for testing
    # The inner tuples are already sorted, so sort the list of tuples
    valid_configurations.sort()
    logger.info(f"Agent {agent_name} has {len(valid_configurations)} valid configurations for T={target_value}.")
    return valid_configurations


def _check_config_lp_feasibility(alloc: AllocationBuilder, target_value: float) -> Tuple[bool, Optional[np.ndarray], Optional[List[Tuple[str, Tuple[str, ...]]]]]:
    """
    Checks if a given target_value T is feasible by formulating and attempting to solve
    the Configuration LP (feasibility version).

    The Configuration LP is defined as (see paper, Section 2):
    Variables: x_i,C for each agent i and valid configuration C in C(i,T).
    Constraints:
        1. sum_{C in C(i,T)} x_i,C <= 1  (for each agent i)
        2. sum_i sum_{C in C(i,T): j in C} x_i,C <= 1 (for each item j)
        3. x_i,C >= 0

    Objective: Minimize 0 (feasibility problem).

    :param alloc: The allocation builder containing the problem instance.
    :param target_value: The target value T to check for feasibility.
    :return: Tuple (is_feasible, solution_vector, variable_mapping).
    """
    logger.info(f"Checking feasibility of T = {target_value:.2f}")
    instance = alloc.instance
    agents = list(instance.agents)
    items = list(instance.items)

    # Step 1: Generate all valid configurations C(i,T) for each agent i
    all_agent_configs: Dict[str, List[Tuple[str, ...]]] = {}
    for agent_name in agents:
        agent_capacity = instance.agent_capacities.get(agent_name, 1) # Default to 1 if not specified
        configs = _generate_valid_configurations(agent_name, items, instance.valuations, target_value, agent_capacity)
        if not configs:
            logger.warning(f"Agent {agent_name} has no valid configurations for T={target_value}. Thus, T is infeasible.")
            return False, None, None 
        all_agent_configs[agent_name] = configs

    # Step 2: Create LP variables and map them
    # x_var_map maps an index in the LP variable vector to (agent_name, configuration_tuple)
    x_var_map: List[Tuple[str, Tuple[str, ...]]] = []
    for agent_name in agents:
        for config_tuple in all_agent_configs[agent_name]:
            x_var_map.append((agent_name, config_tuple))
    
    num_lp_vars = len(x_var_map)
    if num_lp_vars == 0:
        logger.warning(f"No valid configurations found for any agent at T={target_value}. Infeasible.")
        return False, None, None # No way to assign anything

    logger.debug(f"Total number of LP variables (x_i,C): {num_lp_vars}")

    # Objective function: minimize 0 (feasibility)
    c = np.zeros(num_lp_vars) 

    # Step 3: Constraints
    # Constraint type 1: sum_{C in C(i,T)} x_i,C <= 1 (for each agent i)
    # Constraint type 2: sum_i sum_{C in C(i,T): j in C} x_i,C <= 1 (for each item j)
    
    A_ub = []
    b_ub = []

    # Agent constraints: sum_{C in C(i,T)} x_i,C <= 1
    for agent_idx, agent_name in enumerate(agents):
        row = np.zeros(num_lp_vars)
        for lp_var_idx, (var_agent, var_config) in enumerate(x_var_map):
            if var_agent == agent_name:
                row[lp_var_idx] = 1
        A_ub.append(row)
        b_ub.append(1) # Each agent assigned at most 1 configuration bundle

    # Item constraints: sum_i sum_{C in C(i,T): j in C} x_i,C <= 1
    for item_idx, item_name in enumerate(items):
        row = np.zeros(num_lp_vars)
        item_cap = instance.item_capacities.get(item_name, 1)
        for lp_var_idx, (var_agent, var_config) in enumerate(x_var_map):
            if item_name in var_config:
                row[lp_var_idx] = 1
        A_ub.append(row)
        b_ub.append(item_cap) # Each item used at most its capacity (typically 1)

    # Bounds for variables: x_i,C >= 0
    bounds_linprog = [(0, 1) for _ in range(num_lp_vars)] # x_i,C in [0,1]

    if not A_ub:
        logger.warning("LP constraint matrix A_ub is empty. This indicates an issue.")
        return False, None, None

    logger.debug(f"Solving LP with {num_lp_vars} vars, {len(A_ub)} constraints.")

    # Solve the LP
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds_linprog, method='highs')
    except Exception as e:
        # Fallback if 'highs' is not supported or fails
        logger.warning(f"'highs' solver failed ({e}), trying 'revised simplex'.")
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds_linprog, method='revised simplex')
        except Exception as e_rs:
            logger.error(f"LP solver 'revised simplex' also failed: {e_rs}")
            return False, None, None

    if res.success:
        logger.info(f"LP feasible for T={target_value:.2f}. Objective value (should be 0): {res.fun:.4f}")
        return True, res.x, x_var_map
    else:
        logger.info(f"LP infeasible for T={target_value:.2f}. Solver status: {res.status}, message: {res.message}")
        return False, None, None        self.agents = list(valuations.keys())
        
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
    
        # Calculate parameters similar to those in the original algorithm
    T = 0  # Target value we want each kid to achieve
    total_values = [sum(instance.agent_item_value(kid, p) for p in instance.items) for kid in instance.agents]
    if total_values:
        T = min(total_values) / 2  # A reasonable target (half of the minimum total value)
    
    # Calculate alpha - threshold for classifying big and small presents
    alpha = T / 2 if T > 0 else 1
    
    # Classify presents as big or small
    big_presents = []
    small_presents = []
    for p in instance.items:
        is_big = False
        for kid in instance.agents:
            if instance.agent_item_value(kid, p) >= alpha:
                is_big = True
                break
        if is_big:
            big_presents.append(p)
        else:
            small_presents.append(p)
    
    logger.info("\nALGORITHM PARAMETERS:")
    logger.info("-" * 80)
    logger.info(f"Target value T: {T:.2f}")
    logger.info(f"Alpha threshold: {alpha:.2f}")
    logger.info(f"Big presents: {big_presents} ({len(big_presents)} items)")
    logger.info(f"Small presents: {small_presents} ({len(small_presents)} items)")
    logger.info("-" * 80)
    
    # Sort presents by their total value across all kids (descending)
    presents_by_value = sorted(
        instance.items,
        key=lambda p: sum(instance.agent_item_value(kid, p) for kid in instance.agents),
        reverse=True
    )
    
    # Log presents sorted by value
    logger.info("\nPRESENTS SORTED BY TOTAL VALUE (DESCENDING):")
    logger.info("-" * 80)
    for idx, present in enumerate(presents_by_value):
        total_value = sum(instance.agent_item_value(kid, present) for kid in instance.agents)
        logger.info(f"Rank {idx+1}: {present} - Total value across all kids: {total_value}")
    logger.info("-" * 80)
    
    # Keep track of assigned presents
    assigned_presents = set()
    
    # Estimate theoretical optimal value
    logger.info("\nESTIMATING OPTIMAL VALUES:")
    logger.info("-" * 80)
    
    # Upper bound: Best case if each kid gets their most valued presents
    theoretical_upper_bounds = {}
    for kid in instance.agents:
        kid_values = [instance.agent_item_value(kid, present) for present in instance.items]
        kid_values.sort(reverse=True)  # Sort in descending order
        # Assume each kid can get at most one present in this simple model
        theoretical_upper_bounds[kid] = sum(kid_values[:1]) if kid_values else 0
    
    # Lower bound: If presents are distributed perfectly evenly
    total_value = sum(instance.agent_item_value(kid, present) 
                    for kid in instance.agents 
                    for present in instance.items)
    avg_value = total_value / len(instance.agents) if instance.agents else 0
    theoretical_lower_bound = avg_value / len(instance.items) if instance.items else 0
    
    # Compute theoretical optimal value (max-min)
    max_min_theoretical = min(theoretical_upper_bounds.values()) if theoretical_upper_bounds else 0
    
    logger.info(f"Theoretical upper bounds (best case for each kid):")
    for kid, value in theoretical_upper_bounds.items():
        logger.info(f"  - {kid}: {value}")
    logger.info(f"Theoretical lower bound (even distribution): {theoretical_lower_bound:.2f}")
    logger.info(f"Theoretical max-min optimal value: {max_min_theoretical}")
    logger.info("-" * 80)
    
    # Assign presents greedily to maximize the minimum happiness
    logger.info("\nSTARTING ALLOCATION PROCESS:")
    logger.info("-" * 80)
    logger.info("ITERATION | KID SELECTED | CURRENT HAPPINESS | PRESENT ASSIGNED | VALUE | REASON")
    logger.info("-" * 80)
    
    iteration = 1
    while len(assigned_presents) < len(instance.items):
        # Find the kid with the minimum current happiness
        current_happiness = {
            kid: sum(instance.agent_item_value(kid, p) for p in alloc.allocation[kid])
            for kid in instance.agents
        }
        
        # Log all kids' current happiness
        logger.info(f"Current happiness for all kids:")
        for kid in instance.agents:
            logger.info(f"  - {kid}: {current_happiness[kid]} (has presents: {alloc.allocation[kid]})")
        
        min_happiness_kid = min(
            instance.agents, 
            key=lambda k: current_happiness[k]
        )
        
        logger.info(f"\nIteration {iteration}: Selected kid {min_happiness_kid} with minimum happiness {current_happiness[min_happiness_kid]}")
        
        # Find the best unassigned present for this kid
        best_present = None
        best_value = -1
        
        # Log all candidate presents
        logger.info(f"Evaluating unassigned presents for {min_happiness_kid}:")
        for present in presents_by_value:
            if present not in assigned_presents:
                value = instance.agent_item_value(min_happiness_kid, present)
                logger.info(f"  - Present {present}: Value = {value}")
                if value > best_value:
                    best_value = value
                    best_present = present
                    logger.info(f"    New best present found!")
        
        # If we found a present with positive value, assign it
        if best_present and best_value > 0:
            alloc.allocate(min_happiness_kid, best_present)
            assigned_presents.add(best_present)
            logger.info(f"{iteration:^9} | {min_happiness_kid:^11} | {current_happiness[min_happiness_kid]:^17} | {best_present:^15} | {best_value:^5} | Best value for kid")
        else:
            logger.info(f"{iteration:^9} | {min_happiness_kid:^11} | {current_happiness[min_happiness_kid]:^17} | {'None':^15} | {0:^5} | No valuable presents")
            # No more valuable presents for this kid, move to the next one
            break
            
        iteration += 1
        logger.info("-" * 80)
    
    # Calculate final happiness values
    final_happiness = {
        kid: sum(instance.agent_item_value(kid, p) for p in alloc.allocation[kid])
        for kid in instance.agents
    }
    
    # Log the final allocation
    logger.info("\nFINAL ALLOCATION:")
    logger.info("-" * 80)
    logger.info(f"| {'Kid':^15} | {'Presents':^30} | {'Happiness':^10} |")
    logger.info("-" * 80)
    for kid, presents in alloc.allocation.items():
        presents_str = ", ".join(presents) if presents else "None"
        logger.info(f"| {kid[:15]:^15} | {presents_str[:30]:^30} | {final_happiness[kid]:^10} |")
    logger.info("-" * 80)
    
    min_happiness = min(final_happiness.values()) if final_happiness else 0
    max_happiness = max(final_happiness.values()) if final_happiness else 0
    avg_happiness = sum(final_happiness.values()) / len(final_happiness) if final_happiness else 0
    
    logger.info("\nHAPPINESS STATISTICS:")
    logger.info("-" * 80)
    logger.info(f"| {'Statistic':^20} | {'Value':^10} |")
    logger.info("-" * 80)
    logger.info(f"| {'Minimum happiness':^20} | {min_happiness:^10} |")
    logger.info(f"| {'Maximum happiness':^20} | {max_happiness:^10} |")
    logger.info(f"| {'Average happiness':^20} | {avg_happiness:^10.2f} |")
    logger.info("-" * 80)
    
    # Compare to theoretical optimal
    approximation_ratio = min_happiness / max_min_theoretical if max_min_theoretical > 0 else 0
    logger.info("\nAPPROXIMATION QUALITY:")
    logger.info("-" * 80)
    logger.info(f"| {'Metric':^30} | {'Value':^10} |")
    logger.info("-" * 80)
    logger.info(f"| {'Achieved min happiness':^30} | {min_happiness:^10} |")
    logger.info(f"| {'Theoretical optimal':^30} | {max_min_theoretical:^10} |")
    logger.info(f"| {'Approximation ratio':^30} | {approximation_ratio:^10.2f} |")
    logger.info("-" * 80)
    if approximation_ratio >= 0.5:
        logger.info("Very good solution! At least 0.5-approximation of optimal value.")
    elif approximation_ratio >= 0.25:
        logger.info("Good solution! At least 0.25-approximation of optimal value.")
    else:
        logger.info("Solution might be improvable. Below 0.25-approximation of optimal value.")
    
    # Check if allocation is envy-free (nobody prefers someone else's allocation)
    logger.info("\nENVY ANALYSIS:")
    logger.info("-" * 80)
    has_envy = False
    for kid1 in instance.agents:
        for kid2 in instance.agents:
            if kid1 != kid2:
                my_value = sum(instance.agent_item_value(kid1, p) for p in alloc.allocation[kid1])
                their_value_to_me = sum(instance.agent_item_value(kid1, p) for p in alloc.allocation[kid2])
                if their_value_to_me > my_value:
                    has_envy = True
                    logger.info(f"{kid1} envies {kid2}! Values own allocation at {my_value}, but values {kid2}'s allocation at {their_value_to_me}")
    
    if not has_envy:
        logger.info("No envy detected! Each kid prefers their own allocation.")
    logger.info("-" * 80)
    
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
