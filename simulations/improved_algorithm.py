"""
Improved algorithm implementation for the Santa Claus Problem.

This module implements an improved version of the approximation algorithm
for the restricted assignment case of the Santa Claus problem with better
performance for larger problem instances.
"""

import logging
import random
import math
import networkx as nx
import time
from functools import lru_cache
from typing import Dict, List, Set, Tuple, Any, Optional

# Adjusted imports to work from simulations directory
from backend.santa_claus.core import AllocationBuilder
from backend.santa_claus.forest import create_bipartite_graph, eliminate_cycles, form_clusters
from backend.santa_claus.sampling import convert_to_set_system, sample_elements
from backend.santa_claus.leighton import apply_leighton_algorithm, verify_function_quality
from backend.santa_claus.clustering import construct_final_allocation

# Setup logger
logger = logging.getLogger(__name__)


def _generate_valid_configurations_improved(agent_name: str, items: List[str], 
                                  valuations: Dict[str, Dict[str, float]], 
                                  target_value: float, agent_capacity: int,
                                  max_configs: int = 2000) -> List[Tuple[str, ...]]:
    """
    Improved version of the configuration generator that limits the number of configurations
    and prioritizes more valuable configurations.
    
    :param agent_name: The name of the agent.
    :param items: A list of all available item names.
    :param valuations: The full valuation matrix (agents -> items -> value).
    :param target_value: The target value T.
    :param agent_capacity: The maximum number of items the agent can receive.
    :param max_configs: Maximum number of configurations to generate per agent.
    :return: A list of tuples, where each tuple represents a valid configuration.
    """
    agent_valuations = valuations.get(agent_name, {})
    valid_configs = []
    
    # Filter out items with 0 value for this agent (optimization)
    valuable_items = [item for item in items if agent_valuations.get(item, 0) > 0]
    
    # Sort items by their value to this agent (descending)
    valuable_items.sort(key=lambda item: agent_valuations.get(item, 0), reverse=True)
    
    # Helper function to check if a configuration is valid with memoization
    @lru_cache(maxsize=10000)
    def config_truncated_value(config: Tuple[str, ...]) -> float:
        # Calculate truncated value
        return sum(min(agent_valuations.get(item, 0), target_value) for item in config)
    
    # Generate configurations greedily
    def generate_configs_greedy(remaining_items, current_config=(), current_value=0.0):
        # Base cases
        if len(current_config) >= agent_capacity:
            return
        
        if current_value >= target_value:
            valid_configs.append(current_config)
            if len(valid_configs) >= max_configs:
                return
            
        # Try adding each remaining item
        for i, item in enumerate(remaining_items):
            new_config = current_config + (item,)
            item_value = agent_valuations.get(item, 0)
            # Optimization: only consider this item if it could potentially help reach target
            if item_value > 0:
                new_value = current_value + min(item_value, target_value)
                generate_configs_greedy(remaining_items[i+1:], new_config, new_value)
                
                if len(valid_configs) >= max_configs:
                    return
    
    # Start configuration generation
    start_time = time.time()
    generate_configs_greedy(tuple(valuable_items))
    
    # If we need more configurations and we haven't reached max_configs,
    # try combinations approach as fallback
    if len(valid_configs) < min(20, max_configs) and len(valuable_items) <= 15:
        logger.debug(f"Greedy approach for {agent_name} found only {len(valid_configs)} configs, trying combinations.")
        
        import itertools
        # Start with larger sizes which are more likely to meet target value
        for size in range(min(agent_capacity, len(valuable_items)), 0, -1):
            # Use itertools.combinations for subsets of size 'size'
            for config in itertools.combinations(valuable_items, size):
                if len(valid_configs) >= max_configs:
                    break
                    
                # Check if configuration is valid
                if config_truncated_value(config) >= target_value:
                    valid_configs.append(config)
    
    end_time = time.time()
    logger.debug(f"Generated {len(valid_configs)} valid configs for {agent_name} in {end_time - start_time:.3f}s")
    
    return valid_configs[:max_configs]


def _check_config_lp_feasibility_improved(alloc: AllocationBuilder, target_value: float, run_check: bool = True, max_configs: int = 2000) -> Tuple[bool, Optional[Dict[Tuple[str, Tuple[str, ...]], float]]]:
    """
    Improved version of the Configuration LP feasibility check.

    :param alloc: The allocation builder containing the problem instance.
    :param target_value: The target value T to check for feasibility.
    :param run_check: If False, skips the feasibility check and assumes T is feasible.
    :param max_configs: Maximum number of configurations to generate per agent.
    :return: Tuple (is_feasible, solution_dict).
             is_feasible (bool): True if a solution is found, False otherwise.
             solution_dict (Optional[Dict]): The values of x_i,C if feasible.
    """
    logger.info(f"Checking feasibility of T = {target_value:.2f}")
    
    # Generate valid configurations for each agent
    agent_configs = {}
    for agent in alloc.instance.agents:
        agent_capacity = alloc.instance.agent_capacities[agent]
        valid_configs = _generate_valid_configurations_improved(
            agent, 
            list(alloc.instance.items), 
            alloc.instance.valuations, 
            target_value, 
            agent_capacity,
            max_configs
        )
        agent_configs[agent] = valid_configs
        logger.info(f"Agent {agent} has {len(valid_configs)} valid configurations for T={target_value}.")
        
        # Early termination: if any agent has no valid configurations, T is infeasible
        if not valid_configs:
            logger.warning(f"Agent {agent} has no valid configurations for T={target_value}. Thus, T is infeasible.")
            return False, None
    
    # If we're skipping the check, assume T is feasible and return empty solution
    if not run_check:
        logger.info(f"Skipping feasibility check for T={target_value:.4f} since run_check=False (already verified as feasible)")
        return True, {}
    
    # Create PuLP problem
    import pulp as pl
    prob = pl.LpProblem("ConfigurationLP", pl.LpMinimize)
    
    # Create variables
    x_vars = {}
    for agent, configs in agent_configs.items():
        for config in configs:
            x_vars[(agent, config)] = pl.LpVariable(f"x_{agent}_{hash(config)}", lowBound=0)
    
    # Objective function: minimize 0 (feasibility problem)
    prob += 0
    
    # Constraint 1: Each agent gets at least 1 configuration
    for agent in alloc.instance.agents:
        if agent_configs[agent]:  # Only add constraint if agent has valid configurations
            prob += pl.lpSum(x_vars[(agent, config)] for config in agent_configs[agent]) >= 1, f"agent_{agent}_coverage"
    
    # Constraint 2: Each item is used at most once
    for item in alloc.instance.items:
        # Find all configurations that include this item
        item_vars = []
        for agent, configs in agent_configs.items():
            for config in configs:
                if item in config:
                    item_vars.append(x_vars[(agent, config)])
        
        if item_vars:  # Only add constraint if item appears in any configuration
            prob += pl.lpSum(item_vars) <= 1, f"item_{item}_usage"
    
    # Solve the LP
    logger.debug(f"Solving LP with {len(x_vars)} vars, {len(prob.constraints)} constraints.")
    
    # Use solver with time limit
    solver = pl.PULP_CBC_CMD(msg=False, timeLimit=30)  # 30-second time limit
    prob.solve(solver)
    
    # Check if a solution was found
    if prob.status == 1:  # 1 is the value for OPTIMAL in PuLP
        obj_value = pl.value(prob.objective)
        obj_str = f"{obj_value:.4f}" if obj_value is not None else "None"
        logger.info(f"LP feasible for T={target_value:.2f}. Objective value (should be 0): {obj_str}")
        
        # Extract solution
        solution_dict = {}
        for var_key, var in x_vars.items():
            var_value = var.value()
            if var_value is not None and var_value > 1e-6:  # Filter out near-zero values
                solution_dict[var_key] = var_value
                
        return True, solution_dict
    else:
        logger.warning(f"LP infeasible for T={target_value:.2f}. Status code: {prob.status}")
        return False, None


def configuration_lp_solver_improved(alloc: AllocationBuilder, target_value: float, max_configs: int = 2000):
    """
    Improved version of the Configuration LP solver.
    
    :param alloc: The allocation builder containing the problem instance.
    :param target_value: The target value T to use.
    :param max_configs: Maximum number of configurations to generate per agent.
    :return: A dictionary mapping (agent_name, configuration_tuple) to its fractional value (x_i,C).
    """
    logger.info(f"Running improved configuration LP solver with T={target_value:.4f}")
    
    # Run feasibility check
    is_feasible, solution_dict = _check_config_lp_feasibility_improved(alloc, target_value, max_configs=max_configs)
    
    if not is_feasible:
        logger.warning(f"T={target_value:.4f} is infeasible. Returning empty solution.")
        return {}
    
    return solution_dict


def santa_claus_improved(alloc: AllocationBuilder, alpha: float = 3.0, max_configs: int = 2000):
    """
    Improved main algorithm for the Santa Claus Problem, optimized for better performance
    with larger problem instances.
    
    :param alloc: The allocation builder that tracks the allocation process.
    :param alpha: Parameter used for classifying gifts as large or small.
    :param max_configs: Maximum number of configurations to generate per agent.
    """
    start_time = time.time()
    logger.info(f"Starting improved Santa Claus algorithm with {len(alloc.instance.agents)} agents, "
                f"{len(alloc.instance.items)} items, and alpha={alpha}.")
    
    # Step 1: Binary search for the maximum feasible T
    # Note: We'll use a better initial guess based on the problem instance
    max_values = []
    for agent in alloc.instance.agents:
        max_val = max((alloc.instance.agent_item_value(agent, item) for item in alloc.instance.items), default=0)
        if max_val > 0:
            max_values.append(max_val)
    
    if not max_values:
        logger.warning("No positive values found in the instance. Returning empty allocation.")
        return
    
    # Set search bounds based on instance characteristics
    t_min, t_max = 0, max(max_values)
    initial_guess = min(max_values) / 2  # Start with half of the minimum max value as a reasonable guess
    
    logger.info(f"Binary search for T in range [0, {t_max:.4f}], initial guess: {initial_guess:.4f}")
    
    # Improved binary search
    t_values_checked = set()  # Track already checked T values to avoid redundant work
    max_iterations = 10  # Limit number of iterations to avoid excessive computation
    
    current_t = initial_guess
    best_t = 0
    best_solution = {}
    
    for iteration in range(max_iterations):
        logger.info(f"Binary search iteration {iteration+1}, checking T={current_t:.4f}")
        
        # Avoid checking the same T value multiple times
        if current_t in t_values_checked:
            logger.info(f"T={current_t:.4f} already checked, adjusting.")
            # Adjust T slightly to avoid duplicate work
            current_t = (current_t + t_max) / 2 if best_t < current_t else (current_t + t_min) / 2
            continue
            
        t_values_checked.add(current_t)
        
        # Check if current_t is feasible
        is_feasible, solution = _check_config_lp_feasibility_improved(alloc, current_t, max_configs=max_configs)
        
        if is_feasible:
            logger.info(f"T={current_t:.4f} is feasible.")
            best_t = current_t
            best_solution = solution
            t_min = current_t
            # Try a higher T value
            current_t = (current_t + t_max) / 2
        else:
            logger.info(f"T={current_t:.4f} is infeasible.")
            t_max = current_t
            # Try a lower T value
            current_t = (current_t + t_min) / 2
        
        # Early termination conditions
        if t_max - t_min < 0.01 or (iteration > 3 and best_t > 0):
            logger.info(f"Binary search converged or found good enough T={best_t:.4f}.")
            break
    
    T = best_t
    solution_dict = best_solution
    
    logger.info(f"Found maximum feasible T={T:.4f}")
    
    if not solution_dict:
        logger.warning("No feasible solution found. Returning empty allocation.")
        return
    
    # Step 2: Classification of items as large or small
    large_items = set()
    small_items = set()
    for item in alloc.instance.items:
        # Find the maximum valuation any agent has for this item
        max_val = max((alloc.instance.agent_item_value(agent, item) 
                       for agent in alloc.instance.agents), default=0)
        if max_val >= T / alpha:
            large_items.add(item)
        else:
            small_items.add(item)
    
    logger.info(f"Classified {len(large_items)} large items and {len(small_items)} small items.")
    
    # Step 3: Convert to a set system
    set_system = convert_to_set_system(alloc.instance, solution_dict)
    
    # Step 4: Use sampling technique to build a conflict-free allocation
    # We sample elements according to their fractional allocation in solution_dict
    sampled_items = sample_elements(set_system, solution_dict, max_iterations=5)
    
    logger.info(f"Sampled allocation has {sum(1 for agent_items in sampled_items.values() if agent_items)} "
                f"agents with at least one item.")
    
    # Step 5: Create bipartite graph and eliminate cycles
    G = create_bipartite_graph(alloc.instance, large_items, small_items, sampled_items, T)
    
    # Eliminate cycles in the graph to prepare for clustering
    acyclic_graph = eliminate_cycles(G)
    
    # Step 6: Form clusters and construct final allocation
    clusters = form_clusters(acyclic_graph, alloc.instance.agents, T, alpha)
    
    logger.info(f"Formed {len(clusters)} clusters.")
    
    # Step 7: Apply the clustering algorithm to maximize minimum happiness
    # This step heavily depends on the clustering.py module
    allocation = construct_final_allocation(clusters, alloc.instance, large_items, small_items, T, alpha)
    
    # Step 8: Apply Leighton's algorithm to improve solution quality (if available)
    try:
        allocation = apply_leighton_algorithm(allocation, alloc.instance, T)
    except Exception as e:
        logger.warning(f"Error applying Leighton's algorithm: {e}")
    
    # Apply the final allocation to the AllocationBuilder
    for agent, items in allocation.items():
        for item in items:
            alloc.assign_item(agent, item)
    
    # Calculate achieved min happiness
    min_happiness = float('inf')
    for agent in alloc.instance.agents:
        agent_happiness = sum(alloc.instance.agent_item_value(agent, item) 
                              for item in alloc.allocation.get(agent, []))
        min_happiness = min(min_happiness, agent_happiness)
    
    # Prevent inf if no allocations were made
    if min_happiness == float('inf'):
        min_happiness = 0
    
    end_time = time.time()
    logger.info(f"Santa Claus algorithm completed in {end_time - start_time:.2f}s. "
                f"Target T={T:.4f}, achieved min happiness={min_happiness:.4f}.")
    
    # Verify final solution quality
    verify_function_quality(allocation, alloc.instance, T)
    
    
def santa_claus_improved_wrapper(alloc: AllocationBuilder, max_configs: int = 2000):
    """
    Wrapper function for the improved Santa Claus algorithm to ensure consistent interface.
    
    :param alloc: The allocation builder to use.
    :param max_configs: Maximum number of configurations to generate per agent.
    :return: A dictionary mapping each agent (kid) to their allocated items (presents).
    """
    # Run the algorithm with default parameters
    santa_claus_improved(alloc, alpha=3.0, max_configs=max_configs)
    
    # Return the allocation
    return alloc.allocation
