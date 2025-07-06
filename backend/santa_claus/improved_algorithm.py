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
from .core import AllocationBuilder
from .forest import create_bipartite_graph, eliminate_cycles, form_clusters
from .sampling import convert_to_set_system, sample_elements
from .leighton import apply_leighton_algorithm, verify_function_quality
from .clustering import construct_final_allocation

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


def find_optimal_target_value_improved(alloc: AllocationBuilder, max_configs: int = 2000) -> float:
    """
    Improved binary search to find the highest feasible target value T.
    
    :param alloc: The allocation builder containing the problem instance.
    :param max_configs: Maximum number of configurations to generate per agent.
    :return: The highest feasible target value T found. Returns 0.0 if no T > 0 is feasible.
    """
    logger.info("Starting binary search for optimal target value T.")
    
    # Initialize binary search bounds
    low_T = 0.0
    
    # Upper bound: maximum possible value any agent could get
    high_T = 0.0
    for agent in alloc.instance.agents:
        agent_max_value = sum(
            alloc.instance.agent_item_value(agent, item)
            for item in alloc.instance.items
        )
        high_T = max(high_T, agent_max_value)
    
    logger.info(f"Binary search bounds: [{low_T}, {high_T}]")
    
    # Set precision threshold based on high_T
    precision = high_T / 1000.0  # 0.1% of the max value
    
    # Binary search
    T_star = 0.0  # Best feasible T found so far
    while high_T - low_T > precision:
        mid_T = (low_T + high_T) / 2.0
        logger.info(f"Checking mid-point T = {mid_T:.4f}")
        
        is_feasible, _ = _check_config_lp_feasibility_improved(alloc, mid_T, max_configs=max_configs)
        
        if is_feasible:
            T_star = mid_T  # Update best feasible T
            low_T = mid_T   # Search in upper half
            logger.info(f"T = {mid_T:.4f} is FEASIBLE. New range: [{low_T:.4f}, {high_T:.4f}]")
        else:
            high_T = mid_T  # Search in lower half
            logger.info(f"T = {mid_T:.4f} is INFEASIBLE. New range: [{low_T:.4f}, {high_T:.4f}]")
    
    logger.info(f"Binary search completed. Best feasible T found: {T_star:.4f}")
    return T_star


def configuration_lp_solver_improved(alloc: AllocationBuilder, target_value: float, max_configs: int = 2000) -> Dict[Tuple[str, Tuple[str, ...]], float]:
    """
    Improved version of the Configuration LP solver.
    
    :param alloc: The allocation builder containing the problem instance.
    :param target_value: The target value T to use.
    :param max_configs: Maximum number of configurations to generate per agent.
    :return: A dictionary mapping (agent_name, configuration_tuple) to its fractional value (x_i,C).
    """
    # Check if target_value is feasible and get solution
    is_feasible, solution_dict = _check_config_lp_feasibility_improved(
        alloc, target_value, run_check=True, max_configs=max_configs
    )
    
    if not is_feasible or not solution_dict:
        logger.warning(f"No solution found for T={target_value:.4f}")
        return {}
    
    logger.info(f"Configuration LP solution found with {len(solution_dict)} non-zero variables")
    return solution_dict


def santa_claus_improved(alloc: AllocationBuilder, alpha: float = 3.0, max_configs: int = 2000) -> None:
    """
    Improved main algorithm for the Santa Claus Problem, optimized for better performance
    with larger problem instances.
    
    :param alloc: The allocation builder that tracks the allocation process.
    :param alpha: Parameter used for classifying gifts as large or small.
    :param max_configs: Maximum number of configurations to generate per agent.
    """
    logger.info(f"\n================= IMPROVED SANTA CLAUS ALGORITHM STARTS =================\n")
    logger.info(f"Items: {list(alloc.instance.items)}")
    logger.info(f"Agents: {list(alloc.instance.agents)}")
    logger.info(f"Using max_configs={max_configs} to limit configuration space")
    
    if not alloc.instance.items or not alloc.instance.agents:
        logger.warning("No items or agents. Aborting algorithm.")
        return

    # Calculate beta parameter for clustering
    beta = int(math.log(math.log(len(alloc.instance.agents))) / 
               math.log(math.log(math.log(len(alloc.instance.agents)) + 1)) + 1)
    logger.info(f"\nBeta parameter: {beta}")

    # === Phase 1: Initialization and Target Value Determination ===
    
    # Step 1: Find the optimal target value T_star using binary search
    logger.info("\n========== STEP 1: FINDING OPTIMAL TARGET VALUE T ==========\n")
    T_star = find_optimal_target_value_improved(alloc, max_configs=max_configs)
    if T_star <= 1e-9:  # Effectively zero
        logger.warning(f"Optimal target value T_star is {T_star:.4f}. No positive maximin value achievable. Allocating nothing.")
        return
    logger.info(f"Optimal T_star found: {T_star:.4f} (This value will be used throughout the algorithm)")

    # Step 2: Solve the Configuration LP for T_star
    logger.info("\n========== STEP 2: SOLVING CONFIGURATION LP ==========\n")
    fractional_solution_x_star = configuration_lp_solver_improved(alloc, T_star, max_configs=max_configs)
    if not fractional_solution_x_star:
        logger.warning(f"No solution for T_star={T_star:.4f}. No fallback allocation will be made.")
        return
    logger.info(f"Obtained fractional solution with {len(fractional_solution_x_star)} non-zero variables")
    
    # Step 3: Classify gifts as "large" or "small" based on T_star/alpha
    logger.info("\n========== STEP 3: CLASSIFYING GIFTS ==========\n")
    large_gifts = set()
    small_gifts = set()
    threshold = T_star / alpha
    logger.info(f"Classification threshold = T_star/alpha = {T_star:.4f}/{alpha} = {threshold:.4f}")
    
    for item in alloc.instance.items:
        item_is_large = False
        for agent in alloc.instance.agents:
            if alloc.instance.agent_item_value(agent, item) >= threshold:
                item_is_large = True
                large_gifts.add(item)
                break
                
        if not item_is_large:
            small_gifts.add(item)
    
    logger.info(f"Classified gifts: {len(large_gifts)} large, {len(small_gifts)} small")
    
    # Create fractional_assignments_y from x_star for forest construction
    fractional_assignments_y = {}
    for agent in alloc.instance.agents:
        fractional_assignments_y[agent] = {}
        for item in alloc.instance.items:
            fractional_assignments_y[agent][item] = 0.0
            
    for (agent, config), frac_value in fractional_solution_x_star.items():
        for item in config:
            if item in fractional_assignments_y[agent]:
                fractional_assignments_y[agent][item] += frac_value

    # === Phase 3: Forest Construction and Clustering ===
    
    # Step 3.1: Create bipartite graph from fractional solution
    logger.info("\n========== STEP 4: FOREST CONSTRUCTION AND CLUSTERING ==========\n")
    bipartite_graph = create_bipartite_graph(fractional_solution_x_star, large_gifts, alloc)
    logger.info(f"Created bipartite graph with {len(bipartite_graph.nodes)} nodes")
    
    # Step 3.2: Eliminate cycles to create a forest
    forest, permanent_assignments = eliminate_cycles(bipartite_graph)
    logger.info(f"Eliminated cycles and created forest with {len(forest.nodes)} nodes")
    logger.info(f"Made {len(permanent_assignments)} permanent assignments from cycle elimination")
    
    # Apply permanent assignments from cycle elimination
    for job, machine in permanent_assignments.items():
        alloc.give(machine, job)
    
    # Step 3.3: Form clusters with the property |J_i| = |M_i| - 1
    clusters = form_clusters(forest)
    logger.info(f"Formed {len(clusters)} clusters from forest")
    
    # Track configurations per machine for set system conversion
    machine_configs = {}
    for (agent, config), frac_value in fractional_solution_x_star.items():
        if agent not in machine_configs:
            machine_configs[agent] = []
        machine_configs[agent].append((config, frac_value))
    
    # === Phase 4: Set System Conversion and Sampling ===
    
    # Step 4.1: Convert to Set System
    logger.info("\n========== STEP 5: SET SYSTEM CONVERSION AND SAMPLING ==========\n")
    set_system = convert_to_set_system(small_gifts, clusters, machine_configs, alloc)
    logger.info(f"Created set system with {len(set_system.ground_set)} elements and {len(set_system.sets)} sets")
    
    # Step 4.2: Apply random sampling to reduce dimension
    sampled_system = sample_elements(set_system)
    logger.info(f"Created sampled system with {len(sampled_system.ground_set)} elements")
    
    # === Phase 5: Leighton's Algorithm and Function Verification ===
    
    # Step 5.1: Apply Leighton's algorithm to the sampled system
    logger.info("\n========== STEP 6: LEIGHTON'S ALGORITHM AND FUNCTION VERIFICATION ==========\n")
    function_f = apply_leighton_algorithm(sampled_system)
    logger.info(f"Applied Leighton's algorithm, produced function f with {len(function_f)} values")
    
    # Step 5.2: Verify quality of function f
    gamma = 1.0  # Target value for load factor
    is_good, subset_dict = verify_function_quality(sampled_system, function_f, gamma, beta)
    
    if is_good:
        logger.info(f"Function f verification successful with gamma={gamma}")
    else:
        logger.warning("Function f verification failed. Using best available function.")
    
    # === Phase 6: Solution Construction and Final Allocation ===
    
    # Step 6: Construct final allocation
    logger.info("\n========== STEP 7: FINAL ALLOCATION CONSTRUCTION ==========\n")
    
    # Create super_machines structure from clusters for compatibility with construct_final_allocation
    # This is a simplified approach that will work for basic cases
    super_machines = []
    for machines, jobs in clusters:
        super_machines.append((list(machines), list(jobs)))
    
    # Create rounded_solution dict from function_f and subset_dict
    rounded_solution = {}
    if function_f and subset_dict:
        for i, subset in subset_dict.items():
            if subset:  # Only add non-empty subsets
                rounded_solution[i] = subset
    
    # Call with only the required 3 parameters
    construct_final_allocation(alloc, super_machines, rounded_solution)
    
    logger.info("\n================= SANTA CLAUS ALGORITHM COMPLETED =================\n")
    logger.info(f"Final allocation: {alloc.allocation}")


def santa_claus_improved_wrapper(alloc: AllocationBuilder, max_configs: int = 2000) -> Dict[str, List[str]]:
    """
    Wrapper function for the improved Santa Claus algorithm to ensure consistent interface.
    
    :param alloc: The allocation builder to use.
    :param max_configs: Maximum number of configurations to generate per agent.
    :return: A dictionary mapping each agent (kid) to their allocated items (presents).
    """
    santa_claus_improved(alloc, max_configs=max_configs)
    return alloc.allocation
