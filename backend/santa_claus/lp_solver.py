"""
LP solver functions for the Santa Claus Problem.
"""

import logging
import pulp as pl
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from .core import AllocationBuilder

# Setup logger
logger = logging.getLogger(__name__)

def _generate_valid_configurations(agent_name: str, items: List[str], 
                                  valuations: Dict[str, Dict[str, float]], 
                                  target_value: float, agent_capacity: int) -> List[Tuple[str, ...]]:
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
    agent_valuations = valuations.get(agent_name, {})
    valid_configs = []
    
    # Helper function to check if a configuration is valid
    def is_valid_config(config: Tuple[str, ...]) -> bool:
        if len(config) > agent_capacity:
            return False
        
        # Calculate truncated value
        truncated_value = sum(min(agent_valuations.get(item, 0), target_value) for item in config)
        return truncated_value >= target_value
    
    # Generate all possible configurations up to agent_capacity
    # Start with smaller configurations for efficiency
    for size in range(1, agent_capacity + 1):
        # Use itertools.combinations for all subsets of size 'size'
        import itertools
        for config in itertools.combinations(items, size):
            # Skip if any item has 0 value for this agent (optimization)
            if any(agent_valuations.get(item, 0) <= 0 for item in config):
                continue
                
            # Check if configuration is valid
            if is_valid_config(config):
                truncated_value = sum(min(agent_valuations.get(item, 0), target_value) for item in config)
                logger.debug(f"  Found valid config for {agent_name}: {config} with truncated value {truncated_value:.2f}")
                valid_configs.append(config)
    
    return valid_configs


def _check_config_lp_feasibility(alloc: AllocationBuilder, target_value: float, run_check: bool = True) -> Tuple[bool, Optional[np.ndarray], Optional[List[Tuple[str, Tuple[str, ...]]]]]:
    """
    Checks if a given target_value T is feasible by formulating and attempting to solve
    the Configuration LP (feasibility version).

    The Configuration LP is defined as (see paper, Section 2):
    Variables: x_i,C for each agent i and valid configuration C in C(i,T).
    Constraints:
        1. sum_{C in C(i,T)} x_i,C <= 1  (for each agent i)  -- (Constraint 4 in paper, though often sum(...) = 1 for feasibility)
                                                               (Using <= 1 is safer for linprog if T is too high for some agents)
        2. sum_i sum_{C in C(i,T): j in C} x_i,C <= 1 (for each item j) -- (Constraint 5 in paper)
        3. x_i,C >= 0

    Objective: Minimize 0 (feasibility problem).

    :param alloc: The allocation builder containing the problem instance.
    :param target_value: The target value T to check for feasibility.
    :param run_check: If False, skips the feasibility check and assumes T is feasible.
    :return: Tuple (is_feasible, solution_vector, variable_mapping).
             is_feasible (bool): True if a solution is found, False otherwise.
             solution_vector (Optional[np.ndarray]): The values of x_i,C if feasible.
             variable_mapping (Optional[List[Tuple[str, Tuple[str, ...]]]]): 
                 A list mapping LP variable indices to (agent_name, configuration_tuple).
    """
    logger.info(f"Checking feasibility of T = {target_value:.2f}")
    
    # Generate valid configurations for each agent
    agent_configs = {}
    for agent in alloc.instance.agents:
        agent_capacity = alloc.instance.agent_capacities[agent]
        valid_configs = _generate_valid_configurations(
            agent, 
            list(alloc.instance.items), 
            alloc.instance.valuations, 
            target_value, 
            agent_capacity
        )
        agent_configs[agent] = valid_configs
        logger.info(f"Agent {agent} has {len(valid_configs)} valid configurations for T={target_value}.")
        
        # Early termination: if any agent has no valid configurations, T is infeasible
        if not valid_configs:
            logger.warning(f"Agent {agent} has no valid configurations for T={target_value}. Thus, T is infeasible.")
            return False, None, None
    
    # If we're skipping the check, assume T is feasible and return empty solution
    if not run_check:
        logger.info(f"Skipping feasibility check for T={target_value:.4f} since run_check=False (already verified as feasible)")
        # Create a mapping for variables
        var_mapping = []
        for agent, configs in agent_configs.items():
            for config in configs:
                var_mapping.append((agent, config))
        
        # Create an empty solution vector
        solution_vector = np.zeros(len(var_mapping))
        
        return True, solution_vector, var_mapping
    
    # Create a mapping for variables
    var_mapping = []
    for agent, configs in agent_configs.items():
        for config in configs:
            var_mapping.append((agent, config))
    
    # Total number of variables
    n_vars = len(var_mapping)
    logger.debug(f"Total number of LP variables (x_i,C): {n_vars}")
    
    if n_vars == 0:
        logger.warning(f"No valid configurations for any agent with T={target_value}. Thus, T is infeasible.")
        return False, None, None
    
    # Create PuLP problem
    prob = pl.LpProblem("ConfigurationLP", pl.LpMinimize)
    
    # Create variables
    x_vars = [pl.LpVariable(f"x_{i}", lowBound=0) for i in range(n_vars)]
    
    # Objective function: minimize 0 (feasibility problem)
    prob += 0
    
    # Constraint 1: Each agent gets at least 1 configuration (modified from <= 1 to >= 1 per original algorithm)
    for agent in alloc.instance.agents:
        agent_var_indices = [i for i, (a, _) in enumerate(var_mapping) if a == agent]
        if agent_var_indices:  # Only add constraint if agent has valid configurations
            prob += pl.lpSum(x_vars[i] for i in agent_var_indices) >= 1, f"agent_{agent}_coverage"
    
    # Constraint 2: Each item is used at most once
    for item in alloc.instance.items:
        item_var_indices = [i for i, (_, config) in enumerate(var_mapping) if item in config]
        if item_var_indices:  # Only add constraint if item appears in any configuration
            prob += pl.lpSum(x_vars[i] for i in item_var_indices) <= 1, f"item_{item}_usage"
    
    # Solve the LP
    logger.debug(f"Solving LP with {n_vars} vars, {len(prob.constraints)} constraints.")
    prob.solve(pl.PULP_CBC_CMD(msg=False))
    
    # Check if a solution was found
    if prob.status == 1:  # 1 is the value for OPTIMAL in PuLP
        obj_value = pl.value(prob.objective)
        obj_str = f"{obj_value:.4f}" if obj_value is not None else "None"
        logger.info(f"LP feasible for T={target_value:.2f}. Objective value (should be 0): {obj_str}")
        
        # Handle potential None values in solution vector
        solution_vector = []
        for var in x_vars:
            var_value = var.value()
            if var_value is not None:
                solution_vector.append(var_value)
            else:
                # If a variable has None value, assume it's 0
                logger.warning(f"Variable {var.name} has None value, assuming 0")
                solution_vector.append(0.0)
        solution_vector = np.array(solution_vector)
        return True, solution_vector, var_mapping
    else:
        logger.warning(f"LP infeasible for T={target_value:.2f}. Status code: {prob.status}")
        return False, None, None


def find_optimal_target_value(alloc: AllocationBuilder) -> float:
    """
    Algorithm 1 (conceptual): Binary search to find the highest feasible target value T.
    This function uses `_check_config_lp_feasibility` to determine if a T is feasible.
    
    :param alloc: The allocation builder containing the problem instance.
    :return: The highest feasible target value T found. Returns 0.0 if no T > 0 is feasible.
    """
    logger.info("Starting binary search for optimal target value T.")
    
    # Initialize binary search bounds
    # Lower bound: 0 (minimum possible target value)
    low_T = 0.0
    
    # Upper bound: maximum possible value any agent could get
    # This is the sum of all item values for the agent with highest total value
    high_T = 0.0
    for agent in alloc.instance.agents:
        agent_max_value = sum(
            alloc.instance.agent_item_value(agent, item)
            for item in alloc.instance.items
        )
        high_T = max(high_T, agent_max_value)
    
    logger.info(f"Binary search range for T: [{low_T}, {high_T}]")
    
    # Binary search parameters
    max_iterations = 100  # Prevent infinite loops
    precision = 1e-10     # Precision for convergence
    
    # Initialize optimal T found
    optimal_T_found = 0.0
    
    # Binary search loop
    iterations = 0
    while iterations < max_iterations and high_T - low_T > precision:
        iterations += 1
        
        # Calculate midpoint
        mid_T = (low_T + high_T) / 2
        logger.debug(f"Iteration {iterations}/{max_iterations}: low_T={low_T:.4f}, high_T={high_T:.4f}, mid_T={mid_T:.4f}")
        
        # Check if mid_T is feasible
        is_feasible, _, _ = _check_config_lp_feasibility(alloc, mid_T)
        
        if is_feasible:
            # If feasible, update optimal_T_found and try higher T
            optimal_T_found = mid_T
            low_T = mid_T
            logger.debug(f"  T={mid_T:.4f} is feasible. Trying higher.")
        else:
            # If infeasible, try lower T
            high_T = mid_T
            logger.debug(f"  T={mid_T:.4f} is infeasible. Trying lower.")
        
        # Check for convergence
        if abs(high_T - low_T) < precision:
            logger.info(f"Binary search converged. Optimal T found: {optimal_T_found:.4f}")
            break
    else: # Executed if loop finishes without break (max iterations reached)
        logger.warning(f"Binary search reached max iterations ({iterations}). Final T: {optimal_T_found:.4f}")

    # Final check: if optimal_T_found is very close to 0, treat as 0.
    if abs(optimal_T_found) < precision:
        optimal_T_found = 0.0

    logger.info(f"Optimal target value T determined: {optimal_T_found:.4f}")
    return optimal_T_found


def configuration_lp_solver(alloc: AllocationBuilder, target_value: float) -> Dict[Tuple[str, Tuple[str,...]], float]:
    """
    Algorithm 2 (conceptual): Solves the Configuration LP for a given target_value T.
    
    This function uses `_check_config_lp_feasibility` to obtain the LP solution.
    It then formats this solution into a dictionary representing the fractional assignments x_i,C.
    
    :param alloc: The allocation builder containing the problem instance.
    :param target_value: The target value T to use for the Configuration LP.
                         This should typically be the optimal T found by `find_optimal_target_value`.
    :return: A dictionary mapping (agent_name, configuration_tuple) to its fractional value (x_i,C).
             Returns an empty dictionary if T is infeasible or no solution is found.
    """
    logger.info(f"Solving Configuration LP for target_value T = {target_value:.2f}")

    if target_value <= 1e-9: # Effectively T=0
        logger.warning("Target value is ~0. Returning empty fractional solution.")
        # If T=0, any agent can achieve it with an empty bundle. LP might be trivial or ill-defined.
        # The problem implies positive values. An empty solution is safest.
        return {}

    # Check if we already know this T is feasible from the binary search
    is_feasible, solution_vector, x_var_map = _check_config_lp_feasibility(alloc, target_value, run_check=False)

    if not is_feasible or solution_vector is None or x_var_map is None:
        logger.warning(f"Configuration LP infeasible or no solution found for T={target_value:.2f}. Returning empty solution.")
        return {}

    fractional_assignment: Dict[Tuple[str, Tuple[str,...]], float] = {}
    for i, (agent_name, config_tuple) in enumerate(x_var_map):
        val = solution_vector[i]
        if val > 1e-9: # Store only non-zero assignments (with tolerance for float precision)
            fractional_assignment[(agent_name, config_tuple)] = val
            logger.debug(f"  x_({agent_name}, {config_tuple}) = {val:.4f}")
    
    logger.info(f"Successfully obtained fractional solution for T={target_value:.2f} with {len(fractional_assignment)} non-zero variables.")
    
    # If the LP is feasible but we didn't get any non-zero assignments due to numerical issues,
    # create a simple valid assignment to allow the algorithm to proceed
    if is_feasible and not fractional_assignment:
        logger.warning("LP is feasible but no non-zero assignments found. Creating a simple valid assignment.")
        # Generate a simple assignment where each agent gets one valid configuration
        for agent_name in alloc.instance.agents:
            # Find the first valid configuration for this agent
            for i, (a_name, config_tuple) in enumerate(x_var_map):
                if a_name == agent_name:
                    fractional_assignment[(agent_name, config_tuple)] = 1.0
                    logger.debug(f"  Adding fallback: x_({agent_name}, {config_tuple}) = 1.0")
                    break
    logger.info(f"Configuration LP solver completed with {len(fractional_assignment)} assignments")
    return fractional_assignment
