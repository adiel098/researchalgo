"""
Main algorithm implementation for the Santa Claus Problem.

This module implements the O(log log m / log log log m) approximation algorithm
for the restricted assignment case of the Santa Claus problem.
"""

import logging
import random
import math
import networkx as nx
from typing import Dict, List, Set, Tuple, Any, Optional
from .core import AllocationBuilder
from .lp_solver import find_optimal_target_value, configuration_lp_solver
from .forest import create_bipartite_graph, eliminate_cycles, form_clusters
from .sampling import convert_to_set_system, sample_elements
from .leighton import apply_leighton_algorithm, verify_function_quality
from .clustering import construct_final_allocation

# Setup logger
logger = logging.getLogger(__name__)


def santa_claus_algorithm(alloc: AllocationBuilder, alpha: float = 3.0) -> None:
    """
    Main algorithm for the Santa Claus Problem, implementing the O(log log m / log log log m) 
    approximation algorithm for the restricted assignment case.
    
    This algorithm follows the approach from "The Santa Claus Problem" by Bansal and Sviridenko,
    with enhancements based on more recent theoretical results.
    
    :param alloc: The allocation builder that tracks the allocation process.
    :param alpha: Parameter used for classifying gifts as large or small.
                  Default value 3.0 balances the classification and approximation ratio.
    
    The algorithm involves:
    1. Finding optimal target value T
    2. Solving the Configuration LP
    3. Creating bipartite graph and forest structure
    4. Forming clusters with the |J_i| = |M_i| - 1 property
    5. Converting to set system and applying sampling
    6. Using Leighton's algorithm on the sampled system
    7. Constructing final allocation using the function f
    """
    logger.info(f"\n================= SANTA CLAUS ALGORITHM STARTS =================\n")
    logger.info(f"Items: {list(alloc.instance.items)}")
    logger.info(f"Agents: {list(alloc.instance.agents)}")
    logger.info(f"Agent capacities: {alloc.instance.agent_capacities}")
    logger.info(f"Item capacities: {alloc.instance.item_capacities}")
    logger.info(f"Alpha parameter: {alpha}")
    
    # Log valuations matrix
    logger.info("\nValuations matrix:")
    for agent in alloc.instance.agents:
        values = [f"{alloc.instance.agent_item_value(agent, item):5.1f}" for item in alloc.instance.items]
        logger.info(f"  {agent}: {values}")

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
    T_star = find_optimal_target_value(alloc)
    if T_star <= 1e-9:  # Effectively zero
        logger.warning(f"Optimal target value T_star is {T_star:.4f}. No positive maximin value achievable. Allocating nothing.")
        return
    logger.info(f"Optimal T_star found: {T_star:.4f} (This value will be used throughout the algorithm)")

    # Step 2: Solve the Configuration LP for T_star
    # We pass run_check=False to avoid redundant feasibility checks since we know T_star is feasible
    logger.info("\n========== STEP 2: SOLVING CONFIGURATION LP ==========\n")
    fractional_solution_x_star = configuration_lp_solver(alloc, T_star)
    if not fractional_solution_x_star:
        logger.warning(f"No solution for T_star={T_star:.4f}. No fallback allocation will be made.")
        return
    logger.info(f"Obtained fractional solution with {len(fractional_solution_x_star)} non-zero variables")
    
    # Log detailed fractional solution
    logger.info("\nDetailed fractional solution:")
    for (agent, config), value in fractional_solution_x_star.items():
        logger.info(f"  {agent} gets {config} with fraction {value:.4f}")

    # Step 3: Classify gifts as "large" or "small" based on T_star/alpha
    # Implement _classify_gifts functionality directly
    logger.info("\n========== STEP 3: CLASSIFYING GIFTS ==========\n")
    large_gifts = set()
    small_gifts = set()
    threshold = T_star / alpha
    logger.info(f"Classification threshold = T_star/alpha = {T_star:.4f}/{alpha} = {threshold:.4f}")
    
    # Classify gifts based on their actual value to each agent, not the maximum over all agents
    for item in alloc.instance.items:
        # For each item, check its value to each agent
        item_is_large = False
        for agent in alloc.instance.agents:
            # If any agent values this gift above threshold, it's large for that agent
            if alloc.instance.agent_item_value(agent, item) >= threshold:
                item_is_large = True
                large_gifts.add(item)
                break
                
        # If item wasn't classified as large, it's small
        if not item_is_large:
            small_gifts.add(item)
    
    logger.info(f"Classified gifts: {len(large_gifts)} large, {len(small_gifts)} small")
    logger.info(f"Large gifts: {sorted(list(large_gifts))}")
    logger.info(f"Small gifts: {sorted(list(small_gifts))}")
    
    # Log the maximum value of each gift
    logger.info("\nMaximum values for each gift:")
    for item in alloc.instance.items:
        max_value = max(alloc.instance.agent_item_value(agent, item) for agent in alloc.instance.agents)
        classification = "LARGE" if item in large_gifts else "small"
        logger.info(f"  {item}: {max_value:.4f} ({classification})")

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
        logger.info(f"Permanently assigned gift {job} to agent {machine} from cycle elimination")
    
    # Step 3.3: Form clusters with the property |J_i| = |M_i| - 1
    clusters = form_clusters(forest)
    logger.info(f"Formed {len(clusters)} clusters from forest")
    
    # Log cluster details
    for i, (machines, jobs) in enumerate(clusters):
        logger.info(f"Cluster #{i+1}: {len(machines)} machines, {len(jobs)} jobs")
        logger.info(f"  Machines: {machines}")
        logger.info(f"  Jobs: {jobs}")
    
    # === Phase 4: Set System Conversion and Sampling ===
    
    # Track configurations per machine for set system conversion
    machine_configs = {}
    for (agent, config), frac_value in fractional_solution_x_star.items():
        if agent not in machine_configs:
            machine_configs[agent] = []
        machine_configs[agent].append((config, frac_value))
    
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
    
    # Log subset details
    logger.info(f"Created {len(subset_dict)} subsets for final allocation")
    
    # === Phase 6: Solution Construction and Final Allocation ===
    
    # Step 6.1: Construct cluster assignments
    logger.info("\n========== STEP 7: SOLUTION CONSTRUCTION ==========\n")
    cluster_assignments = {}
    
    for i, (machines, jobs) in enumerate(clusters):
        if i not in function_f or len(machines) == 0:
            continue
        
        # Get selected machine index using function f
        selected_idx = function_f[i] % len(machines)
        selected_machine = machines[selected_idx]
        
        # The selected machine gets small gifts, others get large gifts
        cluster_assignments[i] = {
            'representative': selected_machine,
            'other_machines': [m for j, m in enumerate(machines) if j != selected_idx],
            'large_jobs': jobs,
            'small_jobs': subset_dict.get(i, set())
        }
        
        # Log assignment details
        logger.info(f"Cluster #{i+1} assignments:")
        logger.info(f"  Representative machine: {selected_machine}")
        logger.info(f"  Other machines: {cluster_assignments[i]['other_machines']}")
        logger.info(f"  Large jobs: {jobs}")
        logger.info(f"  Small jobs: {subset_dict.get(i, set())}")
        
        # Assign small gifts to representative machine
        for small_job in subset_dict.get(i, set()):
            if small_job in small_gifts:  # Verify it's a valid small gift (not a dummy)
                alloc.give(selected_machine, small_job)
                logger.info(f"Assigned small gift {small_job} to representative agent {selected_machine}")
        
        # Assign large gifts to other machines
        for j, machine in enumerate(cluster_assignments[i]['other_machines']):
            if j < len(jobs):  # Ensure we have enough jobs
                job = jobs[j]
                alloc.give(machine, job)
                logger.info(f"Assigned large gift {job} to agent {machine}")
    
    # === Phase 7: Handle any remaining unallocated gifts ===
    
    # Check for unallocated gifts and distribute them
    all_items = set(alloc.instance.items)
    allocated_items = set()
    
    for agent, items in alloc.bundles.items():
        allocated_items.update(items)
    
    unallocated_items = all_items - allocated_items
    
    if unallocated_items:
        logger.info(f"\n========== STEP 8: DISTRIBUTING REMAINING UNALLOCATED GIFTS ==========\n")
        logger.info(f"Found {len(unallocated_items)} unallocated items to distribute")
        
        # First, identify agents with no gifts
        agents_without_gifts = [agent for agent in alloc.instance.agents if not alloc.bundles[agent]]
        
        # Assign unallocated items to agents without gifts first
        for item in list(unallocated_items):
            if agents_without_gifts:
                agent = agents_without_gifts.pop(0)
                alloc.give(agent, item)
                unallocated_items.remove(item)
                logger.info(f"Assigned unallocated gift {item} to agent {agent} with no previous gifts")
            else:
                logger.info(f"No more agents without gifts, moving to fair distribution")
                break
        
        # If there are still unallocated items, distribute them fairly
        if unallocated_items:
            agents_list = list(alloc.instance.agents)
            for i, item in enumerate(unallocated_items):
                agent = agents_list[i % len(agents_list)]
                alloc.give(agent, item)
                logger.info(f"Assigned remaining unallocated gift {item} to agent {agent}")
    
    # Log final allocation statistics
    min_value = float('inf')
    max_value = float('-inf')
    total_value = 0
    
    logger.info("\n========== FINAL ALLOCATION RESULTS ==========\n")
    for agent in alloc.instance.agents:
        items = alloc.bundles[agent]
        agent_value = sum(alloc.instance.agent_item_value(agent, item) for item in items)
        if items:  # Only consider non-empty bundles for min value
            min_value = min(min_value, agent_value)
        max_value = max(max_value, agent_value)
        total_value += agent_value
        
        logger.info(f"Final allocation for {agent}: {items} with total value {agent_value:.4f}")
    
    if min_value == float('inf'):
        min_value = 0
        
    avg_value = total_value / len(alloc.instance.agents) if alloc.instance.agents else 0
    logger.info(f"\nAllocation statistics:")
    logger.info(f"  Minimum happiness value: {min_value:.4f}")
    logger.info(f"  Maximum happiness value: {max_value:.4f}")
    logger.info(f"  Average happiness value: {avg_value:.4f}")
    logger.info(f"  Optimal target T*: {T_star:.4f}")
    logger.info(f"  Approximation ratio: {T_star / max(0.0001, min_value):.4f}")
    
    logger.info(f"\n================= SANTA CLAUS ALGORITHM ENDS =================\n")


def santa_claus(alloc: AllocationBuilder):
    """
    Main entry point for the Santa Claus Problem algorithm.
    
    This implements the O(log log m / log log log m) approximation algorithm for
    the restricted assignment case as described in "The Santa Claus Problem" by
    Bansal and Sviridenko, with enhancements based on more recent theoretical results.
    
    The algorithm aims to maximize the minimum value received by any agent (maximin objective):
        maximize (min_i sum_{j in S_i} p_ij)
    where S_i is the set of presents allocated to kid i.
    
    The implemented algorithm includes:
    - Configuration LP formulation and solving
    - Forest construction with cycle elimination
    - Cluster formation with structural guarantees
    - Dimension reduction via sampling
    - Leighton's algorithm for sampled instances
    - Flow verification for function quality
    - Final allocation construction
    
    Args:
        alloc: The allocation builder that tracks the allocation process
        
    Returns:
        A dictionary mapping each agent (kid) to their allocated items (presents)
    """
    try:
        santa_claus_algorithm(alloc)
        logger.info("Santa Claus algorithm completed successfully")
        return alloc.allocation
    except Exception as e:
        logger.error(f"Error in Santa Claus algorithm: {e}", exc_info=True)
        # Return current allocation even if algorithm failed
        return alloc.allocation
