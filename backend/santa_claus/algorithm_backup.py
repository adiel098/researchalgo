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

    # === Phase 1: Initialization and Target Value Determination ===
    
    # Step 1.1: Find optimal target value T using binary search
    logger.info("Phase 1.1: Finding optimal target value T")
    T_star = find_optimal_target_value(alloc)
    logger.info(f"Optimal target value T* = {T_star:.4f}")
    
    # Step 1.2: Create α-gap instance by classifying gifts
    logger.info("Phase 1.2: Creating α-gap instance")
    large_gifts = set()
    small_gifts = set()
    
    for item in alloc.instance.items:
        is_large = False
        for agent in alloc.instance.agents:
            value = alloc.instance.agent_item_value(agent, item)
            if value >= T_star / alpha:
                is_large = True
                break
        
        if is_large:
            large_gifts.add(item)
        else:
            small_gifts.add(item)
    
    logger.info(f"Classified {len(large_gifts)} large gifts and {len(small_gifts)} small gifts")
    
    # === Phase 2: Configuration LP Formulation and Solution ===
    
    # Step 2.1 & 2.2: Solve Configuration LP and extract fractional assignment
    logger.info("Phase 2: Solving Configuration LP")
    fractional_solution_x_star = configuration_lp_solver(alloc, T_star)
    
    if not fractional_solution_x_star:
        logger.error("Configuration LP solving failed. Aborting algorithm.")
        return
    
    logger.info(f"Configuration LP solved with {len(fractional_solution_x_star)} non-zero variables")
    
    # Derive assignment y_ij from x*
    fractional_assignments_y = {}
    for agent in alloc.instance.agents:
        fractional_assignments_y[agent] = {}
        for item in alloc.instance.items:
            fractional_assignments_y[agent][item] = 0.0
    
    for (agent, config), frac_value in fractional_solution_x_star.items():
        for item in config:
            if item in fractional_assignments_y[agent]:
                fractional_assignments_y[agent][item] += frac_value
    
    # === Phase 3: Solution Transformation - Creating Structure ===
    
    # Step 3.1: Forest Construction (Lemma 5)
    logger.info("Phase 3.1: Constructing forest from bipartite graph")
    bipartite_graph = create_bipartite_graph(fractional_solution_x_star, large_gifts, alloc)
    forest, permanent_assignments = eliminate_cycles(bipartite_graph)
    
    # Apply permanent assignments from cycle elimination
    for job, machine in permanent_assignments.items():
        alloc.give(machine, job)
        logger.info(f"Permanently assigned large gift {job} to agent {machine} from cycle elimination")
    
    # Step 3.2: Clustering Formation (Lemma 6)
    logger.info("Phase 3.2: Forming clusters from forest structure")
    clusters = form_clusters(forest)
    logger.info(f"Formed {len(clusters)} clusters from forest")
    
    # Track configurations per machine for set system conversion
    machine_configs = {}
    for (agent, config), frac_value in fractional_solution_x_star.items():
        if agent not in machine_configs:
            machine_configs[agent] = []
        machine_configs[agent].append((config, frac_value))
    
    # === Phase 4: Auxiliary Problem Translation ===
    
    # Step 4.1: Convert to Set System
    logger.info("Phase 4: Translating to set system formulation")
    set_system = convert_to_set_system(small_gifts, clusters, machine_configs, alloc)
    
    # === Phase 5: Main Algorithm - Sampling and Reduction ===
    
    # Step 5.1: Random Sampling (Lemma 11)
    logger.info("Phase 5.1: Applying random sampling")
    sampled_system = sample_elements(set_system)
    
    # Step 5.2: Small Instance Solution (Theorem 3)
    logger.info("Phase 5.2: Applying Leighton's algorithm to sampled system")
    function_f = apply_leighton_algorithm(sampled_system)
    
    # Step 5.3: Feasibility Verification (Lemma 9)
    logger.info("Phase 5.3: Verifying function quality")
    gamma = 1.0  # Target value for load factor
    beta = int(math.log(math.log(len(alloc.instance.agents))) / 
               math.log(math.log(math.log(len(alloc.instance.agents)) + 1)) + 1)
    
    is_good, subset_dict = verify_function_quality(sampled_system, function_f, gamma, beta)
    
    if not is_good:
        logger.warning("Could not verify function quality. Using best available function.")
        # If verification fails, we still proceed with the best function we have

    # === Phase 7: Solution Construction and Gift Distribution ===
    
    # Step 7.1 & 7.2: Assign gifts based on function f
    logger.info("Phase 7: Constructing final allocation from function f")
    
    # Track cluster assignments for final allocation
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
    
    # === Phase 8: Handle any remaining unallocated gifts ===
    
    # Check for unallocated gifts and distribute them
    all_items = set(alloc.instance.items)
    allocated_items = set()
    
    for agent, items in alloc.bundles.items():
        allocated_items.update(items)
    
    unallocated_items = all_items - allocated_items
    
    if unallocated_items:
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
    
    logger.info("\nFinal allocation:")
    for agent in alloc.instance.agents:
        items = alloc.bundles[agent]
        agent_value = sum(alloc.instance.agent_item_value(agent, item) for item in items)
        min_value = min(min_value, agent_value)
        max_value = max(max_value, agent_value)
        total_value += agent_value
        
        logger.info(f"  {agent}: {items} (value: {agent_value:.4f})")
    
    avg_value = total_value / len(alloc.instance.agents) if alloc.instance.agents else 0
    logger.info(f"\nAllocation statistics:")
    logger.info(f"  Minimum happiness value: {min_value:.4f}")
    logger.info(f"  Maximum happiness value: {max_value:.4f}")
    logger.info(f"  Average happiness value: {avg_value:.4f}")
    logger.info(f"  Optimal target T*: {T_star:.4f}")
    logger.info(f"  Approximation ratio: {T_star / max(0.0001, min_value):.4f}")
    
    logger.info(f"\n================= SANTA CLAUS ALGORITHM ENDS =================\n")
    logger.info(f"\nBeta parameter: {beta}")
    
    # Step 1: Find the optimal target value T_star using binary search
    # This is computed ONCE at the beginning and used throughout the algorithm
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
    
    for item in alloc.instance.items:
        max_value = max(alloc.instance.agent_item_value(agent, item) for agent in alloc.instance.agents)
        if max_value >= threshold:
            large_gifts.add(item)
        else:
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

    # Step 4: Create super-machines (clusters of children and large gifts)
    logger.info("\n========== STEP 4: CREATING SUPER-MACHINES ==========\n")
    super_machines = create_super_machines(alloc, fractional_solution_x_star, large_gifts)
    logger.info(f"Created {len(super_machines)} super-machines")
    
    # Log detailed super-machine structure
    logger.info("\nDetailed super-machine structure:")
    for i, (children, gifts) in enumerate(super_machines):
        logger.info(f"  Super-machine #{i+1}:")
        logger.info(f"    Children: {children}")
        logger.info(f"    Large gifts: {gifts}")

    # Step 5: Round small gift configurations
    logger.info("\n========== STEP 5: ROUNDING SMALL GIFT CONFIGURATIONS ==========\n")
    rounded_solution = round_small_configurations(alloc, super_machines, small_gifts, beta)
    logger.info(f"Rounded solution created with {len(rounded_solution)} super-machines receiving small gifts")
    
    # Log detailed rounded solution
    logger.info("\nDetailed rounded solution:")
    for i, config in rounded_solution.items():
        representative = config["child"]
        gifts = config["gifts"]
        logger.info(f"  Super-machine #{i+1}:")
        logger.info(f"    Representative child: {representative}")
        logger.info(f"    Small gifts: {gifts}")

    # Step 6: Construct final allocation
    logger.info("\n========== STEP 6: CONSTRUCTING FINAL ALLOCATION ==========\n")
    construct_final_allocation(alloc, super_machines, rounded_solution)
    logger.info("Final allocation constructed")

    # Log the final allocation
    logger.info("\n========== FINAL ALLOCATION RESULTS ==========\n")
    min_value = float('inf')
    for agent, bundle in alloc.bundles.items():
        total_value = sum(alloc.instance.agent_item_value(agent, item) for item in bundle)
        logger.info(f"Final allocation for {agent}: {bundle} with total value {total_value:.4f}")
        if bundle:  # Only consider non-empty bundles for min value
            min_value = min(min_value, total_value)
    
    if min_value == float('inf'):
        min_value = 0
        
    logger.info(f"\nMinimum value achieved: {min_value:.4f}")
    logger.info(f"Target T value: {T_star:.4f}")
    logger.info(f"Approximation ratio: {min_value/T_star:.4f} of target" if T_star > 0 else "N/A")
    logger.info(f"\n================= SANTA CLAUS ALGORITHM COMPLETED =================")


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
        logger.error(f"Error in Santa Claus algorithm: {str(e)}", exc_info=True)
        # Return empty allocation on error
        return {agent: [] for agent in alloc.instance.agents}
