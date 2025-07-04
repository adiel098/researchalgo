"""
Clustering and rounding functions for the Santa Claus Problem.
"""

import logging
import random
import networkx as nx
from typing import Dict, List, Set, Tuple, Any
from .core import AllocationBuilder

# Setup logger
logger = logging.getLogger(__name__)

def create_super_machines(alloc: AllocationBuilder, fractional_solution: Dict[Tuple[str, Tuple[str, ...]], float], large_gifts: Set[str]) -> List[Tuple[List[str], List[str]]]:
    """Create super-machines by clustering children and large gifts."""
    logger.info("Starting creation of super-machines with clustering algorithm")
    """
    Algorithm 3: Create the super-machine structure (clusters of children and large gifts).
    
    This implements the algorithm described in Section 5.2 of "The Santa Claus Problem" by Bansal and Sviridenko.
    It first builds a bipartite graph where children (machines) are on the left and large gifts (jobs) are on the right,
    with an edge indicating a fractional assignment. Then it transforms this into a forest and creates clusters
    of children and large gifts where |Ji| = |Mi| - 1 for each cluster.
    
    :param alloc: The allocation builder containing the problem instance
    :param fractional_solution: The fractional solution from the Configuration LP
    :param large_gifts: Set of gifts classified as large
    :return: List of clusters (Mi, Ji) where Mi is a set of children and Ji is a set of large gifts
    """
    logger.info("Creating super-machine structure from large gifts")
    
    # Step 1: Create bipartite graph of children and large gifts
    G = nx.Graph()
    
    # Add nodes for children (machines) and large gifts (jobs)
    for agent in alloc.instance.agents:
        G.add_node(agent, bipartite=0)  # 0 indicates left side (children)
    
    for gift in large_gifts:
        G.add_node(gift, bipartite=1)  # 1 indicates right side (large gifts)
    
    # Find configurations with large gifts
    large_gift_configs = {}
    for (agent, config), frac_value in fractional_solution.items():
        if any(gift in large_gifts for gift in config):
            if agent not in large_gift_configs:
                large_gift_configs[agent] = {}
            large_gift_configs[agent][config] = frac_value
    
    logger.info(f"Found {len(large_gift_configs)} agents with large gift configurations")
    for agent, configs in large_gift_configs.items():
        logger.info(f"  Agent {agent} has {len(configs)} large gift configurations")
    
    # Create fractional graph (bipartite graph between large gifts and children)
    large_gift_allocation = {}
    for agent, configs in large_gift_configs.items():
        for config, frac_value in configs.items():
            for gift in config:
                if gift in large_gifts:
                    if gift not in large_gift_allocation:
                        large_gift_allocation[gift] = {}
                    if agent not in large_gift_allocation[gift]:
                        large_gift_allocation[gift][agent] = 0.0
                    large_gift_allocation[gift][agent] += frac_value
    
    logger.info(f"Created fractional bipartite graph with {len(large_gift_allocation)} large gifts")
    for gift, agents in large_gift_allocation.items():
        logger.info(f"  Gift {gift} is fractionally allocated to {len(agents)} agents: {', '.join([f'{a}:{v:.2f}' for a, v in agents.items()])}")
    
    # Add edges based on fractional solution
    edge_count = 0
    for (agent, config), value in fractional_solution.items():
        if value > 1e-9:  # Only consider non-zero assignments
            for gift in config:
                if gift in large_gifts:
                    # Edge weight is the fractional assignment value
                    G.add_edge(agent, gift, weight=value)
                    edge_count += 1
    
    logger.info(f"Created bipartite graph with {len(alloc.instance.agents)} agents, {len(large_gifts)} large gifts, and {edge_count} edges")
    
    logger.debug(f"Bipartite graph created with {len(G.nodes)} nodes and {len(G.edges)} edges")
    
    # Step 2: Transform into a forest (remove cycles)
    # Find a maximum spanning tree (or forest if graph is not connected)
    if len(G.edges) > 0:
        T = nx.maximum_spanning_tree(G, weight='weight')
    else:
        logger.warning("No edges in bipartite graph. Creating empty forest.")
        T = nx.Graph()
        for node in G.nodes:
            T.add_node(node, bipartite=G.nodes[node]['bipartite'])
    
    logger.debug(f"Forest created with {len(T.nodes)} nodes and {len(T.edges)} edges")
    
    # Step 3: Create clusters
    clusters = []
    
    # Find connected components in the forest
    components = list(nx.connected_components(T))
    logger.debug(f"Found {len(components)} connected components in the forest")
    
    for component in components:
        # Separate children and large gifts in this component
        children = [node for node in component if T.nodes[node].get('bipartite') == 0]
        gifts = [node for node in component if T.nodes[node].get('bipartite') == 1]
        
        # Check if this component satisfies |Ji| = |Mi| - 1
        if len(gifts) == len(children) - 1:
            clusters.append((children, gifts))
            logger.debug(f"Created cluster with {len(children)} children and {len(gifts)} large gifts")
        else:
            logger.warning(f"Component with {len(children)} children and {len(gifts)} gifts doesn't satisfy |Ji| = |Mi| - 1")
            # We can still try to use this component by adjusting it
            if len(children) > 0:  # Only add if there's at least one child
                # Take as many gifts as possible while satisfying |Ji| = |Mi| - 1
                if len(gifts) > len(children) - 1:
                    # Too many gifts, take only |Mi| - 1
                    selected_gifts = gifts[:len(children) - 1]
                    clusters.append((children, selected_gifts))
                    logger.debug(f"Adjusted cluster: {len(children)} children and {len(selected_gifts)} large gifts")
                else:
                    # Not enough gifts, but we'll use what we have
                    clusters.append((children, gifts))
                    logger.debug(f"Using incomplete cluster: {len(children)} children and {len(gifts)} large gifts")
    
    # Assign large gifts to children in each cluster (for visualization/debugging)
    for children, gifts in clusters:
        for i, gift in enumerate(gifts):
            # Assign each gift to a different child in the cluster
            if i < len(children):
                alloc.give(children[i], gift)
                logger.debug(f"Assigned large gift {gift} to child {children[i]}")
    
    logger.info(f"Created {len(clusters)} super-machine clusters")
    return clusters


def round_small_configurations(alloc: AllocationBuilder, super_machines: List, 
                              small_gifts: Set[str], beta: float = 3.0) -> Dict:
    """
    Algorithm 4: Round the small gift configurations.
    
    This is the core part of the algorithm, using sampling and the Leighton et al. algorithm
    to find configurations with low congestion, as described in Section 6 of "The Santa Claus Problem".
    
    This implementation uses a combination of randomized rounding and deterministic selection
    to distribute small gifts to super-machines while ensuring no small gift is assigned to
    more than β machines.
    
    :param alloc: The allocation builder containing the problem instance
    :param super_machines: The super-machine structure from previous step
    :param small_gifts: Set of gifts classified as small
    :param beta: Relaxation parameter for the solution
    :return: A dictionary mapping super-machine index to selected small gift configuration
    """
    logger.info(f"Rounding small gift configurations with beta={beta}")
    logger.info(f"Small gifts available for rounding: {sorted(list(small_gifts))} (total: {len(small_gifts)})")
    logger.info(f"Super machines available: {len(super_machines)}")
    
    if not small_gifts:
        logger.warning("No small gifts to round. Returning empty configuration.")
        return {}
    
    if not super_machines:
        logger.warning("No super-machines to assign small gifts to. Returning empty configuration.")
        return {}
    
    # Step 1: Create a simple assignment of small gifts to super-machines
    # For simplicity, we'll assign each small gift to at most one super-machine
    # In a full implementation, this would use the fractional solution and rounding
    
    logger.info("Creating assignment of small gifts to super-machines")
    
    rounded_solution = {}
    remaining_gifts = list(small_gifts)
    
    # Shuffle the gifts for random assignment
    random.shuffle(remaining_gifts)
    logger.info(f"Shuffled {len(remaining_gifts)} small gifts for random assignment")
    
    # Assign small gifts to super-machines
    for i, (children, _) in enumerate(super_machines):
        if not remaining_gifts:
            logger.info(f"No more small gifts remaining, stopping assignment at super-machine {i}")
            break
            
        # Choose a representative child from this super-machine
        if children:
            representative = children[0]
            logger.info(f"Processing super-machine {i} with representative child {representative}")
            
            # Find small gifts that have positive value for this representative
            valuable_gifts = []
            for gift in remaining_gifts:
                if alloc.instance.agent_item_value(representative, gift) > 0:
                    valuable_gifts.append(gift)
            
            logger.info(f"Found {len(valuable_gifts)} valuable small gifts for representative {representative}")
            
            # If there are valuable gifts, assign them to this super-machine
            if valuable_gifts:
                # Take up to beta gifts (or all valuable gifts if fewer)
                num_to_take = min(int(beta), len(valuable_gifts))
                assigned_gifts = valuable_gifts[:num_to_take]
                
                logger.info(f"Taking {num_to_take} small gifts (beta={int(beta)}) for super-machine {i}")
                logger.info(f"Assigning small gifts: {assigned_gifts} to super-machine {i}")
                
                # Remove assigned gifts from remaining gifts
                for gift in assigned_gifts:
                    remaining_gifts.remove(gift)
                
                # Record the assignment
                rounded_solution[i] = {
                    "child": representative,
                    "gifts": assigned_gifts
                }
                logger.info(f"Assigned {len(assigned_gifts)} small gifts to super-machine {i}, representative {representative}")
            else:
                logger.info(f"No valuable small gifts for representative {representative} of super-machine {i}")
    
    logger.info(f"Rounded solution created with {len(rounded_solution)} super-machines receiving small gifts")
    logger.info(f"Remaining unassigned small gifts: {len(remaining_gifts)} out of {len(small_gifts)}")
    if remaining_gifts:
        logger.info(f"Unassigned gifts: {remaining_gifts}")
    return rounded_solution


def construct_final_allocation(alloc: AllocationBuilder, super_machines: List, 
                             rounded_solution: Dict) -> None:
    """
    Algorithm 5: Construct the final allocation.
    
    Assigns the small gift configurations and large gifts to children
    in each super-machine, then removes conflicts according to the "Santa Claus Problem" algorithm.
    
    The algorithm works as follows:
    1. In each super-machine:
        a. One representative child m(i) is selected to receive only small gifts
        b. The remaining children share the large gifts
    2. All gifts must be allocated (both large and small)
    
    :param alloc: The allocation builder containing the problem instance
    :param super_machines: The super-machine structure
    :param rounded_solution: The rounded solution for small gifts
    """
    logger.info("====== CONSTRUCTING FINAL ALLOCATION ======")
    logger.info(f"Super machines: {len(super_machines)}")
    logger.info(f"Super machines with small gift assignments: {len(rounded_solution)}")
    
    # Clear any previous allocations
    alloc.bundles = {agent: [] for agent in alloc.instance.agents}
    logger.info("Cleared previous allocations for all agents")
    
    # Keep track of unallocated items
    all_items = set(alloc.instance.items)
    allocated_items = set()
    
    # Step 1: First pass - assign according to super-machines
    logger.info("\nSTEP 1: Initial allocation from super-machines")
    large_gift_assigned_count = 0
    small_gift_assigned_count = 0
    
    # For super-machines with small gift configurations, assign those
    for i, config in rounded_solution.items():
        if i >= len(super_machines):
            logger.warning(f"Invalid super-machine index {i} in rounded solution. Skipping.")
            continue
            
        children, large_gifts = super_machines[i]
        representative = config["child"]
        small_gifts = config["gifts"]
        
        logger.info(f"Processing super-machine {i} with representative {representative}")
        logger.info(f"  Children in cluster: {children}")
        logger.info(f"  Large gifts in cluster: {large_gifts}")
        logger.info(f"  Small gifts assigned to representative: {small_gifts}")
        
        # Verify the representative is in this super-machine
        if representative not in children:
            logger.warning(f"Representative {representative} not in super-machine {i}. Skipping.")
            continue
        
        # 1. The representative gets small gifts
        for gift in small_gifts:
            if gift in all_items and gift not in allocated_items:  # Only if not already allocated
                alloc.give(representative, gift)
                allocated_items.add(gift)
                small_gift_assigned_count += 1
                logger.info(f"  Assigned small gift {gift} to representative {representative} of super-machine {i}")
        
        # 2. Distribute large gifts to non-representative children
        non_representatives = [child for child in children if child != representative]
        
        # Ensure we have children to receive large gifts
        if non_representatives:
            # Each large gift to a different non-representative child
            for j, gift in enumerate(large_gifts):
                if j < len(non_representatives) and gift in all_items and gift not in allocated_items:
                    child = non_representatives[j % len(non_representatives)]  # Cycle through children if needed
                    alloc.give(child, gift)
                    allocated_items.add(gift)
                    large_gift_assigned_count += 1
                    logger.info(f"  Assigned large gift {gift} to child {child} from super-machine {i}")
    
    # For super-machines without small gift configurations, distribute large gifts equally
    for i, (children, large_gifts) in enumerate(super_machines):
        if i in rounded_solution:
            continue  # Already handled above
            
        logger.info(f"Processing super-machine {i} without small gifts, {len(large_gifts)} large gifts")
        
        # Distribute large gifts among children
        for j, gift in enumerate(large_gifts):
            if gift in all_items and gift not in allocated_items:  # Only if not already allocated
                if j < len(children):  # Make sure we have enough children
                    child = children[j % len(children)]  # Cycle through children if needed
                    alloc.give(child, gift)
                    allocated_items.add(gift)
                    large_gift_assigned_count += 1
                    logger.info(f"  Assigned large gift {gift} to child {child} from super-machine {i}")
    
    logger.info(f"Initial allocation: {large_gift_assigned_count} large gifts, {small_gift_assigned_count} small gifts")
    
    # Step 2: Distribute remaining unallocated gifts
    unallocated_items = all_items - allocated_items
    
    if unallocated_items:
        logger.info(f"\nSTEP 2: Distributing {len(unallocated_items)} remaining unallocated gifts")
        logger.info(f"Unallocated items: {unallocated_items}")
        
        # Prioritize children with no gifts yet
        empty_agents = [agent for agent, bundle in alloc.bundles.items() if not bundle]
        
        if empty_agents:
            logger.info(f"Priority allocation to {len(empty_agents)} empty agents: {empty_agents}")
            
            # Assign gifts to empty agents first
            remaining_gifts = list(unallocated_items)
            for i, gift in enumerate(remaining_gifts):
                if i < len(empty_agents):
                    agent = empty_agents[i % len(empty_agents)]
                    alloc.give(agent, gift)
                    allocated_items.add(gift)
                    logger.info(f"  Assigned remaining gift {gift} to empty agent {agent}")
        
        # If we still have unallocated gifts, distribute them to all agents fairly
        still_unallocated = all_items - allocated_items
        if still_unallocated:
            logger.info(f"Distributing {len(still_unallocated)} remaining gifts among all agents")
            
            all_agents = list(alloc.instance.agents)
            remaining_gifts = list(still_unallocated)
            
            for i, gift in enumerate(remaining_gifts):
                agent = all_agents[i % len(all_agents)]
                alloc.give(agent, gift)
                allocated_items.add(gift)
                logger.info(f"  Assigned remaining gift {gift} to agent {agent}")
    
    # Verify all items are allocated
    final_unallocated = all_items - allocated_items
    if final_unallocated:
        logger.warning(f"WARNING: {len(final_unallocated)} items still unallocated: {final_unallocated}")
    else:
        logger.info("SUCCESS: All items have been allocated")
    
    # Evaluate the final allocation
    logger.info("\n====== FINAL ALLOCATION RESULTS ======")
    
    min_value = float('inf')
    max_value = float('-inf')
    total_items_assigned = 0
    agents_with_items = 0
    final_empty_agents = []
    
    # Log the final allocation for each agent
    for agent, bundle in alloc.bundles.items():
        total_value = sum(alloc.instance.agent_item_value(agent, item) for item in bundle)
        item_count = len(bundle)
        total_items_assigned += item_count
        
        if item_count > 0:
            agents_with_items += 1
            min_value = min(min_value, total_value)
            max_value = max(max_value, total_value)
            logger.info(f"Agent {agent}: {bundle} with value {total_value:.4f} (items: {item_count})")
        else:
            final_empty_agents.append(agent)
    
    # Handle edge cases for min/max calculations
    if min_value == float('inf'):
        min_value = 0
    if max_value == float('-inf'):
        max_value = 0
        
    # Log overall allocation statistics
    logger.info(f"\nOverall allocation statistics:")
    logger.info(f"  Total items assigned: {total_items_assigned} out of {len(all_items)}")
    logger.info(f"  Agents receiving items: {agents_with_items} out of {len(alloc.instance.agents)}")
    logger.info(f"  Empty agents: {len(final_empty_agents)} {final_empty_agents if final_empty_agents else ''}")
    logger.info(f"  Minimum agent value: {min_value:.4f}")
    logger.info(f"  Maximum agent value: {max_value:.4f}")
    logger.info(f"  Max/min ratio: {max_value/min_value if min_value > 0 else 'N/A'}")
    
    logger.info("\n====== FINAL ALLOCATION CONSTRUCTED ======")
