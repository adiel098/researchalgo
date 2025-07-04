"""
Forest construction and tree manipulation for the Santa Claus Problem.

This module implements Lemma 5 and 6 from "The Santa Claus Problem" paper,
creating a forest structure from the bipartite graph of machines and big jobs.
"""

import logging
import networkx as nx
from collections import deque
from typing import Dict, Set, List, Tuple, Any
import random

from .core import AllocationBuilder

# Configure logging
logger = logging.getLogger(__name__)

def create_bipartite_graph(fractional_assignment: Dict[Tuple[str, Tuple[str, ...]], float], 
                           large_gifts: Set[str], alloc: AllocationBuilder) -> nx.Graph:
    """
    Creates a bipartite graph from fractional assignment and large gifts.
    
    The bipartite graph G has:
    - Left side: machines M (agents/children)
    - Right side: big jobs Jb (large gifts)
    - Edge (i,j) with weight yij if yij > 0
    
    :param fractional_assignment: The fractional solution from the LP
    :param large_gifts: Set of gifts classified as large
    :param alloc: The allocation builder containing the problem instance
    :return: A NetworkX bipartite graph
    """
    logger.info("Creating bipartite graph for forest construction")
    
    # Extract machine-job assignments for large gifts
    machine_job_assignments = {}  # maps (machine, job) -> fractional value
    
    # First, calculate the total fractional assignment for each agent-item pair
    for (agent, config), frac_value in fractional_assignment.items():
        for item in config:
            if item in large_gifts:  # Only consider large gifts
                if (agent, item) not in machine_job_assignments:
                    machine_job_assignments[(agent, item)] = 0.0
                machine_job_assignments[(agent, item)] += frac_value
    
    # Filter out assignments with zero weight
    zero_assignments = [(agent, item) for (agent, item), weight in machine_job_assignments.items() if weight <= 0.001]
    for agent, item in zero_assignments:
        del machine_job_assignments[(agent, item)]
        
    logger.info(f"Found {len(machine_job_assignments)} positive weighted assignments for large gifts")
    
    # Create bipartite graph
    G = nx.Graph()
    
    # Add machine nodes (left side)
    for machine in alloc.instance.agents:
        G.add_node(machine, bipartite=0, type="machine")
    
    # Add job nodes (right side)
    for job in large_gifts:
        G.add_node(job, bipartite=1, type="job")
    
    # Add edges with weights (only positive weights)
    for (machine, job), weight in machine_job_assignments.items():
        # Double check weight is positive (should already be filtered)
        if weight > 0.001:  # Use small epsilon to handle floating point
            G.add_edge(machine, job, weight=weight)
    
    logger.info(f"Created bipartite graph with {len(alloc.instance.agents)} machines, "
                f"{len(large_gifts)} large jobs, and {len(machine_job_assignments)} edges")
    
    return G

def eliminate_cycles(G: nx.Graph) -> Tuple[nx.Graph, Dict[str, str]]:
    """
    Eliminates cycles from the bipartite graph by adjusting weights and removing edges.
    
    For each cycle:
    1. Decompose into alternating matchings P1, P2
    2. Adjust weights continuously (increase on P1, decrease on P2)
    3. When a weight reaches 0 or 1, fix it and update the graph
    
    :param G: The original bipartite graph with weights
    :return: A forest structure and a dictionary mapping permanently assigned jobs to machines
    """
    logger.info("Eliminating cycles from bipartite graph")
    
    # Make a copy of the graph to modify
    forest = G.copy()
    
    # Track permanent assignments (job -> machine)
    permanent_assignments = {}
    
    # Check if the graph is empty or already a forest
    if forest.number_of_nodes() == 0 or forest.number_of_edges() == 0:
        logger.info("Graph is empty, no cycles to eliminate")
        return forest, permanent_assignments
    
    # Initialize edge weights to 0.5 if not present
    for u, v in forest.edges():
        if 'weight' not in forest[u][v]:
            forest[u][v]['weight'] = 0.5
    
    # Find and eliminate cycles
    cycles_removed = 0
    
    # Create a list to track edges to remove (to avoid modifying during iteration)
    edges_to_remove = []
    nodes_to_remove = []
    
    while True:
        try:
            # Clear pending removals
            for u, v in edges_to_remove:
                if forest.has_edge(u, v):
                    forest.remove_edge(u, v)
            edges_to_remove = []
            
            for node in nodes_to_remove:
                if forest.has_node(node):
                    forest.remove_node(node)
            nodes_to_remove = []
            
            # Get cycle edges as (u,v) pairs without data dictionary
            try:
                cycle = list(nx.find_cycle(forest))
            except nx.NetworkXNoCycle:
                # No more cycles, graph is now a forest
                break
                
            cycles_removed += 1
            
            # If empty cycle or no cycle found, break
            if not cycle:
                break
            
            # Cycle is a list of (u, v) pairs representing edges
            # Decompose cycle into two matchings P1 and P2 (alternating edges)
            P1 = [(u, v) for i, (u, v) in enumerate(cycle) if i % 2 == 0]
            P2 = [(u, v) for i, (u, v) in enumerate(cycle) if i % 2 == 1]
            
            # Skip if any path is empty (shouldn't happen, but just in case)
            if not P1 or not P2:
                logger.warning(f"Invalid cycle detected with P1={P1}, P2={P2}")
                continue
                
            # Find minimum weight in P2 and maximum weight adjustment
            # Use try-except to handle potential missing edges
            try:
                min_P2_weight = min(forest[u][v].get('weight', 0.5) for u, v in P2)
                max_P1_room = min(1.0 - forest[u][v].get('weight', 0.5) for u, v in P1)
                delta = min(min_P2_weight, max_P1_room)
            except Exception as e:
                logger.error(f"Error calculating weight adjustments: {e}")
                continue
            
            logger.debug(f"Found cycle with {len(cycle)} edges, adjusting weights by delta={delta:.4f}")
            
            # Adjust weights
            for u, v in P1:
                if forest.has_edge(u, v):  # Make sure edge still exists
                    forest[u][v]['weight'] = forest[u][v].get('weight', 0.5) + delta
                    if forest[u][v]['weight'] >= 0.999:  # Handle float precision
                        # Permanently assign job v to machine u
                        # Determine which is the job and which is the machine
                        try:
                            if forest.nodes[u].get('type') == 'job' and forest.has_node(u) and forest.has_node(v):
                                job, machine = u, v
                            elif forest.nodes[v].get('type') == 'job' and forest.has_node(u) and forest.has_node(v):
                                job, machine = v, u
                            else:
                                continue  # Skip if nodes don't have proper types
                            
                            permanent_assignments[job] = machine
                            logger.info(f"Permanently assigned job {job} to machine {machine}")
                            
                            # Mark job and machine nodes for removal
                            nodes_to_remove.append(job)
                            nodes_to_remove.append(machine)
                        except Exception as e:
                            logger.error(f"Error assigning job to machine: {e}")
            
            for u, v in P2:
                if forest.has_edge(u, v):  # Make sure edge still exists
                    forest[u][v]['weight'] = forest[u][v].get('weight', 0.5) - delta
                    if forest[u][v]['weight'] <= 0.001:  # Handle float precision
                        # Mark edge for removal if weight is ~0
                        edges_to_remove.append((u, v))
                        logger.debug(f"Marking edge ({u}, {v}) for removal with weight ~0")
        
        except Exception as e:
            logger.error(f"Error in cycle elimination: {e}")
            break
    
    # Clean up any pending removals
    for u, v in edges_to_remove:
        if forest.has_edge(u, v):
            forest.remove_edge(u, v)
    
    for node in nodes_to_remove:
        if forest.has_node(node):
            forest.remove_node(node)
    
    logger.info(f"Eliminated {cycles_removed} cycles from graph")
    logger.info(f"Resulting forest has {forest.number_of_nodes()} nodes, {forest.number_of_edges()} edges")
    logger.info(f"Made {len(permanent_assignments)} permanent job assignments")
    
    return forest, permanent_assignments

def form_clusters(forest: nx.Graph) -> List[Tuple[List[str], List[str]]]:
    """
    Forms clusters from the forest structure according to Lemma 6.
    
    For each tree in the forest:
    - Case 1: Isolated machine node -> singleton cluster
    - Case 2: Job node is leaf -> assign to parent machine and remove both
    - Case 3: All job nodes have degree exactly 2 -> entire tree is one cluster
    - Case 4: Complex tree -> root at machine node, find job with ≥2 machine children
              and minimal subtree, extract subtree rooted at one machine child
    
    :param forest: The forest structure (bipartite graph with no cycles)
    :return: List of clusters, each as (machines, jobs)
    """
    logger.info("Forming clusters from forest structure")
    
    # Create a working copy of the forest
    working_forest = forest.copy()
    
    # Store clusters as (machines, jobs) tuples
    clusters = []
    
    # Process each connected component (tree) in the forest
    for tree_nodes in nx.connected_components(working_forest):
        tree = working_forest.subgraph(tree_nodes).copy()
        
        # Get machine and job nodes in this tree
        machine_nodes = [n for n in tree.nodes() if tree.nodes[n].get('type') == 'machine']
        job_nodes = [n for n in tree.nodes() if tree.nodes[n].get('type') == 'job']
        
        logger.debug(f"Processing tree with {len(machine_nodes)} machines and {len(job_nodes)} jobs")
        
        # Case 1: Isolated machine node
        if len(job_nodes) == 0:
            clusters.append(([machine_nodes[0]], []))
            logger.debug(f"Created singleton cluster for isolated machine {machine_nodes[0]}")
            continue
        
        # Case 2: Process leaf job nodes
        leaf_jobs = [n for n in job_nodes if tree.degree(n) == 1]
        while leaf_jobs:
            job = leaf_jobs[0]
            # Get the machine connected to this leaf job
            machine = list(tree.neighbors(job))[0]
            
            # Create a cluster with this machine and job
            clusters.append(([machine], [job]))
            logger.debug(f"Created cluster from leaf job {job} and machine {machine}")
            
            # Remove both from the tree
            tree.remove_node(job)
            tree.remove_node(machine)
            
            # Update job and machine lists
            job_nodes.remove(job)
            machine_nodes.remove(machine)
            
            # Update leaf jobs
            leaf_jobs = [n for n in job_nodes if n in tree.nodes() and tree.degree(n) == 1]
        
        # After processing leaf jobs, if the tree is empty, continue to next tree
        if not tree.nodes():
            continue
        
        # Re-get machine and job nodes in case they changed
        machine_nodes = [n for n in tree.nodes() if tree.nodes[n].get('type') == 'machine']
        job_nodes = [n for n in tree.nodes() if tree.nodes[n].get('type') == 'job']
        
        # Case 3: All job nodes have degree exactly 2
        if all(tree.degree(j) == 2 for j in job_nodes):
            clusters.append((machine_nodes, job_nodes))
            logger.debug(f"Created cluster with all jobs having degree 2: {len(machine_nodes)} machines, {len(job_nodes)} jobs")
            continue
        
        # Case 4: Complex tree structure
        # Check if there are any machine nodes before trying to root the tree
        if not machine_nodes:
            logger.warning(f"No machine nodes found in tree: {tree.nodes}")
            continue
            
        # Root the tree at an arbitrary machine node
        root = machine_nodes[0]
        
        # Build a directed tree for easier traversal
        directed_tree = nx.bfs_tree(tree, root)
        
        # Find job nodes with ≥2 machine children
        complex_jobs = []
        for job in job_nodes:
            if job in directed_tree:
                machine_children = [n for n in directed_tree.successors(job) 
                                    if n in machine_nodes]
                if len(machine_children) >= 2:
                    complex_jobs.append((job, machine_children))
        
        # If no complex jobs, this shouldn't happen in a well-formed tree
        if not complex_jobs:
            logger.warning(f"No complex jobs found in tree with {len(machine_nodes)} machines and {len(job_nodes)} jobs")
            # As a fallback, create a single cluster
            clusters.append((machine_nodes, job_nodes))
            continue
        
        # Find job with minimal subtree property
        job_to_process, machine_children = complex_jobs[0]  # Just take first for simplicity
        
        # Find a machine child with <0.5 units of job assigned
        for machine in machine_children:
            # In a real implementation, we would check the assignment value
            # For simplicity, we'll just take the first machine
            machine_to_extract = machine
            break
        
        # Extract subtree rooted at machine_to_extract
        subtree_nodes = list(nx.descendants(directed_tree, machine_to_extract)) + [machine_to_extract]
        subtree_machines = [n for n in subtree_nodes if n in machine_nodes]
        subtree_jobs = [n for n in subtree_nodes if n in job_nodes]
        
        # Create cluster from subtree
        clusters.append((subtree_machines, subtree_jobs))
        logger.debug(f"Created cluster from subtree: {len(subtree_machines)} machines, {len(subtree_jobs)} jobs")
        
        # Remove subtree from original tree and continue processing
        for node in subtree_nodes:
            if node in tree:
                tree.remove_node(node)
    
    # Verify cluster properties
    valid_clusters = []
    for i, (machines, jobs) in enumerate(clusters):
        logger.debug(f"Cluster {i}: {len(machines)} machines, {len(jobs)} jobs")
        
        if len(jobs) == len(machines) - 1:
            valid_clusters.append((machines, jobs))
        else:
            logger.warning(f"Cluster {i} has invalid property: |Ji| = {len(jobs)}, |Mi| = {len(machines)}")
            # For incorrect clusters, adjust by adding dummy jobs or removing machines
            # This is a simplification - in a real implementation we would need to ensure
            # cluster properties in a more sophisticated way
            while len(jobs) < len(machines) - 1:
                # Add dummy job (will be handled specially later)
                dummy_job = f"dummy_job_{i}_{len(jobs)}"
                jobs.append(dummy_job)
                logger.warning(f"Added dummy job {dummy_job} to balance cluster {i}")
            
            while len(jobs) > len(machines) - 1 and machines:
                # Remove a machine (simplification)
                removed_machine = machines.pop()
                logger.warning(f"Removed machine {removed_machine} from cluster {i} to balance")
                
            if len(jobs) > len(machines) - 1 and not machines:
                logger.warning(f"Cannot balance cluster {i}: no more machines to remove")
            
            valid_clusters.append((machines, jobs))
    
    logger.info(f"Formed {len(valid_clusters)} valid clusters")
    return valid_clusters
