"""
Implementation of Leighton's algorithm and network flow verification for the Santa Claus Problem.

This module implements Theorem 3 and Lemma 9 from the paper, providing algorithms
for solving small instances of the set system and verifying function quality.
"""

import logging
import random
import networkx as nx
from typing import Dict, List, Set, Tuple, Optional, Any
from .sampling import SetSystem

# Setup logger
logger = logging.getLogger(__name__)

def apply_leighton_algorithm(system: SetSystem) -> Dict[int, int]:
    """
    Applies Leighton's algorithm to solve a small instance of the set system.
    
    Based on Theorem 3 from the paper, uses a constructive proof via Lovász Local Lemma
    to find a good assignment function.
    
    :param system: The set system to solve
    :return: A function f: {1,...,p} → {1,...,l} that maps collections to set indices
    """
    logger.info("Applying Leighton's algorithm to sampled set system")
    
    # The algorithm guarantees finding a (1, O(η log ksl / log log ksl))-good function
    # where k = set size, s = sampling probability factor, l = sets per collection
    
    # Implementation note: For practical purposes, we'll use a randomized approach 
    # with verification, rather than the full derandomized Lovász Local Lemma algorithm
    
    max_attempts = 1000  # Limit number of attempts
    best_f = None
    best_quality = float('-inf')
    
    for attempt in range(max_attempts):
        # Generate a random function f
        f = {}
        for i in system.collections:
            # Find valid set indices for this collection
            valid_indices = [j for j in range(system.l) if (i, j) in system.sets]
            if not valid_indices:
                # If no valid sets for this collection, assign a random index
                f[i] = random.randint(0, system.l - 1)
            else:
                f[i] = random.choice(valid_indices)
        
        # Verify function quality
        gamma = 1.0  # Target value for load factor
        beta_estimate = (system.eta * system.k * system.l) / system.p  # Theoretical estimate
        beta = int(beta_estimate) + 1  # Round up to ensure integer
        
        is_good, _ = verify_function_quality(system, f, gamma, beta)
        
        if is_good:
            logger.info(f"Found good function after {attempt+1} attempts with β={beta}")
            return f
        
        # If not good, evaluate how close we are to being good
        quality_score = evaluate_function_quality(system, f, gamma)
        if quality_score > best_quality:
            best_quality = quality_score
            best_f = f.copy()
    
    logger.warning(f"Could not find optimal function after {max_attempts} attempts")
    logger.warning(f"Using best found function with quality score {best_quality}")
    
    return best_f


def verify_function_quality(system: SetSystem, f: Dict[int, int], 
                            gamma: float, beta: int) -> Tuple[bool, Dict[int, Set[str]]]:
    """
    Verifies if a function is (γ,β)-good for the set system using network flow.
    
    Based on Lemma 9 from the paper, constructs a network flow instance and checks
    if maximum flow equals kp/γ.
    
    :param system: The set system
    :param f: Function mapping collections to set indices
    :param gamma: Load factor parameter
    :param beta: Congestion parameter
    :return: Tuple of (is_good, subset_dict) where subset_dict maps collections to subsets
    """
    logger.info(f"Verifying function quality with γ={gamma}, β={beta}")
    
    # Construct network flow instance
    G = nx.DiGraph()
    
    # Add source and sink
    source = 'source'
    sink = 'sink'
    G.add_node(source)
    G.add_node(sink)
    
    # Add vertices for collections and elements
    for i in system.collections:
        G.add_node(f"V_{i}")
        # Edge from source to collection with capacity k/gamma
        capacity_i = max(1, int(system.k / gamma))
        G.add_edge(source, f"V_{i}", capacity=capacity_i)
    
    for elem in system.ground_set:
        G.add_node(f"U_{elem}")
        # Edge from element to sink with capacity beta
        G.add_edge(f"U_{elem}", sink, capacity=beta)
    
    # Add edges from collections to elements based on selected sets
    for i in system.collections:
        j = f.get(i, 0)  # Get selected set index, default to 0
        selected_set = system.get_set(i, j)
        for elem in selected_set:
            G.add_edge(f"V_{i}", f"U_{elem}", capacity=1)
    
    # Compute maximum flow
    try:
        max_flow_value = nx.maximum_flow_value(G, source, sink)
        target_flow = max(1, int(system.k * len(system.collections) / gamma))
        
        logger.info(f"Max flow: {max_flow_value}, Target flow: {target_flow}")
        
        if max_flow_value >= target_flow:
            # Extract the selected subsets from the flow
            flow_dict = nx.maximum_flow(G, source, sink)[1]
            
            subset_dict = {}
            for i in system.collections:
                subset = set()
                collection_node = f"V_{i}"
                for elem in system.ground_set:
                    elem_node = f"U_{elem}"
                    if (collection_node in flow_dict and 
                        elem_node in flow_dict[collection_node] and 
                        flow_dict[collection_node][elem_node] > 0):
                        subset.add(elem)
                subset_dict[i] = subset
            
            logger.info(f"Function is ({gamma},{beta})-good")
            return True, subset_dict
        else:
            logger.info(f"Function is not ({gamma},{beta})-good")
            return False, {}
    
    except nx.NetworkXError:
        logger.error("Network flow computation failed")
        return False, {}


def evaluate_function_quality(system: SetSystem, f: Dict[int, int], 
                              gamma: float) -> float:
    """
    Evaluates how close a function is to being good.
    
    :param system: The set system
    :param f: Function mapping collections to set indices
    :param gamma: Load factor parameter
    :return: Quality score (higher is better)
    """
    # Count element occurrences in selected sets
    elem_counts = {}
    for elem in system.ground_set:
        elem_counts[elem] = 0
    
    total_elements_in_sets = 0
    
    # Count occurrences
    for i in system.collections:
        j = f.get(i, 0)  # Get selected set index, default to 0
        selected_set = system.get_set(i, j)
        total_elements_in_sets += len(selected_set)
        for elem in selected_set:
            elem_counts[elem] = elem_counts.get(elem, 0) + 1
    
    # Calculate congestion (lower is better)
    if elem_counts:
        max_congestion = max(elem_counts.values())
        avg_congestion = sum(elem_counts.values()) / len(elem_counts)
    else:
        max_congestion = 0
        avg_congestion = 0
    
    # Calculate coverage (higher is better)
    target_elements = system.k * len(system.collections) / gamma
    coverage_ratio = total_elements_in_sets / max(1, target_elements)
    
    # Combined quality score
    quality_score = coverage_ratio - (max_congestion / system.l)
    
    return quality_score
