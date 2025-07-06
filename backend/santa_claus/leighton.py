"""
Implementation of Leighton's algorithm and network flow verification for the Santa Claus Problem.

This module implements Theorem 3 and Lemma 9 from the paper, providing algorithms
for solving small instances of the set system and verifying function quality.
"""

import logging
import random
import networkx as nx
import sys
import codecs
from typing import Dict, List, Set, Tuple, Optional, Any
from .sampling import SetSystem

# Setup logger with Unicode support
logger = logging.getLogger(__name__)

# Configure Unicode handling for logging on Windows
if sys.platform == 'win32':
    # Ensure stdout can handle Unicode
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
    # Configure handlers to use UTF-8 where possible
    for handler in logging.root.handlers:
        if hasattr(handler, 'stream') and hasattr(handler.stream, 'reconfigure'):
            handler.stream.reconfigure(encoding='utf-8', errors='backslashreplace')

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
    
    max_attempts = 100  # Reduced max attempts to prevent excessive logging
    best_f = None
    best_quality = float('-inf')
    
    # Calculate beta once outside the loop
    gamma = 1.0  # Target value for load factor
    beta_estimate = (system.eta * system.k * system.l) / system.p  # Theoretical estimate
    beta = max(1, int(beta_estimate) + 1)  # Round up and ensure at least 1
    
    logger.info(f"Starting search with gamma={gamma}, beta={beta}, eta={system.eta}, k={system.k}, l={system.l}, p={system.p}")
    
    # We'll log less frequently to avoid excessive output
    log_frequency = max(1, max_attempts // 10)  
    
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
        
        # Only log verification at specific intervals
        should_log = (attempt % log_frequency == 0) or (attempt == max_attempts - 1) 
        
        # Verify function quality
        is_good, subset_dict = verify_function_quality(system, f, gamma, beta, should_log)
        
        if is_good:
            logger.info(f"Found good function after {attempt+1} attempts with beta={beta}")
            return f
        
        # If not good, evaluate how close we are to being good
        quality_score = evaluate_function_quality(system, f, gamma)
        if quality_score > best_quality:
            best_quality = quality_score
            best_f = f.copy()
            
        # Only log progress at intervals
        if should_log:
            logger.info(f"Attempt {attempt+1}/{max_attempts}, best quality so far: {best_quality:.4f}")
    
    logger.warning(f"Could not find optimal function after {max_attempts} attempts")
    logger.warning(f"Using best found function with quality score {best_quality}")
    
    return best_f


def verify_function_quality(system: SetSystem, f: Dict[int, int], 
                            gamma: float, beta: int, should_log: bool = True) -> Tuple[bool, Dict[int, Set[str]]]:
    """
    Verifies if a function is (γ,β)-good for the set system using network flow.
    
    Based on Lemma 9 from the paper, constructs a network flow instance and checks
    if maximum flow equals kp/γ.
    
    :param system: The set system
    :param f: Function mapping collections to set indices
    :param gamma: Load factor parameter
    :param beta: Congestion parameter
    :param should_log: Whether to log info messages during verification
    :return: Tuple of (is_good, subset_dict) where subset_dict maps collections to subsets
    """
    if should_log:
        logger.info(f"Verifying function quality with gamma={gamma}, beta={beta}")
    
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
    
    # Check if the network would be empty
    edges_added = 0
    
    # Add edges from collections to elements based on selected sets
    for i in system.collections:
        j = f.get(i, 0)  # Get selected set index, default to 0
        selected_set = system.get_set(i, j)
        for elem in selected_set:
            G.add_edge(f"V_{i}", f"U_{elem}", capacity=1)
            edges_added += 1
    
    # If no edges were added between collections and elements,
    # we know the max flow will be 0
    if edges_added == 0:
        if should_log:
            logger.warning("Network has no edges between collections and elements")
            logger.info(f"Max flow: 0, Target flow: {max(1, int(system.k * len(system.collections) / gamma))}")
            logger.info(f"Function is not ({gamma},{beta})-good")
        return False, {}
    
    # Compute maximum flow
    try:
        max_flow_value = nx.maximum_flow_value(G, source, sink)
        target_flow = max(1, int(system.k * len(system.collections) / gamma))
        
        if should_log:
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
            
            if should_log:
                logger.info(f"Function is ({gamma},{beta})-good")
            return True, subset_dict
        else:
            if should_log:
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
