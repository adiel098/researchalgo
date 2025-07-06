"""
Sampling and set system conversion for the Santa Claus Problem.

This module implements the sampling-based techniques described in the paper
for reducing the instance size and applying specialized algorithms.
"""

import logging
import random
import networkx as nx
import numpy as np
from typing import Dict, List, Set, Tuple, Any
from .core import AllocationBuilder

# Setup logger
logger = logging.getLogger(__name__)

class SetSystem:
    """
    Represents a (k,l,p,η)-system for the Santa Claus Problem.
    
    The set system consists of:
    - Ground set U: small jobs as elements
    - Collections Ci: super-machines (clusters) as set families
    - Sets Si,j: machine-configuration pairs as subsets of U
    
    Parameters:
    - k: size of each set (e.g., n/ε)
    - l: number of sets per collection (e.g., n+m)
    - p: number of clusters
    - η: relaxation bound (e.g., 3)
    """
    
    def __init__(self, k: int, l: int, p: int, eta: float):
        self.k = k          # Target size of each set
        self.l = l          # Number of sets per collection
        self.p = p          # Number of clusters/collections
        self.eta = eta      # Relaxation bound
        
        self.ground_set = set()  # Elements (small jobs)
        self.collections = []     # List of collections (super-machines)
        self.sets = {}            # Maps (i, j) -> subset of ground set
        
        logger.info(f"Created set system with parameters: k={k}, l={l}, p={p}, eta={eta}")
    
    def add_element(self, element: str) -> None:
        """Adds an element to the ground set."""
        self.ground_set.add(element)
    
    def add_collection(self, collection_idx: int, sets_dict: Dict[int, Set[str]]) -> None:
        """
        Adds a collection (super-machine) with its sets.
        
        :param collection_idx: Index of the collection
        :param sets_dict: Dictionary mapping set indices to sets of elements
        """
        self.collections.append(collection_idx)
        for set_idx, elements in sets_dict.items():
            self.sets[(collection_idx, set_idx)] = set(elements)
    
    def get_set(self, collection_idx: int, set_idx: int) -> Set[str]:
        """Gets the set at position j in collection i."""
        return self.sets.get((collection_idx, set_idx), set())
    
    def verify_properties(self) -> bool:
        """
        Verifies that the set system satisfies its required properties:
        - Each set Si,j has cardinality k
        - Each element appears in at most η*l sets
        
        :return: True if properties are satisfied, False otherwise
        """
        # Check set sizes
        for (i, j), s in self.sets.items():
            if len(s) != self.k:
                logger.warning(f"Set ({i},{j}) has size {len(s)}, expected {self.k}")
                return False
        
        # Check element appearances
        for elem in self.ground_set:
            appearances = sum(1 for s in self.sets.values() if elem in s)
            if appearances > self.eta * self.l:
                logger.warning(f"Element {elem} appears in {appearances} sets, max allowed: {self.eta * self.l}")
                return False
        
        logger.info("Set system properties verified successfully")
        return True


def convert_to_set_system(small_gifts: Set[str], 
                          clusters: List[Tuple[List[str], List[str]]],
                          configurations: Dict[str, List[Tuple[Tuple[str, ...], float]]],
                          alloc: AllocationBuilder,
                          eta: float = 3.0) -> SetSystem:
    """
    Converts the Santa Claus instance to a (k,l,p,η)-system.
    
    :param small_gifts: Set of small gifts
    :param clusters: List of clusters from forest decomposition
    :param configurations: Configurations per machine with their fractional values
    :param alloc: Allocation builder with instance data
    :param eta: Relaxation bound
    :return: A set system representation
    """
    logger.info("Converting problem to set system representation")
    
    n = len(alloc.instance.items)
    m = len(alloc.instance.agents)
    p = len(clusters)
    
    # Set system parameters
    k = len(small_gifts) // m  # Approximation of n/ε
    if k == 0:
        k = 1  # Ensure k is at least 1
    l = n + m
    
    # Create set system
    system = SetSystem(k=k, l=l, p=p, eta=eta)
    
    # Add small gifts as elements to ground set
    for gift in small_gifts:
        system.add_element(gift)
    
    # For each cluster (super-machine), create a collection
    for cluster_idx, (machines, _) in enumerate(clusters):
        collection_sets = {}
        
        # For each machine in the cluster
        for set_idx, machine in enumerate(machines):
            # Get configurations for this machine
            machine_configs = configurations.get(machine, [])
            
            # Create a set of small gifts for this machine based on configurations
            gift_set = set()
            for config_tuple, frac_value in machine_configs:
                for gift in config_tuple:
                    if gift in small_gifts:
                        gift_set.add(gift)
                        if len(gift_set) >= k:
                            break
                if len(gift_set) >= k:
                    break
            
            # If not enough gifts collected from configurations, add random small gifts
            if len(gift_set) < k:
                remaining_gifts = small_gifts - gift_set
                gift_set.update(random.sample(list(remaining_gifts), 
                                              min(k - len(gift_set), len(remaining_gifts))))
            
            # If still not enough (unlikely), duplicate some gifts
            if len(gift_set) < k:
                logger.warning(f"Not enough unique small gifts for set ({cluster_idx},{set_idx}), duplicating")
                additional_needed = k - len(gift_set)
                gift_list = list(gift_set)
                for _ in range(additional_needed):
                    gift_set.add(f"{random.choice(gift_list)}_duplicate")
            
            # Add the set to the collection
            collection_sets[set_idx] = gift_set
        
        # Add the collection to the system
        system.add_collection(cluster_idx, collection_sets)
    
    logger.info(f"Created set system with {len(system.ground_set)} elements in ground set")
    logger.info(f"System has {len(system.collections)} collections and {len(system.sets)} sets")
    
    # Verify system properties
    if not system.verify_properties():
        logger.warning("Set system does not satisfy required properties")
    
    return system


def sample_elements(system: SetSystem, sampling_prob: float = None) -> SetSystem:
    """
    Creates a sampled version of the set system with fewer elements.
    
    :param system: Original set system
    :param sampling_prob: Probability for sampling elements (if None, calculated automatically)
    :return: A sampled set system with much fewer elements
    """
    logger.info("Sampling elements from set system")
    
    # Calculate sampling probability if not provided
    if sampling_prob is None:
        sampling_prob = (system.eta ** 2 * (system.l ** 2) * (np.log(system.p) ** 2)) / system.k
        sampling_prob = min(sampling_prob, 0.5)  # Cap at 0.5 to avoid excessive sampling
    
    logger.info(f"Using sampling probability: {sampling_prob:.4f}")
    
    # Sample elements
    sampled_elements = set()
    for elem in system.ground_set:
        if random.random() < sampling_prob:
            sampled_elements.add(elem)
    
    logger.info(f"Sampled {len(sampled_elements)} elements out of {len(system.ground_set)}")
    
    # Create new sampled system
    sampled_system = SetSystem(k=1, l=system.l, p=system.p, eta=2*system.eta)
    
    # Add sampled elements to ground set
    for elem in sampled_elements:
        sampled_system.add_element(elem)
    
    # Restrict sets to sampled elements
    for (i, j), s in system.sets.items():
        sampled_set = s.intersection(sampled_elements)
        if sampled_set:  # Only add non-empty sets
            if (i, j) not in sampled_system.sets:
                sampled_system.sets[(i, j)] = set()
            sampled_system.sets[(i, j)] = sampled_set
    
    # Filter out empty sets and recalculate actual k
    non_empty_sets = {k: v for k, v in sampled_system.sets.items() if v}
    sampled_system.sets = non_empty_sets
    
    if non_empty_sets:
        # Update k to be the average set size
        sampled_system.k = sum(len(v) for v in non_empty_sets.values()) // len(non_empty_sets)
        if sampled_system.k == 0:
            sampled_system.k = 1  # Ensure k is at least 1
    
    logger.info(f"Created sampled system with k={sampled_system.k}")
    
    return sampled_system
