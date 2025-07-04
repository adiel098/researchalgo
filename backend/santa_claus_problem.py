"""
Algorithm implementation for 'The Santa Claus Problem'.

Paper: The Santa Claus Problem
Authors: Nikhil Bansal, Maxim Sviridenko
Link: https://dl.acm.org/doi/10.1145/1132516.1132557 (Proceedings of the 38th Annual ACM Symposium on Theory of Computing, 2006)

The problem involves distributing n gifts (items) among m children (agents).
Each child i has an arbitrary value pij for each present j.
The Santa's goal is to distribute presents such that the least lucky kid (agent)
is as happy as possible (maximin objective function):
    maximize (min_i sum_{j in S_i} p_ij)
where S_i is the set of presents received by kid i.

This file focuses on implementing the O(log log m / log log log m) approximation
algorithm for the restricted assignment case (p_ij in {p_j, 0}).

Programmers: Roey and Adiel
Date: 2025-05-29

This file is now a wrapper that imports from the modular implementation in the santa_claus package.
"""

import logging

# Import from the modular implementation
from santa_claus.core import Instance, AllocationBuilder, divide, example_instance, restricted_example
from santa_claus.algorithm import santa_claus, santa_claus_algorithm
from santa_claus.lp_solver import find_optimal_target_value, configuration_lp_solver
from santa_claus.clustering import create_super_machines, round_small_configurations, construct_final_allocation

# Setup logger
logger = logging.getLogger(__name__)
