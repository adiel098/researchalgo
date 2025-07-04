"""
Core classes for the Santa Claus Problem.
"""

from typing import Dict, List, Set, Tuple, Iterable, Any, Optional
import logging

# Setup logger
logger = logging.getLogger(__name__)

class Instance:
    """
    A class representing a fair division instance.
    
    :param agents: A list of agent names (strings).
    :param items: A list of item names (strings).
    :param valuations: A dict of dicts: {agent_name: {item_name: value}}.
    :param agent_capacities: A dict mapping agent names to their capacity (max number of items).
    :param item_capacities: A dict mapping item names to their capacity (max number of copies).
    """
    
    def __init__(self, agents: List[str] = None, items: List[str] = None, 
                 valuations: Dict[str, Dict[str, float]] = None, 
                 agent_capacities: Dict[str, int] = None, 
                 item_capacities: Dict[str, int] = None):
        self.agents = agents if agents is not None else []
        self.items = items if items is not None else []
        self.valuations = valuations if valuations is not None else {}
        
        # Default capacities: 1 for each agent and item
        self.agent_capacities = agent_capacities if agent_capacities is not None else {agent: 1 for agent in self.agents}
        self.item_capacities = item_capacities if item_capacities is not None else {item: 1 for item in self.items}
        
        logger.info(f"Created new Instance with {len(self.agents)} agents and {len(self.items)} items")
        logger.debug(f"Agents: {self.agents}")
        logger.debug(f"Items: {self.items}")
        
        # Log value statistics
        if self.valuations and self.agents and self.items:
            total_values = 0
            nonzero_values = 0
            max_value = 0
            min_nonzero_value = float('inf')
            
            for agent, values in self.valuations.items():
                for item, value in values.items():
                    if value > 0:
                        total_values += value
                        nonzero_values += 1
                        max_value = max(max_value, value)
                        min_nonzero_value = min(min_nonzero_value, value)
            
            if nonzero_values == 0:
                min_nonzero_value = 0
                
            logger.info(f"Valuation statistics: {nonzero_values} nonzero values")
            logger.info(f"  Average nonzero value: {total_values/nonzero_values if nonzero_values > 0 else 0:.4f}")
            logger.info(f"  Max value: {max_value:.4f}")
            logger.info(f"  Min nonzero value: {min_nonzero_value:.4f}")
    
    def agent_item_value(self, agent: str, item: str) -> float:
        """
        Returns the value that the given agent assigns to the given item.
        If the agent or item is not in the instance, returns 0.
        """
        value = self.valuations.get(agent, {}).get(item, 0)
        return value


class AllocationBuilder:
    """
    A class for building an allocation (assignment of items to agents).
    
    :param instance: The fair division instance.
    """
    
    def __init__(self, instance: Instance):
        self.instance = instance
        self.bundles = {agent: [] for agent in instance.agents}
        logger.info(f"Created new AllocationBuilder for instance with {len(instance.agents)} agents")
    
    def give(self, agent: str, item: str) -> None:
        """
        Give the specified item to the specified agent.
        """
        if item in self.bundles[agent]:
            logger.debug(f"Item {item} already allocated to agent {agent}, skipping")
            return  # Item already allocated to this agent
            
        # Remove item from other agents
        for a in self.instance.agents:
            if a != agent and item in self.bundles[a]:
                self.bundles[a].remove(item)
                logger.debug(f"Removed item {item} from agent {a} to give to {agent}")
                
        # Add item to target agent
        self.bundles[agent].append(item)
        value = self.instance.agent_item_value(agent, item)
        logger.debug(f"Gave item {item} to agent {agent} with value {value:.4f}")
    
    @property
    def allocation(self) -> Dict[str, List[str]]:
        """
        Returns the current allocation as a dictionary from agent names to their bundles.
        """
        return {agent: bundle.copy() for agent, bundle in self.bundles.items()}


def divide(algorithm, instance: Instance, return_builder: bool = False) -> Dict[str, List[str]]:
    """
    Apply the given algorithm to the given instance.
    
    :param algorithm: A function that takes an AllocationBuilder and returns a dict mapping agent names to their allocated items.
    :param instance: The fair division instance.
    :param return_builder: If True, returns the AllocationBuilder instead of the allocation.
    :return: A dictionary mapping each agent to a list of items.
    """
    logger.info(f"Dividing instance with {len(instance.agents)} agents and {len(instance.items)} items using {algorithm.__name__ if hasattr(algorithm, '__name__') else 'unknown'} algorithm")
    
    alloc_builder = AllocationBuilder(instance)
    result = algorithm(alloc_builder)
    
    if return_builder:
        logger.info("Returning allocation builder")
        return alloc_builder
    
    final_allocation = result if result is not None else alloc_builder.allocation
    
    # Log statistics about the final allocation
    item_count = sum(len(bundle) for bundle in final_allocation.values())
    logger.info(f"Division complete. Allocated {item_count} items to {len(final_allocation)} agents")
    
    return final_allocation


# Example instance for testing
example_instance = Instance(
    agents=["Child1", "Child2", "Child3"],
    items=["gift1", "gift2", "gift3"],
    valuations={
        "Child1": {"gift1": 5, "gift2": 5, "gift3": 5},
        "Child2": {"gift1": 5, "gift2": 5, "gift3": 10},
        "Child3": {"gift1": 5, "gift2": 5, "gift3": 10}
    },
    agent_capacities={"Child1": 1, "Child2": 1, "Child3": 1},
    item_capacities={"gift1": 1, "gift2": 1, "gift3": 1}
)

# Example instance with restricted assignment (values in {p_j, 0})
restricted_example = Instance(
    agents=["Child1", "Child2", "Child3"],
    items=["gift1", "gift2", "gift3", "gift4", "gift5"],
    valuations={
        "Child1": {"gift1": 5, "gift2": 0, "gift3": 0, "gift4": 0, "gift5": 10},
        "Child2": {"gift1": 0, "gift2": 5, "gift3": 0, "gift4": 0, "gift5": 0},
        "Child3": {"gift1": 0, "gift2": 0, "gift3": 10, "gift4": 5, "gift5": 0}
    },
    agent_capacities={"Child1": 2, "Child2": 2, "Child3": 2},
    item_capacities={"gift1": 1, "gift2": 1, "gift3": 1, "gift4": 1, "gift5": 1}
)
