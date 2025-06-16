"""Compatibility shim – some modules mistakenly import `networkz` instead of `networkx`.
Installing this module prevents ImportError by forwarding all attributes to the
real `networkx` package.
"""
import sys
import types

try:
    import networkx as _nx
except ImportError:  # pragma: no cover
    raise ImportError("networkx must be installed to use this project")

module = types.ModuleType(__name__)
module.__dict__.update(_nx.__dict__)

sys.modules[__name__] = module
