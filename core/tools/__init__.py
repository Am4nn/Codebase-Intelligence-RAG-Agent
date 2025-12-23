"""Tools package for modular agent tooling.

Each tool module should expose a `build_<name>_tool(agent)` function that returns a callable decorated with `@tool`.
Factory functions in `factory.py` will use these builders to assemble a toolset for an agent.
"""

from .get_tools import get_tools

__all__ = ["get_tools"]
