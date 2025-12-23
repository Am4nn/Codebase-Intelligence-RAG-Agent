from typing import List

from .codebase_tools import build_codebase_tools
from core.models import ToolBuilder, ToolCallable, AgentProtocol

def get_local_tools(agent_protocol: AgentProtocol) -> List[ToolCallable]:
    """Return the default list of tool callables for the given agent."""
    builders: List[ToolBuilder] = [
      build_codebase_tools
    ]
    tools = [tool for builder in builders for tool in builder(agent_protocol)]
    return tools
