import logging
from typing import List

from core.models import ToolCallable
from core.models.models import AgentProtocol
from .mcp_tools import mcp_tools_instance
from .factory import get_local_tools

logger = logging.getLogger(__name__)

async def get_tools(agent_protocol: AgentProtocol) -> List[ToolCallable]:
    mcp_tools = await mcp_tools_instance.get_mcp_tools()
    default_tools = get_local_tools(agent_protocol)
    tools: List[ToolCallable] = [*default_tools, *mcp_tools]
    logger.info(f"✅ Loaded total {len(tools)} tools for the agent.")
    return tools
