"""
MCP Tools Integration using langchain_mcp_adapters

Provides MCP tools from multiple servers including Playwright for browser automation.
"""
import logging
from typing import List
from langchain_core.tools.base import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import Connection

logger = logging.getLogger(__name__)

mcp_connections: dict[str, Connection] = {
    "mcp_utility": {
        "url": "http://localhost:8001/mcp/",
        "transport": "streamable_http",
    },
    "playwright": {
        "transport": "stdio",
        "command": "npx",
        "args": ["@playwright/mcp@latest"],
    },
}

class MCPTools:
    """Class to manage MCP tools integration."""

    def __init__(self, connections: dict[str, Connection] = {}):
      self.connections: dict[str, Connection] = connections
      self.client = self.mcp_client()

    def mcp_client(self) -> MultiServerMCPClient:
        """Create and configure MCP client with multiple servers."""
        return MultiServerMCPClient(connections=self.connections)

    async def get_mcp_tools(self) -> List[BaseTool]:
        """Get all tools from configured MCP servers."""
        try:
            tools = await self.client.get_tools()
            logger.info(f"✅ Loaded {len(tools)} MCP tools.")
            return tools
        except Exception as e:
            logger.error(f"❌ Failed to load MCP tools: {e}")
            raise e

mcp_tools_instance = MCPTools()
