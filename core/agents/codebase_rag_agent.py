import logging
import os
from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.language_models.chat_models import BaseChatModel
from core.agents.system_prompt import SYSTEM_PROMPT
from core.models import CodeChange, AgentProtocol, ToolCallable, ToolBuilder
from langchain_chroma import Chroma
from core.utils.response_formatter import format_response
from core.agents.build_langgraph_agent import build_langgraph_agent
from core.tools import get_tools
from core.utils.display_graph import display_graph

logger = logging.getLogger(__name__)

class CodebaseRAGAgent(AgentProtocol):
    """RAG Agent that uses LangGraph and a provided vector store"""

    def __init__(self, vector_store: Chroma):
        self.vector_store = vector_store
        self.change_log: List[CodeChange] = []

    async def initialize(self, llm: Optional[BaseChatModel] = None, model_name: Optional[str] = None):
        """Initialize the agent by preparing tools and other components."""
        self.llm = llm if llm is not None else self.setup_llm(model_name=model_name)
        self.tools: List[ToolCallable] = await get_tools(self)
        self.agent = build_langgraph_agent(self.llm, self.tools, system_prompt=SYSTEM_PROMPT, name="CodebaseRAGAgent")
        # display_graph(self.agent, open_in_viewer=True)

    def setup_llm(self, model_name: Optional[str] = None) -> BaseChatModel:
        """Create and return the LLM instance for the agent."""
        chosen_model = model_name or os.environ.get("LLM_MODEL", "gpt-5-nano")
        return ChatOpenAI(
            name="codebase-rag-agent",
            model=chosen_model,
            temperature=0.3,
            streaming=False
        )

    async def execute_query(self, query: str, conversation_id: str) -> str:
        """Execute query using planner (if enabled) or clean pipeline."""
        if not query:
            return ""
        response = await self.agent.ainvoke(
          {"messages": [HumanMessage(content=query)]},
          {"configurable": {"thread_id": conversation_id}},  
        )
        return format_response(response)

    def log_code_change(self, file_path: str, original: str, new: str, reason: str) -> None:
        """Log a code change to the change log."""
        change = CodeChange.from_parts(file_path, original, new, reason)
        self.change_log.append(change)

    def export_change_log(self, output_file: str = "change_log.json") -> None:
        """Export the change log to JSON."""
        import json
        with open(output_file, 'w') as f:
            json.dump([c.to_serializable() for c in self.change_log], f, indent=2)
