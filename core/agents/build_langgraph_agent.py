import logging
from typing import List

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import InMemorySaver  
from langchain.agents.middleware import SummarizationMiddleware

from core.models import ToolCallable

logger = logging.getLogger(__name__)

def build_langgraph_agent(llm: BaseChatModel, tools: List[ToolCallable], system_prompt: str, name: str) -> CompiledStateGraph:
    """Construct and return a LangGraph agent with the given LLM, tools, and system prompt."""
    try:
        return create_agent(
            model=llm,
            tools=tools,
            system_prompt=system_prompt,
            name=name,
            middleware=[
              SummarizationMiddleware(
                  model=llm,
                  trigger=("tokens", 4000),
                  keep=("messages", 20)
              )
            ],
            checkpointer=InMemorySaver()
        )
    except Exception as e:
        logger.exception("Failed to construct LangGraph agent: %s", e)
        raise
