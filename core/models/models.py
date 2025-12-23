from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Protocol, Callable, Any, Dict, TYPE_CHECKING, Union
from langchain_core.messages import AIMessage
from langchain_core.language_models.chat_models import BaseChatModel

if TYPE_CHECKING:
    # Import lazily for type checking only to avoid runtime dependency issues
    from langchain_chroma import Chroma
    from langchain.tools import BaseTool
    from langgraph.graph.state import CompiledStateGraph

@dataclass
class CodeChange:
    file_path: str
    original_code: str
    new_code: str
    reason: str
    timestamp: datetime
    reviewed: bool = False

    @classmethod
    def from_parts(cls, file_path: str, original: str, new: str, reason: str) -> "CodeChange":
        return cls(
            file_path=file_path,
            original_code=original,
            new_code=new,
            reason=reason,
            timestamp=datetime.now(),
        )

    def to_serializable(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.isoformat()
        return payload


# Minimal protocol that describes the parts of the agent used by tool builders
class AgentProtocol(Protocol):
    vector_store: "Chroma"
    llm: BaseChatModel
    tools: list
    agent: CompiledStateGraph
    async def execute_query(self, query: str, conversation_id: str) -> str: ...
    def export_change_log(self, output_file: str = "change_log.json") -> None: ...


# Tool types used across the codebase
# Tools may be callables or langchain BaseTool objects
ToolCallable = Union[Callable[..., str], "BaseTool"]
ToolBuilder = Callable[[AgentProtocol], List[ToolCallable]]
