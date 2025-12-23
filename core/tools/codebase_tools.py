import logging
from typing import List
from langchain.tools import tool

from core.models import AgentProtocol, ToolCallable

logger = logging.getLogger(__name__)

def build_codebase_tools(agent_protocol: AgentProtocol) -> List[ToolCallable]:
    @tool
    def search_codebase(query: str) -> str:
        """Search the codebase vector store for relevant code snippets and return formatted results."""
        # Log the tool invocation with a concise parameter summary
        try:
            short_q = (query[:200] + '...') if len(query) > 200 else query
            logger.info("🔎 Tool 'search_codebase' called with query=%r", short_q)
            results = agent_protocol.vector_store.similarity_search_with_score(query, k=5)
            if not results:
                logger.info("🔎 Tool 'search_codebase' returned 0 results for query=%r", short_q)
                return "❌ No relevant code found."
            logger.info("🔎 Tool 'search_codebase' returned %d results", len(results))
            formatted_results = []
            for doc, score in results:
                source = doc.metadata.get('source') or doc.metadata.get('file_path') or 'unknown'
                try:
                    confidence = f"{(1 - float(score)) * 100:.1f}%"
                except Exception:
                    confidence = f"{float(score):.4f}"
                formatted_results.append(f"📄 File: {source} (score: {confidence})\n```\n{doc.page_content}\n```")
            return "\n\n".join(formatted_results)
        except Exception as e:
            logger.exception("Search error in search_codebase tool: %s", e)
            return f"❌ Search error: {str(e)}"

    return [search_codebase]
