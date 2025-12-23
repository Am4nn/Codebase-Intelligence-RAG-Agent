"""
Core API for Codebase Intelligence System

This is the main black-box API that can be used by any interface (CLI, web server, UI, etc.)
"""
import logging
from typing import Optional, List, Dict, Any, cast
from core.ingestion import CodebaseLoader, SemanticChunker
from core.storage import CodeVectorStore
from core.agents import CodebaseRAGAgent
from typing import TYPE_CHECKING
from langchain_core.runnables import RunnableConfig
from core.utils.fs import db_persist_directory, projects_directory

if TYPE_CHECKING:
    from langchain_chroma import Chroma

logger = logging.getLogger(__name__)

class CodebaseIntelligence:
    """
    Core API for codebase intelligence operations.
    
    This class provides a clean, interface-agnostic API for:
    - Loading and indexing codebases
    - Querying code with natural language
    - Managing vector stores and embeddings
    - Accessing the RAG agent
    
    Usage:
        # Initialize system
        system = CodebaseIntelligence()
        system.initialize()
        
        # Query the codebase
        result = system.query("How does authentication work?")
        
        # Access the agent directly for advanced use
        agent = system.get_agent()
    """

    def __init__(
        self, 
        include_extensions: Optional[List[str]] = None,
        model_name: Optional[str] = None
    ):
        """
        Initialize the Codebase Intelligence system.
        
        Args:
            repo_path: Path to the repository to analyze
            persist_dir: Directory for storing vector embeddings
            include_extensions: List of file extensions to include (e.g., ['py', 'js'])
            model_name: LLM model name to use (defaults to gpt-5-nano)
        """
        self.persist_dir = db_persist_directory()
        self.repo_path = projects_directory()
        self.model_name = model_name
        self.loader = CodebaseLoader(repo_path=self.repo_path, include_extensions=include_extensions)
        self.chunker = SemanticChunker()
        self.vector_store_manager = CodeVectorStore(persist_directory=self.persist_dir)
        self.agent: CodebaseRAGAgent | None = None
        self.vector_store: Chroma | None = None
        self._initialized = False

    async def initialize(self, force_reload: bool = False, skip_embeddings: bool = False) -> 'CodebaseIntelligence':
        """
        Initialize the system: load/create vector store and instantiate the agent.
        
        Args:
            force_reload: If True, rebuild vector store from scratch
            skip_embeddings: If True, skip embedding creation (useful for testing)
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If OpenAI API key is not set and skip_embeddings is False
        """
        logger.info("Initializing Codebase Intelligence Engine...")

        if not force_reload:
            self.vector_store = self.vector_store_manager.load_existing()

        if self.vector_store is None:
            if skip_embeddings:
                logger.info("Skipping embedding creation as requested.")
                return self

            if not self.vector_store_manager.has_api_key():
                raise ValueError(
                    "OpenAI API key is not set. Set OPENAI_API_KEY environment variable "
                    "or run with skip_embeddings=True"
                )

            logger.info("Building vector store from repository...")
            documents = self.loader.load_repository()
            if not documents:
                logger.error("No documents loaded from repository")
                return self
            
            chunked_docs = self.chunker.chunk_documents(documents)
            self.vector_store = self.vector_store_manager.create_from_documents(chunked_docs)

        if self.vector_store:
            self.agent = CodebaseRAGAgent(self.vector_store)
            await self.agent.initialize(model_name=self.model_name)
            self._initialized = True
            logger.info("System initialized successfully!")

        return self

    async def query(self, question: str, conversation_id: str) -> str:
        """
        Query the codebase with a natural language question.
        
        Args:
            question: Natural language question about the codebase
            
        Returns:
            Answer from the RAG agent
            
        Raises:
            RuntimeError: If system is not initialized
        """
        if not self._initialized or self.agent is None:
            raise RuntimeError(
                "System not initialized. Call initialize() first."
            )
        return str(await self.agent.execute_query(question, conversation_id))

    def get_agent(self) -> Optional[CodebaseRAGAgent]:
        """
        Get direct access to the underlying RAG agent.
        
        Returns:
            The CodebaseRAGAgent instance or None if not initialized
        """
        return self.agent

    def get_vector_store(self) -> Optional['Chroma']:
        """
        Get direct access to the vector store.
        
        Returns:
            The Chroma vector store instance or None if not initialized
        """
        return self.vector_store

    def is_initialized(self) -> bool:
        """Check if the system has been initialized."""
        return self._initialized

    def export_change_log(self, output_file: str = "change_log.json") -> None:
        """
        Export the agent's change log to a file.
        
        Args:
            output_file: Path to the output JSON file
            
        Raises:
            RuntimeError: If system is not initialized
        """
        if not self._initialized or self.agent is None:
            raise RuntimeError(
                "System not initialized. Call initialize() first."
            )
        self.agent.export_change_log(output_file)

    async def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Get the full conversation history for a given conversation ID.
        
        Args:
            conversation_id: The conversation ID to retrieve history for
            
        Returns:
            List of message dictionaries with role and content
            
        Raises:
            RuntimeError: If system is not initialized
        """
        if not self._initialized or self.agent is None:
            raise RuntimeError(
                "System not initialized. Call initialize() first."
            )
        
        try:
            checkpointer = self.agent.agent.checkpointer
            if checkpointer is None or isinstance(checkpointer, bool):
                logger.warning("No checkpointer available")
                return []
            
            config: RunnableConfig = {"configurable": {"thread_id": conversation_id}}
            
            # Get the checkpoint for this conversation
            checkpoint = await checkpointer.aget(config)
            
            if checkpoint is None:
                return []
            
            # Extract messages from the checkpoint
            messages = checkpoint.get("channel_values", {}).get("messages", [])
            
            # Format messages for API response
            history = []
            for msg in messages:
                if hasattr(msg, 'type'):
                    history.append({
                        "role": msg.type,  # 'human', 'ai', 'system', 'tool'
                        "content": msg.content,
                        "timestamp": getattr(msg, 'timestamp', None)
                    })
            
            return history
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {e}")
            return []

    async def list_conversations(self) -> List[str]:
        """
        List all conversation IDs that have been stored.
        
        Returns:
            List of conversation IDs
            
        Raises:
            RuntimeError: If system is not initialized
        """
        if not self._initialized or self.agent is None:
            raise RuntimeError(
                "System not initialized. Call initialize() first."
            )
        
        try:
            checkpointer = self.agent.agent.checkpointer
            if checkpointer is None:
                logger.warning("No checkpointer available")
                return []
            
            # InMemorySaver stores checkpoints in memory
            # Access the internal storage to list all conversation IDs
            if hasattr(checkpointer, 'storage'):
                return list(getattr(checkpointer, 'storage').keys())
            else:
                logger.warning("Checkpointer does not expose storage. Cannot list conversations.")
                return []
        except Exception as e:
            logger.error(f"Error listing conversations: {e}")
            return []

    async def get_conversation_state(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the complete state/checkpoint for a conversation.
        
        Args:
            conversation_id: The conversation ID to retrieve state for
            
        Returns:
            Dictionary containing the full conversation state or None if not found
            
        Raises:
            RuntimeError: If system is not initialized
        """
        if not self._initialized or self.agent is None:
            raise RuntimeError(
                "System not initialized. Call initialize() first."
            )
        
        try:
            checkpointer = self.agent.agent.checkpointer
            if checkpointer is None or isinstance(checkpointer, bool):
                logger.warning("No checkpointer available")
                return None
            
            config: RunnableConfig = {"configurable": {"thread_id": conversation_id}}
            checkpoint = await checkpointer.aget(config)
            
            if checkpoint is None:
                return None
            
            return {
                "conversation_id": conversation_id,
                "checkpoint_id": checkpoint.get("id"),
                "channel_values": checkpoint.get("channel_values", {}),
                "metadata": checkpoint.get("metadata", {}),
                "message_count": len(checkpoint.get("channel_values", {}).get("messages", []))
            }
        except Exception as e:
            logger.error(f"Error retrieving conversation state: {e}")
            return None

    async def clear_conversation(self, conversation_id: str) -> bool:
        """
        Clear/delete a specific conversation history.
        
        Args:
            conversation_id: The conversation ID to clear
            
        Returns:
            True if successfully cleared, False otherwise
            
        Raises:
            RuntimeError: If system is not initialized
        """
        if not self._initialized or self.agent is None:
            raise RuntimeError(
                "System not initialized. Call initialize() first."
            )
        
        try:
            checkpointer = self.agent.agent.checkpointer
            if checkpointer is None:
                logger.warning("No checkpointer available")
                return False
            
            # InMemorySaver stores data in memory, we can remove it
            if hasattr(checkpointer, 'storage'):
                # Remove the conversation from storage
                keys_to_remove = [k for k in getattr(checkpointer, 'storage').keys() if conversation_id in str(k)]
                for key in keys_to_remove:
                    del getattr(checkpointer, 'storage')[key]
                logger.info(f"Cleared conversation: {conversation_id}")
                return True
            else:
                logger.warning("Checkpointer does not support clearing conversations.")
                return False
        except Exception as e:
            logger.error(f"Error clearing conversation: {e}")
            return False

    async def get_conversation_summary(self, conversation_id: str) -> Dict[str, Any]:
        """
        Get a summary of a conversation including message count and participants.
        
        Args:
            conversation_id: The conversation ID to summarize
            
        Returns:
            Dictionary with conversation summary
            
        Raises:
            RuntimeError: If system is not initialized
        """
        if not self._initialized or self.agent is None:
            raise RuntimeError(
                "System not initialized. Call initialize() first."
            )
        
        history = await self.get_conversation_history(conversation_id)
        
        if not history:
            return {
                "conversation_id": conversation_id,
                "exists": False,
                "message_count": 0
            }
        
        # Count messages by role
        role_counts: dict[str, int] = {}
        for msg in history:
            role = msg.get("role", "unknown")
            role_counts[role] = role_counts.get(role, 0) + 1
        
        return {
            "conversation_id": conversation_id,
            "exists": True,
            "message_count": len(history),
            "role_counts": role_counts,
            "first_message": history[0].get("content", "")[:100] if history else None,
            "last_message": history[-1].get("content", "")[:100] if history else None
        }
