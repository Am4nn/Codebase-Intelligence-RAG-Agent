import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class CodeVectorStore:
    """Manage vector embeddings with latest Chroma"""

    def __init__(self, persist_directory: str, embedding_model: str = "text-embedding-3-small"):
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.vector_store: Optional[Chroma] = None
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            api_key=lambda: os.getenv("OPENAI_API_KEY", "")
        )

    def has_api_key(self) -> bool:
        """Return True if an OpenAI API key is available in the environment."""
        return bool(os.getenv("OPENAI_API_KEY"))

    def create_from_documents(self, documents: List[Document]) -> Chroma:
        """Create and persist a Chroma vector store from the provided Documents."""
        # Verbose operation; surface when logging level is DEBUG
        logger.info("🔍 Creating vector embeddings (%s)...", self.embedding_model)
        try:
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name="codebase_intelligence"
            )
            logger.info("✅ Vector store created at %s", self.persist_directory)
            return self.vector_store
        except Exception as e:
            # If it's an auth error from OpenAI, provide a helpful message
            if '401' in str(e) or 'You didn\'t provide an API key' in str(e):
                logger.error("❌ OpenAI API authentication failed: %s", e)
                logger.error("Please set OPENAI_API_KEY in your environment: e.g. `setx OPENAI_API_KEY \"sk-...\"` on Windows or `export OPENAI_API_KEY=sk-...` on Linux/macOS")
            logger.exception("❌ Error creating vector store: %s", e)
            raise

    def load_existing(self) -> Optional[Chroma]:
        """Load an existing persistent Chroma vector store if present; otherwise return None."""
        if Path(self.persist_directory).exists():
            try:
                # Verbose: show when debug is enabled
                logger.info("📂 Loading vector store from %s...", self.persist_directory)
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings,
                    collection_name="codebase_intelligence"
                )
                logger.info("✅ Vector store loaded")
                return self.vector_store
            except Exception as e:
                logger.exception("⚠️  Could not load existing store: %s", e)
                return None
        return None

    def search_similar(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            return results
        except Exception as e:
            logger.exception("❌ Search error: %s", e)
            return []
