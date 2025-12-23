import logging
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

class SemanticChunker:
    """Smart chunking that preserves semantic meaning"""

    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        # Chunking is a verbose operation; show details only at DEBUG level
        logger.info("✂️  Chunking %d documents...", len(documents))

        separators = [
            "\nclass ",
            "\ndef ",
            "\n\n",
            "\n",
            "\n{\n",
        ]

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=separators,
        )

        chunked = splitter.split_documents(documents)
        logger.info("✅ Created %d chunks", len(chunked))
        return chunked
