import logging
from pathlib import Path
from typing import Any, List, Optional
from datetime import datetime

from langchain_core.documents import Document
from core.ingestion.parser import CodeParser

logger = logging.getLogger(__name__)

class CodebaseLoader:
    """Walk and load a repository as documents."""

    def __init__(self, repo_path: str, include_extensions: Optional[List[str]] = None):
        self.repo_path = Path(repo_path)
        self.include_extensions = include_extensions
        self.exclude_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'dist', 'build'}
        self.documents: List[Document] = []
        self._binary_exts = {'.png', '.jpg', '.jpeg', '.gif', '.exe', '.dll', '.so', '.pyc', '.class', '.jar', '.zip', '.tar', '.gz'}

    def _is_excluded(self, path: Path) -> bool:
        parts = {p.lower() for p in path.parts}
        return bool(parts & self.exclude_dirs)

    def _is_text_file(self, path: Path) -> bool:
        ext = path.suffix.lower()
        if ext in self._binary_exts:
            return False
        try:
            with open(path, 'rb') as f:
                chunk = f.read(2048)
                if b'\x00' in chunk:
                    return False
        except Exception:
            return False
        return True

    def load_repository(self) -> List[Document]:
        """Load text files from the repository into a list of langchain Documents.
        Binary or unreadable files are skipped; file metadata is attached to each Document.
        """
        logger.info("🔄 Loading repository from: %s", self.repo_path)
        if not self.repo_path.exists():
            logger.error("❌ Repository path not found: %s", self.repo_path)
            return []

        docs = []
        for path in self.repo_path.rglob('*'):
            if not path.is_file():
                continue
            if self._is_excluded(path):
                continue
            if not self._is_text_file(path):
                continue

            ext = path.suffix.lower().lstrip('.')
            if self.include_extensions is not None and ext not in self.include_extensions:
                continue

            try:
                text = path.read_text(encoding='utf-8')
            except Exception:
                try:
                    text = path.read_text(encoding='latin-1')
                except Exception as e:
                    logger.warning("⚠️  Skipping unreadable file %s: %s", path, e)
                    continue

            # Use the CodeParser to split files into logical chunks and attach richer metadata
            try:
                chunks = CodeParser.parse_file(str(path), text, repo_root=str(self.repo_path))
            except Exception:
                # fallback: single chunk
                chunks = [(text, {
                    'file_path': str(path),
                    'language': ext or 'unknown',
                    'name': path.stem,
                })]

            for chunk_text, chunk_meta in chunks:
                # Add repo-relative path and load info
                try:
                    repo_rel = str(path.relative_to(self.repo_path))
                except Exception:
                    repo_rel = str(path)

                metadata = {
                    **chunk_meta,
                    'repo_relative_path': repo_rel,
                    'repo_path': str(self.repo_path),
                    'load_timestamp': datetime.now().isoformat(),
                    'character_count': len(chunk_text),
                }

                # Sanitize metadata values so they are safe for vector stores like Chroma
                def _sanitize_value(v: Any) -> str | int | float | bool | None:
                    # Allowed primitive types: str, int, float, bool, None
                    if v is None or isinstance(v, (str, int, float, bool)):
                        return v
                    # Lists -> comma-separated string
                    if isinstance(v, (list, tuple)):
                        try:
                            return ','.join(str(x) for x in v)
                        except Exception:
                            return str(v)
                    # Dicts -> JSON string (compact)
                    if isinstance(v, dict):
                        try:
                            import json
                            return json.dumps(v, separators=(',', ':'))
                        except Exception:
                            return str(v)
                    # Fallback: convert to string representation
                    return str(v)

                sanitized_metadata = {k: _sanitize_value(val) for k, val in metadata.items()}

                docs.append(Document(page_content=chunk_text, metadata=sanitized_metadata))

        self.documents = docs
        # More verbose; surface when debugging repository loading
        logger.info("✅ Loaded %d chunks", len(docs))
        return docs
