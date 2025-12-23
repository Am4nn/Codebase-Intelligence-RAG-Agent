from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

def fin_root() -> str:
    """Return the root directory of the project."""
    return str(Path(__file__).parent.parent.parent.resolve())

def projects_directory() -> str:
    """Return the preferred projects root directory."""
    base = Path(fin_root())
    candidate = base / "data" / "projects"
    if candidate.exists():
        return str(candidate)

    logger.warning("⚠️ Default projects directory not found; using %s", base)
    return str(base)

def db_persist_directory() -> str:
    """Return the default directory for vector store persistence."""
    base = Path(fin_root())
    candidate = base / "data" / "codebase_intelligence_db"
    return str(candidate)
