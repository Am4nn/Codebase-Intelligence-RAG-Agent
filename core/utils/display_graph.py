from __future__ import annotations

import logging
import os
import sys
import tempfile
import webbrowser
from pathlib import Path
from typing import Optional
from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)


def _open_file_with_default_viewer(path: Path) -> None:
    try:
        # Supported only for windows
        if sys.platform.startswith("win"):
            os.startfile(str(path))
    except Exception as exc:
        logger.debug("Failed to open image with default viewer: %s", exc)
        # fallback to webbrowser (works in many desktop environments)
        try:
            webbrowser.open(path.as_uri())
        except Exception:
            logger.debug("Fallback webbrowser open also failed for %s", path)

def render_graph_png(app: CompiledStateGraph) -> bytes:
    """Return PNG bytes for the agent's state graph.

    Raises:
        RuntimeError: if rendering fails or returned data isn't bytes.
    """
    try:
        png = app.get_graph().draw_mermaid_png()
    except Exception as exc:
        logger.exception("Failed to render graph PNG: %s", exc)
        raise RuntimeError("Failed to render agent graph to PNG") from exc

    if not isinstance(png, (bytes, bytearray)):
        raise RuntimeError("Rendered graph did not return bytes")
    return bytes(png)

def save_graph_png(app: CompiledStateGraph, path: Optional[Path | str] = None) -> Path:
    """Save the agent graph PNG to disk and return the path.

    If path is None, a temporary file is created and returned.
    """
    png = render_graph_png(app)
    if path is None:
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tf.write(png)
        tf.close()
        return Path(tf.name)
    p = Path(path)
    p.write_bytes(png)
    return p

def display_graph(app: CompiledStateGraph, *, open_in_viewer: bool = False) -> Optional[Path]:
    """Display the agent graph.

    - In Jupyter: show inline image (returns None).
    - Outside notebook:
        - If open_in_viewer=True: save to temp file and open with OS viewer; return saved Path.
        - Otherwise: save to temp file and return the Path for caller to handle.
    """
    path = save_graph_png(app)
    logger.info("Saved agent graph to %s", path)
    if open_in_viewer:
        _open_file_with_default_viewer(path)
    return path
