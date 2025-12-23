"""Preferred ASGI entrypoint.

Use with:
- `uvicorn src.app:app --reload --port 8000`

The `python -m server` launcher also points here.
"""

from server.app import CodebaseIntelligenceServer

server = CodebaseIntelligenceServer()
app = server.create_app()

