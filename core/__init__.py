"""
Core Module - Codebase Intelligence Black Box

This is a pure library module with NO CLI or server code.
Import this in your CLI, server, or any other application.

Usage:
    from core import CodebaseIntelligence

    # initialize() and query() are async
    # await system.initialize(); await system.query(...)
"""

from core.api import CodebaseIntelligence

__all__ = ["CodebaseIntelligence"]
