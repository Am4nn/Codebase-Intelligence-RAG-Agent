"""Data ingestion pipeline: load → parse → chunk"""
from .loader import CodebaseLoader
from .parser import CodeParser
from .chunker import SemanticChunker

__all__ = ["CodebaseLoader", "CodeParser", "SemanticChunker"]
