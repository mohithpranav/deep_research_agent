"""
Core components for the Deep Researcher Agent
"""
from .document_manager import DocumentManager
from .text_processor import TextProcessor

# Try to import embedding engine, but don't fail if dependencies are missing
try:
    from .embedding_engine import EmbeddingEngine
    EMBEDDING_AVAILABLE = True
except ImportError:
    EmbeddingEngine = None
    EMBEDDING_AVAILABLE = False

__all__ = ['DocumentManager', 'TextProcessor']
if EMBEDDING_AVAILABLE:
    __all__.append('EmbeddingEngine')