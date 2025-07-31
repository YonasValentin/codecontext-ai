"""
CodeContext AI - Privacy-first local AI for code documentation

Enhanced with Qwen 3 models, RAG capabilities, and Ollama integration.
"""

from .inference import InferenceEngine, ArchitecturalGuideAI
from .rag import RAGEnhancedDocumentationAI, RAGConfig, LocalRAGEngine
from .ollama_integration import (
    OllamaDocumentationAI, 
    OllamaConfig, 
    OllamaInferenceEngine,
    setup_ollama_for_codecontext
)
from .evaluation import ModelEvaluator, Benchmarks, EvaluationMetrics

# Backward compatibility
try:
    from .datasets import DocumentationDataset, DataProcessor
    _HAS_DATASETS = True
except ImportError:
    _HAS_DATASETS = False
    DocumentationDataset = None
    DataProcessor = None

__version__ = "2.0.0"
__author__ = "CodeContext Team"

__all__ = [
    "InferenceEngine",
    "ArchitecturalGuideAI", 
    "RAGEnhancedDocumentationAI",
    "RAGConfig",
    "LocalRAGEngine",
    "OllamaDocumentationAI",
    "OllamaConfig",
    "OllamaInferenceEngine",
    "setup_ollama_for_codecontext",
    "ModelEvaluator",
    "Benchmarks",
    "EvaluationMetrics"
]

# Add datasets to exports if available
if _HAS_DATASETS:
    __all__.extend(["DocumentationDataset", "DataProcessor"])

# Legacy alias for backward compatibility
DocumentationAI = ArchitecturalGuideAI