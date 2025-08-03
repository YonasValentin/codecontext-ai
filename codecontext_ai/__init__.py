"""
CodeContext AI - Privacy-first local AI for code documentation

Enhanced with Qwen 3 models, RAG capabilities, and Ollama integration.
"""

# Core inference and advisory components
try:
    from .inference import InferenceEngine, ArchitecturalGuideAI
    from .advisory import AdvisoryEngine, AdvisoryType, AdvisoryReport
    _HAS_INFERENCE = True
except ImportError:
    _HAS_INFERENCE = False
    InferenceEngine = None
    ArchitecturalGuideAI = None
    AdvisoryEngine = None
    AdvisoryType = None
    AdvisoryReport = None

# Enhanced features (optional)
try:
    from .rag import RAGEnhancedDocumentationAI, RAGConfig, LocalRAGEngine
    _HAS_RAG = True
except ImportError:
    _HAS_RAG = False
    RAGEnhancedDocumentationAI = None
    RAGConfig = None
    LocalRAGEngine = None

try:
    from .ollama_integration import (
        OllamaDocumentationAI, 
        OllamaConfig, 
        OllamaInferenceEngine,
        setup_ollama_for_codecontext
    )
    _HAS_OLLAMA = True
except ImportError:
    _HAS_OLLAMA = False
    OllamaDocumentationAI = None
    OllamaConfig = None
    OllamaInferenceEngine = None
    setup_ollama_for_codecontext = None

try:
    from .evaluation import ModelEvaluator, Benchmarks, EvaluationMetrics
    _HAS_EVALUATION = True
except ImportError:
    _HAS_EVALUATION = False
    ModelEvaluator = None
    Benchmarks = None
    EvaluationMetrics = None

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

__all__ = []

# Add core components if available
if _HAS_INFERENCE:
    __all__.extend([
        "InferenceEngine",
        "ArchitecturalGuideAI",
        "AdvisoryEngine", 
        "AdvisoryType",
        "AdvisoryReport"
    ])

# Add enhanced features if available
if _HAS_RAG:
    __all__.extend([
        "RAGEnhancedDocumentationAI",
        "RAGConfig", 
        "LocalRAGEngine"
    ])

if _HAS_OLLAMA:
    __all__.extend([
        "OllamaDocumentationAI",
        "OllamaConfig",
        "OllamaInferenceEngine",
        "setup_ollama_for_codecontext"
    ])

if _HAS_EVALUATION:
    __all__.extend([
        "ModelEvaluator",
        "Benchmarks",
        "EvaluationMetrics"
    ])

# Add datasets to exports if available
if _HAS_DATASETS:
    __all__.extend(["DocumentationDataset", "DataProcessor"])

# Legacy alias for backward compatibility
DocumentationAI = ArchitecturalGuideAI