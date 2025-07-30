"""
CodeContext AI - Privacy-focused AI models for code documentation
"""

from .inference import DocumentationAI, InferenceEngine
from .datasets import DocumentationDataset, DataProcessor
from .evaluation import ModelEvaluator, Benchmarks

__version__ = "0.1.0"
__author__ = "CodeContext Team"

__all__ = [
    "DocumentationAI",
    "InferenceEngine", 
    "DocumentationDataset",
    "DataProcessor",
    "ModelEvaluator",
    "Benchmarks"
]