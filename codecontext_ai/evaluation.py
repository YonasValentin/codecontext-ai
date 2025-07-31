"""
Advanced model evaluation framework with comprehensive benchmarks
"""

import json
import time
import asyncio
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
import logging
from concurrent.futures import ThreadPoolExecutor
import statistics
import re
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import subprocess

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

logger = logging.getLogger(__name__)

@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics for documentation generation"""
    bleu_score: float
    rouge_1: float
    rouge_2: float  
    rouge_l: float
    semantic_similarity: float
    readability_score: float
    completeness_score: float
    accuracy_score: float
    fluency_score: float
    structure_score: float
    code_quality_score: float
    
@dataclass
class BenchmarkResult:
    model_name: str
    task_type: str
    metrics: EvaluationMetrics
    inference_time: float
    tokens_per_second: float
    memory_usage: float
    sample_count: int
    timestamp: str

class TextQualityAnalyzer:
    """Advanced text quality analysis using multiple metrics"""
    
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
        )
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def calculate_bleu_score(self, reference: str, candidate: str) -> float:
        """Calculate BLEU score with smoothing"""
        try:
            reference_tokens = nltk.word_tokenize(reference.lower())
            candidate_tokens = nltk.word_tokenize(candidate.lower())
            
            smoothing = SmoothingFunction().method1
            score = sentence_bleu(
                [reference_tokens], 
                candidate_tokens,
                smoothing_function=smoothing
            )
            return score
        except:
            return 0.0
    
    def calculate_rouge_scores(self, reference: str, candidate: str) -> Dict[str, float]:
        """Calculate ROUGE scores"""
        try:
            scores = self.rouge_scorer.score(reference, candidate)
            return {
                'rouge_1': scores['rouge1'].fmeasure,
                'rouge_2': scores['rouge2'].fmeasure,
                'rouge_l': scores['rougeL'].fmeasure
            }
        except:
            return {'rouge_1': 0.0, 'rouge_2': 0.0, 'rouge_l': 0.0}
    
    def calculate_semantic_similarity(self, reference: str, candidate: str) -> float:
        """Calculate semantic similarity using sentence embeddings"""
        try:
            ref_embedding = self.sentence_model.encode([reference])
            cand_embedding = self.sentence_model.encode([candidate])
            similarity = cosine_similarity(ref_embedding, cand_embedding)[0][0]
            return float(similarity)
        except:
            return 0.0
    
    def calculate_readability_score(self, text: str) -> float:
        """Calculate readability using Flesch Reading Ease approximation"""
        try:
            sentences = len(re.findall(r'[.!?]+', text))
            words = len(text.split())
            syllables = sum([self._count_syllables(word) for word in text.split()])
            
            if sentences == 0 or words == 0:
                return 0.0
            
            flesch = 206.835 - (1.015 * words / sentences) - (84.6 * syllables / words)
            return max(0.0, min(100.0, flesch)) / 100.0
        except:
            return 0.0
    
    def _count_syllables(self, word: str) -> int:
        """Approximate syllable count"""
        word = word.lower()
        count = 0
        vowels = 'aeiouy'
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                count += 1
            prev_was_vowel = is_vowel
        
        if word.endswith('e'):
            count -= 1
        
        return max(1, count)
    
    def calculate_completeness_score(self, text: str) -> float:
        """Analyze documentation completeness"""
        score = 0.0
        
        # Check for key sections
        sections = {
            'title': r'^#\s+\w+',
            'description': r'(?i)(description|overview|about)',
            'installation': r'(?i)(install|setup|getting started)',
            'usage': r'(?i)(usage|example|how to use)',
            'api': r'(?i)(api|methods|functions)',
            'license': r'(?i)license'
        }
        
        for section, pattern in sections.items():
            if re.search(pattern, text, re.MULTILINE):
                score += 1/len(sections)
        
        # Code examples bonus
        code_blocks = len(re.findall(r'```[\s\S]*?```', text))
        if code_blocks >= 1:
            score += 0.1
        if code_blocks >= 3:
            score += 0.1
        
        return min(1.0, score)
    
    def calculate_structure_score(self, text: str) -> float:
        """Analyze document structure quality"""
        score = 0.0
        
        # Header hierarchy
        headers = re.findall(r'^(#{1,6})\s+', text, re.MULTILINE)
        if headers:
            score += 0.3
            
            # Check for proper hierarchy
            levels = [len(h) for h in headers]
            if levels == sorted(levels):
                score += 0.2
        
        # Lists presence
        if re.search(r'^\s*[-\*\+]\s+', text, re.MULTILINE):
            score += 0.2
        
        # Links and references
        links = len(re.findall(r'\[.*?\]\(.*?\)', text))
        if links >= 3:
            score += 0.3
        elif links >= 1:
            score += 0.15
        
        return score
    
    def calculate_code_quality_score(self, text: str) -> float:
        """Analyze code examples quality"""
        code_blocks = re.findall(r'```(\w+)?\n(.*?)```', text, re.DOTALL)
        
        if not code_blocks:
            return 0.5  # Neutral score if no code
        
        score = 0.0
        for lang, code in code_blocks:
            # Language specification
            if lang:
                score += 0.2
            
            # Code length appropriateness
            lines = len(code.strip().split('\n'))
            if 3 <= lines <= 20:
                score += 0.3
            elif 1 <= lines <= 30:
                score += 0.2
            
            # Comments in code
            if '//' in code or '#' in code or '/*' in code:
                score += 0.2
            
            # Proper indentation
            if re.search(r'^\s+', code, re.MULTILINE):
                score += 0.1
        
        return min(1.0, score / len(code_blocks))

class ModelEvaluator:
    """Enhanced evaluation framework for CodeContext AI models with Qwen3 and Ollama support"""
    
    def __init__(self, model_path: str, model_type: str = "huggingface"):
        self.model_path = model_path
        self.model_type = model_type  # "huggingface", "ollama", or "gguf"
        self.quality_analyzer = TextQualityAnalyzer()
        self.results = []
        self.inference_engine = None
        
    async def evaluate_model(self, test_dataset: List[Dict], 
                           batch_size: int = 1) -> BenchmarkResult:
        """Evaluate model on test dataset with multi-backend support"""
        
        # Load appropriate model based on type
        if self.model_type == "ollama":
            from .ollama_integration import OllamaDocumentationAI, OllamaConfig
            config = OllamaConfig(model_name=self.model_path)
            model = OllamaDocumentationAI(config)
        elif self.model_type == "rag":
            from .rag import RAGEnhancedDocumentationAI, RAGConfig
            rag_config = RAGConfig()
            model = RAGEnhancedDocumentationAI(self.model_path, rag_config)
        else:
            from .inference import ArchitecturalGuideAI
            model = ArchitecturalGuideAI(self.model_path)
        
        # Evaluation metrics
        all_metrics = []
        inference_times = []
        memory_usage = []
        
        for i in range(0, len(test_dataset), batch_size):
            batch = test_dataset[i:i+batch_size]
            
            for sample in batch:
                start_time = time.time()
                
                # Generate prediction based on model type
                try:
                    if self.model_type == "ollama":
                        prediction = model.generate_documentation(
                            doc_type=sample['doc_type'],
                            context=sample.get('context', ''),
                            requirements=sample.get('requirements', ''),
                            thinking_mode=True
                        )
                    elif self.model_type == "rag":
                        prediction = model.generate_context_aware_documentation(
                            codebase_path=sample.get('codebase_path', './'),
                            doc_type=sample['doc_type'],
                            additional_context=sample.get('context', ''),
                            use_rag=True
                        )
                    else:
                        # Traditional evaluation
                        if sample['doc_type'] == 'readme':
                            prediction = model.generate_readme(
                                sample.get('codebase_path', ''),
                                sample.get('context', '')
                            )
                        elif sample['doc_type'] == 'api':
                            prediction = model.generate_api_docs(
                                sample.get('api_info', {})
                            )
                        elif sample['doc_type'] == 'architecture':
                            prediction = model.generate_architecture_guide(
                                sample.get('codebase_path', ''),
                                sample.get('guide_type', 'full')
                            )
                        else:
                            prediction = model.engine.generate(sample.get('context', ''))
                        
                except Exception as e:
                    logger.error(f"Generation failed: {e}")
                    prediction = ""
                
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # Calculate metrics
                reference = sample['target']
                metrics = self._calculate_comprehensive_metrics(reference, prediction)
                all_metrics.append(metrics)
                
                # Memory usage (approximate)
                if torch.cuda.is_available():
                    memory_usage.append(torch.cuda.memory_allocated() / 1024**2)  # MB
        
        # Aggregate metrics
        avg_metrics = self._aggregate_metrics(all_metrics)
        
        # Calculate performance metrics
        avg_inference_time = statistics.mean(inference_times)
        tokens_per_second = self._estimate_tokens_per_second(test_dataset, inference_times)
        avg_memory = statistics.mean(memory_usage) if memory_usage else 0.0
        
        result = BenchmarkResult(
            model_name=Path(self.model_path).name,
            task_type="documentation",
            metrics=avg_metrics,
            inference_time=avg_inference_time,
            tokens_per_second=tokens_per_second,
            memory_usage=avg_memory,
            sample_count=len(test_dataset),
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
        )
        
        self.results.append(result)
        return result
    
    def _calculate_comprehensive_metrics(self, reference: str, 
                                       candidate: str) -> EvaluationMetrics:
        """Calculate all evaluation metrics"""
        
        # Basic NLP metrics
        bleu = self.quality_analyzer.calculate_bleu_score(reference, candidate)
        rouge_scores = self.quality_analyzer.calculate_rouge_scores(reference, candidate)
        semantic_sim = self.quality_analyzer.calculate_semantic_similarity(reference, candidate)
        
        # Quality metrics
        readability = self.quality_analyzer.calculate_readability_score(candidate)
        completeness = self.quality_analyzer.calculate_completeness_score(candidate)
        structure = self.quality_analyzer.calculate_structure_score(candidate)
        code_quality = self.quality_analyzer.calculate_code_quality_score(candidate)
        
        # Derived metrics
        accuracy = (bleu + rouge_scores['rouge_l'] + semantic_sim) / 3
        fluency = (readability + structure) / 2
        
        return EvaluationMetrics(
            bleu_score=bleu,
            rouge_1=rouge_scores['rouge_1'],
            rouge_2=rouge_scores['rouge_2'],
            rouge_l=rouge_scores['rouge_l'],
            semantic_similarity=semantic_sim,
            readability_score=readability,
            completeness_score=completeness,
            accuracy_score=accuracy,
            fluency_score=fluency,
            structure_score=structure,
            code_quality_score=code_quality
        )
    
    def _aggregate_metrics(self, metrics_list: List[EvaluationMetrics]) -> EvaluationMetrics:
        """Aggregate metrics across all samples"""
        if not metrics_list:
            return EvaluationMetrics(**{field: 0.0 for field in EvaluationMetrics.__annotations__})
        
        aggregated = {}
        for field in EvaluationMetrics.__annotations__:
            values = [getattr(m, field) for m in metrics_list]
            aggregated[field] = statistics.mean(values)
        
        return EvaluationMetrics(**aggregated)
    
    def _estimate_tokens_per_second(self, dataset: List[Dict], 
                                  times: List[float]) -> float:
        """Estimate tokens per second with proper tokenizer selection"""
        try:
            # Use appropriate tokenizer based on model type
            if self.model_type == "ollama" or "qwen" in self.model_path.lower():
                # For Qwen models or Ollama, use a generic tokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    "Qwen/Qwen2-7B", 
                    trust_remote_code=True
                )
            else:
                # Fallback to CodeLlama tokenizer
                tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
            
            total_tokens = 0
            
            for sample in dataset[:len(times)]:
                target = sample.get('target', '')
                tokens = tokenizer.encode(target)
                total_tokens += len(tokens)
            
            total_time = sum(times)
            return total_tokens / total_time if total_time > 0 else 0.0
        except Exception as e:
            logger.warning(f"Token estimation failed: {e}")
            # Fallback estimation: ~4 chars per token average
            total_chars = sum(len(sample.get('target', '')) for sample in dataset[:len(times)])
            total_time = sum(times)
            return (total_chars / 4) / total_time if total_time > 0 else 0.0
    
    def compare_models(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Compare multiple model results"""
        comparison = {
            'models': [r.model_name for r in results],
            'metrics_comparison': {},
            'performance_comparison': {},
            'recommendations': []
        }
        
        # Metrics comparison
        for field in EvaluationMetrics.__annotations__:
            comparison['metrics_comparison'][field] = {
                r.model_name: getattr(r.metrics, field) for r in results
            }
        
        # Performance comparison
        comparison['performance_comparison'] = {
            'inference_time': {r.model_name: r.inference_time for r in results},
            'tokens_per_second': {r.model_name: r.tokens_per_second for r in results},
            'memory_usage': {r.model_name: r.memory_usage for r in results}
        }
        
        # Generate recommendations
        best_accuracy = max(results, key=lambda r: r.metrics.accuracy_score)
        fastest = min(results, key=lambda r: r.inference_time)
        most_complete = max(results, key=lambda r: r.metrics.completeness_score)
        
        comparison['recommendations'] = [
            f"Best accuracy: {best_accuracy.model_name} ({best_accuracy.metrics.accuracy_score:.3f})",
            f"Fastest inference: {fastest.model_name} ({fastest.inference_time:.3f}s)",
            f"Most complete outputs: {most_complete.model_name} ({most_complete.metrics.completeness_score:.3f})"
        ]
        
        return comparison
    
    def save_results(self, output_path: str):
        """Save evaluation results"""
        results_data = [asdict(result) for result in self.results]
        
        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")

class Benchmarks:
    """Standard benchmarks for documentation models"""
    
    @staticmethod
    def load_readme_benchmark() -> List[Dict]:
        """Load standard README generation benchmark"""
        # This would load a curated test set
        return [
            {
                'doc_type': 'readme',
                'context': 'JavaScript library for data visualization',
                'target': '# DataViz\n\nA modern JavaScript library...',
                'codebase_path': '',
                'metadata': {'language': 'javascript'}
            }
            # Add more benchmark samples
        ]
    
    @staticmethod
    def load_api_benchmark() -> List[Dict]:
        """Load standard API documentation benchmark"""
        return [
            {
                'doc_type': 'api',
                'api_info': {
                    'endpoints': [{'method': 'GET', 'path': '/users'}]
                },
                'target': '## GET /users\n\nRetrieve all users...',
                'metadata': {'framework': 'express'}
            }
        ]
    
    @staticmethod 
    def load_human_eval_benchmark() -> List[Dict]:
        """Load human-evaluated benchmark for quality assessment"""
        # This would contain human-rated documentation samples
        return []

# CLI interface
async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate CodeContext AI models")
    parser.add_argument("--model", required=True, help="Path to model")
    parser.add_argument("--benchmark", choices=['readme', 'api', 'all'], 
                       default='all', help="Benchmark to run")
    parser.add_argument("--output", default="evaluation_results.json", 
                       help="Output file")
    
    args = parser.parse_args()
    
    evaluator = ModelEvaluator(args.model)
    
    # Load benchmarks
    benchmarks = []
    if args.benchmark in ['readme', 'all']:
        benchmarks.extend(Benchmarks.load_readme_benchmark())
    if args.benchmark in ['api', 'all']:
        benchmarks.extend(Benchmarks.load_api_benchmark())
    
    # Run evaluation
    result = await evaluator.evaluate_model(benchmarks)
    
    # Print results
    print(f"\nEvaluation Results for {result.model_name}")
    print("=" * 50)
    print(f"BLEU Score: {result.metrics.bleu_score:.3f}")
    print(f"ROUGE-L: {result.metrics.rouge_l:.3f}")
    print(f"Semantic Similarity: {result.metrics.semantic_similarity:.3f}")
    print(f"Completeness: {result.metrics.completeness_score:.3f}")
    print(f"Structure Quality: {result.metrics.structure_score:.3f}")
    print(f"Inference Time: {result.inference_time:.3f}s")
    print(f"Tokens/Second: {result.tokens_per_second:.1f}")
    print(f"Memory Usage: {result.memory_usage:.1f}MB")
    
    # Save results
    evaluator.save_results(args.output)

if __name__ == "__main__":
    asyncio.run(main())