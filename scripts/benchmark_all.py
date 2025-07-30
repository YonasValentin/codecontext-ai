#!/usr/bin/env python3

import asyncio
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any
import logging
from concurrent.futures import ThreadPoolExecutor
import statistics
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from codecontext_ai.evaluation import ModelEvaluator, Benchmarks, BenchmarkResult
from codecontext_ai.inference import DocumentationAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensiveBenchmark:
    """Run comprehensive benchmarks across all models and tasks"""
    
    def __init__(self, models_dir: str, output_dir: str = "./benchmark_results"):
        self.models_dir = Path(models_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load all available models
        self.models = self._discover_models()
        
        # Load benchmark datasets
        self.benchmarks = {
            'readme': Benchmarks.load_readme_benchmark(),
            'api': Benchmarks.load_api_benchmark(),
            'human_eval': Benchmarks.load_human_eval_benchmark()
        }
        
    def _discover_models(self) -> Dict[str, str]:
        """Discover all available model files"""
        models = {}
        
        for model_file in self.models_dir.rglob("*.gguf"):
            model_name = model_file.stem
            models[model_name] = str(model_file)
            
        logger.info(f"Discovered {len(models)} models: {list(models.keys())}")
        return models
    
    async def run_all_benchmarks(self) -> Dict[str, List[BenchmarkResult]]:
        """Run comprehensive benchmarks on all models"""
        all_results = {}
        
        for model_name, model_path in self.models.items():
            logger.info(f"Benchmarking {model_name}...")
            
            evaluator = ModelEvaluator(model_path)
            model_results = []
            
            # Run on each benchmark
            for benchmark_name, dataset in self.benchmarks.items():
                if not dataset:  # Skip empty benchmarks
                    continue
                    
                logger.info(f"  Running {benchmark_name} benchmark...")
                try:
                    result = await evaluator.evaluate_model(dataset)
                    result.task_type = benchmark_name
                    model_results.append(result)
                    
                except Exception as e:
                    logger.error(f"Benchmark {benchmark_name} failed for {model_name}: {e}")
                    continue
            
            all_results[model_name] = model_results
            
        return all_results
    
    def analyze_results(self, results: Dict[str, List[BenchmarkResult]]) -> Dict[str, Any]:
        """Comprehensive analysis of benchmark results"""
        analysis = {
            'summary': {},
            'performance_rankings': {},
            'task_analysis': {},
            'recommendations': []
        }
        
        # Overall summary
        total_models = len(results)
        total_benchmarks = sum(len(model_results) for model_results in results.values())
        
        analysis['summary'] = {
            'total_models': total_models,
            'total_benchmarks': total_benchmarks,
            'benchmark_types': list(self.benchmarks.keys()),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Performance rankings
        for benchmark_name in self.benchmarks.keys():
            benchmark_results = []
            
            for model_name, model_results in results.items():
                for result in model_results:
                    if result.task_type == benchmark_name:
                        benchmark_results.append((model_name, result))
            
            if benchmark_results:
                # Rank by accuracy score
                ranked = sorted(benchmark_results, 
                              key=lambda x: x[1].metrics.accuracy_score, 
                              reverse=True)
                
                analysis['performance_rankings'][benchmark_name] = [
                    {
                        'model': model_name,
                        'accuracy': result.metrics.accuracy_score,
                        'completeness': result.metrics.completeness_score,
                        'inference_time': result.inference_time,
                        'tokens_per_second': result.tokens_per_second
                    }
                    for model_name, result in ranked
                ]
        
        # Task-specific analysis
        for benchmark_name in self.benchmarks.keys():
            task_results = []
            
            for model_results in results.values():
                for result in model_results:
                    if result.task_type == benchmark_name:
                        task_results.append(result)
            
            if task_results:
                # Calculate statistics
                accuracy_scores = [r.metrics.accuracy_score for r in task_results]
                inference_times = [r.inference_time for r in task_results]
                completeness_scores = [r.metrics.completeness_score for r in task_results]
                
                analysis['task_analysis'][benchmark_name] = {
                    'accuracy': {
                        'mean': statistics.mean(accuracy_scores),
                        'std': statistics.stdev(accuracy_scores) if len(accuracy_scores) > 1 else 0,
                        'min': min(accuracy_scores),
                        'max': max(accuracy_scores)
                    },
                    'inference_time': {
                        'mean': statistics.mean(inference_times),
                        'std': statistics.stdev(inference_times) if len(inference_times) > 1 else 0,
                        'min': min(inference_times),
                        'max': max(inference_times)
                    },
                    'completeness': {
                        'mean': statistics.mean(completeness_scores),
                        'std': statistics.stdev(completeness_scores) if len(completeness_scores) > 1 else 0,
                        'min': min(completeness_scores),
                        'max': max(completeness_scores)
                    }
                }
        
        # Generate recommendations
        recommendations = []
        
        # Best overall model
        all_results = []
        for model_name, model_results in results.items():
            avg_accuracy = statistics.mean([r.metrics.accuracy_score for r in model_results])
            all_results.append((model_name, avg_accuracy))
        
        if all_results:
            best_model = max(all_results, key=lambda x: x[1])
            recommendations.append(f"Best overall model: {best_model[0]} (avg accuracy: {best_model[1]:.3f})")
        
        # Fastest model
        all_speed_results = []
        for model_name, model_results in results.items():
            avg_speed = statistics.mean([r.inference_time for r in model_results])
            all_speed_results.append((model_name, avg_speed))
        
        if all_speed_results:
            fastest_model = min(all_speed_results, key=lambda x: x[1])
            recommendations.append(f"Fastest model: {fastest_model[0]} ({fastest_model[1]:.3f}s avg)")
        
        # Task-specific recommendations
        for benchmark_name, rankings in analysis['performance_rankings'].items():
            if rankings:
                best_for_task = rankings[0]
                recommendations.append(
                    f"Best for {benchmark_name}: {best_for_task['model']} "
                    f"(accuracy: {best_for_task['accuracy']:.3f})"
                )
        
        analysis['recommendations'] = recommendations
        
        return analysis
    
    def generate_visualizations(self, results: Dict[str, List[BenchmarkResult]], 
                              analysis: Dict[str, Any]):
        """Generate comprehensive visualizations"""
        plt.style.use('seaborn-v0_8')
        
        # 1. Accuracy comparison across models and tasks
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy heatmap
        accuracy_data = []
        models = list(results.keys())
        tasks = list(self.benchmarks.keys())
        
        for model in models:
            model_row = []
            for task in tasks:
                # Find result for this model and task
                score = 0.0
                for result in results.get(model, []):
                    if result.task_type == task:
                        score = result.metrics.accuracy_score
                        break
                model_row.append(score)
            accuracy_data.append(model_row)
        
        if accuracy_data:
            sns.heatmap(accuracy_data, 
                       xticklabels=tasks,
                       yticklabels=models,
                       annot=True, 
                       fmt='.3f',
                       cmap='YlOrRd',
                       ax=axes[0,0])
            axes[0,0].set_title('Accuracy Scores by Model and Task')
        
        # Inference time comparison
        inference_times = []
        model_names = []
        
        for model_name, model_results in results.items():
            for result in model_results:
                inference_times.append(result.inference_time)
                model_names.append(f"{model_name}\n({result.task_type})")
        
        if inference_times:
            axes[0,1].bar(range(len(inference_times)), inference_times)
            axes[0,1].set_xticks(range(len(model_names)))
            axes[0,1].set_xticklabels(model_names, rotation=45, ha='right')
            axes[0,1].set_ylabel('Inference Time (seconds)')
            axes[0,1].set_title('Inference Time by Model and Task')
        
        # Completeness vs Accuracy scatter
        completeness_scores = []
        accuracy_scores = []
        colors = []
        
        color_map = {'readme': 'red', 'api': 'blue', 'human_eval': 'green'}
        
        for model_results in results.values():
            for result in model_results:
                completeness_scores.append(result.metrics.completeness_score)
                accuracy_scores.append(result.metrics.accuracy_score)
                colors.append(color_map.get(result.task_type, 'gray'))
        
        if completeness_scores and accuracy_scores:
            axes[1,0].scatter(completeness_scores, accuracy_scores, c=colors, alpha=0.7)
            axes[1,0].set_xlabel('Completeness Score')
            axes[1,0].set_ylabel('Accuracy Score')
            axes[1,0].set_title('Completeness vs Accuracy')
            
            # Add legend
            for task, color in color_map.items():
                axes[1,0].scatter([], [], c=color, label=task)
            axes[1,0].legend()
        
        # Performance metrics comparison
        metrics_data = []
        for model_name, model_results in results.items():
            for result in model_results:
                metrics_data.append({
                    'Model': model_name,
                    'Task': result.task_type,
                    'BLEU': result.metrics.bleu_score,
                    'ROUGE-L': result.metrics.rouge_l,
                    'Semantic Sim': result.metrics.semantic_similarity,
                    'Structure': result.metrics.structure_score
                })
        
        if metrics_data:
            df = pd.DataFrame(metrics_data)
            metric_cols = ['BLEU', 'ROUGE-L', 'Semantic Sim', 'Structure']
            
            # Box plot of metrics
            df[metric_cols].boxplot(ax=axes[1,1])
            axes[1,1].set_title('Distribution of Quality Metrics')
            axes[1,1].set_ylabel('Score')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'benchmark_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {self.output_dir}/benchmark_visualization.png")
    
    def save_results(self, results: Dict[str, List[BenchmarkResult]], 
                    analysis: Dict[str, Any], output_file: str):
        """Save comprehensive results"""
        
        # Convert results to serializable format
        serializable_results = {}
        for model_name, model_results in results.items():
            serializable_results[model_name] = [
                {
                    'model_name': r.model_name,
                    'task_type': r.task_type,
                    'metrics': {
                        'bleu_score': r.metrics.bleu_score,
                        'rouge_1': r.metrics.rouge_1,
                        'rouge_2': r.metrics.rouge_2,
                        'rouge_l': r.metrics.rouge_l,
                        'semantic_similarity': r.metrics.semantic_similarity,
                        'readability_score': r.metrics.readability_score,
                        'completeness_score': r.metrics.completeness_score,
                        'accuracy_score': r.metrics.accuracy_score,
                        'fluency_score': r.metrics.fluency_score,
                        'structure_score': r.metrics.structure_score,
                        'code_quality_score': r.metrics.code_quality_score
                    },
                    'inference_time': r.inference_time,
                    'tokens_per_second': r.tokens_per_second,
                    'memory_usage': r.memory_usage,
                    'sample_count': r.sample_count,
                    'timestamp': r.timestamp
                }
                for r in model_results
            ]
        
        final_data = {
            'results': serializable_results,
            'analysis': analysis,
            'metadata': {
                'benchmark_version': '1.0.0',
                'total_models': len(results),
                'total_benchmarks': sum(len(r) for r in results.values()),
                'generated_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(final_data, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")

async def main():
    parser = argparse.ArgumentParser(description="Comprehensive CodeContext AI benchmarks")
    parser.add_argument("--models-dir", required=True, help="Directory containing model files")
    parser.add_argument("--output", default="benchmark_results.json", help="Output file")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    
    args = parser.parse_args()
    
    # Run comprehensive benchmark
    benchmark = ComprehensiveBenchmark(args.models_dir)
    
    logger.info("Starting comprehensive benchmark suite...")
    results = await benchmark.run_all_benchmarks()
    
    logger.info("Analyzing results...")
    analysis = benchmark.analyze_results(results)
    
    # Print summary
    print("\n" + "="*60)
    print("CODECONTEXT AI BENCHMARK RESULTS")
    print("="*60)
    
    print(f"\nModels tested: {analysis['summary']['total_models']}")
    print(f"Total benchmarks: {analysis['summary']['total_benchmarks']}")
    
    print("\nRECOMMENDATIONS:")
    for rec in analysis['recommendations']:
        print(f"  • {rec}")
    
    print("\nTASK ANALYSIS:")
    for task, stats in analysis['task_analysis'].items():
        print(f"  {task.upper()}:")
        print(f"    Avg Accuracy: {stats['accuracy']['mean']:.3f} ± {stats['accuracy']['std']:.3f}")
        print(f"    Avg Inference: {stats['inference_time']['mean']:.3f}s ± {stats['inference_time']['std']:.3f}s")
        print(f"    Avg Completeness: {stats['completeness']['mean']:.3f} ± {stats['completeness']['std']:.3f}")
    
    # Generate visualizations
    if args.visualize:
        logger.info("Generating visualizations...")
        benchmark.generate_visualizations(results, analysis)
    
    # Save results
    benchmark.save_results(results, analysis, args.output)
    
    logger.info("Benchmark complete!")

if __name__ == "__main__":
    asyncio.run(main())