#!/usr/bin/env python3
"""
CodeContext AI - Command Line Interface
"""

import sys
import argparse
import asyncio
from pathlib import Path
from typing import Optional

from .inference import ArchitecturalGuideAI
from .evaluation import ModelEvaluator, Benchmarks


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="CodeContext AI - Privacy-first documentation generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  codecontext-ai generate readme /path/to/code
  codecontext-ai evaluate --model my-model.gguf --benchmark readme
  codecontext-ai benchmark --models-dir ./models
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate guides and documentation')
    generate_parser.add_argument('type', choices=['readme', 'api', 'changelog', 'architecture', 'implementation', 'component', 'best-practices'],
                                help='Type of guide to generate')
    generate_parser.add_argument('path', help='Path to codebase or input file')
    generate_parser.add_argument('--model', help='Path to GGUF model file')
    generate_parser.add_argument('--output', help='Output file path')
    generate_parser.add_argument('--difficulty', choices=['easy', 'medium', 'hard'], default='medium',
                                help='Implementation difficulty level')
    generate_parser.add_argument('--framework', default='react', help='Target framework for component guides')
    generate_parser.add_argument('--project-type', default='web', help='Project type for best practices')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model performance')
    eval_parser.add_argument('--model', required=True, help='Path to model file')
    eval_parser.add_argument('--benchmark', choices=['readme', 'api', 'all'], 
                            default='all', help='Benchmark to run')
    eval_parser.add_argument('--output', default='evaluation_results.json',
                            help='Output file for results')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run comprehensive benchmarks')
    benchmark_parser.add_argument('--models-dir', required=True,
                                 help='Directory containing model files')
    benchmark_parser.add_argument('--output', default='benchmark_results.json',
                                 help='Output file for results')
    benchmark_parser.add_argument('--visualize', action='store_true',
                                 help='Generate visualization charts')
    
    # Version command
    version_parser = subparsers.add_parser('version', help='Show version information')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == 'generate':
            return handle_generate(args)
        elif args.command == 'evaluate':
            return asyncio.run(handle_evaluate(args))
        elif args.command == 'benchmark':
            return asyncio.run(handle_benchmark(args))
        elif args.command == 'version':
            return handle_version()
        else:
            print(f"Unknown command: {args.command}")
            return 1
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


def handle_generate(args) -> int:
    """Handle documentation generation"""
    if not args.model:
        print("Error: --model is required for generation")
        return 1
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found: {args.model}")
        return 1
    
    print(f"Loading model: {args.model}")
    ai = ArchitecturalGuideAI(str(model_path))
    
    print(f"Generating {args.type} guide for: {args.path}")
    
    try:
        if args.type == 'readme':
            result = ai.generate_readme(args.path)
        elif args.type == 'api':
            api_info = {"endpoints": [], "models": []}
            result = ai.generate_api_docs(api_info)
        elif args.type == 'changelog':
            changes = []
            result = ai.generate_changelog(changes)
        elif args.type == 'architecture':
            result = ai.generate_architecture_guide(args.path)
        elif args.type == 'implementation':
            requirements = {"features": [], "tech_stack": []}
            result = ai.generate_implementation_guide(requirements, args.difficulty)
        elif args.type == 'component':
            component_info = {"type": "functional", "props": []}
            result = ai.generate_component_guide(component_info, args.framework)
        elif args.type == 'best-practices':
            tech_stack = []
            result = ai.generate_best_practices_guide(tech_stack, args.project_type)
        else:
            print(f"Unsupported guide type: {args.type}")
            return 1
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(result)
            print(f"Documentation saved to: {args.output}")
        else:
            print("\n" + "="*50)
            print("GENERATED DOCUMENTATION")
            print("="*50)
            print(result)
        
        return 0
        
    except Exception as e:
        print(f"Generation failed: {e}")
        return 1


async def handle_evaluate(args) -> int:
    """Handle model evaluation"""
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found: {args.model}")
        return 1
    
    print(f"Evaluating model: {args.model}")
    evaluator = ModelEvaluator(str(model_path))
    
    # Load benchmarks
    benchmarks = []
    if args.benchmark in ['readme', 'all']:
        benchmarks.extend(Benchmarks.load_readme_benchmark())
    if args.benchmark in ['api', 'all']:
        benchmarks.extend(Benchmarks.load_api_benchmark())
    
    if not benchmarks:
        print("Warning: No benchmark data available")
        return 1
    
    print(f"Running evaluation on {len(benchmarks)} samples...")
    result = await evaluator.evaluate_model(benchmarks)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Model: {result.model_name}")
    print(f"Samples: {result.sample_count}")
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
    print(f"\nDetailed results saved to: {args.output}")
    
    return 0


async def handle_benchmark(args) -> int:
    """Handle comprehensive benchmarking"""
    from .scripts.benchmark_all import ComprehensiveBenchmark
    
    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        print(f"Error: Models directory not found: {args.models_dir}")
        return 1
    
    print(f"Running comprehensive benchmarks on models in: {args.models_dir}")
    
    benchmark = ComprehensiveBenchmark(str(models_dir))
    
    print("Starting benchmark suite...")
    results = await benchmark.run_all_benchmarks()
    
    print("Analyzing results...")
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
    
    # Generate visualizations if requested
    if args.visualize:
        print("\nGenerating visualizations...")
        benchmark.generate_visualizations(results, analysis)
    
    # Save results
    benchmark.save_results(results, analysis, args.output)
    print(f"\nComplete results saved to: {args.output}")
    
    return 0


def handle_version() -> int:
    """Handle version command"""
    print("CodeContext AI v1.0.0")
    print("Privacy-first AI models for code documentation")
    print("GitHub: https://github.com/codecontext/codecontext-ai")
    return 0


if __name__ == "__main__":
    sys.exit(main())