"""
CLI interface for CodeContext AI™ guidance system.
Provides structured code analysis without generation.
"""

import click
import json
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax

from .advisory import AdvisoryEngine, AdvisoryType


console = Console()


@click.group()
def guidance():
    """CodeContext AI™ guidance system."""
    pass


@guidance.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--type', 'advisory_type', 
              type=click.Choice(['refactor', 'architecture', 'performance', 'security', 'testing']),
              default='refactor',
              help='Type of analysis to perform')
@click.option('--model', default='./models/codecontext-advisory-qwen3-8b.gguf',
              help='Path to advisory model')
@click.option('--format', type=click.Choice(['table', 'json', 'detailed']), 
              default='table',
              help='Output format')
def analyze(file_path: str, advisory_type: str, model: str, format: str):
    """Analyze file and provide improvement guidance."""
    
    if not Path(model).exists():
        console.print(f"[red]Model not found: {model}[/red]")
        console.print("Run: codecontext-ai download --model advisory")
        return
    
    console.print(f"[blue]Analyzing {file_path}...[/blue]")
    
    try:
        engine = AdvisoryEngine(model)
        report = engine.analyze_file(file_path, AdvisoryType(advisory_type))
        
        if format == 'json':
            _output_json(report)
        elif format == 'detailed':
            _output_detailed(report)
        else:
            _output_table(report)
            
    except Exception as e:
        console.print(f"[red]Analysis failed: {e}[/red]")


@guidance.command()
@click.argument('directory', type=click.Path(exists=True, file_okay=False))
@click.option('--pattern', default='*.py',
              help='File pattern to analyze')
@click.option('--type', 'advisory_type',
              type=click.Choice(['refactor', 'architecture', 'performance', 'security', 'testing']),
              default='refactor')
@click.option('--model', default='./models/codecontext-advisory-qwen3-8b.gguf')
@click.option('--threshold', type=int, default=3,
              help='Minimum priority threshold (1-5)')
def scan(directory: str, pattern: str, advisory_type: str, model: str, threshold: int):
    """Scan directory for improvement opportunities."""
    
    console.print(f"[blue]Scanning {directory} for {advisory_type} opportunities...[/blue]")
    
    dir_path = Path(directory)
    files = list(dir_path.glob(f"**/{pattern}"))
    
    if not files:
        console.print(f"[yellow]No files matching {pattern} found[/yellow]")
        return
    
    engine = AdvisoryEngine(model)
    all_recommendations = []
    
    with console.status(f"Analyzing {len(files)} files..."):
        for file_path in files:
            try:
                report = engine.analyze_file(str(file_path), AdvisoryType(advisory_type))
                high_priority = [r for r in report.recommendations if r.priority <= threshold]
                
                if high_priority:
                    all_recommendations.extend([
                        (str(file_path.relative_to(dir_path)), r) 
                        for r in high_priority
                    ])
                    
            except Exception as e:
                console.print(f"[yellow]Skipped {file_path}: {e}[/yellow]")
    
    _output_scan_results(all_recommendations, advisory_type)


@guidance.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--model', default='./models/codecontext-advisory-qwen3-8b.gguf')
def quick(file_path: str, model: str):
    """Quick analysis with top 3 recommendations."""
    
    engine = AdvisoryEngine(model)
    report = engine.analyze_file(file_path, AdvisoryType.REFACTOR)
    
    console.print(Panel(f"[bold]{Path(file_path).name}[/bold]\n{report.summary}"))
    
    if report.recommendations:
        top_3 = sorted(report.recommendations, key=lambda r: r.priority)[:3]
        
        for i, rec in enumerate(top_3, 1):
            console.print(f"\n[bold cyan]{i}. {rec.issue}[/bold cyan]")
            console.print(f"   → {rec.solution}")
            console.print(f"   → {rec.impact}")


@guidance.command()
@click.argument('config_file', type=click.Path(exists=True))
def batch(config_file: str):
    """Run batch analysis from configuration file."""
    
    with open(config_file) as f:
        config = json.load(f)
    
    model = config.get('model', './models/codecontext-advisory-qwen3-8b.gguf')
    engine = AdvisoryEngine(model)
    
    results = {}
    
    for analysis in config['analyses']:
        file_path = analysis['file']
        advisory_type = AdvisoryType(analysis.get('type', 'refactor'))
        
        console.print(f"[blue]Analyzing {file_path}...[/blue]")
        
        try:
            report = engine.analyze_file(file_path, advisory_type)
            results[file_path] = {
                'summary': report.summary,
                'recommendations_count': len(report.recommendations),
                'high_priority': len([r for r in report.recommendations if r.priority <= 2])
            }
        except Exception as e:
            results[file_path] = {'error': str(e)}
    
    # Output batch results
    table = Table(title="Batch Analysis Results")
    table.add_column("File")
    table.add_column("Summary")
    table.add_column("Issues")
    table.add_column("High Priority")
    
    for file_path, result in results.items():
        if 'error' in result:
            table.add_row(file_path, f"[red]Error: {result['error']}[/red]", "-", "-")
        else:
            table.add_row(
                file_path,
                result['summary'][:50] + "..." if len(result['summary']) > 50 else result['summary'],
                str(result['recommendations_count']),
                str(result['high_priority'])
            )
    
    console.print(table)


def _output_table(report):
    """Output recommendations as table."""
    
    table = Table(title=f"Analysis: {Path(report.file_path).name}")
    table.add_column("Priority", style="cyan", width=8)
    table.add_column("Category", style="magenta", width=12)
    table.add_column("Location", style="yellow", width=15)
    table.add_column("Issue", style="white", width=40)
    table.add_column("Solution", style="green", width=40)
    table.add_column("Impact", style="blue", width=15)
    
    for rec in sorted(report.recommendations, key=lambda r: r.priority):
        table.add_row(
            str(rec.priority),
            rec.category,
            rec.location,
            rec.issue[:37] + "..." if len(rec.issue) > 40 else rec.issue,
            rec.solution[:37] + "..." if len(rec.solution) > 40 else rec.solution,
            rec.complexity
        )
    
    console.print(table)
    
    # Summary panel
    console.print(Panel(
        f"[bold]Summary:[/bold] {report.summary}\n\n"
        f"[bold]Next Steps:[/bold]\n" + 
        "\n".join(f"• {step}" for step in report.next_steps),
        title="Recommendations"
    ))


def _output_detailed(report):
    """Output detailed recommendations."""
    
    console.print(Panel(f"[bold]File:[/bold] {report.file_path}\n"
                       f"[bold]Language:[/bold] {report.language}\n"
                       f"[bold]Summary:[/bold] {report.summary}"))
    
    for i, rec in enumerate(sorted(report.recommendations, key=lambda r: r.priority), 1):
        console.print(f"\n[bold cyan]{i}. {rec.category.upper()} - Priority {rec.priority}[/bold cyan]")
        console.print(f"[bold]Location:[/bold] {rec.location}")
        console.print(f"[bold]Issue:[/bold] {rec.issue}")
        console.print(f"[bold]Solution:[/bold] {rec.solution}")
        console.print(f"[bold]Impact:[/bold] {rec.impact}")
        console.print(f"[bold]Complexity:[/bold] {rec.complexity}")
    
    if report.next_steps:
        console.print(f"\n[bold green]Recommended Next Steps:[/bold green]")
        for step in report.next_steps:
            console.print(f"→ {step}")


def _output_json(report):
    """Output as JSON."""
    
    data = {
        'file_path': report.file_path,
        'language': report.language,
        'summary': report.summary,
        'recommendations': [
            {
                'priority': rec.priority,
                'category': rec.category,
                'location': rec.location,
                'issue': rec.issue,
                'solution': rec.solution,
                'impact': rec.impact,
                'complexity': rec.complexity
            }
            for rec in report.recommendations
        ],
        'next_steps': report.next_steps
    }
    
    console.print(json.dumps(data, indent=2))


def _output_scan_results(recommendations, advisory_type):
    """Output directory scan results."""
    
    if not recommendations:
        console.print(f"[green]No {advisory_type} issues found above threshold[/green]")
        return
    
    # Group by file
    by_file = {}
    for file_path, rec in recommendations:
        if file_path not in by_file:
            by_file[file_path] = []
        by_file[file_path].append(rec)
    
    table = Table(title=f"Directory Scan - {advisory_type.title()} Issues")
    table.add_column("File", style="cyan", width=30)
    table.add_column("Priority", style="red", width=8)
    table.add_column("Category", style="magenta", width=12)
    table.add_column("Issue", style="white", width=50)
    
    for file_path, file_recs in by_file.items():
        for i, rec in enumerate(sorted(file_recs, key=lambda r: r.priority)):
            table.add_row(
                file_path if i == 0 else "",
                str(rec.priority),
                rec.category,
                rec.issue[:47] + "..." if len(rec.issue) > 50 else rec.issue
            )
    
    console.print(table)
    
    # Summary
    total_issues = len(recommendations)
    high_priority = len([r for _, r in recommendations if r.priority <= 2])
    
    console.print(f"\n[bold]Summary:[/bold] {total_issues} issues found")
    console.print(f"[bold]High priority:[/bold] {high_priority}")
    console.print(f"[bold]Files affected:[/bold] {len(by_file)}")


if __name__ == '__main__':
    guidance()