#!/usr/bin/env python3
"""
Prepare training dataset for advisory guidance models.
Creates structured training examples from code analysis patterns.
"""

import json
import re
import ast
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed


@dataclass
class AdvisoryExample:
    """Training example for advisory model."""
    file_path: str
    language: str
    code_content: str
    analysis_type: str
    function_count: int
    class_count: int
    complexity_level: str
    line_count: int
    focus_areas: str
    recommendations: str


class AdvisoryDatasetBuilder:
    """Build training dataset for advisory guidance."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Advisory patterns for different issues
        self.advisory_patterns = self._load_advisory_patterns()
    
    def build_dataset(self, 
                     code_repos: List[str], 
                     max_examples: int = 10000) -> None:
        """Build advisory training dataset from code repositories."""
        
        examples = []
        
        print(f"Processing {len(code_repos)} repositories...")
        
        for repo_path in code_repos:
            repo_examples = self._process_repository(repo_path)
            examples.extend(repo_examples)
            
            if len(examples) >= max_examples:
                break
        
        # Split into train/eval
        train_split = int(len(examples) * 0.9)
        train_examples = examples[:train_split]
        eval_examples = examples[train_split:]
        
        # Save datasets
        self._save_jsonl(train_examples, "advisory_train.jsonl")
        self._save_jsonl(eval_examples, "advisory_eval.jsonl")
        
        print(f"Generated {len(train_examples)} training examples")
        print(f"Generated {len(eval_examples)} evaluation examples")
    
    def _process_repository(self, repo_path: str) -> List[AdvisoryExample]:
        """Process single repository for advisory examples."""
        
        repo = Path(repo_path)
        if not repo.exists():
            print(f"Repository not found: {repo_path}")
            return []
        
        examples = []
        
        # Find source files
        patterns = ['**/*.py', '**/*.js', '**/*.ts', '**/*.java', '**/*.cpp']
        files = []
        
        for pattern in patterns:
            files.extend(repo.glob(pattern))
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self._analyze_file, file_path): file_path 
                for file_path in files[:100]  # Limit per repo
            }
            
            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    file_examples = future.result()
                    examples.extend(file_examples)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
        
        return examples
    
    def _analyze_file(self, file_path: Path) -> List[AdvisoryExample]:
        """Analyze single file and generate advisory examples."""
        
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            
            # Skip very large or very small files
            if len(content) < 100 or len(content) > 10000:
                return []
            
            language = self._detect_language(file_path)
            structure = self._analyze_structure(content, language)
            
            examples = []
            
            # Generate examples for different advisory types
            advisory_types = ['refactor', 'performance', 'security', 'architecture']
            
            for advisory_type in advisory_types:
                recommendations = self._generate_recommendations(
                    content, structure, language, advisory_type
                )
                
                if recommendations:
                    example = AdvisoryExample(
                        file_path=str(file_path),
                        language=language,
                        code_content=content[:3000],  # Truncate for training
                        analysis_type=advisory_type,
                        function_count=structure['functions'],
                        class_count=structure['classes'],
                        complexity_level=structure['complexity'],
                        line_count=len(content.split('\n')),
                        focus_areas=self._get_focus_areas(advisory_type),
                        recommendations=recommendations
                    )
                    examples.append(example)
            
            return examples
            
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return []
    
    def _analyze_structure(self, code: str, language: str) -> Dict:
        """Analyze code structure for metrics."""
        
        if language == 'python':
            return self._analyze_python_structure(code)
        elif language in ['javascript', 'typescript']:
            return self._analyze_js_structure(code)
        else:
            return self._analyze_generic_structure(code)
    
    def _analyze_python_structure(self, code: str) -> Dict:
        """Analyze Python code structure."""
        try:
            tree = ast.parse(code)
            
            functions = len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)])
            classes = len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)])
            
            # Complexity indicators
            max_depth = self._calculate_max_depth(tree)
            complexity = 'high' if max_depth > 4 else 'medium' if max_depth > 2 else 'low'
            
            return {
                'functions': functions,
                'classes': classes,
                'complexity': complexity,
                'imports': len([n for n in ast.walk(tree) if isinstance(n, ast.Import)])
            }
        except:
            return {'functions': 0, 'classes': 0, 'complexity': 'low', 'imports': 0}
    
    def _analyze_js_structure(self, code: str) -> Dict:
        """Analyze JavaScript structure."""
        
        # Pattern-based analysis
        function_count = len(re.findall(r'function\s+\w+|const\s+\w+\s*=.*=>', code))
        class_count = len(re.findall(r'class\s+\w+', code))
        
        # Estimate complexity from nesting
        brace_depth = 0
        max_depth = 0
        for char in code:
            if char == '{':
                brace_depth += 1
                max_depth = max(max_depth, brace_depth)
            elif char == '}':
                brace_depth -= 1
        
        complexity = 'high' if max_depth > 4 else 'medium' if max_depth > 2 else 'low'
        
        return {
            'functions': function_count,
            'classes': class_count,
            'complexity': complexity,
            'imports': len(re.findall(r'import.*from', code))
        }
    
    def _analyze_generic_structure(self, code: str) -> Dict:
        """Generic structure analysis."""
        lines = [l for l in code.split('\n') if l.strip()]
        
        return {
            'functions': len(re.findall(r'def\s+\w+|function\s+\w+', code)),
            'classes': len(re.findall(r'class\s+\w+', code)),
            'complexity': 'medium',
            'imports': 0
        }
    
    def _calculate_max_depth(self, tree) -> int:
        """Calculate maximum nesting depth."""
        
        def depth(node, current=0):
            if hasattr(node, 'body') and node.body:
                child_depths = [depth(child, current + 1) for child in node.body]
                return max(child_depths) if child_depths else current
            return current
        
        return depth(tree)
    
    def _generate_recommendations(self, 
                                code: str, 
                                structure: Dict, 
                                language: str,
                                advisory_type: str) -> str:
        """Generate synthetic recommendations based on code patterns."""
        
        patterns = self.advisory_patterns[advisory_type]
        recommendations = []
        
        # Apply pattern matching for different issues
        if advisory_type == 'refactor':
            recommendations = self._generate_refactor_recommendations(code, structure)
        elif advisory_type == 'performance':
            recommendations = self._generate_performance_recommendations(code, structure)
        elif advisory_type == 'security':
            recommendations = self._generate_security_recommendations(code, structure)
        elif advisory_type == 'architecture':
            recommendations = self._generate_architecture_recommendations(code, structure)
        
        if not recommendations:
            return ""
        
        # Format as numbered list
        formatted = []
        for i, (issue, solution, impact) in enumerate(recommendations[:3], 1):
            formatted.append(f"{i}. {issue}\n→ {solution}\n→ {impact}")
        
        return "\n\n".join(formatted)
    
    def _generate_refactor_recommendations(self, code: str, structure: Dict) -> List[Tuple[str, str, str]]:
        """Generate refactoring recommendations."""
        recommendations = []
        
        # Long functions
        lines = code.split('\n')
        if len(lines) > 50:
            recommendations.append((
                "Function length exceeds 50 lines",
                "Extract logical sections into separate functions",
                "Improves readability and testability"
            ))
        
        # Nested complexity
        if structure['complexity'] == 'high':
            recommendations.append((
                "High cyclomatic complexity detected", 
                "Reduce nesting levels using early returns or guard clauses",
                "Reduces cognitive load and bug likelihood"
            ))
        
        # Magic numbers
        if re.search(r'\b\d{2,}\b', code):
            recommendations.append((
                "Magic numbers found in code",
                "Extract numeric constants to named variables",
                "Improves code maintainability and clarity"
            ))
        
        return recommendations
    
    def _generate_performance_recommendations(self, code: str, structure: Dict) -> List[Tuple[str, str, str]]:
        """Generate performance recommendations."""
        recommendations = []
        
        # Loop optimizations
        if 'for' in code.lower() and 'append' in code.lower():
            recommendations.append((
                "Inefficient list operations in loop",
                "Consider list comprehensions or pre-allocate list size",
                "Reduces time complexity and memory allocations"
            ))
        
        # Database queries in loops
        if re.search(r'for.*query|while.*select', code, re.IGNORECASE):
            recommendations.append((
                "Potential N+1 query problem detected",
                "Batch database queries or use joins",
                "Dramatically reduces database load"
            ))
        
        return recommendations
    
    def _generate_security_recommendations(self, code: str, structure: Dict) -> List[Tuple[str, str, str]]:
        """Generate security recommendations."""
        recommendations = []
        
        # Input validation
        if re.search(r'input\(|request\.|params', code, re.IGNORECASE):
            recommendations.append((
                "User input used without validation",
                "Add input validation and sanitization",
                "Prevents injection attacks and data corruption"
            ))
        
        # Hardcoded secrets
        if re.search(r'password\s*=|api_key\s*=|secret\s*=', code, re.IGNORECASE):
            recommendations.append((
                "Potential hardcoded credentials detected",
                "Move sensitive data to environment variables",
                "Prevents credential exposure in source code"
            ))
        
        return recommendations
    
    def _generate_architecture_recommendations(self, code: str, structure: Dict) -> List[Tuple[str, str, str]]:
        """Generate architecture recommendations."""
        recommendations = []
        
        # Large classes
        if structure['functions'] > 10:
            recommendations.append((
                "Class has too many methods (>10)",
                "Consider splitting into multiple focused classes",
                "Improves maintainability and follows SRP"
            ))
        
        # Tight coupling
        if structure['imports'] > 20:
            recommendations.append((
                "High number of dependencies indicates tight coupling",
                "Review dependencies and extract common interfaces",
                "Reduces coupling and improves testability"
            ))
        
        return recommendations
    
    def _get_focus_areas(self, advisory_type: str) -> str:
        """Get focus areas for advisory type."""
        
        areas = {
            'refactor': 'code structure, readability, maintainability',
            'performance': 'algorithmic efficiency, resource usage, scalability',
            'security': 'input validation, authentication, data protection',
            'architecture': 'design patterns, coupling, extensibility'
        }
        
        return areas.get(advisory_type, 'general code quality')
    
    def _detect_language(self, file_path: Path) -> str:
        """Detect programming language from file extension."""
        
        ext_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c'
        }
        
        return ext_map.get(file_path.suffix.lower(), 'unknown')
    
    def _save_jsonl(self, examples: List[AdvisoryExample], filename: str):
        """Save examples as JSONL format."""
        
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            for example in examples:
                # Format for training
                training_data = {
                    'instruction': f"""### System:
You are a senior software architect providing code analysis guidance.
Provide structured recommendations without generating code.
Focus on actionable improvements with specific locations and impacts.

### Code Analysis Request:
Language: {example.language}
File: {Path(example.file_path).name}
Analysis Type: {example.analysis_type}

Code Structure:
- Functions: {example.function_count}
- Classes: {example.class_count}  
- Complexity: {example.complexity_level}
- Lines: {example.line_count}

### Code:
{example.code_content}

### Instructions:
Analyze the code and provide numbered recommendations in this exact format:

1. [Specific issue description with location]
→ [Actionable improvement step]
→ [Expected impact and benefit]

Focus on: {example.focus_areas}
Prioritize by impact and feasibility.

### Analysis:""",
                    'output': example.recommendations
                }
                
                f.write(json.dumps(training_data) + '\n')
        
        print(f"Saved {len(examples)} examples to {output_path}")
    
    def _load_advisory_patterns(self) -> Dict:
        """Load advisory pattern templates."""
        
        return {
            'refactor': {
                'long_function': 'Function exceeds recommended length',
                'high_complexity': 'Cyclomatic complexity too high',
                'magic_numbers': 'Magic numbers reduce readability'
            },
            'performance': {
                'n_plus_one': 'N+1 query pattern detected',
                'inefficient_loop': 'Inefficient loop operations',
                'memory_leak': 'Potential memory leak'
            },
            'security': {
                'input_validation': 'Missing input validation',
                'hardcoded_secrets': 'Hardcoded credentials',
                'sql_injection': 'SQL injection vulnerability'
            },
            'architecture': {
                'tight_coupling': 'High coupling between modules',
                'large_class': 'Class violates single responsibility',
                'missing_abstraction': 'Missing abstraction layer'
            }
        }


def main():
    parser = argparse.ArgumentParser(description='Prepare advisory training dataset')
    parser.add_argument('--repos', nargs='+', required=True,
                       help='Paths to code repositories')
    parser.add_argument('--output', default='./data',
                       help='Output directory for dataset')
    parser.add_argument('--max-examples', type=int, default=10000,
                       help='Maximum number of examples to generate')
    
    args = parser.parse_args()
    
    builder = AdvisoryDatasetBuilder(args.output)
    builder.build_dataset(args.repos, args.max_examples)


if __name__ == '__main__':
    main()