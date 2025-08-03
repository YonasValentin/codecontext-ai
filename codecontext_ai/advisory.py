"""
Advisory inference engine for CodeContext AI™ guidance system.
Provides structured recommendations without code generation.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum
import json
import ast
import re
from pathlib import Path

from .inference import InferenceEngine


class AdvisoryType(Enum):
    REFACTOR = "refactor"
    ARCHITECTURE = "architecture" 
    PERFORMANCE = "performance"
    SECURITY = "security"
    TESTING = "testing"


@dataclass
class Recommendation:
    priority: int
    category: str
    location: str
    issue: str
    solution: str
    impact: str
    complexity: str


@dataclass
class AdvisoryReport:
    file_path: str
    language: str
    recommendations: List[Recommendation]
    summary: str
    next_steps: List[str]


class AdvisoryEngine:
    """Local AI engine providing guidance without code generation."""
    
    def __init__(self, model_path: str):
        self.engine = InferenceEngine(model_path)
        self.templates = self._load_templates()
    
    def analyze_file(self, 
                    file_path: str, 
                    advisory_type: AdvisoryType = AdvisoryType.REFACTOR) -> AdvisoryReport:
        """Analyze file and provide structured guidance."""
        
        code_content = Path(file_path).read_text()
        language = self._detect_language(file_path)
        
        # Extract structural information
        structure = self._analyze_structure(code_content, language)
        
        # Generate advisory prompt
        prompt = self._build_advisory_prompt(
            code_content, structure, language, advisory_type
        )
        
        # Get AI recommendations
        response = self.engine.generate(prompt, max_tokens=512, temperature=0.3)
        
        # Parse structured output
        recommendations = self._parse_recommendations(response)
        
        return AdvisoryReport(
            file_path=file_path,
            language=language,
            recommendations=recommendations,
            summary=self._generate_summary(recommendations),
            next_steps=self._prioritize_actions(recommendations)
        )
    
    def _build_advisory_prompt(self, 
                              code: str, 
                              structure: Dict, 
                              language: str,
                              advisory_type: AdvisoryType) -> str:
        """Build advisory-focused prompt for guidance generation."""
        
        template = self.templates[advisory_type.value][language]
        
        return template.format(
            code=code[:2000],  # Truncate for context window
            functions=len(structure.get('functions', [])),
            classes=len(structure.get('classes', [])),
            complexity=structure.get('complexity_score', 'medium'),
            loc=len(code.split('\n'))
        )
    
    def _analyze_structure(self, code: str, language: str) -> Dict:
        """Extract structural patterns from code."""
        
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
            
            functions = [node.name for node in ast.walk(tree) 
                        if isinstance(node, ast.FunctionDef)]
            classes = [node.name for node in ast.walk(tree) 
                      if isinstance(node, ast.ClassDef)]
            
            # Calculate complexity indicators
            nested_depth = self._calculate_nesting_depth(tree)
            
            return {
                'functions': functions,
                'classes': classes,
                'complexity_score': 'high' if nested_depth > 4 else 'medium',
                'imports': len([n for n in ast.walk(tree) if isinstance(n, ast.Import)])
            }
        except:
            return self._analyze_generic_structure(code)
    
    def _analyze_js_structure(self, code: str) -> Dict:
        """Analyze JavaScript/TypeScript structure."""
        
        # Pattern-based analysis for JS/TS
        function_pattern = r'(?:function\s+\w+|const\s+\w+\s*=.*=>|\w+\s*\([^)]*\)\s*{)'
        class_pattern = r'class\s+\w+'
        
        functions = re.findall(function_pattern, code)
        classes = re.findall(class_pattern, code)
        
        # Estimate complexity from nesting
        brace_depth = max(code[:i].count('{') - code[:i].count('}')
                         for i in range(len(code)))
        
        return {
            'functions': functions,
            'classes': classes,
            'complexity_score': 'high' if brace_depth > 4 else 'medium',
            'imports': len(re.findall(r'import.*from', code))
        }
    
    def _analyze_generic_structure(self, code: str) -> Dict:
        """Generic structure analysis for unknown languages."""
        
        lines = code.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        
        return {
            'functions': [],
            'classes': [],
            'complexity_score': 'medium',
            'loc': len(non_empty_lines)
        }
    
    def _calculate_nesting_depth(self, tree) -> int:
        """Calculate maximum nesting depth in AST."""
        
        def depth(node, current=0):
            if hasattr(node, 'body') and node.body:
                return max(depth(child, current + 1) for child in node.body)
            return current
        
        return depth(tree)
    
    def _parse_recommendations(self, response: str) -> List[Recommendation]:
        """Parse AI response into structured recommendations."""
        
        recommendations = []
        
        # Pattern for structured recommendations
        pattern = r'(\d+)\.\s*(.*?)\n\s*→\s*(.*?)\n\s*→\s*(.*?)(?=\n\d+\.|\n💡|\Z)'
        
        matches = re.findall(pattern, response, re.DOTALL)
        
        for match in matches:
            priority, issue, solution, impact = match
            
            rec = Recommendation(
                priority=int(priority.strip()),
                category=self._classify_recommendation(issue),
                location=self._extract_location(issue),
                issue=issue.strip(),
                solution=solution.strip(),
                impact=impact.strip(),
                complexity=self._estimate_complexity(solution)
            )
            recommendations.append(rec)
        
        return recommendations
    
    def _classify_recommendation(self, issue: str) -> str:
        """Classify recommendation by category."""
        
        patterns = {
            'performance': r'(slow|performance|optimize|cache|memory)',
            'security': r'(security|vulnerable|sanitize|validation)',
            'maintainability': r'(complex|readable|maintainable|refactor)',
            'testing': r'(test|coverage|mock|assertion)',
            'architecture': r'(structure|design|pattern|coupling)'
        }
        
        for category, pattern in patterns.items():
            if re.search(pattern, issue.lower()):
                return category
        
        return 'general'
    
    def _extract_location(self, issue: str) -> str:
        """Extract location information from issue description."""
        
        # Look for line numbers, function names, etc.
        line_match = re.search(r'line[s]?\s*(\d+(?:-\d+)?)', issue.lower())
        if line_match:
            return f"lines {line_match.group(1)}"
        
        func_match = re.search(r'function\s+(\w+)', issue.lower())
        if func_match:
            return f"function {func_match.group(1)}"
        
        return "general"
    
    def _estimate_complexity(self, solution: str) -> str:
        """Estimate implementation complexity."""
        
        indicators = {
            'low': ['rename', 'move', 'extract', 'inline'],
            'medium': ['refactor', 'restructure', 'add', 'implement'],
            'high': ['redesign', 'rewrite', 'architecture', 'framework']
        }
        
        solution_lower = solution.lower()
        
        for complexity, keywords in indicators.items():
            if any(keyword in solution_lower for keyword in keywords):
                return complexity
        
        return 'medium'
    
    def _generate_summary(self, recommendations: List[Recommendation]) -> str:
        """Generate executive summary of recommendations."""
        
        if not recommendations:
            return "No significant improvements identified."
        
        high_priority = [r for r in recommendations if r.priority <= 2]
        categories = set(r.category for r in recommendations)
        
        summary = f"Found {len(recommendations)} improvement opportunities. "
        
        if high_priority:
            summary += f"{len(high_priority)} high-priority issues identified. "
        
        summary += f"Focus areas: {', '.join(categories)}."
        
        return summary
    
    def _prioritize_actions(self, recommendations: List[Recommendation]) -> List[str]:
        """Generate prioritized next steps."""
        
        if not recommendations:
            return ["Code appears well-structured."]
        
        # Sort by priority and impact
        sorted_recs = sorted(recommendations, key=lambda r: r.priority)
        
        next_steps = []
        for rec in sorted_recs[:3]:  # Top 3 priorities
            step = f"Address {rec.category} issue: {rec.issue.split('.')[0]}"
            next_steps.append(step)
        
        return next_steps
    
    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension."""
        
        ext_map = {
            '.py': 'python',
            '.js': 'javascript', 
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.go': 'go',
            '.rs': 'rust'
        }
        
        ext = Path(file_path).suffix.lower()
        return ext_map.get(ext, 'unknown')
    
    def _load_templates(self) -> Dict:
        """Load advisory prompt templates."""
        
        return {
            'refactor': {
                'python': """Analyze this Python code for refactoring opportunities.
Focus on structure, readability, and maintainability.

Code ({loc} lines, {functions} functions, {classes} classes):
{code}

Provide numbered recommendations in this format:
1. Issue description
→ Specific improvement action
→ Expected impact and benefit

Complexity: {complexity}
Be specific about locations and actionable steps.""",
                
                'javascript': """Analyze this JavaScript code for refactoring opportunities.
Focus on modern patterns, performance, and maintainability.

Code ({loc} lines, estimated complexity: {complexity}):
{code}

Provide numbered recommendations:
1. Issue description
→ Specific improvement action  
→ Expected impact and benefit

Consider ES6+, async patterns, and best practices.""",
                
                'unknown': """Analyze this code for improvement opportunities.

Code ({loc} lines):
{code}

Provide numbered recommendations:
1. Issue description
→ Improvement suggestion
→ Expected benefit"""
            },
            
            'architecture': {
                'python': """Review architectural patterns in this Python code.
Focus on design principles, coupling, and extensibility.

Code structure: {functions} functions, {classes} classes
{code}

Identify architectural improvements:
1. Design issue
→ Architectural solution
→ Long-term benefit""",
                
                'javascript': """Review architectural patterns in this JavaScript code.
Focus on modularity, separation of concerns, and scalability.

Code complexity: {complexity}
{code}

Identify architectural improvements:
1. Design issue  
→ Architectural solution
→ Long-term benefit""",
                
                'unknown': """Review code architecture and design patterns.

{code}

Suggest architectural improvements:
1. Design issue
→ Improvement approach
→ Expected benefit"""
            },
            
            'performance': {
                'python': """Analyze performance characteristics of this Python code.
Focus on algorithmic efficiency and resource usage.

Code: {functions} functions, complexity {complexity}
{code}

Performance recommendations:
1. Performance bottleneck
→ Optimization approach
→ Expected improvement""",
                
                'javascript': """Analyze performance of this JavaScript code.
Focus on runtime efficiency and memory usage.

{code}

Performance recommendations:
1. Performance issue
→ Optimization strategy
→ Expected improvement""",
                
                'unknown': """Analyze code performance characteristics.

{code}

Performance recommendations:
1. Potential bottleneck
→ Optimization approach
→ Expected benefit"""
            },
            
            'security': {
                'python': """Security review of Python code.
Focus on common vulnerabilities and secure coding practices.

{code}

Security recommendations:
1. Security concern
→ Mitigation approach
→ Risk reduction""",
                
                'javascript': """Security review of JavaScript code.
Focus on input validation, XSS, and secure patterns.

{code}

Security recommendations:
1. Security risk
→ Secure implementation
→ Protection gained""",
                
                'unknown': """Security analysis of code.

{code}

Security recommendations:
1. Potential vulnerability
→ Security improvement
→ Risk mitigation"""
            },
            
            'testing': {
                'python': """Test strategy analysis for Python code.
Focus on testability, coverage, and test design.

Code: {functions} functions, {classes} classes
{code}

Testing recommendations:
1. Testing gap
→ Test strategy
→ Quality improvement""",
                
                'javascript': """Test strategy for JavaScript code.
Focus on unit testing, mocking, and test structure.

{code}

Testing recommendations:
1. Testing opportunity
→ Test approach
→ Quality benefit""",
                
                'unknown': """Testing strategy analysis.

{code}

Testing recommendations:
1. Testing need
→ Test approach
→ Quality improvement"""
            }
        }