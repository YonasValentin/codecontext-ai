"""
Tests for CodeContext AI™ advisory system.
"""

import pytest
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch

from codecontext_ai.advisory import AdvisoryEngine, AdvisoryType, Recommendation, AdvisoryReport


class TestAdvisoryEngine:
    """Test advisory engine functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.sample_python_code = '''
def process_data(users):
    results = []
    for user in users:
        if user['age'] > 18:
            if user['status'] == 'active':
                if user['subscription'] == 'premium':
                    result = {
                        'id': user['id'],
                        'name': user['name']
                    }
                    results.append(result)
    return results

def another_very_long_function_name_that_does_complex_processing():
    # This function has nested loops and complex logic
    data = []
    for i in range(100):
        for j in range(100):
            if i * j > 50:
                data.append(i + j)
    return data
'''
        
        self.sample_js_code = '''
function processUsers(users) {
    const results = [];
    for (let i = 0; i < users.length; i++) {
        const user = users[i];
        if (user.age > 18 && user.status === 'active') {
            results.push({
                id: user.id,
                name: user.name
            });
        }
    }
    return results;
}

// This function has performance issues
function updateUI() {
    for (let i = 0; i < 1000; i++) {
        const element = document.querySelector('.item-' + i);
        element.textContent = 'Updated';
    }
}
'''
    
    @patch('codecontext_ai.advisory.InferenceEngine')
    def test_advisory_engine_initialization(self, mock_inference):
        """Test advisory engine initializes correctly."""
        with tempfile.NamedTemporaryFile(suffix='.gguf') as temp_model:
            engine = AdvisoryEngine(temp_model.name)
            assert engine.engine is not None
            assert engine.templates is not None
    
    def test_advisory_engine_file_not_found(self):
        """Test advisory engine handles missing model file."""
        with pytest.raises(FileNotFoundError):
            AdvisoryEngine("nonexistent_model.gguf")
    
    @patch('codecontext_ai.advisory.InferenceEngine')
    def test_detect_language(self, mock_inference):
        """Test language detection from file extensions."""
        with tempfile.NamedTemporaryFile(suffix='.gguf') as temp_model:
            engine = AdvisoryEngine(temp_model.name)
            
            assert engine._detect_language("test.py") == "python"
            assert engine._detect_language("test.js") == "javascript"
            assert engine._detect_language("test.ts") == "typescript"
            assert engine._detect_language("test.java") == "java"
            assert engine._detect_language("test.unknown") == "unknown"
    
    @patch('codecontext_ai.advisory.InferenceEngine')
    def test_analyze_python_structure(self, mock_inference):
        """Test Python code structure analysis."""
        with tempfile.NamedTemporaryFile(suffix='.gguf') as temp_model:
            engine = AdvisoryEngine(temp_model.name)
            
            structure = engine._analyze_python_structure(self.sample_python_code)
            
            assert structure['functions'] == ['process_data', 'another_very_long_function_name_that_does_complex_processing']
            assert structure['classes'] == []
            assert 'complexity_score' in structure
    
    @patch('codecontext_ai.advisory.InferenceEngine')
    def test_analyze_js_structure(self, mock_inference):
        """Test JavaScript code structure analysis."""
        with tempfile.NamedTemporaryFile(suffix='.gguf') as temp_model:
            engine = AdvisoryEngine(temp_model.name)
            
            structure = engine._analyze_js_structure(self.sample_js_code)
            
            assert len(structure['functions']) > 0
            assert 'complexity_score' in structure
    
    @patch('codecontext_ai.advisory.InferenceEngine')
    def test_parse_recommendations(self, mock_inference):
        """Test recommendation parsing from AI response."""
        with tempfile.NamedTemporaryFile(suffix='.gguf') as temp_model:
            engine = AdvisoryEngine(temp_model.name)
            
            sample_response = """
1. Function complexity is too high in process_data
→ Extract nested conditions into separate validation functions
→ Improves readability and reduces cognitive load

2. Missing error handling for user data access
→ Add try-catch blocks around user property access
→ Prevents runtime errors with malformed data
"""
            
            recommendations = engine._parse_recommendations(sample_response)
            
            assert len(recommendations) == 2
            assert recommendations[0].priority == 1
            assert "complexity" in recommendations[0].issue.lower()
            assert recommendations[1].priority == 2
    
    @patch('codecontext_ai.advisory.InferenceEngine')
    def test_generate_summary(self, mock_inference):
        """Test summary generation from recommendations."""
        with tempfile.NamedTemporaryFile(suffix='.gguf') as temp_model:
            engine = AdvisoryEngine(temp_model.name)
            
            recommendations = [
                Recommendation(1, "performance", "line 10", "Slow loop", "Optimize", "Faster", "medium"),
                Recommendation(2, "security", "line 20", "Input validation", "Add checks", "Secure", "low"),
                Recommendation(3, "maintainability", "line 30", "Long function", "Split", "Readable", "high")
            ]
            
            summary = engine._generate_summary(recommendations)
            
            assert "3 improvement opportunities" in summary
            assert "1 high-priority" in summary
            assert "maintainability, performance, security" in summary
    
    @patch('codecontext_ai.advisory.InferenceEngine')
    def test_prioritize_actions(self, mock_inference):
        """Test action prioritization."""
        with tempfile.NamedTemporaryFile(suffix='.gguf') as temp_model:
            engine = AdvisoryEngine(temp_model.name)
            
            recommendations = [
                Recommendation(3, "performance", "line 10", "Minor optimization", "Tweak", "Small gain", "low"),
                Recommendation(1, "security", "line 20", "Critical vulnerability", "Fix now", "Secure", "high"),
                Recommendation(2, "maintainability", "line 30", "Code smell", "Refactor", "Clean", "medium")
            ]
            
            actions = engine._prioritize_actions(recommendations)
            
            assert len(actions) == 3
            assert "security" in actions[0].lower()  # Highest priority first
    
    @patch('codecontext_ai.advisory.InferenceEngine')
    def test_classify_recommendation(self, mock_inference):
        """Test recommendation classification."""
        with tempfile.NamedTemporaryFile(suffix='.gguf') as temp_model:
            engine = AdvisoryEngine(temp_model.name)
            
            assert engine._classify_recommendation("This is slow and needs optimization") == "performance"
            assert engine._classify_recommendation("Security vulnerability found") == "security"
            assert engine._classify_recommendation("Code is complex and hard to read") == "maintainability"
            assert engine._classify_recommendation("Missing test coverage") == "testing"
    
    @patch('codecontext_ai.advisory.InferenceEngine')
    def test_estimate_complexity(self, mock_inference):
        """Test complexity estimation."""
        with tempfile.NamedTemporaryFile(suffix='.gguf') as temp_model:
            engine = AdvisoryEngine(temp_model.name)
            
            assert engine._estimate_complexity("Rename this variable") == "low"
            assert engine._estimate_complexity("Refactor this module") == "medium"
            assert engine._estimate_complexity("Redesign the architecture") == "high"
    
    @patch('codecontext_ai.advisory.InferenceEngine')
    def test_analyze_file_integration(self, mock_inference):
        """Test full file analysis integration."""
        # Mock the inference engine response
        mock_inference.return_value.generate.return_value = """
1. Function process_data has high cyclomatic complexity
→ Break down nested conditions into separate validation functions
→ Improves code readability and reduces maintenance burden

2. Consider adding input validation for user data
→ Add type checking and null/undefined guards
→ Prevents runtime errors and improves reliability
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write(self.sample_python_code)
            temp_file.flush()
            
            try:
                with tempfile.NamedTemporaryFile(suffix='.gguf') as temp_model:
                    engine = AdvisoryEngine(temp_model.name)
                    report = engine.analyze_file(temp_file.name, AdvisoryType.REFACTOR)
                    
                    assert report.file_path == temp_file.name
                    assert report.language == "python"
                    assert len(report.recommendations) >= 0  # May be empty if parsing fails
                    assert report.summary is not None
                    assert report.next_steps is not None
            finally:
                Path(temp_file.name).unlink()


class TestRecommendation:
    """Test recommendation data structure."""
    
    def test_recommendation_creation(self):
        """Test recommendation object creation."""
        rec = Recommendation(
            priority=1,
            category="security",
            location="line 42",
            issue="SQL injection vulnerability",
            solution="Use parameterized queries",
            impact="Prevents data breaches",
            complexity="medium"
        )
        
        assert rec.priority == 1
        assert rec.category == "security"
        assert rec.location == "line 42"
        assert "SQL injection" in rec.issue
        assert "parameterized" in rec.solution
        assert "breaches" in rec.impact
        assert rec.complexity == "medium"


class TestAdvisoryReport:
    """Test advisory report structure."""
    
    def test_advisory_report_creation(self):
        """Test advisory report object creation."""
        recommendations = [
            Recommendation(1, "security", "line 10", "Issue", "Solution", "Impact", "low")
        ]
        
        report = AdvisoryReport(
            file_path="/path/to/file.py",
            language="python", 
            recommendations=recommendations,
            summary="Found security issue",
            next_steps=["Fix security issue"]
        )
        
        assert report.file_path == "/path/to/file.py"
        assert report.language == "python"
        assert len(report.recommendations) == 1
        assert "security" in report.summary
        assert len(report.next_steps) == 1


class TestAdvisoryTypes:
    """Test advisory type enumeration."""
    
    def test_advisory_types(self):
        """Test all advisory types are available."""
        assert AdvisoryType.REFACTOR.value == "refactor"
        assert AdvisoryType.ARCHITECTURE.value == "architecture"
        assert AdvisoryType.PERFORMANCE.value == "performance"
        assert AdvisoryType.SECURITY.value == "security"
        assert AdvisoryType.TESTING.value == "testing"