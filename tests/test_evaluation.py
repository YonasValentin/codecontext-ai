"""
Test suite for evaluation framework
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch

from codecontext_ai.evaluation import (
    TextQualityAnalyzer,
    ModelEvaluator,
    EvaluationMetrics,
    BenchmarkResult,
    Benchmarks
)


class TestTextQualityAnalyzer:
    """Test text quality analysis functions"""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance"""
        return TextQualityAnalyzer()
    
    def test_bleu_score_calculation(self, analyzer):
        """Test BLEU score calculation"""
        reference = "The quick brown fox jumps over the lazy dog"
        candidate = "A quick brown fox jumps over a lazy dog"
        
        score = analyzer.calculate_bleu_score(reference, candidate)
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should be reasonably high for similar sentences
    
    def test_rouge_scores_calculation(self, analyzer):
        """Test ROUGE scores calculation"""
        reference = "The documentation should be comprehensive and clear"
        candidate = "Documentation must be comprehensive and very clear"
        
        scores = analyzer.calculate_rouge_scores(reference, candidate)
        
        assert "rouge_1" in scores
        assert "rouge_2" in scores
        assert "rouge_l" in scores
        
        for score in scores.values():
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0
    
    @patch('codecontext_ai.evaluation.SentenceTransformer')
    def test_semantic_similarity(self, mock_transformer, analyzer):
        """Test semantic similarity calculation"""
        # Mock the sentence transformer
        mock_model = Mock()
        mock_model.encode.side_effect = [
            [[0.5, 0.3, 0.8]],  # reference embedding
            [[0.6, 0.4, 0.9]]   # candidate embedding
        ]
        mock_transformer.return_value = mock_model
        analyzer.sentence_model = mock_model
        
        reference = "This is a test document"
        candidate = "This document is for testing"
        
        similarity = analyzer.calculate_semantic_similarity(reference, candidate)
        
        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0
    
    def test_readability_score(self, analyzer):
        """Test readability score calculation"""
        # Simple, readable text
        simple_text = "This is easy to read. Short sentences work well."
        score_simple = analyzer.calculate_readability_score(simple_text)
        
        # Complex, hard to read text
        complex_text = "The implementation of sophisticated algorithmic paradigms necessitates comprehensive understanding of multifaceted computational complexities."
        score_complex = analyzer.calculate_readability_score(complex_text)
        
        assert isinstance(score_simple, float)
        assert isinstance(score_complex, float)
        assert 0.0 <= score_simple <= 1.0
        assert 0.0 <= score_complex <= 1.0
        # Simple text should be more readable
        assert score_simple > score_complex
    
    def test_completeness_score(self, analyzer):
        """Test documentation completeness scoring"""
        # Complete documentation
        complete_doc = """
        # Project Title
        
        ## Description
        This project does amazing things.
        
        ## Installation  
        pip install package
        
        ## Usage
        Use it like this:
        ```python
        import package
        package.run()
        ```
        
        ## API Reference
        - method1(): Does something
        - method2(): Does something else
        
        ## License
        MIT License
        """
        
        # Incomplete documentation
        incomplete_doc = "# Title\n\nSome text here."
        
        score_complete = analyzer.calculate_completeness_score(complete_doc)
        score_incomplete = analyzer.calculate_completeness_score(incomplete_doc)
        
        assert score_complete > score_incomplete
        assert score_complete > 0.8  # Should be high for complete doc
        assert score_incomplete < 0.5  # Should be low for incomplete doc
    
    def test_structure_score(self, analyzer):
        """Test document structure scoring"""
        # Well-structured document
        structured_doc = """
        # Main Title
        
        ## Section 1
        Content here.
        
        ### Subsection
        More content.
        
        - List item 1
        - List item 2
        
        [Link text](http://example.com)
        [Another link](http://test.com)
        [Third link](http://demo.com)
        """
        
        # Poorly structured document
        unstructured_doc = "Some text without any structure or formatting."
        
        score_structured = analyzer.calculate_structure_score(structured_doc)
        score_unstructured = analyzer.calculate_structure_score(unstructured_doc)
        
        assert score_structured > score_unstructured
        assert score_structured > 0.5
        assert score_unstructured < 0.3
    
    def test_code_quality_score(self, analyzer):
        """Test code examples quality scoring"""
        # Good code examples
        good_code_doc = """
        Here's how to use it:
        
        ```python
        # Import the module
        import mypackage
        
        # Create instance
        instance = mypackage.MyClass()
        
        # Use the method
        result = instance.process_data(data)
        ```
        
        ```javascript
        // JavaScript example
        const lib = require('mylib');
        lib.initialize();
        ```
        """
        
        # Poor code examples
        poor_code_doc = """
        Use it like this:
        
        ```
        x=1
        y=2
        z=x+y
        ```
        """
        
        score_good = analyzer.calculate_code_quality_score(good_code_doc)
        score_poor = analyzer.calculate_code_quality_score(poor_code_doc)
        
        assert score_good > score_poor
        assert score_good > 0.6
        assert score_poor < 0.4


class TestModelEvaluator:
    """Test model evaluation framework"""
    
    @pytest.fixture
    def mock_ai_model(self):
        """Mock DocumentationAI model"""
        with patch('codecontext_ai.evaluation.DocumentationAI') as mock:
            mock_instance = Mock()
            mock_instance.generate_readme.return_value = "# Generated README\n\nThis is generated documentation."
            mock_instance.generate_api_docs.return_value = "# API Documentation\n\nEndpoints and methods."
            mock_instance.engine.generate.return_value = "Generated content."
            mock.return_value = mock_instance
            yield mock_instance
    
    @pytest.fixture
    def test_dataset(self):
        """Create test dataset"""
        return [
            {
                "doc_type": "readme",
                "context": "A Python library for data processing",
                "target": "# DataProcessor\n\nA comprehensive library for processing data efficiently.",
                "codebase_path": "/test/path"
            },
            {
                "doc_type": "api",
                "api_info": {"endpoints": [{"method": "GET", "path": "/data"}]},
                "target": "## GET /data\n\nRetrieve data from the system.",
                "context": "API documentation"
            }
        ]
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.memory_allocated')
    def test_model_evaluation(self, mock_memory, mock_cuda, mock_ai_model, test_dataset):
        """Test complete model evaluation"""
        mock_cuda.return_value = True
        mock_memory.return_value = 1024 * 1024 * 100  # 100MB
        
        evaluator = ModelEvaluator("test-model.gguf")
        
        # Run evaluation
        result = evaluator.evaluate_model(test_dataset)
        
        assert isinstance(result, BenchmarkResult)
        assert result.model_name == "test-model.gguf"
        assert result.sample_count == 2
        assert result.inference_time > 0
        assert isinstance(result.metrics, EvaluationMetrics)
    
    def test_metrics_aggregation(self):
        """Test metrics aggregation"""
        evaluator = ModelEvaluator("test-model.gguf")
        
        # Create test metrics
        metrics_list = [
            EvaluationMetrics(
                bleu_score=0.5, rouge_1=0.6, rouge_2=0.4, rouge_l=0.55,
                semantic_similarity=0.7, readability_score=0.8, completeness_score=0.9,
                accuracy_score=0.6, fluency_score=0.75, structure_score=0.65, code_quality_score=0.7
            ),
            EvaluationMetrics(
                bleu_score=0.7, rouge_1=0.8, rouge_2=0.6, rouge_l=0.75,
                semantic_similarity=0.9, readability_score=0.6, completeness_score=0.7,
                accuracy_score=0.8, fluency_score=0.65, structure_score=0.85, code_quality_score=0.9
            )
        ]
        
        aggregated = evaluator._aggregate_metrics(metrics_list)
        
        assert aggregated.bleu_score == 0.6  # (0.5 + 0.7) / 2
        assert aggregated.rouge_1 == 0.7     # (0.6 + 0.8) / 2
        assert aggregated.completeness_score == 0.8  # (0.9 + 0.7) / 2
    
    def test_model_comparison(self):
        """Test model comparison functionality"""
        evaluator = ModelEvaluator("test-model.gguf")
        
        # Create mock results
        result1 = BenchmarkResult(
            model_name="model1", task_type="readme",
            metrics=EvaluationMetrics(
                bleu_score=0.6, rouge_1=0.7, rouge_2=0.5, rouge_l=0.65,
                semantic_similarity=0.8, readability_score=0.7, completeness_score=0.9,
                accuracy_score=0.7, fluency_score=0.75, structure_score=0.8, code_quality_score=0.6
            ),
            inference_time=2.5, tokens_per_second=50.0, memory_usage=500.0,
            sample_count=10, timestamp="2024-01-01 12:00:00"
        )
        
        result2 = BenchmarkResult(
            model_name="model2", task_type="readme", 
            metrics=EvaluationMetrics(
                bleu_score=0.8, rouge_1=0.6, rouge_2=0.7, rouge_l=0.75,
                semantic_similarity=0.7, readability_score=0.8, completeness_score=0.8,
                accuracy_score=0.75, fluency_score=0.8, structure_score=0.7, code_quality_score=0.9
            ),
            inference_time=1.8, tokens_per_second=70.0, memory_usage=400.0,
            sample_count=10, timestamp="2024-01-01 12:00:00"
        )
        
        comparison = evaluator.compare_models([result1, result2])
        
        assert "models" in comparison
        assert "metrics_comparison" in comparison
        assert "performance_comparison" in comparison
        assert "recommendations" in comparison
        
        assert len(comparison["models"]) == 2
        assert "model1" in comparison["models"]
        assert "model2" in comparison["models"]


class TestBenchmarks:
    """Test benchmark datasets"""
    
    def test_readme_benchmark_structure(self):
        """Test README benchmark data structure"""
        benchmark = Benchmarks.load_readme_benchmark()
        
        assert isinstance(benchmark, list)
        if benchmark:  # If not empty
            sample = benchmark[0]
            assert "doc_type" in sample
            assert "context" in sample
            assert "target" in sample
            assert sample["doc_type"] == "readme"
    
    def test_api_benchmark_structure(self):
        """Test API benchmark data structure"""
        benchmark = Benchmarks.load_api_benchmark()
        
        assert isinstance(benchmark, list)
        if benchmark:  # If not empty
            sample = benchmark[0]
            assert "doc_type" in sample
            assert "api_info" in sample
            assert "target" in sample
            assert sample["doc_type"] == "api"
    
    def test_human_eval_benchmark(self):
        """Test human evaluation benchmark"""
        benchmark = Benchmarks.load_human_eval_benchmark()
        
        assert isinstance(benchmark, list)
        # This benchmark is expected to be empty in the base implementation


class TestErrorHandling:
    """Test error handling in evaluation"""
    
    def test_empty_dataset_evaluation(self):
        """Test evaluation with empty dataset"""
        evaluator = ModelEvaluator("test-model.gguf")
        
        result = evaluator.evaluate_model([])
        
        assert isinstance(result, BenchmarkResult)
        assert result.sample_count == 0
    
    def test_malformed_dataset(self):
        """Test evaluation with malformed dataset"""
        evaluator = ModelEvaluator("test-model.gguf")
        
        malformed_data = [
            {"doc_type": "readme"},  # Missing required fields
            {"target": "Some text"}  # Missing doc_type
        ]
        
        # Should handle gracefully without crashing
        with patch('codecontext_ai.evaluation.DocumentationAI'):
            result = evaluator.evaluate_model(malformed_data)
            assert isinstance(result, BenchmarkResult)
    
    def test_generation_failure_handling(self):
        """Test handling of generation failures"""
        with patch('codecontext_ai.evaluation.DocumentationAI') as mock_ai:
            mock_instance = Mock()
            mock_instance.generate_readme.side_effect = Exception("Generation failed")
            mock_ai.return_value = mock_instance
            
            evaluator = ModelEvaluator("test-model.gguf")
            test_data = [{
                "doc_type": "readme",
                "context": "test",
                "target": "test target"
            }]
            
            # Should not crash, should handle the error
            result = evaluator.evaluate_model(test_data)
            assert isinstance(result, BenchmarkResult)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])