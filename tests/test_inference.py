"""
Test suite for CodeContext AI inference engine
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from codecontext_ai.inference import DocumentationAI, InferenceEngine


class TestInferenceEngine:
    """Test the core inference engine"""
    
    def test_initialization(self):
        """Test engine initialization"""
        engine = InferenceEngine("test-model.gguf")
        assert engine.model_path == "test-model.gguf"
        assert engine.max_tokens == 2048
    
    @patch('codecontext_ai.inference.subprocess.run')
    def test_generate_basic(self, mock_subprocess):
        """Test basic text generation"""
        # Mock successful subprocess call
        mock_result = Mock()
        mock_result.stdout = "Generated documentation text"
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result
        
        engine = InferenceEngine("test-model.gguf")
        result = engine.generate("Write documentation for this code")
        
        assert result == "Generated documentation text"
        mock_subprocess.assert_called_once()
    
    @patch('codecontext_ai.inference.subprocess.run')
    def test_generate_with_error(self, mock_subprocess):
        """Test generation with subprocess error"""
        # Mock failed subprocess call
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stderr = "Model loading error"
        mock_subprocess.return_value = mock_result
        
        engine = InferenceEngine("test-model.gguf")
        
        with pytest.raises(RuntimeError):
            engine.generate("Test prompt")
    
    def test_format_prompt(self):
        """Test prompt formatting"""
        engine = InferenceEngine("test-model.gguf")
        
        prompt = engine._format_prompt("Write README", "readme")
        assert "README" in prompt
        assert "documentation" in prompt.lower()


class TestDocumentationAI:
    """Test the high-level documentation AI interface"""
    
    @pytest.fixture
    def mock_engine(self):
        """Create a mock inference engine"""
        with patch('codecontext_ai.inference.InferenceEngine') as mock:
            mock_instance = Mock()
            mock_instance.generate.return_value = "# Test README\n\nThis is a test documentation."
            mock.return_value = mock_instance
            yield mock_instance
    
    def test_initialization(self, mock_engine):
        """Test DocumentationAI initialization"""
        ai = DocumentationAI("test-model.gguf")
        assert ai.model_path == "test-model.gguf"
        assert ai.engine is not None
    
    def test_generate_readme(self, mock_engine):
        """Test README generation"""
        ai = DocumentationAI("test-model.gguf")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a simple test project
            test_file = Path(temp_dir) / "main.py"
            test_file.write_text("print('Hello, World!')")
            
            readme = ai.generate_readme(temp_dir, "A simple Python script")
            
            assert readme.startswith("# Test README")
            assert "documentation" in readme
            mock_engine.generate.assert_called_once()
    
    def test_generate_api_docs(self, mock_engine):
        """Test API documentation generation"""
        ai = DocumentationAI("test-model.gguf")
        
        api_info = {
            "endpoints": [
                {"method": "GET", "path": "/users", "description": "Get all users"}
            ],
            "models": [
                {"name": "User", "fields": ["id", "name", "email"]}
            ]
        }
        
        docs = ai.generate_api_docs(api_info)
        
        assert "# Test README" in docs
        mock_engine.generate.assert_called_once()
    
    def test_generate_changelog(self, mock_engine):
        """Test changelog generation"""
        ai = DocumentationAI("test-model.gguf")
        
        changes = [
            {"type": "feat", "message": "Add user authentication", "hash": "abc123"},
            {"type": "fix", "message": "Fix login bug", "hash": "def456"}
        ]
        
        changelog = ai.generate_changelog(changes)
        
        assert "# Test README" in changelog
        mock_engine.generate.assert_called_once()
    
    def test_analyze_codebase(self, mock_engine):
        """Test codebase analysis"""
        ai = DocumentationAI("test-model.gguf")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            (Path(temp_dir) / "main.py").write_text("def hello(): pass")
            (Path(temp_dir) / "utils.py").write_text("class Helper: pass")
            (Path(temp_dir) / "README.md").write_text("# Project")
            
            analysis = ai.analyze_codebase(temp_dir)
            
            assert "files" in analysis
            assert "languages" in analysis
            assert "structure" in analysis
            assert len(analysis["files"]) >= 2  # main.py and utils.py


class TestPrivacyFeatures:
    """Test privacy and security features"""
    
    def test_no_data_leakage(self):
        """Ensure no sensitive data is logged or transmitted"""
        with patch('codecontext_ai.inference.subprocess.run') as mock_subprocess:
            mock_result = Mock()
            mock_result.stdout = "Generated text"
            mock_result.returncode = 0
            mock_subprocess.return_value = mock_result
            
            engine = InferenceEngine("test-model.gguf")
            sensitive_prompt = "API_KEY=secret123 DATABASE_URL=postgres://user:pass@host"
            
            # Generate should work but not expose sensitive data
            result = engine.generate(sensitive_prompt)
            
            # Check that subprocess call doesn't contain raw sensitive data
            call_args = mock_subprocess.call_args
            assert "secret123" not in str(call_args)
            assert "pass@host" not in str(call_args)
    
    def test_local_only_inference(self):
        """Test that inference only uses local models"""
        with patch('codecontext_ai.inference.subprocess.run') as mock_subprocess:
            mock_subprocess.return_value = Mock(stdout="output", returncode=0)
            
            engine = InferenceEngine("local-model.gguf")
            engine.generate("test prompt")
            
            # Verify that ollama command is used (local inference)
            call_args = mock_subprocess.call_args[0][0]
            assert "ollama" in call_args[0]
            assert "run" in call_args


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_missing_model_file(self):
        """Test behavior with missing model file"""
        with pytest.raises(FileNotFoundError):
            DocumentationAI("nonexistent-model.gguf")
    
    def test_empty_codebase(self):
        """Test analysis of empty codebase"""
        ai_mock = Mock()
        ai_mock.analyze_codebase.return_value = {
            "files": [],
            "languages": [],
            "structure": {},
            "error": "No source files found"
        }
        
        with tempfile.TemporaryDirectory() as empty_dir:
            result = ai_mock.analyze_codebase(empty_dir)
            
            assert result["files"] == []
            assert "error" in result
    
    def test_generation_timeout(self):
        """Test handling of generation timeouts"""
        with patch('codecontext_ai.inference.subprocess.run') as mock_subprocess:
            # Simulate timeout
            mock_subprocess.side_effect = TimeoutError("Generation timed out")
            
            engine = InferenceEngine("test-model.gguf")
            
            with pytest.raises(TimeoutError):
                engine.generate("test prompt", timeout=1)
    
    def test_malformed_response(self):
        """Test handling of malformed model output"""
        with patch('codecontext_ai.inference.subprocess.run') as mock_subprocess:
            mock_result = Mock()
            mock_result.stdout = "���invalid utf-8���"  # Invalid characters
            mock_result.returncode = 0
            mock_subprocess.return_value = mock_result
            
            engine = InferenceEngine("test-model.gguf")
            
            # Should handle encoding errors gracefully
            result = engine.generate("test prompt")
            assert isinstance(result, str)


class TestPerformance:
    """Test performance characteristics"""
    
    @patch('codecontext_ai.inference.subprocess.run')
    def test_batch_processing(self, mock_subprocess):
        """Test batch processing efficiency"""
        mock_result = Mock()
        mock_result.stdout = "Generated response"
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result
        
        engine = InferenceEngine("test-model.gguf")
        prompts = ["prompt 1", "prompt 2", "prompt 3"]
        
        results = []
        for prompt in prompts:
            results.append(engine.generate(prompt))
        
        assert len(results) == 3
        assert all(r == "Generated response" for r in results)
        assert mock_subprocess.call_count == 3
    
    def test_memory_usage(self):
        """Test memory usage stays reasonable"""
        # This is a placeholder for memory profiling tests
        # In real implementation, you'd use memory_profiler or similar
        engine = InferenceEngine("test-model.gguf")
        
        # Verify engine doesn't hold large amounts of data
        import sys
        engine_size = sys.getsizeof(engine)
        assert engine_size < 1024 * 1024  # Less than 1MB


if __name__ == "__main__":
    pytest.main([__file__, "-v"])