"""
Enhanced Ollama integration for CodeContext AI with Qwen 3 optimization
"""

import os
import json
import requests
import subprocess
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import logging
import time

logger = logging.getLogger(__name__)

@dataclass
class OllamaConfig:
    """Configuration for Ollama integration"""
    base_url: str = "http://localhost:11434"
    model_name: str = "qwen3:8b"
    context_window: int = 8192
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    timeout: int = 300
    thinking_mode: bool = False
    stream: bool = False

class OllamaModelManager:
    """Enhanced Ollama model management with Qwen 3 support"""
    
    def __init__(self, config: OllamaConfig = None):
        self.config = config or OllamaConfig()
        self.session = requests.Session()
        self.session.timeout = self.config.timeout
        
    def is_server_running(self) -> bool:
        """Check if Ollama server is running"""
        try:
            response = self.session.get(f"{self.config.base_url}/api/tags")
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def start_server(self, wait_for_ready: bool = True) -> bool:
        """Start Ollama server if not running"""
        if self.is_server_running():
            logger.info("Ollama server already running")
            return True
        
        try:
            logger.info("Starting Ollama server...")
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            if wait_for_ready:
                # Wait for server to be ready
                for _ in range(30):  # 30 second timeout
                    time.sleep(1)
                    if self.is_server_running():
                        logger.info("Ollama server started successfully")
                        return True
                
                logger.error("Ollama server failed to start within timeout")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start Ollama server: {e}")
            return False
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List available models"""
        try:
            response = self.session.get(f"{self.config.base_url}/api/tags")
            response.raise_for_status()
            return response.json().get("models", [])
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def pull_model(self, model_name: str, stream_progress: bool = True) -> bool:
        """Pull a model from Ollama registry"""
        try:
            logger.info(f"Pulling model: {model_name}")
            
            data = {"name": model_name}
            response = self.session.post(
                f"{self.config.base_url}/api/pull",
                json=data,
                stream=stream_progress
            )
            
            if stream_progress:
                for line in response.iter_lines():
                    if line:
                        try:
                            progress = json.loads(line.decode('utf-8'))
                            if progress.get("status"):
                                logger.info(f"Pull progress: {progress['status']}")
                            if progress.get("error"):
                                logger.error(f"Pull error: {progress['error']}")
                                return False
                        except json.JSONDecodeError:
                            continue
            else:
                response.raise_for_status()
            
            logger.info(f"Model {model_name} pulled successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            return False
    
    def model_exists(self, model_name: str) -> bool:
        """Check if model exists locally"""
        models = self.list_models()
        return any(model.get("name") == model_name for model in models)
    
    def ensure_model_available(self, model_name: str = None) -> bool:
        """Ensure required model is available, pull if necessary"""
        model_name = model_name or self.config.model_name
        
        if not self.is_server_running():
            if not self.start_server():
                return False
        
        if not self.model_exists(model_name):
            logger.info(f"Model {model_name} not found, pulling...")
            return self.pull_model(model_name)
        
        return True

class OllamaInferenceEngine:
    """Enhanced inference engine using Ollama with Qwen 3 optimizations"""
    
    def __init__(self, config: OllamaConfig = None):
        self.config = config or OllamaConfig()
        self.model_manager = OllamaModelManager(config)
        self.session = requests.Session()
        self.session.timeout = self.config.timeout
    
    def generate(
        self, 
        prompt: str, 
        model: str = None,
        system_prompt: str = None,
        thinking_mode: bool = None,
        stream: bool = None,
        **kwargs
    ) -> str:
        """Generate text using Ollama with Qwen 3 optimizations"""
        
        model = model or self.config.model_name
        thinking_mode = thinking_mode if thinking_mode is not None else self.config.thinking_mode
        stream = stream if stream is not None else self.config.stream
        
        # Ensure model is available
        if not self.model_manager.ensure_model_available(model):
            raise RuntimeError(f"Model {model} not available")
        
        # Apply Qwen 3 thinking mode if requested
        if thinking_mode and "/think" not in prompt:
            prompt += " /think"
        elif not thinking_mode and "/no_think" not in prompt:
            prompt += " /no_think"
        
        # Prepare request data
        data = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "num_ctx": self.config.context_window,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                **kwargs
            }
        }
        
        if system_prompt:
            data["system"] = system_prompt
        
        try:
            response = self.session.post(
                f"{self.config.base_url}/api/generate",
                json=data,
                stream=stream
            )
            response.raise_for_status()
            
            if stream:
                return self._handle_streaming_response(response)
            else:
                result = response.json()
                return result.get("response", "")
                
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise RuntimeError(f"Ollama generation failed: {e}")
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        thinking_mode: bool = None,
        stream: bool = None,
        **kwargs
    ) -> str:
        """Chat completion using Ollama"""
        
        model = model or self.config.model_name
        thinking_mode = thinking_mode if thinking_mode is not None else self.config.thinking_mode
        stream = stream if stream is not None else self.config.stream
        
        # Ensure model is available
        if not self.model_manager.ensure_model_available(model):
            raise RuntimeError(f"Model {model} not available")
        
        # Apply thinking mode to the last user message
        if thinking_mode and messages and messages[-1].get("role") == "user":
            content = messages[-1]["content"]
            if "/think" not in content:
                messages[-1]["content"] = content + " /think"
        
        data = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": {
                "num_ctx": self.config.context_window,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                **kwargs
            }
        }
        
        try:
            response = self.session.post(
                f"{self.config.base_url}/api/chat",
                json=data,
                stream=stream
            )
            response.raise_for_status()
            
            if stream:
                return self._handle_streaming_response(response)
            else:
                result = response.json()
                return result.get("message", {}).get("content", "")
                
        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            raise RuntimeError(f"Ollama chat failed: {e}")
    
    def _handle_streaming_response(self, response) -> str:
        """Handle streaming response from Ollama"""
        full_response = ""
        
        try:
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line.decode('utf-8'))
                        if chunk.get("response"):
                            full_response += chunk["response"]
                        elif chunk.get("message", {}).get("content"):
                            full_response += chunk["message"]["content"]
                        
                        if chunk.get("done", False):
                            break
                            
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            logger.error(f"Streaming response error: {e}")
            
        return full_response

class OllamaDocumentationAI:
    """Documentation AI using Ollama with RAG capabilities"""
    
    def __init__(self, config: OllamaConfig = None):
        self.config = config or OllamaConfig()
        self.inference_engine = OllamaInferenceEngine(config)
        
    def generate_documentation(
        self,
        doc_type: str,
        context: str,
        requirements: str = "",
        thinking_mode: bool = True
    ) -> str:
        """Generate documentation using Ollama with context awareness"""
        
        system_prompt = self._get_system_prompt(doc_type)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self._create_user_prompt(doc_type, context, requirements)}
        ]
        
        return self.inference_engine.chat(
            messages=messages,
            thinking_mode=thinking_mode,
            temperature=0.7,  # Balanced creativity for documentation
            top_p=0.9
        )
    
    def _get_system_prompt(self, doc_type: str) -> str:
        """Get specialized system prompt for documentation type"""
        
        system_prompts = {
            "readme": """You are an expert technical writer specializing in creating comprehensive, professional README files. 
            Focus on clarity, completeness, and user-friendliness. Include proper installation instructions, usage examples, 
            and clear project structure explanations.""",
            
            "api": """You are an expert API documentation specialist. Create clear, comprehensive API documentation 
            following OpenAPI standards. Include detailed endpoint descriptions, request/response schemas, 
            authentication methods, and practical examples.""",
            
            "architecture": """You are a senior software architect creating detailed architectural documentation. 
            Focus on system design, component relationships, data flow, technology decisions, and implementation guidance. 
            Provide both high-level overview and technical details.""",
            
            "changelog": """You are a technical writer specializing in release documentation. Create clear, 
            well-structured changelogs following semantic versioning principles. Group changes logically and 
            use user-focused language.""",
            
            "implementation": """You are a senior developer creating implementation guides. Provide step-by-step 
            instructions, code examples, best practices, and troubleshooting guidance. Focus on practical, 
            actionable content."""
        }
        
        return system_prompts.get(doc_type, "You are an expert technical writer creating professional documentation.")
    
    def _create_user_prompt(self, doc_type: str, context: str, requirements: str) -> str:
        """Create user prompt for documentation generation"""
        
        base_prompt = f"""Based on the following codebase context, generate a comprehensive {doc_type}:

CODEBASE CONTEXT:
{context}

ADDITIONAL REQUIREMENTS:
{requirements}

Please create professional, well-structured documentation that is:
- Clear and easy to understand
- Comprehensive and complete
- Properly formatted with appropriate headings
- Includes practical examples where relevant
- Follows industry best practices

{doc_type.upper()}:"""
        
        return base_prompt

def get_recommended_qwen3_models() -> Dict[str, Dict[str, Any]]:
    """Get recommended Qwen 3 models for different use cases"""
    return {
        "qwen3:8b": {
            "description": "Balanced performance and resource usage",
            "ram_required": "8GB",
            "use_cases": ["general documentation", "readme generation", "api docs"],
            "performance": "high"
        },
        "qwen3:4b": {
            "description": "Lighter model for resource-constrained environments",
            "ram_required": "4GB", 
            "use_cases": ["basic documentation", "simple guides"],
            "performance": "medium"
        },
        "qwen3:14b": {
            "description": "Higher quality for complex documentation",
            "ram_required": "16GB",
            "use_cases": ["complex architecture docs", "comprehensive guides"],
            "performance": "very high"
        },
        "qwen3:30b-a3b": {
            "description": "MoE model with excellent reasoning capabilities",
            "ram_required": "16GB",
            "use_cases": ["advanced architecture", "complex implementation guides"],
            "performance": "excellent"
        }
    }

def setup_ollama_for_codecontext(model_preference: str = "qwen3:8b") -> bool:
    """Set up Ollama with recommended models for CodeContext AI"""
    
    config = OllamaConfig(model_name=model_preference)
    manager = OllamaModelManager(config)
    
    logger.info(f"Setting up Ollama with model: {model_preference}")
    
    # Start server if needed
    if not manager.start_server():
        logger.error("Failed to start Ollama server")
        return False
    
    # Pull recommended model
    if not manager.ensure_model_available(model_preference):
        logger.error(f"Failed to set up model: {model_preference}")
        return False
    
    # Pull embedding model for RAG
    embedding_model = "nomic-embed-text"
    if not manager.ensure_model_available(embedding_model):
        logger.warning(f"Failed to pull embedding model: {embedding_model}")
    
    logger.info("Ollama setup completed successfully")
    return True