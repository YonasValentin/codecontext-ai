"""
Privacy-focused local inference engine for CodeContext AI models
"""

import os
import json
from typing import Dict, List, Optional, Union
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass

@dataclass
class GenerationConfig:
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True

class InferenceEngine:
    """Local inference engine with privacy guarantees"""
    
    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load model and tokenizer with optimal settings"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map=self.device if self.device != "auto" else "auto",
                load_in_4bit=True if torch.cuda.is_available() else False
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {self.model_path}: {e}")
    
    def generate(self, prompt: str, config: GenerationConfig = None) -> str:
        """Generate text with privacy-first local inference"""
        if config is None:
            config = GenerationConfig()
            
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=config.max_length,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                repetition_penalty=config.repetition_penalty,
                do_sample=config.do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text[len(prompt):].strip()

class DocumentationAI:
    """High-level interface for documentation generation"""
    
    def __init__(self, model_path: str, model_type: str = "readme"):
        self.model_type = model_type
        self.engine = InferenceEngine(model_path)
        self.prompts = self._load_prompts()
    
    def _load_prompts(self) -> Dict[str, str]:
        """Load optimized prompts for different documentation types"""
        prompts_file = Path(__file__).parent / "prompts" / f"{self.model_type}.json"
        
        if prompts_file.exists():
            with open(prompts_file) as f:
                return json.load(f)
        
        # Default prompts
        return {
            "readme": "Generate a comprehensive README.md for the following codebase:\n\n{context}\n\nREADME.md:\n",
            "api": "Generate API documentation for:\n\n{context}\n\nAPI Documentation:\n",
            "changelog": "Generate a changelog based on the following commits:\n\n{context}\n\nCHANGELOG.md:\n"
        }
    
    def generate_readme(self, codebase_path: str, additional_context: str = "") -> str:
        """Generate README with codebase analysis"""
        context = self._analyze_codebase(codebase_path)
        if additional_context:
            context += f"\n\nAdditional context: {additional_context}"
            
        prompt = self.prompts["readme"].format(context=context)
        return self.engine.generate(prompt)
    
    def generate_api_docs(self, api_info: Dict) -> str:
        """Generate API documentation from structured data"""
        context = self._format_api_context(api_info)
        prompt = self.prompts["api"].format(context=context)
        return self.engine.generate(prompt)
    
    def generate_changelog(self, commits: List[Dict]) -> str:
        """Generate changelog from commit history"""
        context = self._format_commit_context(commits)
        prompt = self.prompts["changelog"].format(context=context)
        return self.engine.generate(prompt)
    
    def _analyze_codebase(self, codebase_path: str) -> str:
        """Extract relevant codebase information"""
        analysis = []
        
        # Find key files
        key_files = ["package.json", "pyproject.toml", "Cargo.toml", "go.mod"]
        for file in key_files:
            file_path = Path(codebase_path) / file
            if file_path.exists():
                with open(file_path) as f:
                    content = f.read()[:1000]  # Limit size
                    analysis.append(f"{file}:\n{content}")
        
        # Directory structure
        try:
            structure = []
            for root, dirs, files in os.walk(codebase_path):
                level = root.replace(codebase_path, '').count(os.sep)
                if level < 3:  # Limit depth
                    indent = ' ' * 2 * level
                    structure.append(f"{indent}{os.path.basename(root)}/")
                    subindent = ' ' * 2 * (level + 1)
                    for file in files[:5]:  # Limit files per directory
                        structure.append(f"{subindent}{file}")
            
            analysis.append(f"Directory structure:\n{chr(10).join(structure[:50])}")
        except Exception:
            pass
        
        return "\n\n".join(analysis)
    
    def _format_api_context(self, api_info: Dict) -> str:
        """Format API information for generation"""
        context = []
        
        if api_info.get("endpoints"):
            context.append("Endpoints:")
            for endpoint in api_info["endpoints"]:
                context.append(f"- {endpoint.get('method', 'GET')} {endpoint.get('path', '')}")
        
        if api_info.get("functions"):
            context.append("Functions:")
            for func in api_info["functions"]:
                context.append(f"- {func.get('name', '')}({func.get('params', '')})")
        
        return "\n".join(context)
    
    def _format_commit_context(self, commits: List[Dict]) -> str:
        """Format commit history for changelog generation"""
        context = []
        
        for commit in commits[:20]:  # Limit commits
            hash_short = commit.get("hash", "")[:7]
            message = commit.get("message", "")
            context.append(f"- {hash_short} {message}")
        
        return "\n".join(context)