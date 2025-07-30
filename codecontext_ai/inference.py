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

class ArchitecturalGuideAI:
    """Production-ready architectural guide and documentation generation"""
    
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
        
        return {
            "readme": "Generate a comprehensive README.md for the following codebase:\n\n{context}\n\nREADME.md:\n",
            "api": "Generate API documentation for:\n\n{context}\n\nAPI Documentation:\n",
            "changelog": "Generate a changelog based on the following commits:\n\n{context}\n\nCHANGELOG.md:\n",
            "architecture": "Generate professional architecture guide:\n\n{context}\n\nArchitecture Guide:\n",
            "implementation": "Generate step-by-step implementation guide:\n\n{context}\n\nImplementation Guide:\n",
            "component": "Generate component architecture guide:\n\n{context}\n\nComponent Guide:\n",
            "best_practices": "Generate best practices guide:\n\n{context}\n\nBest Practices:\n"
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
    
    def generate_architecture_guide(self, codebase_path: str, guide_type: str = "full") -> str:
        """Generate architecture guide with project analysis"""
        context = self._analyze_architecture(codebase_path, guide_type)
        return self.engine.generate(self.prompts["architecture"].format(context=context))
    
    def generate_implementation_guide(self, requirements: Dict, difficulty: str = "medium") -> str:
        """Generate implementation guide from requirements"""
        context = self._format_requirements(requirements, difficulty)
        return self.engine.generate(self.prompts["implementation"].format(context=context))
    
    def generate_component_guide(self, component_info: Dict, framework: str = "react") -> str:
        """Generate component guide for framework"""
        context = self._format_component_info(component_info, framework)
        return self.engine.generate(self.prompts["component"].format(context=context))
    
    def generate_best_practices_guide(self, tech_stack: List[str], project_type: str = "web") -> str:
        """Generate best practices for tech stack"""
        context = self._format_tech_stack(tech_stack, project_type)
        return self.engine.generate(self.prompts["best_practices"].format(context=context))
    
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
    
    def _analyze_architecture(self, codebase_path: str, guide_type: str) -> str:
        """Analyze project architecture with priority assessment"""
        components = [
            self._get_project_structure(codebase_path),
            self._detect_tech_stack(codebase_path),
            self._analyze_dependencies(codebase_path)
        ]
        return "\n\n".join(filter(None, components))
    
    def _format_requirements(self, requirements: Dict, difficulty: str) -> str:
        """Format requirements with priority indicators"""
        lines = [f"Difficulty: {difficulty.upper()}"]
        
        if features := requirements.get("features"):
            lines.append("Features:")
            lines.extend(f"- [{f.get('priority', 'MEDIUM')}] {f.get('name')}: {f.get('description')}" 
                        for f in features)
        
        if stack := requirements.get("tech_stack"):
            lines.append(f"Stack: {', '.join(stack)}")
            
        return "\n".join(lines)
    
    def _format_component_info(self, component_info: Dict, framework: str) -> str:
        """Format component information for guide generation"""
        lines = [f"Framework: {framework}", f"Type: {component_info.get('type', 'functional')}"]
        
        if props := component_info.get("props"):
            lines.append("Props:")
            lines.extend(f"- {p.get('name')}: {p.get('type', 'string')} {'(required)' if p.get('required') else '(optional)'}"
                        for p in props)
                        
        return "\n".join(lines)
    
    def _format_tech_stack(self, tech_stack: List[str], project_type: str) -> str:
        """Categorize and format technology stack"""
        categories = {
            'frontend': ['react', 'vue', 'angular', 'svelte', 'nextjs'],
            'backend': ['nodejs', 'python', 'nestjs', 'fastapi', 'django'],
            'database': ['postgresql', 'mongodb', 'redis', 'supabase']
        }
        
        lines = [f"Project: {project_type}", f"Stack: {', '.join(tech_stack)}"]
        
        for category, techs in categories.items():
            if matches := [t for t in tech_stack if t.lower() in techs]:
                lines.append(f"{category.title()}: {', '.join(matches)}")
                
        return "\n".join(lines)
    
    def _get_project_structure(self, codebase_path: str) -> str:
        """Analyze project structure with priority indicators"""
        structure = []
        priority_folders = {'src', 'app', 'components', 'pages', 'api', 'lib', 'utils'}
        
        try:
            for root, dirs, files in os.walk(codebase_path):
                level = root.replace(codebase_path, '').count(os.sep)
                if level >= 3: continue
                
                folder = os.path.basename(root)
                priority = "HIGH" if folder.lower() in priority_folders else "MEDIUM" if any(f.endswith(('.ts', '.tsx', '.js', '.jsx', '.py')) for f in files) else "LOW"
                
                indent = '  ' * level
                structure.append(f"{indent}{folder}/ ({priority})")
                
                for file in files[:8]:
                    file_priority = "HIGH" if file in {'package.json', 'tsconfig.json', 'index.tsx'} else "MEDIUM" if file.endswith(('.ts', '.tsx', '.js', '.jsx', '.py')) else "LOW"
                    structure.append(f"{indent}  {file} ({file_priority})")
                    
        except Exception:
            return "Structure analysis failed"
            
        return "\n".join(structure[:80])
    
    def _detect_tech_stack(self, codebase_path: str) -> str:
        """Detect frameworks and generate dependency insights"""
        detected = set()
        
        package_json = Path(codebase_path) / "package.json"
        if package_json.exists():
            try:
                with open(package_json) as f:
                    data = json.load(f)
                    deps = {**data.get("dependencies", {}), **data.get("devDependencies", {})}
                    
                tech_map = {
                    'react': ['react', '@types/react'], 'nextjs': ['next'], 'typescript': ['typescript'],
                    'tailwind': ['tailwindcss'], 'prisma': ['prisma'], 'supabase': ['@supabase/supabase-js']
                }
                
                for tech, indicators in tech_map.items():
                    if any(ind in deps for ind in indicators):
                        detected.add(tech)
            except: pass
                        
        return f"Detected: {', '.join(detected)}" if detected else ""
    
    def _analyze_dependencies(self, codebase_path: str) -> str:
        """Analyze key dependencies with purposes"""
        package_json = Path(codebase_path) / "package.json"
        if not package_json.exists():
            return ""
            
        try:
            with open(package_json) as f:
                deps = json.load(f).get("dependencies", {})
                
            key_deps = {
                'react': 'UI Framework', 'next': 'Full-stack Framework', 'typescript': 'Type Safety',
                'tailwindcss': 'CSS Framework', '@supabase/supabase-js': 'Backend Service',
                'prisma': 'Database ORM', 'zustand': 'State Management'
            }
            
            found = [f"- {dep}: {purpose}" for dep, purpose in key_deps.items() if dep in deps]
            return "Dependencies:\n" + "\n".join(found) if found else ""
            
        except:
            return ""