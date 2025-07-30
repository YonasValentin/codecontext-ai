#!/usr/bin/env python3

import os
import json
import asyncio
import aiohttp
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import subprocess
import tempfile
import shutil
import re
from concurrent.futures import ThreadPoolExecutor
import hashlib
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RepoMetadata:
    owner: str
    name: str
    stars: int
    language: str
    size: int
    readme_score: float
    api_docs_present: bool
    changelog_present: bool
    has_tests: bool
    doc_quality_score: float

@dataclass
class DocumentationSample:
    repo_id: str
    doc_type: str  # readme, api, changelog, docstring
    context: str   # codebase context
    target: str    # expected documentation
    quality_score: float
    metadata: Dict

class GitHubCrawler:
    """High-performance GitHub repository crawler with intelligent filtering"""
    
    def __init__(self, token: str, max_concurrent: int = 10):
        self.token = token
        self.max_concurrent = max_concurrent
        self.session = None
        
    async def __aenter__(self):
        connector = aiohttp.TCPConnector(limit=self.max_concurrent)
        self.session = aiohttp.ClientSession(
            connector=connector,
            headers={'Authorization': f'token {self.token}'},
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def search_repositories(self, language: str, min_stars: int = 100) -> List[Dict]:
        """Search for high-quality repositories with excellent documentation"""
        query = f"language:{language} stars:>{min_stars} size:<10000"
        url = f"https://api.github.com/search/repositories?q={query}&sort=stars&per_page=100"
        
        repos = []
        page = 1
        
        while len(repos) < 1000 and page <= 10:  # Limit to prevent rate limits
            try:
                async with self.session.get(f"{url}&page={page}") as response:
                    if response.status != 200:
                        logger.error(f"GitHub API error: {response.status}")
                        break
                        
                    data = await response.json()
                    if not data.get('items'):
                        break
                        
                    repos.extend(data['items'])
                    page += 1
                    
            except Exception as e:
                logger.error(f"Error fetching repositories: {e}")
                break
                
        return repos
    
    async def get_repository_content(self, owner: str, repo: str, path: str = "") -> Optional[Dict]:
        """Fetch repository content with error handling"""
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
        
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    return await response.json()
        except Exception as e:
            logger.debug(f"Error fetching {owner}/{repo}/{path}: {e}")
            
        return None

class DocumentationQualityAnalyzer:
    """Advanced documentation quality analysis using NLP and heuristics"""
    
    @staticmethod
    def analyze_readme_quality(content: str, repo_metadata: Dict) -> float:
        """Comprehensive README quality scoring (0-1)"""
        score = 0.0
        
        # Length appropriateness (optimal 1000-3000 chars)
        length = len(content)
        if 1000 <= length <= 3000:
            score += 0.2
        elif 500 <= length <= 5000:
            score += 0.1
        
        # Required sections
        sections = {
            'installation': r'(?i)(install|setup|getting started)',
            'usage': r'(?i)(usage|example|how to)',
            'api': r'(?i)(api|documentation|docs)',
            'contributing': r'(?i)(contribut|develop)',
            'license': r'(?i)license'
        }
        
        for section, pattern in sections.items():
            if re.search(pattern, content):
                score += 0.1
        
        # Code examples presence
        code_blocks = len(re.findall(r'```[\s\S]*?```', content))
        if code_blocks >= 2:
            score += 0.15
        elif code_blocks >= 1:
            score += 0.1
        
        # Badge presence (indicates maintenance)
        badges = len(re.findall(r'!\[.*?\]\(.*?\)', content))
        if badges >= 3:
            score += 0.1
        elif badges >= 1:
            score += 0.05
        
        # Link quality
        links = len(re.findall(r'\[.*?\]\(.*?\)', content))
        if links >= 5:
            score += 0.1
        
        # Repository stars factor
        stars = repo_metadata.get('stargazers_count', 0)
        if stars >= 1000:
            score += 0.15
        elif stars >= 100:
            score += 0.1
        
        return min(1.0, score)
    
    @staticmethod
    def extract_context_from_repo(files: List[Dict], language: str) -> str:
        """Extract relevant context from repository files"""
        context_parts = []
        
        # Package files
        package_files = {
            'javascript': ['package.json'],
            'python': ['pyproject.toml', 'setup.py', 'requirements.txt'],
            'rust': ['Cargo.toml'],
            'go': ['go.mod'],
            'java': ['pom.xml', 'build.gradle']
        }
        
        target_files = package_files.get(language.lower(), [])
        
        for file_data in files:
            if file_data['name'] in target_files:
                try:
                    content = file_data.get('content', '')
                    if file_data.get('encoding') == 'base64':
                        import base64
                        content = base64.b64decode(content).decode('utf-8')
                    context_parts.append(f"{file_data['name']}:\n{content[:1000]}")
                except:
                    continue
        
        # Directory structure
        structure = []
        for file_data in files[:20]:  # Limit to first 20 files
            if file_data['type'] == 'file':
                structure.append(file_data['name'])
        
        if structure:
            context_parts.append(f"Project structure:\n{chr(10).join(structure)}")
        
        return "\n\n".join(context_parts)

class DatasetBuilder:
    """Build high-quality training datasets for documentation models"""
    
    def __init__(self, github_token: str, output_dir: str = "./data"):
        self.github_token = github_token
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.quality_analyzer = DocumentationQualityAnalyzer()
        
    async def build_readme_dataset(self, target_samples: int = 10000) -> str:
        """Build comprehensive README generation dataset"""
        logger.info(f"Building README dataset with {target_samples} samples")
        
        samples = []
        languages = ['javascript', 'python', 'go', 'rust', 'java', 'typescript']
        
        async with GitHubCrawler(self.github_token) as crawler:
            for language in languages:
                lang_samples = target_samples // len(languages)
                logger.info(f"Collecting {lang_samples} samples for {language}")
                
                repos = await crawler.search_repositories(language, min_stars=50)
                
                for repo in repos[:lang_samples * 2]:  # Get 2x to filter
                    try:
                        owner = repo['owner']['login']
                        name = repo['name']
                        
                        # Get repository contents
                        contents = await crawler.get_repository_content(owner, name)
                        if not contents:
                            continue
                        
                        # Find README
                        readme_file = None
                        for item in contents:
                            if item['name'].lower().startswith('readme'):
                                readme_content = await crawler.get_repository_content(
                                    owner, name, item['name']
                                )
                                if readme_content:
                                    readme_file = readme_content
                                    break
                        
                        if not readme_file:
                            continue
                        
                        # Decode README content
                        try:
                            import base64
                            readme_text = base64.b64decode(
                                readme_file['content']
                            ).decode('utf-8')
                        except:
                            continue
                        
                        # Quality check
                        quality_score = self.quality_analyzer.analyze_readme_quality(
                            readme_text, repo
                        )
                        
                        if quality_score < 0.6:  # Only high-quality samples
                            continue
                        
                        # Extract context
                        context = self.quality_analyzer.extract_context_from_repo(
                            contents, language
                        )
                        
                        sample = DocumentationSample(
                            repo_id=f"{owner}/{name}",
                            doc_type="readme",
                            context=context,
                            target=readme_text,
                            quality_score=quality_score,
                            metadata={
                                'language': language,
                                'stars': repo.get('stargazers_count', 0),
                                'size': repo.get('size', 0)
                            }
                        )
                        
                        samples.append(sample)
                        
                        if len(samples) >= target_samples:
                            break
                            
                    except Exception as e:
                        logger.debug(f"Error processing {repo.get('full_name', 'unknown')}: {e}")
                        continue
        
        # Save dataset
        dataset_file = self.output_dir / "readme_dataset.jsonl"
        with open(dataset_file, 'w') as f:
            for sample in samples:
                f.write(json.dumps(asdict(sample)) + '\n')
        
        logger.info(f"Created README dataset with {len(samples)} samples")
        return str(dataset_file)
    
    async def build_api_dataset(self, target_samples: int = 5000) -> str:
        """Build API documentation dataset from OpenAPI specs and code"""
        logger.info(f"Building API dataset with {target_samples} samples")
        
        samples = []
        
        # Search for repositories with OpenAPI specs
        async with GitHubCrawler(self.github_token) as crawler:
            query_terms = [
                "openapi.yaml", "swagger.yaml", "api.yaml",
                "OpenAPI", "FastAPI", "Express", "Flask"
            ]
            
            for term in query_terms:
                repos = await crawler.search_repositories("", min_stars=20)
                
                for repo in repos[:target_samples // len(query_terms) * 2]:
                    try:
                        owner = repo['owner']['login']
                        name = repo['name']
                        
                        # Look for API documentation
                        contents = await crawler.get_repository_content(owner, name)
                        api_docs = []
                        
                        for item in contents:
                            if any(api_term in item['name'].lower() 
                                  for api_term in ['api', 'docs', 'swagger', 'openapi']):
                                if item['type'] == 'dir':
                                    # Check directory contents
                                    dir_contents = await crawler.get_repository_content(
                                        owner, name, item['name']
                                    )
                                    if dir_contents:
                                        api_docs.extend(dir_contents)
                                else:
                                    api_docs.append(item)
                        
                        if not api_docs:
                            continue
                        
                        # Process API documentation
                        for doc in api_docs[:2]:  # Limit per repo
                            try:
                                content = await crawler.get_repository_content(
                                    owner, name, doc['path']
                                )
                                if not content:
                                    continue
                                
                                import base64
                                doc_text = base64.b64decode(
                                    content['content']
                                ).decode('utf-8')
                                
                                # Extract context from codebase
                                context = self.quality_analyzer.extract_context_from_repo(
                                    contents, repo.get('language', 'unknown')
                                )
                                
                                sample = DocumentationSample(
                                    repo_id=f"{owner}/{name}",
                                    doc_type="api",
                                    context=context,
                                    target=doc_text,
                                    quality_score=0.8,  # Assume good quality for API docs
                                    metadata={
                                        'language': repo.get('language', 'unknown'),
                                        'stars': repo.get('stargazers_count', 0),
                                        'file_name': doc['name']
                                    }
                                )
                                
                                samples.append(sample)
                                
                            except Exception as e:
                                logger.debug(f"Error processing API doc: {e}")
                                continue
                        
                        if len(samples) >= target_samples:
                            break
                            
                    except Exception as e:
                        logger.debug(f"Error processing repo: {e}")
                        continue
        
        # Save dataset
        dataset_file = self.output_dir / "api_dataset.jsonl"
        with open(dataset_file, 'w') as f:
            for sample in samples:
                f.write(json.dumps(asdict(sample)) + '\n')
        
        logger.info(f"Created API dataset with {len(samples)} samples")
        return str(dataset_file)
    
    def create_synthetic_samples(self, base_samples: List[DocumentationSample], 
                                count: int = 1000) -> List[DocumentationSample]:
        """Generate synthetic training samples using templates and variations"""
        synthetic_samples = []
        
        # Load templates from guided approach
        from ..utils.guide_generator import guideGenerator
        
        guide_types = ['readme', 'api', 'changelog', 'documentation']
        
        for i in range(count):
            guide_type = guide_types[i % len(guide_types)]
            
            # Create synthetic context
            context = {
                'projectName': f'SampleProject{i}',
                'packageName': f'sample-package-{i}',
                'framework': ['React', 'Vue', 'Angular'][i % 3],
                'language': ['JavaScript', 'Python', 'Go'][i % 3]
            }
            
            # Generate guide
            guide = guideGenerator.generateArchitecturalGuide({
                'type': guide_type,
                'context': context
            })
            
            # Convert guide to training format
            synthetic_context = f"Project: {context['projectName']}\nFramework: {context['framework']}\nLanguage: {context['language']}"
            synthetic_target = guideGenerator.formatGuideAsMarkdown(guide)
            
            sample = DocumentationSample(
                repo_id=f"synthetic/{i}",
                doc_type=guide_type,
                context=synthetic_context,
                target=synthetic_target,
                quality_score=0.9,  # High quality for synthetic
                metadata={'synthetic': True, **context}
            )
            
            synthetic_samples.append(sample)
        
        return synthetic_samples

async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Build CodeContext AI training datasets")
    parser.add_argument("--github-token", required=True, help="GitHub API token")
    parser.add_argument("--output", default="./data", help="Output directory")
    parser.add_argument("--readme-samples", type=int, default=10000, help="README samples")
    parser.add_argument("--api-samples", type=int, default=5000, help="API samples")
    parser.add_argument("--synthetic-samples", type=int, default=2000, help="Synthetic samples")
    
    args = parser.parse_args()
    
    builder = DatasetBuilder(args.github_token, args.output)
    
    # Build datasets
    tasks = [
        builder.build_readme_dataset(args.readme_samples),
        builder.build_api_dataset(args.api_samples)
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Dataset building failed: {result}")
        else:
            logger.info(f"Dataset saved: {result}")
    
    logger.info("Dataset preparation complete")

if __name__ == "__main__":
    asyncio.run(main())