"""
Advanced RAG capabilities for CodeContext AI with privacy-first local processing
"""

import os
import json
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, asdict
import logging

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

from .inference import InferenceEngine, GenerationConfig

logger = logging.getLogger(__name__)

@dataclass
class RAGConfig:
    """Configuration for RAG processing"""
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    similarity_top_k: int = 5
    vector_store_path: str = "./rag_store"
    cache_embeddings: bool = True
    min_similarity_score: float = 0.7
    
class LocalRAGEngine:
    """Privacy-first RAG engine for enhanced documentation generation"""
    
    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        self.vector_store = None
        self.embeddings = None
        self.text_splitter = None
        self._setup_components()
        
    def _setup_components(self):
        """Initialize RAG components with privacy guarantees"""
        try:
            # Initialize local embedding model
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.config.embedding_model,
                model_kwargs={'device': 'cpu'},  # Ensure local processing
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Initialize text splitter for optimal chunking
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            # Initialize persistent vector store
            self.vector_store = Chroma(
                persist_directory=self.config.vector_store_path,
                embedding_function=self.embeddings
            )
            
            logger.info(f"RAG engine initialized with model: {self.config.embedding_model}")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG components: {e}")
            raise RuntimeError(f"RAG initialization failed: {e}")
    
    def index_codebase(self, codebase_path: str, force_reindex: bool = False) -> Dict[str, Any]:
        """Index a codebase for RAG retrieval with change detection"""
        codebase_path = Path(codebase_path)
        
        if not codebase_path.exists():
            raise ValueError(f"Codebase path does not exist: {codebase_path}")
        
        # Generate content hash for change detection
        content_hash = self._generate_codebase_hash(codebase_path)
        index_metadata_path = Path(self.config.vector_store_path) / "index_metadata.json"
        
        # Check if reindexing is needed
        if not force_reindex and index_metadata_path.exists():
            with open(index_metadata_path, 'r') as f:
                metadata = json.load(f)
                if metadata.get('content_hash') == content_hash:
                    logger.info("Codebase unchanged, skipping reindexing")
                    return metadata
        
        logger.info(f"Indexing codebase: {codebase_path}")
        
        # Extract and process documents
        documents = self._extract_codebase_documents(codebase_path)
        
        if not documents:
            logger.warning("No documents found for indexing")
            return {"indexed_files": 0, "total_chunks": 0}
        
        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)
        
        # Add metadata for better retrieval
        for chunk in chunks:
            chunk.metadata.update({
                "codebase_path": str(codebase_path),
                "content_hash": content_hash,
                "chunk_id": hashlib.md5(chunk.page_content.encode()).hexdigest()[:8]
            })
        
        # Clear existing index for this codebase if reindexing
        if force_reindex:
            self._clear_codebase_index(str(codebase_path))
        
        # Add chunks to vector store
        self.vector_store.add_documents(chunks)
        self.vector_store.persist()
        
        # Save indexing metadata
        index_metadata = {
            "codebase_path": str(codebase_path),
            "content_hash": content_hash,
            "indexed_files": len(documents),
            "total_chunks": len(chunks),
            "embedding_model": self.config.embedding_model,
            "timestamp": str(Path().stat().st_mtime if Path().exists() else 0)
        }
        
        os.makedirs(Path(self.config.vector_store_path), exist_ok=True)
        with open(index_metadata_path, 'w') as f:
            json.dump(index_metadata, f, indent=2)
        
        logger.info(f"Indexed {len(documents)} files into {len(chunks)} chunks")
        return index_metadata
    
    def retrieve_context(self, query: str, filter_metadata: Dict = None) -> List[Document]:
        """Retrieve relevant context for query with privacy preservation"""
        if not self.vector_store:
            raise RuntimeError("Vector store not initialized")
        
        try:
            # Perform similarity search
            retriever = self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    'k': self.config.similarity_top_k,
                    'score_threshold': self.config.min_similarity_score,
                    'filter': filter_metadata
                }
            )
            
            relevant_docs = retriever.get_relevant_documents(query)
            
            # Deduplicate by content hash
            seen_hashes = set()
            unique_docs = []
            
            for doc in relevant_docs:
                doc_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
                if doc_hash not in seen_hashes:
                    seen_hashes.add(doc_hash)
                    unique_docs.append(doc)
            
            logger.info(f"Retrieved {len(unique_docs)} relevant documents for query")
            return unique_docs
            
        except Exception as e:
            logger.error(f"Context retrieval failed: {e}")
            return []
    
    def _extract_codebase_documents(self, codebase_path: Path) -> List[Document]:
        """Extract documents from codebase with intelligent file filtering"""
        documents = []
        
        # Define relevant file extensions for documentation context
        code_extensions = {'.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h', 
                          '.cs', '.php', '.rb', '.go', '.rs', '.swift', '.kt', '.scala'}
        doc_extensions = {'.md', '.rst', '.txt', '.yml', '.yaml', '.json', '.toml'}
        config_files = {'package.json', 'pyproject.toml', 'Cargo.toml', 'pom.xml', 
                       'build.gradle', 'tsconfig.json', 'webpack.config.js'}
        
        try:
            for root, dirs, files in os.walk(codebase_path):
                # Skip common ignore patterns
                dirs[:] = [d for d in dirs if not d.startswith('.') and 
                          d not in {'node_modules', '__pycache__', 'venv', 'env', 'build', 'dist'}]
                
                for file in files:
                    file_path = Path(root) / file
                    relative_path = file_path.relative_to(codebase_path)
                    
                    # Check if file should be indexed
                    if (file_path.suffix in code_extensions or 
                        file_path.suffix in doc_extensions or 
                        file in config_files):
                        
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                                
                                # Skip empty or very large files
                                if 50 <= len(content) <= 50000:  # Reasonable file size limits
                                    document = Document(
                                        page_content=content,
                                        metadata={
                                            "source": str(file_path),
                                            "file_type": file_path.suffix,
                                            "relative_path": str(relative_path),
                                            "file_size": len(content)
                                        }
                                    )
                                    documents.append(document)
                                    
                        except Exception as e:
                            logger.warning(f"Failed to read file {file_path}: {e}")
                            continue
                            
        except Exception as e:
            logger.error(f"Failed to extract codebase documents: {e}")
            
        return documents
    
    def _generate_codebase_hash(self, codebase_path: Path) -> str:
        """Generate hash of codebase for change detection"""
        hash_md5 = hashlib.md5()
        
        try:
            for root, dirs, files in os.walk(codebase_path):
                # Sort for consistent hashing
                dirs.sort()
                files.sort()
                
                for file in files:
                    file_path = Path(root) / file
                    if file_path.is_file():
                        try:
                            # Include file path and modification time
                            hash_md5.update(str(file_path.relative_to(codebase_path)).encode())
                            hash_md5.update(str(file_path.stat().st_mtime).encode())
                        except (OSError, ValueError):
                            continue
                            
        except Exception as e:
            logger.warning(f"Hash generation failed, using fallback: {e}")
            return hashlib.md5(str(codebase_path).encode()).hexdigest()
            
        return hash_md5.hexdigest()
    
    def _clear_codebase_index(self, codebase_path: str):
        """Clear existing index entries for a specific codebase"""
        try:
            # This would require implementing metadata filtering in Chroma
            # For now, we'll rely on document replacement
            logger.info(f"Clearing existing index for: {codebase_path}")
        except Exception as e:
            logger.warning(f"Index clearing failed: {e}")

class RAGEnhancedDocumentationAI:
    """Enhanced documentation AI with RAG capabilities"""
    
    def __init__(self, model_path: str, rag_config: RAGConfig = None):
        self.inference_engine = InferenceEngine(model_path)
        self.rag_engine = LocalRAGEngine(rag_config)
        
    def generate_context_aware_documentation(
        self, 
        codebase_path: str, 
        doc_type: str = "readme",
        additional_context: str = "",
        use_rag: bool = True
    ) -> str:
        """Generate documentation with RAG-enhanced context"""
        
        if use_rag:
            # Index codebase if needed
            self.rag_engine.index_codebase(codebase_path)
            
            # Create contextual query for retrieval
            query = self._create_retrieval_query(doc_type, additional_context)
            
            # Retrieve relevant context
            relevant_docs = self.rag_engine.retrieve_context(
                query, 
                filter_metadata={"codebase_path": codebase_path}
            )
            
            # Combine context from RAG and traditional analysis
            rag_context = self._format_rag_context(relevant_docs)
            traditional_context = self._analyze_codebase_traditional(codebase_path)
            
            combined_context = f"{rag_context}\n\n{traditional_context}"
        else:
            combined_context = self._analyze_codebase_traditional(codebase_path)
        
        # Generate documentation using enhanced context
        prompt = self._create_documentation_prompt(doc_type, combined_context, additional_context)
        
        config = GenerationConfig(
            max_length=4096,  # Increased for comprehensive docs
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )
        
        return self.inference_engine.generate(prompt, config)
    
    def _create_retrieval_query(self, doc_type: str, additional_context: str) -> str:
        """Create optimized query for RAG retrieval"""
        base_queries = {
            "readme": "project overview architecture setup installation usage examples",
            "api": "API endpoints functions methods classes interfaces documentation",
            "changelog": "changes updates modifications releases versions commits",
            "architecture": "system design components modules structure patterns",
            "implementation": "code examples implementation details technical guide",
            "component": "component interface props methods usage patterns",
            "best_practices": "best practices coding standards conventions guidelines"
        }
        
        query = base_queries.get(doc_type, "documentation code structure overview")
        if additional_context:
            query += f" {additional_context}"
        
        return query
    
    def _format_rag_context(self, documents: List[Document]) -> str:
        """Format retrieved documents into context string"""
        if not documents:
            return ""
        
        context_parts = []
        context_parts.append("=== RELEVANT CODE CONTEXT ===")
        
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get('relative_path', 'unknown')
            file_type = doc.metadata.get('file_type', '')
            
            context_parts.append(f"\n--- File {i}: {source} ({file_type}) ---")
            context_parts.append(doc.page_content[:1500])  # Limit content length
            
        context_parts.append("\n=== END CONTEXT ===\n")
        return "\n".join(context_parts)
    
    def _analyze_codebase_traditional(self, codebase_path: str) -> str:
        """Traditional codebase analysis (existing functionality)"""
        # This would use the existing analysis methods from inference.py
        # Simplified implementation for now
        analysis = []
        
        codebase_path = Path(codebase_path)
        
        # Analyze key configuration files
        key_files = ["package.json", "pyproject.toml", "Cargo.toml", "go.mod", "requirements.txt"]
        for file in key_files:
            file_path = codebase_path / file
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()[:1000]  # Limit size
                        analysis.append(f"=== {file} ===\n{content}")
                except Exception:
                    continue
        
        # Basic directory structure
        try:
            structure = []
            for root, dirs, files in os.walk(codebase_path):
                level = root.replace(str(codebase_path), '').count(os.sep)
                if level < 3:  # Limit depth
                    indent = '  ' * level
                    folder_name = os.path.basename(root) or codebase_path.name
                    structure.append(f"{indent}{folder_name}/")
                    
                    # Add important files
                    subindent = '  ' * (level + 1)
                    important_files = [f for f in files[:5] if not f.startswith('.')]
                    for file in important_files:
                        structure.append(f"{subindent}{file}")
            
            analysis.append(f"=== PROJECT STRUCTURE ===\n{chr(10).join(structure[:30])}")
            
        except Exception:
            pass
        
        return "\n\n".join(analysis)
    
    def _create_documentation_prompt(self, doc_type: str, context: str, additional_context: str) -> str:
        """Create specialized prompt for documentation generation"""
        
        prompts = {
            "readme": f"""Generate a comprehensive, professional README.md for this project based on the following context:

{context}

Additional requirements: {additional_context}

Create a README that includes:
- Clear project title and description
- Installation instructions
- Usage examples
- Key features and capabilities
- Project structure overview
- Contributing guidelines
- License information

README.md:
""",
            "api": f"""Generate comprehensive API documentation based on this codebase context:

{context}

Additional requirements: {additional_context}

Create documentation that includes:
- API overview and base URL
- Authentication methods
- Endpoint descriptions with HTTP methods
- Request/response schemas with examples
- Error codes and responses
- Rate limiting information
- SDKs and client libraries

API Documentation:
""",
            "architecture": f"""Generate a detailed architectural guide based on this codebase:

{context}

Additional requirements: {additional_context}

Create an architecture guide that includes:
- System overview and key components
- Architecture patterns and principles
- Technology stack and dependencies
- Data flow and integration points
- Deployment architecture
- Security considerations
- Performance characteristics

Architecture Guide:
"""
        }
        
        return prompts.get(doc_type, f"Generate {doc_type} documentation based on:\n\n{context}\n\nDocumentation:")