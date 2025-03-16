import os
import yaml
import shutil
import warnings
import concurrent.futures
from typing import List, Dict, Any, Optional, Callable
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    UnstructuredWordDocumentLoader,
    CSVLoader,
    UnstructuredMarkdownLoader
)
from langchain.text_splitter import (
    MarkdownTextSplitter, 
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter
)

# Use updated imports to fix deprecation warnings
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    HUGGINGFACE_AVAILABLE = False
    warnings.warn(
        "Using deprecated HuggingFaceEmbeddings from langchain_community. "
        "Install langchain-huggingface for the updated version."
    )

try:
    from langchain_chroma import Chroma
    CHROMA_AVAILABLE = True
except ImportError:
    from langchain_community.vectorstores import Chroma
    CHROMA_AVAILABLE = False
    warnings.warn(
        "Using deprecated Chroma from langchain_community. "
        "Install langchain-chroma for the updated version."
    )

# Conditionally import Azure clients
try:
    from src.azure_clients import AzureOpenAIEmbeddings, AzureInferenceEmbeddings
except ImportError:
    # Create placeholder classes to avoid errors when Azure modules are not available
    class AzureOpenAIEmbeddings:
        def __init__(self, **kwargs):
            raise ImportError("Azure OpenAI modules not available. Please install with 'pip install langchain-openai'")
    
    class AzureInferenceEmbeddings:
        def __init__(self, **kwargs):
            raise ImportError("Azure AI Inference modules not available. Please install with 'pip install azure-ai-inference'")

class DocumentProcessor:
    """Base class for document processing strategies"""
    
    def __init__(self, text_splitter_config: Dict[str, Any]):
        """Initialize with text splitter configuration"""
        self.text_splitter_config = text_splitter_config
        self._init_text_splitter()
    
    def _init_text_splitter(self):
        """Initialize the text splitter based on configuration"""
        chunk_size = self.text_splitter_config.get('chunk_size', 1000)
        chunk_overlap = self.text_splitter_config.get('chunk_overlap', 200)
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def process(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process the documents"""
        return self.text_splitter.split_documents(documents)

class ChunkSemanticProcessor(DocumentProcessor):
    """Process documents by splitting into semantic chunks"""
    
    def process(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split documents into semantic chunks"""
        # Default to basic splitting
        return super().process(documents)

class SummaryProcessor(DocumentProcessor):
    """Process documents by generating summaries"""
    
    def __init__(self, text_splitter_config: Dict[str, Any], llm=None):
        super().__init__(text_splitter_config)
        self.llm = llm
    
    def process(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate summaries for documents"""
        # First, split documents into manageable chunks
        chunks = super().process(documents)
        
        # For demonstration only - would need actual LLM integration to generate summaries
        # This would be replaced with actual summary generation using the LLM
        if self.llm:
            # This is a placeholder - in a real implementation, we would use the LLM
            # to generate summaries for each chunk
            pass
            
        return chunks

class HybridProcessor(DocumentProcessor):
    """Process documents using both chunk-based and summary-based approaches"""
    
    def __init__(self, text_splitter_config: Dict[str, Any], llm=None):
        super().__init__(text_splitter_config)
        self.llm = llm
        self.chunk_processor = ChunkSemanticProcessor(text_splitter_config)
        self.summary_processor = SummaryProcessor(text_splitter_config, llm)
    
    def process(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process using both chunk and summary methods"""
        chunks = self.chunk_processor.process(documents)
        
        # In a real implementation, we would generate summaries and add them as additional chunks
        # with appropriate metadata to distinguish them
        
        return chunks

class StructuredProcessor(DocumentProcessor):
    """Process structured documents like CSV or JSON"""
    
    def __init__(self, text_splitter_config: Dict[str, Any], **kwargs):
        super().__init__(text_splitter_config)
        self.kwargs = kwargs
    
    def process(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process structured documents"""
        # For structured data, we might want custom processing
        # For now, we'll use the default text splitter
        return super().process(documents)

class DocumentIndexer:
    """Handles the pipeline for loading, splitting, embedding, and storing documents."""
    
    def __init__(self, config_path: str, reset_db: bool = False):
        """
        Initialize with configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            reset_db: Whether to reset the vector database
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_dir = self.config['data_directory']
        self.db_dir = self.config['vector_db_directory']
        
        # Make sure directories exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Reset vector DB if requested
        if reset_db and os.path.exists(self.db_dir):
            print(f"Resetting vector database at {self.db_dir}")
            shutil.rmtree(self.db_dir)
        
        os.makedirs(self.db_dir, exist_ok=True)
        
        # Initialize embedding model
        self._initialize_embedding_model()
        
        # Initialize default text splitter
        self.default_text_splitter_config = self.config['text_splitter']
        
        # Configure file loaders and processors
        self._configure_file_handlers()
        
        # Initialize parallel processing settings
        self.parallel_config = self.config.get('parallel_processing', {})
        self.parallel_enabled = self.parallel_config.get('enabled', False)
        self.max_workers = self.parallel_config.get('max_workers', 4)
        
    def _initialize_embedding_model(self):
        """Initialize the embedding model based on configuration."""
        embedding_config = self.config['embedding_model']
        
        # Check embedding model type
        embedding_type = embedding_config.get('type', 'huggingface')
        
        if embedding_type == 'azure_inference':
            # Use Azure AI Inference SDK embeddings
            print("Using Azure AI Inference embeddings")
            self.embedding_model = AzureInferenceEmbeddings(
                endpoint=embedding_config.get('endpoint'),
                model_name=embedding_config.get('model_name')
            )
        elif embedding_type == 'azure_openai':
            # Use Azure OpenAI embeddings
            print("Using Azure OpenAI embeddings")
            self.embedding_model = AzureOpenAIEmbeddings(
                azure_endpoint=embedding_config.get('azure_endpoint'),
                azure_deployment=embedding_config.get('azure_deployment'),
                api_version=embedding_config.get('api_version', '2023-05-15')
            )
        else:
            # Default to HuggingFace embeddings
            print("Using HuggingFace embeddings")
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=embedding_config.get('name', 'sentence-transformers/all-mpnet-base-v2'),
                model_kwargs=embedding_config.get('kwargs', {})
            )
    
    def _configure_file_handlers(self):
        """Configure file loaders and processors for different file types"""
        # Define file loaders by extension
        self.file_loaders = {
            '.pdf': PyPDFLoader,
            '.txt': TextLoader,
            '.docx': UnstructuredWordDocumentLoader,
            '.doc': UnstructuredWordDocumentLoader,
            '.csv': CSVLoader,
            '.md': UnstructuredMarkdownLoader,
        }
        
        # Get file processor configurations
        self.file_processors_config = self.config.get('file_processors', {})
        
    def _get_processor_for_file_type(self, file_extension: str) -> DocumentProcessor:
        """Get the appropriate processor for the file type"""
        # Remove the dot from extension if present
        if file_extension.startswith('.'):
            file_extension = file_extension[1:]
        
        # Get processor config for this file type, or use default
        processor_config = self.file_processors_config.get(file_extension, {})
        processor_type = processor_config.get('processor', 'chunk_semantic')
        
        # Get text splitter config, fallback to default if not specified
        text_splitter_config = processor_config.get('text_splitter', self.default_text_splitter_config)
        
        # Create and return the appropriate processor
        if processor_type == 'summary':
            return SummaryProcessor(text_splitter_config)
        elif processor_type == 'hybrid':
            return HybridProcessor(text_splitter_config)
        elif processor_type == 'structured':
            return StructuredProcessor(text_splitter_config, 
                                       separator=processor_config.get('separator', ','))
        else:  # Default to chunk_semantic
            return ChunkSemanticProcessor(text_splitter_config)
    
    def _get_loader_for_file(self, file_path: str):
        """Get the appropriate loader for the file type"""
        _, file_extension = os.path.splitext(file_path.lower())
        
        if file_extension in self.file_loaders:
            return self.file_loaders[file_extension](file_path)
        else:
            # Default to text loader if extension not recognized
            print(f"Warning: No specific loader for {file_extension}, using TextLoader as default")
            return TextLoader(file_path)
    
    def load_and_process_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Load and process a single file"""
        # Get file extension
        _, file_extension = os.path.splitext(file_path.lower())
        
        # Get loader for this file type
        try:
            loader = self._get_loader_for_file(file_path)
            documents = loader.load()
            
            # Add metadata
            filename = os.path.basename(file_path)
            for i, doc in enumerate(documents):
                doc.metadata.update({
                    'source': filename,
                    'page': i,
                    'file_path': file_path,
                    'file_type': file_extension[1:] if file_extension.startswith('.') else file_extension
                })
            
            # Process documents based on file type
            processor = self._get_processor_for_file_type(file_extension)
            processed_docs = processor.process(documents)
            
            print(f"Processed {file_path}: {len(documents)} original pages, {len(processed_docs)} chunks")
            return processed_docs
            
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return []
    
    def load_documents(self) -> List[Dict[str, Any]]:
        """Load all supported documents from the data directory."""
        all_documents = []
        files_to_process = []
        
        # Collect all supported files
        for filename in os.listdir(self.data_dir):
            file_path = os.path.join(self.data_dir, filename)
            if not os.path.isfile(file_path):
                continue
                
            _, file_extension = os.path.splitext(filename.lower())
            if file_extension in self.file_loaders:
                files_to_process.append(file_path)
        
        print(f"Found {len(files_to_process)} files to process")
        
        # Process files - either in parallel or sequentially
        if self.parallel_enabled and len(files_to_process) > 1:
            print(f"Processing files in parallel with {self.max_workers} workers")
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all files for processing
                future_to_file = {
                    executor.submit(self.load_and_process_file, file_path): file_path 
                    for file_path in files_to_process
                }
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        processed_docs = future.result()
                        all_documents.extend(processed_docs)
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
        else:
            print("Processing files sequentially")
            for file_path in files_to_process:
                processed_docs = self.load_and_process_file(file_path)
                all_documents.extend(processed_docs)
        
        return all_documents
    
    def create_vector_store(self, documents: List[Dict[str, Any]]) -> Chroma:
        """Create or update the vector store with document embeddings."""
        if not documents:
            print("Warning: No documents to index")
            return None
            
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            persist_directory=self.db_dir
        )
        return vector_store
    
    def index(self, reset_db: bool = False) -> None:
        """
        Run the full indexing pipeline.
        
        Args:
            reset_db: Whether to reset the vector database
        """
        if reset_db and os.path.exists(self.db_dir):
            print(f"Resetting vector database at {self.db_dir}")
            shutil.rmtree(self.db_dir)
            os.makedirs(self.db_dir, exist_ok=True)
            
        print("Loading and processing documents...")
        documents = self.load_documents()
        print(f"Total processed documents: {len(documents)} chunks")
        
        if documents:
            print("Computing embeddings and storing in vector database...")
            try:
                self.create_vector_store(documents)
                print("Indexing complete!")
            except Exception as e:
                print(f"Error during indexing: {e}")
                print("Try running with reset_db=True to recreate the vector store.")
        else:
            print("No documents were processed. Indexing aborted.")
