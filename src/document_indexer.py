import os
import yaml
import shutil
import warnings
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import MarkdownTextSplitter

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
        
        # Initialize text splitter
        self.text_splitter = MarkdownTextSplitter(
            chunk_size=self.config['text_splitter']['chunk_size'],
            chunk_overlap=self.config['text_splitter']['chunk_overlap']
        )
        
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
    
    def load_documents(self) -> List[Dict[str, Any]]:
        """Load all PDF documents from the data directory."""
        documents = []
        for filename in os.listdir(self.data_dir):
            if filename.lower().endswith('.pdf'):
                file_path = os.path.join(self.data_dir, filename)
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                
                # Add metadata
                for i, doc in enumerate(docs):
                    doc.metadata.update({
                        'source': filename,
                        'page': i,
                        'file_path': file_path
                    })
                
                documents.extend(docs)
        
        return documents
    
    def split_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split documents into smaller chunks while preserving metadata."""
        return self.text_splitter.split_documents(documents)
    
    def create_vector_store(self, documents: List[Dict[str, Any]]) -> Chroma:
        """Create or update the vector store with document embeddings."""
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
            
        print("Loading documents...")
        documents = self.load_documents()
        print(f"Loaded {len(documents)} document pages.")
        
        print("Splitting documents...")
        chunks = self.split_documents(documents)
        print(f"Created {len(chunks)} document chunks.")
        
        print("Computing embeddings and storing in vector database...")
        try:
            self.create_vector_store(chunks)
            print("Indexing complete!")
        except Exception as e:
            print(f"Error during indexing: {e}")
            print("Try running with reset_db=True to recreate the vector store.")
