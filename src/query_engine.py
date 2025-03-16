import yaml
import warnings
from typing import List, Dict, Any, Tuple

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

class QueryEngine:
    """Handles querying the vector store for relevant documents."""
    
    def __init__(self, config_path: str):
        """Initialize with configuration from YAML file."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.db_dir = self.config['vector_db_directory']
        self.k = self.config['retrieval']['top_k']
        
        # Initialize embedding model
        self._initialize_embedding_model()
        
        # Load vector store
        self.vector_store = Chroma(
            persist_directory=self.db_dir,
            embedding_function=self.embedding_model
        )
    
    def _initialize_embedding_model(self):
        """Initialize the embedding model based on configuration."""
        embedding_config = self.config['embedding_model']
        
        # Check embedding model type
        embedding_type = embedding_config.get('type', 'huggingface')
        
        if embedding_type == 'azure_inference':
            # Use Azure AI Inference SDK embeddings
            self.embedding_model = AzureInferenceEmbeddings(
                endpoint=embedding_config.get('endpoint'),
                model_name=embedding_config.get('model_name')
            )
        elif embedding_type == 'azure_openai':
            # Use Azure OpenAI embeddings
            self.embedding_model = AzureOpenAIEmbeddings(
                azure_endpoint=embedding_config.get('azure_endpoint'),
                azure_deployment=embedding_config.get('azure_deployment'),
                api_version=embedding_config.get('api_version', '2023-05-15')
            )
        else:
            # Default to HuggingFace embeddings
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=embedding_config['name'],
                model_kwargs=embedding_config.get('kwargs', {})
            )
    
    def query_vector_store(self, query: str) -> List[Tuple[Any, float]]:
        """
        Query the vector store for relevant documents.
        
        Args:
            query: The user query string
            
        Returns:
            List of tuples containing (document, similarity_score)
        """
        # Retrieve documents with similarity scores
        docs_with_scores = self.vector_store.similarity_search_with_score(
            query=query,
            k=self.k
        )
        
        return docs_with_scores
    
    def format_results(self, results: List[Tuple[Any, float]]) -> List[Dict[str, Any]]:
        """Format the query results for better readability."""
        formatted_results = []
        
        for doc, score in results:
            formatted_results.append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                'similarity_score': score
            })
            
        return formatted_results
