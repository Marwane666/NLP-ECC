import yaml
import os
import traceback
from typing import List, Dict, Any, Tuple, Optional
from langchain_community.vectorstores import Chroma

# Use updated imports to fix deprecation warnings
try:
    from langchain_chroma import Chroma
    CHROMA_AVAILABLE = True
except ImportError:
    from langchain_community.vectorstores import Chroma
    CHROMA_AVAILABLE = False

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    HUGGINGFACE_AVAILABLE = False

# Conditionally import Azure clients
try:
    from src.azure_clients import AzureOpenAIEmbeddings, AzureInferenceEmbeddings
except ImportError:
    # Placeholder classes to avoid errors when Azure modules are not available
    class AzureOpenAIEmbeddings:
        def __init__(self, **kwargs):
            raise ImportError("Azure OpenAI modules not available. Please install with 'pip install langchain-openai'")
    
    class AzureInferenceEmbeddings:
        def __init__(self, **kwargs):
            raise ImportError("Azure AI Inference modules not available. Please install with 'pip install azure-ai-inference'")


class QueryEngine:
    """Handles document retrieval from the vector store."""
    
    def __init__(self, config_path: str):
        """
        Initialize with configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.db_dir = self.config['vector_db_directory']
        self.top_k = self.config['retrieval'].get('top_k', 5)
        
        # Initialize embedding model
        self._initialize_embedding_model()
        
        # Check if vector store exists
        if not os.path.exists(self.db_dir):
            print(f"Warning: Vector store directory {self.db_dir} does not exist.")
            self.vector_store = None
        else:
            try:
                self.vector_store = Chroma(
                    persist_directory=self.db_dir,
                    embedding_function=self.embedding_model
                )
            except Exception as e:
                print(f"Error loading vector store: {e}")
                self.vector_store = None
        
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
    
    def query_vector_store(self, query: str) -> List[Tuple[Any, float]]:
        """
        Query the vector store and return relevant documents with similarity scores.
        
        Args:
            query: The query text
            
        Returns:
            List of tuples (document, similarity_score)
        """
        if not self.vector_store:
            print("Vector store is not initialized.")
            return []
            
        try:
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=self.top_k
            )
            return results
        except Exception as e:
            print(f"Error querying vector store: {e}")
            return []
    
    def get_document_metadata(self) -> List[Dict[str, Any]]:
        """
        Get metadata for all documents in the vector store.
        
        Returns:
            List of document metadata dictionaries
        """
        if not self.vector_store:
            print("Vector store is not initialized.")
            return []
            
        try:
            # Use the Chroma client's get_collection method to access metadata
            collection = self.vector_store._collection
            
            # Get all metadata
            result = collection.get()
            
            # Extract and return just the metadata
            if result and 'metadatas' in result:
                return result['metadatas']
            else:
                return []
                
        except Exception as e:
            print(f"Error retrieving document metadata: {e}")
            return []

    def delete_documents(self, filter_dict: Dict[str, Any]) -> bool:
        """
        Delete documents from the vector store based on filter criteria.
        
        Args:
            filter_dict: Dictionary of metadata fields to match for deletion
            
        Returns:
            True if deletion was successful, False otherwise
        """
        if not self.vector_store:
            print("Vector store is not initialized.")
            return False
            
        try:
            # Get a reference to the underlying collection
            collection = self.vector_store._collection
            
            # First get IDs of documents matching the filter
            if not filter_dict:
                print("Warning: No filter provided. This would delete all documents.")
                return False
            
            # Convert filter dict to where clause
            where_clause = {}
            for key, value in filter_dict.items():
                where_clause[f"metadata.{key}"] = value
            
            # Get matching documents
            result = collection.get(where=where_clause)
            
            if not result or 'ids' not in result or not result['ids']:
                print("No matching documents found.")
                return False
            
            # Delete the matching documents
            collection.delete(ids=result['ids'])
            print(f"Deleted {len(result['ids'])} documents matching filter: {filter_dict}")
            
            # Force persistence
            self.vector_store.persist()
            
            # Output IDs for debugging
            print(f"Deleted document IDs: {result['ids'][:5]}...")
            
            return True
        except Exception as e:
            print(f"Error deleting documents: {str(e)}")
            traceback.print_exc()
            return False
