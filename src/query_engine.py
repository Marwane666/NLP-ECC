import yaml
from typing import List, Dict, Any, Tuple
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

class QueryEngine:
    """Handles querying the vector store for relevant documents."""
    
    def __init__(self, config_path: str):
        """Initialize with configuration from YAML file."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.db_dir = self.config['vector_db_directory']
        self.k = self.config['retrieval']['top_k']
        
        # Initialize embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.config['embedding_model']['name'],
            model_kwargs=self.config['embedding_model'].get('kwargs', {})
        )
        
        # Load vector store
        self.vector_store = Chroma(
            persist_directory=self.db_dir,
            embedding_function=self.embedding_model
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
