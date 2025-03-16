import os
import yaml
from typing import List, Dict, Any
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import MarkdownTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

class DocumentIndexer:
    """Handles the pipeline for loading, splitting, embedding, and storing documents."""
    
    def __init__(self, config_path: str):
        """Initialize with configuration from YAML file."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_dir = self.config['data_directory']
        self.db_dir = self.config['vector_db_directory']
        
        # Initialize embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.config['embedding_model']['name'],
            model_kwargs=self.config['embedding_model'].get('kwargs', {})
        )
        
        # Initialize text splitter
        self.text_splitter = MarkdownTextSplitter(
            chunk_size=self.config['text_splitter']['chunk_size'],
            chunk_overlap=self.config['text_splitter']['chunk_overlap']
        )
        
        # Make sure directories exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.db_dir, exist_ok=True)
    
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
    
    def index(self) -> None:
        """Run the full indexing pipeline."""
        print("Loading documents...")
        documents = self.load_documents()
        print(f"Loaded {len(documents)} document pages.")
        
        print("Splitting documents...")
        chunks = self.split_documents(documents)
        print(f"Created {len(chunks)} document chunks.")
        
        print("Computing embeddings and storing in vector database...")
        self.create_vector_store(chunks)
        print("Indexing complete!")
