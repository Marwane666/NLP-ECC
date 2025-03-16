import yaml
import os
from typing import Dict, Any, List
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.llms import CTransformers
import torch

from src.prompt_template import get_qa_prompt
from src.query_engine import QueryEngine

class QASystem:
    """Question-answering system that combines retrieval with language model generation."""
    
    def __init__(self, config_path: str):
        """Initialize with configuration from YAML file."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize query engine
        self.query_engine = QueryEngine(config_path)
        
        # Initialize LLM
        self._initialize_llm()
        
        # Create LLM chain with prompt template
        self.prompt = get_qa_prompt()
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def _initialize_llm(self):
        """Initialize the language model based on configuration."""
        llm_config = self.config['llm']
        
        if llm_config.get('use_gpu', False) and torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        
        # Using CTransformers for local inference with GGML models
        self.llm = CTransformers(
            model=llm_config['model_path'],
            model_type=llm_config['model_type'],
            config={
                'max_new_tokens': llm_config.get('max_new_tokens', 512),
                'temperature': llm_config.get('temperature', 0.7),
                'context_length': llm_config.get('context_length', 2048),
                'gpu_layers': llm_config.get('gpu_layers', 0) if device == 'cuda' else 0
            }
        )
    
    def _prepare_context(self, query: str) -> str:
        """Retrieve relevant documents and prepare context for the LLM."""
        results = self.query_engine.query_vector_store(query)
        
        # Combine all relevant documents into a single context string
        context_parts = []
        for doc, score in results:
            # Add document content and source information
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'Unknown')
            context_parts.append(f"Document: {source}, Page: {page}\n{doc.page_content}")
        
        return "\n\n".join(context_parts)
    
    def generate_response(self, query: str) -> Dict[str, Any]:
        """
        Generate a response to the given query using the RAG approach.
        
        Args:
            query: The user's question
            
        Returns:
            Dictionary containing the response and additional information
        """
        # Prepare context from relevant documents
        context = self._prepare_context(query)
        
        # Generate response using the LLM
        response = self.chain.run(context=context, query=query)
        
        return {
            'query': query,
            'response': response,
            'context_used': context
        }
