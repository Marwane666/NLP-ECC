import yaml
import os
from typing import Dict, Any, List
from langchain.chains import LLMChain

from src.prompt_template import get_qa_prompt
from src.query_engine import QueryEngine

# Conditionally import Azure clients
try:
    from src.azure_clients import get_azure_chat_openai, get_azure_inference_chat_llm
except ImportError:
    # Create placeholder functions to avoid errors when Azure modules are not available
    def get_azure_chat_openai(config):
        raise ImportError("Azure OpenAI modules not available. Please install with 'pip install langchain-openai'")
    
    def get_azure_inference_chat_llm(config):
        raise ImportError("Azure AI Inference modules not available. Please install with 'pip install azure-ai-inference'")

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
        print("QA System initialized successfully with Azure AI Inference LLM")
    
    def _initialize_llm(self):
        """Initialize the language model based on configuration."""
        llm_config = self.config['llm']
        
        # Always use Azure AI Inference
        print("Initializing Azure AI Inference LLM...")
        self.llm = get_azure_inference_chat_llm(llm_config)
        
        # Verify the LLM was initialized
        if not self.llm:
            raise ValueError("Azure AI Inference LLM initialization failed")
    
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
        try:
            # Prepare context from relevant documents
            context = self._prepare_context(query)
            print(f"Retrieved context of length: {len(context)}")
            
            # Generate response using the LLM
            print(f"Generating response to query: {query}")
            response = self.chain.run(context=context, query=query)
            print("Response generated successfully")
            
            return {
                'query': query,
                'response': response,
                'context_used': context
            }
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return {
                'query': query,
                'response': f"I encountered an error: {str(e)}. Please try again.",
                'context_used': "Error retrieving context"
            }
