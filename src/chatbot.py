import yaml
from typing import Dict, Any, List
from langchain.chains import LLMChain

from src.prompt_template import get_chat_prompt
from src.query_engine import QueryEngine

class Chatbot:
    """Chatbot system that maintains conversation history and provides context-aware responses."""
    
    def __init__(self, config_path: str, llm):
        """
        Initialize the chatbot.
        
        Args:
            config_path: Path to configuration file
            llm: Language model instance (shared with QA system)
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize query engine
        self.query_engine = QueryEngine(config_path)
        
        # Use shared LLM instance
        self.llm = llm
        
        # Create LLM chain with chat prompt template
        self.prompt = get_chat_prompt()
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
        
        # Initialize conversation history
        self.history = []
    
    def _prepare_context(self, query: str) -> str:
        """Retrieve relevant documents and prepare context for the current query."""
        results = self.query_engine.query_vector_store(query)
        
        # Combine all relevant documents into a single context string
        context_parts = []
        for doc, score in results:
            # Add document content and source information
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'Unknown')
            context_parts.append(f"Document: {source}, Page: {page}\n{doc.page_content}")
        
        return "\n\n".join(context_parts)
    
    def _format_history(self) -> str:
        """Format conversation history for inclusion in the prompt."""
        formatted_history = []
        for entry in self.history:
            formatted_history.append(f"User: {entry['user']}")
            formatted_history.append(f"Assistant: {entry['assistant']}")
        
        return "\n".join(formatted_history)
    
    def chat(self, user_input: str) -> str:
        """
        Process user input and generate a response, maintaining conversation history.
        
        Args:
            user_input: The user's message or question
            
        Returns:
            The chatbot's response
        """
        # Get context for the current query
        context = self._prepare_context(user_input)
        
        # Format the conversation history
        history_text = self._format_history()
        
        # Generate response using the LLM
        response = self.chain.run(
            history=history_text,
            context=context,
            query=user_input
        )
        
        # Update conversation history
        self.history.append({
            'user': user_input,
            'assistant': response
        })
        
        # Limit history length to prevent context overflow
        max_history = self.config.get('max_chat_history', 10)
        if len(self.history) > max_history:
            self.history = self.history[-max_history:]
        
        return response
    
    def reset(self) -> None:
        """Reset the conversation history."""
        self.history = []
